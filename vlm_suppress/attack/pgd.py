# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/attack/pgd.py
#
# Constrained Surrogate-Ensemble PGD — region-aware version.
#
# Key change from uniform version:
#   The l-inf projection is now per-pixel via an epsilon map built from
#   word_boxes. Text pixels use epsilon_text (large budget), background pixels
#   use epsilon_bg (small budget). This concentrates suppression energy on
#   text regions where the model extracts content.
#
# When cfg.region_aware=False (or word_boxes is empty), behaviour is
# identical to the original uniform projection — safe fallback.
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vlm_suppress.attack.constraints import compute_LH, constraint_satisfied, penalty_term
from vlm_suppress.attack.ensemble import compute_FM_ens
from vlm_suppress.attack.masks import build_epsilon_map, project_onto_region_ball
from vlm_suppress.attack.salience import build_salience_budget_map
from vlm_suppress.config import AttackConfig
from vlm_suppress.models.base import SurrogateModel


@dataclass
class StepLog:
    step:           int
    fm_ens:         float
    lh:             float
    penalty:        float
    objective:      float
    constraint_ok:  bool
    mu:             float


@dataclass
class AttackResult:
    image_id:   str
    transcript: str
    epsilon:    float          # effective text-region epsilon
    kappa:      float

    x_adv:                torch.Tensor   # (3, H, W) float32 [0,1] CPU
    constraint_satisfied: bool
    fm_ens_final:         float
    lh_final:             float

    # Whether region-aware mode was active for this run
    region_aware: bool

    trajectory: list[StepLog] = field(default_factory=list)


def run_attack(
    image_id:       str,
    x_orig:         torch.Tensor,          # (3, H, W) float32 [0,1] CPU
    transcript:     str,
    surrogates:     list[SurrogateModel],
    cfg:            AttackConfig,
    word_boxes:     list[list[int]] | None = None,
    font_embedder:  object = None,
    all_surrogates: list[SurrogateModel] | None = None,
) -> AttackResult:
    """
    Run the constrained PGD attack for one image.

    Region-aware mode (cfg.region_aware=True):
        Builds a per-pixel epsilon map from word_boxes.
        Text pixels  → cfg.epsilon_text  (large budget)
        Background   → cfg.epsilon_bg    (small budget)

    Uniform mode (cfg.region_aware=False):
        Uses cfg.epsilon globally for all pixels.
        Identical to the original implementation.
    """
    device   = surrogates[0].device
    x_orig_d = x_orig.to(device)
    H, W     = x_orig_d.shape[-2], x_orig_d.shape[-1]

    # ── Build epsilon map ──────────────────────────────────────────────────────
    use_region = cfg.region_aware and word_boxes is not None and len(word_boxes) > 0
    if use_region:
        # [ABLATION: uniform-text budget]
        eps_map = build_epsilon_map(
            height=H, width=W,
            word_boxes=word_boxes,
            epsilon_text=cfg.epsilon_text,
            epsilon_bg=cfg.epsilon_bg,
            dilation=cfg.mask_dilation,
            device=device,
        )
    else:
        # Uniform fallback — scalar broadcast over (3, H, W)
        eps_map = torch.full(
            (1, H, W), cfg.epsilon,
            dtype=torch.float32, device=device,
        )

    if cfg.salience_budget:
        if not word_boxes:
            import warnings
            warnings.warn(
                "salience_budget=True but word_boxes is empty — "
                "falling back to non-salience eps_map. "
                "Check that the dataset sample has bounding box annotations.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # [SALIENCE BUDGET] Phase 1: compute salience map before PGD loop
            # Select the surrogate pool for salience estimation.
            # cfg.salience_surrogate_indices picks specific models from the
            # full pool (opt + held-out); None falls back to opt pool only.
            _pool = all_surrogates if all_surrogates is not None else surrogates
            if cfg.salience_surrogate_indices is not None:
                sal_surrogates = [
                    _pool[i] for i in cfg.salience_surrogate_indices
                    if i < len(_pool)
                ]
                if not sal_surrogates:
                    sal_surrogates = surrogates
            else:
                sal_surrogates = surrogates

            alpha_weights = [1.0 / len(sal_surrogates)] * len(sal_surrogates)
            eps_map = build_salience_budget_map(
                image_tensor  = x_orig_d.unsqueeze(0),
                transcript    = transcript,
                word_boxes    = word_boxes,
                surrogates    = sal_surrogates,
                alpha_weights = alpha_weights,
                epsilon_min   = cfg.epsilon_min,
                epsilon_max   = cfg.epsilon,        # reuse cfg.epsilon as ceiling
                epsilon_bg    = cfg.epsilon_bg,
                dilation      = cfg.mask_dilation,
                device        = device,
            )
    # else: eps_map already set by the existing block above

    # ── Initialise delta ───────────────────────────────────────────────────────
    delta = torch.zeros_like(x_orig_d)
    mu    = cfg.mu_init
    trajectory: list[StepLog] = []

    for step in range(1, cfg.pgd_steps + 1):
        delta   = delta.detach().requires_grad_(True)
        x_delta = (x_orig_d + delta).clamp(0.0, 1.0)

        # ── Machine failure objective ──────────────────────────────────────────
        fm = compute_FM_ens(surrogates, x_delta, transcript, cfg)

        # ── Readability proxy ──────────────────────────────────────────────────
        lh = compute_LH(x_delta, x_orig_d, cfg, word_boxes, font_embedder)

        # ── Penalised objective ────────────────────────────────────────────────
        pen = penalty_term(lh, cfg.kappa, mu)
        obj = fm - pen

        obj.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta_new = delta + cfg.pgd_step_size * grad_sign

            # Region-aware l-inf projection
            delta_new = project_onto_region_ball(delta_new, eps_map)

            # Keep x in [0, 1]
            delta_new = (x_orig_d + delta_new).clamp(0.0, 1.0) - x_orig_d

        delta = delta_new

        trajectory.append(StepLog(
            step=step,
            fm_ens=fm.item(),
            lh=lh.item(),
            penalty=pen.item(),
            objective=obj.item(),
            constraint_ok=constraint_satisfied(lh.item(), cfg.kappa),
            mu=mu,
        ))

        mu = mu * cfg.mu_growth

    # ── Final result ───────────────────────────────────────────────────────────
    with torch.no_grad():
        x_adv     = (x_orig_d + delta).clamp(0.0, 1.0).cpu()
        lh_final  = compute_LH(
            x_adv.to(device), x_orig_d, cfg, word_boxes
        ).item()
        fm_final  = trajectory[-1].fm_ens

    return AttackResult(
        image_id=image_id,
        transcript=transcript,
        epsilon=cfg.epsilon_text if use_region else cfg.epsilon,
        kappa=cfg.kappa,
        x_adv=x_adv,
        constraint_satisfied=constraint_satisfied(lh_final, cfg.kappa),
        fm_ens_final=fm_final,
        lh_final=lh_final,
        region_aware=use_region,
        trajectory=trajectory,
    )