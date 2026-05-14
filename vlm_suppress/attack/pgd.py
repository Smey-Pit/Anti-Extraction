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
from vlm_suppress.attack.ensemble import compute_FM_ens, compute_FM_ens_eot
from vlm_suppress.attack.masks import build_epsilon_map, project_onto_region_ball
from vlm_suppress.attack.salience import build_salience_budget_map
from vlm_suppress.config import AttackConfig
from vlm_suppress.models.base import SurrogateModel
from vlm_suppress.models.lazy import LazySurrogate


class _OffloadedSurrogate(LazySurrogate):
    """
    Wraps a CPU-offloaded eager model for the salience pass.

    Inherits from LazySurrogate so salience.py's isinstance check routes it
    through the context-manager path, giving one-model-at-a-time VRAM use.

    __enter__ → move weights GPU, return eager model
    __exit__  → move weights back to CPU, clear CUDA cache
    """

    def __init__(self, model: SurrogateModel, gpu_device: torch.device) -> None:
        self._model      = model
        self._gpu_device = gpu_device
        # LazySurrogate.name / .device read from self.cfg — duck-type to model
        self.cfg = model
        self.cls = None   # prevents base-class __exit__ from trying to unload

    def __enter__(self) -> SurrogateModel:
        self._model.model.to(self._gpu_device)
        return self._model

    def __exit__(self, *args) -> None:
        self._model.model.to("cpu")
        torch.cuda.empty_cache()


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
    surrogates:     list,
    cfg:            AttackConfig,
    word_boxes:     list[list[int]] | None = None,
    font_embedder:  object = None,
    all_surrogates: list | None = None,
    lazy:           bool = False,
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

            # salience_offload: CPU-offload eager opt-surrogates before the
            # salience pass so peak VRAM = one model at a time.  After the
            # pass, each model is restored to GPU for the PGD loop.
            _offloaded: list[tuple] = []   # (eager_model, gpu_device) for restore
            sal_final = sal_surrogates

            if getattr(cfg, "salience_offload", False):
                if lazy:
                    import warnings
                    warnings.warn(
                        "salience_offload=True is redundant when lazy_loading=True "
                        "— surrogates already load/unload via LazySurrogate.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    _wrapped = []
                    for s in sal_surrogates:
                        if isinstance(s, LazySurrogate):
                            _wrapped.append(s)
                        else:
                            gpu_dev = s.device
                            s.model.to("cpu")
                            torch.cuda.empty_cache()
                            _offloaded.append((s, gpu_dev))
                            _wrapped.append(_OffloadedSurrogate(s, gpu_dev))
                    sal_final = _wrapped
            elif cfg.salience_lazy:
                import warnings
                non_lazy = [s for s in sal_surrogates if not isinstance(s, LazySurrogate)]
                if non_lazy:
                    warnings.warn(
                        f"salience_lazy=True but {len(non_lazy)} surrogate(s) are "
                        "already-loaded SurrogateModel instances — VRAM will NOT be "
                        "freed between salience passes.  Set salience_offload: true "
                        "or lazy_loading: true to actually reduce peak VRAM.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            alpha_weights = [1.0 / len(sal_final)] * len(sal_final)
            eps_map = build_salience_budget_map(
                image_tensor  = x_orig_d.unsqueeze(0),
                transcript    = transcript,
                word_boxes    = word_boxes,
                surrogates    = sal_final,
                alpha_weights = alpha_weights,
                epsilon_min   = cfg.epsilon_min,
                epsilon_max   = cfg.epsilon,        # reuse cfg.epsilon as ceiling
                epsilon_bg    = cfg.epsilon_bg,
                dilation      = cfg.mask_dilation,
                device        = device,
            )

            # Restore CPU-offloaded models to GPU for the PGD loop
            for eager_model, gpu_dev in _offloaded:
                eager_model.model.to(gpu_dev)
    # else: eps_map already set by the existing block above

    # ── Initialise delta ───────────────────────────────────────────────────────
    delta = torch.zeros_like(x_orig_d)
    mu    = cfg.mu_init
    trajectory: list[StepLog] = []

    for step in range(1, cfg.pgd_steps + 1):
        delta   = delta.detach().requires_grad_(True)
        x_delta = (x_orig_d + delta).clamp(0.0, 1.0)

        # ── Machine failure objective ──────────────────────────────────────────
        if cfg.eot.enabled:
            # EOT: average FM over n_samples JPEG-compressed copies via STE.
            # Gradients accumulated into delta.grad inside compute_FM_ens_eot;
            # retain_graph=True keeps x_delta→delta alive for the penalty below.
            fm_val = compute_FM_ens_eot(surrogates, x_delta, transcript, cfg, lazy=lazy)
            lh     = compute_LH(x_delta, x_orig_d, cfg, word_boxes, font_embedder)
            pen    = penalty_term(lh, cfg.kappa, mu)
            (-pen).backward()           # adds penalty grad; frees the shared graph
            obj_val = fm_val - pen.item()
        else:
            fm = compute_FM_ens(surrogates, x_delta, transcript, cfg, lazy=lazy)
            lh = compute_LH(x_delta, x_orig_d, cfg, word_boxes, font_embedder)
            pen = penalty_term(lh, cfg.kappa, mu)
            obj = fm - pen
            obj.backward()
            fm_val  = fm.item()
            obj_val = obj.item()

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
            fm_ens=fm_val,
            lh=lh.item(),
            penalty=pen.item(),
            objective=obj_val,
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