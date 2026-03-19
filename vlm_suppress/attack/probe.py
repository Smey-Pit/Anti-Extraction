# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/attack/probe.py
#
# Unconstrained white-box probe — region-aware version.
#
# Same purpose as before: validate F_M^ens is optimisable before introducing
# the readability constraint.
#
# Region-aware change: the l-inf projection now uses a per-pixel epsilon map
# built from word_boxes. Text pixels use epsilon_text (tight budget),
# background uses epsilon_bg (loose budget).
#
# This matters even for the probe because it sets up the correct gradient
# flow and projection geometry that the full attack will use. Running the
# probe with uniform epsilon and the attack with region-aware epsilon would
# make the probe results misleading.
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vlm_suppress.attack.ensemble import compute_FM_ens
from vlm_suppress.attack.masks import build_epsilon_map, project_onto_region_ball
from vlm_suppress.config import AttackConfig, ObjectiveConfig
from vlm_suppress.models.base import SurrogateModel


# ── L2 projection (used when norm="l2") ───────────────────────────────────────

def _proj_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    flat = delta.view(-1)
    n = flat.norm(p=2)
    if n > epsilon:
        flat = flat * (epsilon / n)
    return flat.view(delta.shape)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ProbeStepLog:
    step:            int
    fm_ens:          float
    fm_ce:           float
    fm_align:        float
    delta_norm_l2:   float
    delta_norm_linf: float


@dataclass
class ProbeResult:
    image_id:    str
    transcript:  str
    norm:        str
    epsilon:     float       # text-region epsilon (or global if uniform)
    region_aware: bool

    x_adv:    torch.Tensor   # (3, H, W) float32 [0,1] CPU
    fm_clean: float
    fm_final: float
    fm_delta: float

    trajectory: list[ProbeStepLog] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.fm_delta > 0.0


# ── Main probe function ────────────────────────────────────────────────────────

def run_probe(
    image_id:   str,
    x_orig:     torch.Tensor,            # (3, H, W) float32 [0,1] CPU
    transcript: str,
    surrogates: list[SurrogateModel],
    cfg:        AttackConfig,
    word_boxes: list[list[int]] | None = None,
    norm:       str   = "l2",
    epsilon:    float | None = None,     # overrides cfg.epsilon_text if set
    pgd_steps:  int   | None = None,
) -> ProbeResult:
    """
    Unconstrained PGD probe with region-aware l-inf projection.

    norm="l2":   global L2 ball projection (no region split — useful for
                 checking gradient flow cleanly before adding complexity).
    norm="linf": per-pixel l-inf projection using the epsilon map from
                 word_boxes. This is the geometry the full attack uses.

    For the first probe run, use norm="l2" to confirm gradient flow,
    then norm="linf" to confirm the region-aware projection works.
    """
    device = surrogates[0].device
    steps  = pgd_steps if pgd_steps is not None else cfg.pgd_steps
    H, W   = x_orig.shape[-2], x_orig.shape[-1]

    x_orig_d = x_orig.to(device)

    # ── Build epsilon map for linf mode ───────────────────────────────────────
    use_region = (
        norm == "linf"
        and cfg.region_aware
        and word_boxes is not None
        and len(word_boxes) > 0
    )

    if use_region:
        # Override epsilon_text if caller passed explicit epsilon
        eps_text = epsilon if epsilon is not None else cfg.epsilon_text
        eps_bg   = cfg.epsilon_bg
        eps_map  = build_epsilon_map(
            height=H, width=W,
            word_boxes=word_boxes,
            epsilon_text=eps_text,
            epsilon_bg=eps_bg,
            dilation=cfg.mask_dilation,
            device=device,
        )
        eps_display = eps_text   # for logging
    else:
        # Uniform fallback — used for l2 norm or when boxes unavailable
        eps_uniform = epsilon if epsilon is not None else cfg.epsilon
        eps_map = torch.full(
            (1, H, W), eps_uniform,
            dtype=torch.float32, device=device,
        )
        eps_display = eps_uniform

    # ── Clean baseline ─────────────────────────────────────────────────────────
    fm_clean = _measure_fm(x_orig_d, transcript, surrogates, cfg)

    # ── PGD loop ───────────────────────────────────────────────────────────────
    delta = torch.zeros_like(x_orig_d)
    trajectory: list[ProbeStepLog] = []

    for step in range(1, steps + 1):
        delta   = delta.detach().requires_grad_(True)
        x_delta = (x_orig_d + delta).clamp(0.0, 1.0)

        fm = compute_FM_ens(surrogates, x_delta, transcript, cfg)
        fm.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta_new = delta + cfg.pgd_step_size * grad_sign

            if norm == "l2":
                # Global L2 projection
                delta_new = _proj_l2(delta_new, eps_display)
            else:
                # Per-pixel l-inf projection (region-aware)
                delta_new = project_onto_region_ball(delta_new, eps_map)

            delta_new = (x_orig_d + delta_new).clamp(0.0, 1.0) - x_orig_d

        with torch.no_grad():
            fm_ce, fm_align = _measure_components(
                (x_orig_d + delta_new).clamp(0.0, 1.0),
                transcript, surrogates, cfg,
            )
            trajectory.append(ProbeStepLog(
                step=step,
                fm_ens=fm.item(),
                fm_ce=fm_ce,
                fm_align=fm_align,
                delta_norm_l2=delta_new.norm(p=2).item(),
                delta_norm_linf=delta_new.abs().max().item(),
            ))

        delta = delta_new

    with torch.no_grad():
        x_adv    = (x_orig_d + delta).clamp(0.0, 1.0).cpu()
        fm_final = trajectory[-1].fm_ens

    return ProbeResult(
        image_id=image_id,
        transcript=transcript,
        norm=norm,
        epsilon=eps_display,
        region_aware=use_region,
        x_adv=x_adv,
        fm_clean=fm_clean,
        fm_final=fm_final,
        fm_delta=fm_final - fm_clean,
        trajectory=trajectory,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _measure_fm(
    x: torch.Tensor, transcript: str,
    surrogates: list[SurrogateModel], cfg: AttackConfig,
) -> float:
    x_g = x.detach().requires_grad_(True)
    return compute_FM_ens(surrogates, x_g, transcript, cfg).item()


def _measure_components(
    x: torch.Tensor, transcript: str,
    surrogates: list[SurrogateModel], cfg: AttackConfig,
) -> tuple[float, float]:
    fm_ce = fm_align = 0.0
    alpha = 1.0 / len(surrogates)
    for m in surrogates:
        x_g = x.detach().requires_grad_(False)
        if cfg.objective in (ObjectiveConfig.CE_ONLY, ObjectiveConfig.CE_AND_ALIGN):
            try:
                fm_ce += alpha * m.ce_loss(x_g, transcript).item()
            except Exception:
                pass
        if cfg.objective in (ObjectiveConfig.ALIGN_ONLY, ObjectiveConfig.CE_AND_ALIGN):
            try:
                fm_align += alpha * m.align_loss(x_g, transcript).item()
            except Exception:
                pass
    return fm_ce, fm_align