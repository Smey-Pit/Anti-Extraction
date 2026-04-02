"""
F_M^ens(delta): ensemble machine failure objective.

Aggregates F_ce^k and F_align^k across all optimisation surrogates.
Sign convention: higher = more suppression success.
"""

from __future__ import annotations

import torch

from vlm_suppress.config import AttackConfig, EnsembleWeighting, ObjectiveConfig
from vlm_suppress.models.base import SurrogateModel


def _to_scalar(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure t is a 0-dim scalar tensor regardless of what shape the
    surrogate wrapper returned.  Handles (), (1,), (1,1) etc.
    Any surrogate returning a non-unit tensor is a bug, but we guard
    here so the PGD loop never crashes on a shape mismatch.
    """
    t = t.float()
    if t.numel() != 1:
        raise RuntimeError(
            f"Surrogate loss returned {t.numel()} elements (shape {tuple(t.shape)}). "
            "ce_loss and align_loss must return a scalar tensor."
        )
    return t.reshape(())   # guaranteed 0-dim


def compute_FM_ens(
    surrogates: list[SurrogateModel],
    image_tensor: torch.Tensor,
    transcript: str,
    cfg: AttackConfig,
) -> torch.Tensor:
    """
    Computes F_M^ens(delta) = sum_k alpha_k (lambda_ce * F_ce^k + lambda_align * F_align^k)

    Returns a 0-dim scalar tensor with gradients w.r.t. image_tensor.
    """
    alphas = _compute_alphas(surrogates, cfg)
    total  = torch.zeros((), device=image_tensor.device, dtype=torch.float32)

    for model, alpha in zip(surrogates, alphas):
        contribution = torch.zeros((), device=image_tensor.device, dtype=torch.float32)

        if cfg.objective in (ObjectiveConfig.CE_ONLY, ObjectiveConfig.CE_AND_ALIGN):
            f_ce = _to_scalar(model.ce_loss(image_tensor, transcript))
            contribution = contribution + cfg.lambda_ce * f_ce

        if cfg.objective in (ObjectiveConfig.ALIGN_ONLY, ObjectiveConfig.CE_AND_ALIGN):
            f_align = _to_scalar(model.align_loss(image_tensor, transcript))
            contribution = contribution + cfg.lambda_align * f_align

        total = total + alpha * contribution

    return total


def _compute_alphas(
    surrogates: list[SurrogateModel],
    cfg: AttackConfig,
) -> list[float]:
    if cfg.ensemble_weighting == EnsembleWeighting.UNIFORM or len(surrogates) == 1:
        return [1.0 / len(surrogates)] * len(surrogates)

    # Diversity-weighted stub — falls back to uniform until CKA implemented
    return [1.0 / len(surrogates)] * len(surrogates)