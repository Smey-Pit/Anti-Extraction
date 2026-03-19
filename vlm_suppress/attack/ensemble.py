"""
F_M^ens(delta): ensemble machine failure objective.

Aggregates F_ce^k and F_align^k across all optimisation surrogates.
Sign convention: higher = more suppression success.

Supports uniform and diversity-weighted (CKA-based) ensemble weights.
CKA weighting is a stub for now — enabled in week 4+ analysis.
"""

from __future__ import annotations

import torch

from vlm_suppress.config import AttackConfig, EnsembleWeighting, ObjectiveConfig
from vlm_suppress.models.base import SurrogateModel


def compute_FM_ens(
    surrogates: list[SurrogateModel],
    image_tensor: torch.Tensor,   # (3, H, W), requires_grad=True
    transcript: str,
    cfg: AttackConfig,
) -> torch.Tensor:
    """
    Computes F_M^ens(delta) = sum_k alpha_k (lambda_ce * F_ce^k + lambda_align * F_align^k)

    Returns scalar tensor with gradients w.r.t. image_tensor.
    """
    alphas = _compute_alphas(surrogates, cfg)
    total = torch.tensor(0.0, device=image_tensor.device, dtype=torch.float32)

    for k, (model, alpha) in enumerate(zip(surrogates, alphas)):
        contribution = torch.tensor(0.0, device=image_tensor.device, dtype=torch.float32)

        if cfg.objective in (ObjectiveConfig.CE_ONLY, ObjectiveConfig.CE_AND_ALIGN):
            f_ce = model.ce_loss(image_tensor, transcript).float()
            contribution = contribution + cfg.lambda_ce * f_ce

        if cfg.objective in (ObjectiveConfig.ALIGN_ONLY, ObjectiveConfig.CE_AND_ALIGN):
            f_align = model.align_loss(image_tensor, transcript).float()
            contribution = contribution + cfg.lambda_align * f_align

        total = total + alpha * contribution

    return total


def _compute_alphas(
    surrogates: list[SurrogateModel],
    cfg: AttackConfig,
) -> list[float]:
    if cfg.ensemble_weighting == EnsembleWeighting.UNIFORM or len(surrogates) == 1:
        return [1.0 / len(surrogates)] * len(surrogates)

    # Diversity-weighted: stub — falls back to uniform until CKA implemented (week 4+)
    # TODO: compute pairwise CKA between surrogate visual encoders,
    #       set alpha_k proportional to mean dissimilarity from other members.
    return [1.0 / len(surrogates)] * len(surrogates)

