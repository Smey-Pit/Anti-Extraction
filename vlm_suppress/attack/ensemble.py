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


def _weighted_contribution(
    model: SurrogateModel,
    image_tensor: torch.Tensor,
    transcript: str,
    cfg: AttackConfig,
    alpha: float,
) -> torch.Tensor:
    c = torch.zeros((), device=image_tensor.device, dtype=torch.float32)
    if cfg.objective in (ObjectiveConfig.CE_ONLY, ObjectiveConfig.CE_AND_ALIGN):
        c = c + cfg.lambda_ce * _to_scalar(model.ce_loss(image_tensor, transcript))
    if cfg.objective in (ObjectiveConfig.ALIGN_ONLY, ObjectiveConfig.CE_AND_ALIGN):
        c = c + cfg.lambda_align * _to_scalar(model.align_loss(image_tensor, transcript))
    return alpha * c


def compute_FM_ens(
    surrogates: list,
    image_tensor: torch.Tensor,
    transcript: str,
    cfg: AttackConfig,
    lazy: bool = False,
) -> torch.Tensor:
    """
    Computes F_M^ens(delta) = Σ_k alpha_k (lambda_ce·F_ce^k + lambda_align·F_align^k)

    Eager mode (lazy=False, default):
        Returns a 0-dim scalar with a live gradient graph w.r.t. image_tensor.
        Caller calls .backward() on the returned value (or a derived objective).

    Lazy mode (lazy=True):
        Loads each LazySurrogate, computes its contribution, calls .backward()
        within the context so gradients accumulate into delta.grad via the chain
        image_tensor → delta, then unloads.  Returns a detached scalar for
        trajectory logging.  Caller must NOT call .backward() for FM again —
        only the penalty gradient still needs to be added via obj = 0 - pen.
    """
    from vlm_suppress.models.lazy import LazySurrogate

    alphas = _compute_alphas(surrogates, cfg)

    if lazy:
        fm_total = torch.tensor(0.0, device=image_tensor.device, dtype=torch.float32)
        for surrogate, alpha in zip(surrogates, alphas):
            if isinstance(surrogate, LazySurrogate):
                with surrogate as model:
                    loss_k = _weighted_contribution(model, image_tensor, transcript, cfg, alpha)
                    loss_k.backward()
            else:
                # Eagerly-loaded model in a lazy run (e.g. a held-out surrogate
                # selected for this step) — call directly and backward immediately.
                loss_k = _weighted_contribution(surrogate, image_tensor, transcript, cfg, alpha)
                loss_k.backward()
            fm_total = fm_total + loss_k.detach()
        return fm_total
    else:
        total = torch.zeros((), device=image_tensor.device, dtype=torch.float32)
        for surrogate, alpha in zip(surrogates, alphas):
            total = total + _weighted_contribution(surrogate, image_tensor, transcript, cfg, alpha)
        return total


def _compute_alphas(
    surrogates: list,
    cfg: AttackConfig,
) -> list[float]:
    if cfg.ensemble_weighting == EnsembleWeighting.UNIFORM or len(surrogates) == 1:
        return [1.0 / len(surrogates)] * len(surrogates)

    # Diversity-weighted stub — falls back to uniform until CKA implemented
    return [1.0 / len(surrogates)] * len(surrogates)