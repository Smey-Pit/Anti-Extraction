"""
F_M^ens(delta): ensemble machine failure objective.

Aggregates F_ce^k and F_align^k across all optimisation surrogates.
Sign convention: higher = more suppression success.
"""

from __future__ import annotations

import random

import torch

from vlm_suppress.attack.transforms import jpeg_compress_ste
from vlm_suppress.config import AttackConfig, EOTConfig, EnsembleWeighting, ObjectiveConfig
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
        # All surrogates share the same x_delta = clamp(x_orig_d + delta) node.
        # Each backward would free clamp's saved mask, breaking the second
        # surrogate's backward.  retain_graph=True keeps the shared path alive
        # across all FM backward calls; the final obj.backward() in the PGD loop
        # (carrying only the penalty, since fm is detached) is the last consumer
        # and frees everything with the default retain_graph=False.
        fm_total = torch.tensor(0.0, device=image_tensor.device, dtype=torch.float32)
        for surrogate, alpha in zip(surrogates, alphas):
            if isinstance(surrogate, LazySurrogate):
                with surrogate as model:
                    loss_k = _weighted_contribution(model, image_tensor, transcript, cfg, alpha)
                    loss_k.backward(retain_graph=True)
            else:
                # Eagerly-loaded model in a lazy run (e.g. a held-out surrogate
                # selected for this step) — call directly and backward immediately.
                loss_k = _weighted_contribution(surrogate, image_tensor, transcript, cfg, alpha)
                loss_k.backward(retain_graph=True)
            fm_total = fm_total + loss_k.detach()
        return fm_total
    else:
        total = torch.zeros((), device=image_tensor.device, dtype=torch.float32)
        for surrogate, alpha in zip(surrogates, alphas):
            total = total + _weighted_contribution(surrogate, image_tensor, transcript, cfg, alpha)
        return total


def compute_FM_ens_eot(
    surrogates:   list,
    x_delta:      torch.Tensor,   # (3, H, W) on device, requires_grad
    transcript:   str,
    cfg:          AttackConfig,
    lazy:         bool = False,
) -> float:
    """
    EOT version of compute_FM_ens.

    Draws cfg.eot.n_samples random JPEG qualities, applies each via STE to
    produce x_ste_s, calls compute_FM_ens on each sample, and accumulates
    gradients into x_delta (and thus delta) via per-sample .backward().

    Returns the total FM value as a plain float for trajectory logging.
    Callers must NOT call .backward() on the returned value — gradients are
    already accumulated.  The penalty backward (-pen).backward() must be
    called separately after this function returns.

    retain_graph=True is used on every sample so the shared x_delta → delta
    graph edge survives for the penalty backward that follows.
    """
    eot    = cfg.eot
    device = x_delta.device
    fm_total = 0.0

    for s in range(eot.n_samples):
        quality = random.randint(eot.quality_min, eot.quality_max)
        x_ste   = jpeg_compress_ste(x_delta, quality, device)
        fm_s    = compute_FM_ens(surrogates, x_ste, transcript, cfg, lazy=lazy)
        # retain_graph=True: keep x_delta → delta alive for penalty backward
        (fm_s / eot.n_samples).backward(retain_graph=True)
        fm_total += fm_s.detach().item()

    return fm_total / eot.n_samples


def _compute_alphas(
    surrogates: list,
    cfg: AttackConfig,
) -> list[float]:
    if cfg.ensemble_weighting == EnsembleWeighting.UNIFORM or len(surrogates) == 1:
        return [1.0 / len(surrogates)] * len(surrogates)

    # Diversity-weighted stub — falls back to uniform until CKA implemented
    return [1.0 / len(surrogates)] * len(surrogates)