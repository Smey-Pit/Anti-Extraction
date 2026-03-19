"""
L_H combined readability proxy.

This is the CONSTRAINT side of the optimisation — separate from the
machine failure objective F_M. All four terms feed into one scalar L_H.
The constraint is L_H(delta) <= kappa.
"""

from __future__ import annotations

import torch

from vlm_suppress.attack.losses import l_conf, l_edge, l_shape, l_stroke
from vlm_suppress.config import AttackConfig, ProxyStage


def compute_LH(
    x_delta: torch.Tensor,          # (3, H, W), perturbed, on device
    x_orig: torch.Tensor,           # (3, H, W), original, on device
    cfg: AttackConfig,
    word_boxes: list[list[int]] | None = None,
    font_embedder=None,
) -> torch.Tensor:
    """
    Computes L_H(delta) according to the active proxy stage.

    Returns a scalar tensor (gradient flows through x_delta).
    Higher = more structural damage = worse readability.

    Stage 1 (default): L_edge + L_stroke
    Stage 2:           + L_shape
    Stage 3:           + L_conf (requires font_embedder)
    """
    lh = torch.tensor(0.0, device=x_delta.device, dtype=x_delta.dtype)

    # Stage 1 always active
    lh = lh + cfg.beta_edge   * l_edge(x_delta, x_orig)
    lh = lh + cfg.beta_stroke * l_stroke(x_delta, x_orig)

    if cfg.proxy_stage in (ProxyStage.STAGE2, ProxyStage.STAGE3):
        lh = lh + cfg.beta_shape * l_shape(x_delta, x_orig, word_boxes)

    if cfg.proxy_stage == ProxyStage.STAGE3 and cfg.beta_conf > 0:
        lh = lh + cfg.beta_conf * l_conf(x_delta, x_orig, font_embedder)

    return lh


def constraint_satisfied(lh_value: float, kappa: float) -> bool:
    return lh_value <= kappa


def penalty_term(
    lh: torch.Tensor,
    kappa: float,
    mu: float,
) -> torch.Tensor:
    """
    Squared penalty for the readability constraint violation:
      mu * [L_H(delta) - kappa]_+^2

    This is the penalty term subtracted from F_M^ens in the penalised objective.
    (See Eq. 10 in paper.)
    """
    violation = F.relu(lh - kappa)  # [L_H - kappa]_+
    return mu * violation ** 2


import torch.nn.functional as F