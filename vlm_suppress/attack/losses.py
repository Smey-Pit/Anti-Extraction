"""
Readability proxy loss terms L_H.

Sign convention: ALL terms here are COSTS — higher = more structural damage.
The constraint is L_H(delta) <= kappa.

Staged implementation matches the paper exactly:
  Stage 1: L_edge + L_stroke
  Stage 2: + L_shape
  Stage 3: + L_conf  (conditional; beta_conf=0 disables it)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Sobel edge detector.
    x: (..., C, H, W) float tensor
    Returns: (..., H, W) edge magnitude map
    """
    gray = x.mean(dim=-3, keepdim=True)  # (..., 1, H, W)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx ** 2 + gy ** 2).sqrt().squeeze(-3)  # (..., H, W)


def l_edge(x_delta: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
    """
    L_edge(delta) = || E(X+delta) - E(X) ||_1  (normalised by numel)
    Measures how much the perturbation shifts edge structure.
    """
    e_delta = _sobel_edges(x_delta)
    e_orig  = _sobel_edges(x_orig.detach())
    return (e_delta - e_orig).abs().mean()


def _morphological_stroke(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Differentiable stroke approximation via max-pooling on binarised image.
    x: (..., C, H, W) in [0,1]
    Returns stroke map (..., H, W).
    """
    gray = x.mean(dim=-3, keepdim=True)
    # Soft binarisation: pixels darker than 0.7 are likely ink
    binary = 1.0 - torch.sigmoid(20 * (gray - 0.7))
    pad = kernel_size // 2
    strokes = F.max_pool2d(binary, kernel_size, stride=1, padding=pad)
    return strokes.squeeze(-3)


def l_stroke(x_delta: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
    """
    L_stroke(delta) = || S(X+delta) - S(X) ||_1
    """
    s_delta = _morphological_stroke(x_delta)
    s_orig  = _morphological_stroke(x_orig.detach())
    return (s_delta - s_orig).abs().mean()


def l_shape(
    x_delta: torch.Tensor,
    x_orig: torch.Tensor,
    word_boxes: list[list[int]] | None,
) -> torch.Tensor:
    """
    L_shape(delta) = sum_w || B_w(X+delta) - B_w(X) ||_1

    Preserves word-region silhouette (bouma).
    If word_boxes is None or empty, falls back to whole-image region loss.
    word_boxes: list of [x0, y0, x1, y1] in pixel coords.
    """
    if not word_boxes:
        # Fallback: full-image channel-mean difference
        return (x_delta.mean(0) - x_orig.detach().mean(0)).abs().mean()

    total = torch.tensor(0.0, device=x_delta.device, dtype=x_delta.dtype)
    for box in word_boxes:
        x0, y0, x1, y1 = box
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(x_delta.shape[-1], x1)
        y1 = min(x_delta.shape[-2], y1)
        if x1 <= x0 or y1 <= y0:
            continue
        region_d = x_delta[..., y0:y1, x0:x1]
        region_o = x_orig[..., y0:y1, x0:x1].detach()
        total = total + (region_d - region_o).abs().mean()

    return total / max(len(word_boxes), 1)


def l_conf(
    x_delta: torch.Tensor,
    x_orig: torch.Tensor,
    font_embedder: torch.nn.Module | None = None,
    gamma: float = 0.3,
) -> torch.Tensor:
    """
    L_conf(delta): character confusability hinge loss.
    Penalises perturbations that push a glyph across identity boundaries.

    Currently a stub — returns 0 when font_embedder is None.
    Full implementation added in Stage 3 when/if needed.
    font_embedder: maps image crop -> glyph embedding in R^d.
    """
    if font_embedder is None:
        return torch.tensor(0.0, device=x_delta.device, dtype=x_delta.dtype)
    raise NotImplementedError("Stage 3 confusability not yet implemented.")

