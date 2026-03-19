# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/attack/masks.py
#
# Builds spatial binary masks that separate text regions from background.
#
# These masks drive the region-aware perturbation budget split:
#   - text pixels    → tight epsilon (epsilon_text)
#   - background     → loose epsilon (epsilon_bg)
#
# The masks are built from word_boxes (or char_boxes if available) which
# are already part of your dataset schema. No model inference needed.
#
# Dilation is applied to the text mask to cover anti-aliasing fringe
# pixels at character edges — without it, the tight budget would clip
# right at the character boundary and leave a visible artefact halo.
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_text_mask(
    height:        int,
    width:         int,
    word_boxes:    list[list[int]],   # flat list of [x0, y0, x1, y1] scaled to (H, W)
    dilation:      int = 3,           # cfg.mask_dilation
    device:        torch.device = torch.device("cpu"),
    dtype:         torch.dtype  = torch.float32,
) -> torch.Tensor:
    """
    Returns a binary mask of shape (1, H, W) on `device`:
        1.0  →  text region  (tight epsilon applies)
        0.0  →  background   (loose epsilon applies)

    If word_boxes is empty, falls back to a full-ones mask (uniform budget,
    same behaviour as the old implementation).

    Args:
        height, width:  spatial dimensions matching the image tensor.
        word_boxes:     scaled word boxes [[x0,y0,x1,y1], ...].
                        Must already be in the coordinate space of the image
                        tensor — use sample.scaled_word_boxes().
        dilation:       number of pixels to expand the text mask outward.
                        Covers anti-aliasing fringe. 0 = no dilation.
    """
    if not word_boxes:
        # No boxes available — fall back to uniform (full-image text mask)
        return torch.ones(1, height, width, dtype=dtype, device=device)

    # Build mask on CPU as numpy, move to device at the end
    mask = np.zeros((height, width), dtype=np.float32)

    for box in word_boxes:
        x0, y0, x1, y1 = box
        # Clamp to image bounds
        x0 = max(0, x0);  y0 = max(0, y0)
        x1 = min(width,  x1);  y1 = min(height, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0

    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    # Morphological dilation via max-pooling
    if dilation > 0:
        kernel = 2 * dilation + 1
        mask_t = F.max_pool2d(
            mask_t, kernel_size=kernel, stride=1, padding=dilation
        )

    return mask_t.squeeze(0).to(device=device, dtype=dtype)   # (1, H, W)


def build_epsilon_map(
    height:       int,
    width:        int,
    word_boxes:   list[list[int]],
    epsilon_text: float,
    epsilon_bg:   float,
    dilation:     int = 3,
    device:       torch.device = torch.device("cpu"),
    dtype:        torch.dtype  = torch.float32,
) -> torch.Tensor:
    """
    Returns a per-pixel epsilon map of shape (1, H, W):
        epsilon_text  at text pixels
        epsilon_bg    at background pixels

    This map is used directly in the projection step — each pixel is
    clipped to its own budget rather than a single global epsilon.

    When epsilon_text == epsilon_bg the result is identical to the
    original uniform projection (useful for ablation).
    """
    text_mask = build_text_mask(
        height, width, word_boxes, dilation, device, dtype
    )
    # Blend: eps_map = mask * eps_text + (1 - mask) * eps_bg
    eps_map = text_mask * epsilon_text + (1.0 - text_mask) * epsilon_bg
    return eps_map   # (1, H, W)


def project_onto_region_ball(
    delta:     torch.Tensor,   # (3, H, W)
    eps_map:   torch.Tensor,   # (1, H, W) — per-pixel epsilon
) -> torch.Tensor:
    """
    Per-pixel l-inf projection using the region-aware epsilon map.

    Each pixel channel is clipped independently to [-eps_map, +eps_map].
    This is the drop-in replacement for the global delta.clamp(-eps, eps).

    Args:
        delta:    current perturbation (3, H, W)
        eps_map:  per-pixel budget     (1, H, W), broadcasts over channels
    """
    return delta.clamp(-eps_map, eps_map)