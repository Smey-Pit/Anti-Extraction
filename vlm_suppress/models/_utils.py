"""Shared utilities for surrogate wrappers."""

from __future__ import annotations

import torch
from PIL import Image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    (3, H, W) float32 [0,1] → PIL RGB.

    Always detaches and moves to CPU. Used only for inference paths in
    Phase 0; later phases that need gradient flow must NOT route through
    this function.
    """
    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            f"Expected image_tensor of shape (3, H, W), got {tuple(image_tensor.shape)}"
        )
    arr = (image_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")