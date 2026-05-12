"""
Input transformation utilities for Expectation over Transformations (EOT).

All transforms operate on (3, H, W) float32 [0, 1] tensors.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image


def jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    JPEG encode/decode roundtrip via PIL.

    x:       (3, H, W) float32 [0, 1] — any device, detached
    quality: JPEG quality factor [1, 95]
    Returns: (3, H, W) float32 [0, 1] on CPU
    """
    arr = (
        x.detach().cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)           # (H, W, 3)
        .mul(255.0)
        .byte()
        .numpy()
    )
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    arr_dec = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr_dec).permute(2, 0, 1)   # (3, H, W) on CPU


def jpeg_compress_ste(
    x_delta: torch.Tensor,
    quality:  int,
    device:   torch.device,
) -> torch.Tensor:
    """
    Straight-Through Estimator (STE) wrapper around jpeg_compress.

    Forward:  returns the JPEG-compressed image (what the surrogate sees).
    Backward: gradient flows through x_delta unchanged (identity Jacobian).

    x_delta: (3, H, W) float32 [0, 1] on device, may have requires_grad
    Returns: same shape/device/dtype as x_delta, with STE backward
    """
    x_jpeg = jpeg_compress(x_delta, quality).to(device=device, dtype=x_delta.dtype)
    # STE: forward = x_jpeg, backward = d/d(x_delta) [identity]
    return x_delta + (x_jpeg - x_delta).detach()
