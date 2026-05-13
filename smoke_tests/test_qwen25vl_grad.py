"""
smoke_tests/test_qwen25vl_grad.py

Verifies that Qwen2_5VL satisfies the full SurrogateModel contract:
  1. ce_loss    — scalar, gradient reaches image_tensor
  2. align_loss — scalar, gradient reaches image_tensor
  3. transcribe — returns a non-empty string
  4. salience   — build_salience_budget_map produces a valid (1,H,W) budget map
  5. lazy       — LazySurrogate wrapping loads/unloads cleanly

Usage (on Spartan, from project root):
    uv run python smoke_tests/test_qwen25vl_grad.py \
        --model_id Qwen/Qwen2.5-VL-3B-Instruct \
        [--image_size 512]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.models.qwen2_5vl import Qwen2_5VL
from vlm_suppress.config import SurrogateConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cfg(model_id: str) -> SurrogateConfig:
    return SurrogateConfig(
        name="qwen2_5vl",
        model_id=model_id,
        dtype="bfloat16",
        device_map="auto",
        max_new_tokens=128,
    )


def _make_image(size: int, device: torch.device) -> torch.Tensor:
    """Random RGB image tensor (3, H, W) in [0,1] with requires_grad."""
    img = torch.rand(3, size, size, dtype=torch.float32, device=device)
    img.requires_grad_(True)
    return img


def _check_grad(tensor: torch.Tensor, name: str) -> None:
    assert tensor.grad is not None, f"[FAIL] {name}: .grad is None — gradient did not reach image_tensor"
    assert tensor.grad.abs().sum().item() > 0, f"[FAIL] {name}: .grad is all-zeros — gradient vanished"
    print(f"  [PASS] {name}: grad norm = {tensor.grad.norm().item():.6f}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ce_loss(model: Qwen2_5VL, image_size: int) -> None:
    print("\n── Test 1: ce_loss gradient ──────────────────────────────────────────")
    img = _make_image(image_size, model.device)
    transcript = "Hello World"

    loss = model.ce_loss(img, transcript)

    assert loss.shape == (), f"[FAIL] ce_loss shape {loss.shape} is not scalar"
    assert loss.item() > 0, f"[FAIL] ce_loss = {loss.item()} is non-positive"
    print(f"  ce_loss = {loss.item():.6f}")

    loss.backward()
    _check_grad(img, "ce_loss")


def test_align_loss(model: Qwen2_5VL, image_size: int) -> None:
    print("\n── Test 2: align_loss gradient ───────────────────────────────────────")
    img = _make_image(image_size, model.device)
    transcript = "Hello World"

    loss = model.align_loss(img, transcript)

    assert loss.shape == (), f"[FAIL] align_loss shape {loss.shape} is not scalar"
    print(f"  align_loss = {loss.item():.6f}")

    loss.backward()
    _check_grad(img, "align_loss")


def test_transcribe(model: Qwen2_5VL, image_size: int) -> None:
    print("\n── Test 3: transcribe ────────────────────────────────────────────────")
    img = torch.rand(3, image_size, image_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        text = model.transcribe(img)

    assert isinstance(text, str), f"[FAIL] transcribe returned {type(text)}, not str"
    print(f"  transcribe output: {text[:80]!r}")
    print("  [PASS] transcribe returned a string")


def test_salience_map(model: Qwen2_5VL, image_size: int) -> None:
    print("\n── Test 4: salience budget map ───────────────────────────────────────")
    from vlm_suppress.attack.salience import build_salience_budget_map

    img_4d     = torch.rand(1, 3, image_size, image_size, dtype=torch.float32)
    transcript = "Hello World"
    word_boxes = [[10, 10, image_size // 2, image_size // 3]]

    budget_map = build_salience_budget_map(
        image_tensor  = img_4d,
        transcript    = transcript,
        word_boxes    = word_boxes,
        surrogates    = [model],
        alpha_weights = [1.0],
        epsilon_min   = 4.0  / 255.0,
        epsilon_max   = 16.0 / 255.0,
        epsilon_bg    = 1.0  / 255.0,
        dilation      = 3,
        device        = model.device,
    )

    assert budget_map.shape == (1, image_size, image_size), (
        f"[FAIL] budget_map shape {budget_map.shape}, expected (1, {image_size}, {image_size})"
    )
    assert budget_map.min().item() >= 1.0 / 255.0 - 1e-7, (
        f"[FAIL] budget_map min {budget_map.min().item():.6f} < epsilon_bg"
    )
    assert budget_map.max().item() <= 16.0 / 255.0 + 1e-7, (
        f"[FAIL] budget_map max {budget_map.max().item():.6f} > epsilon_max"
    )
    print(
        f"  budget_map — "
        f"min: {budget_map.min().item():.5f}  "
        f"mean: {budget_map.mean().item():.5f}  "
        f"max: {budget_map.max().item():.5f}"
    )
    print("  [PASS] build_salience_budget_map produced a valid budget map")


def test_lazy_loading(cfg: SurrogateConfig) -> None:
    print("\n── Test 5: LazySurrogate wrapping ────────────────────────────────────")
    from vlm_suppress.models.lazy import LazySurrogate

    lazy = LazySurrogate(cfg, Qwen2_5VL)
    assert lazy.name == cfg.name, f"[FAIL] lazy.name = {lazy.name!r}, expected {cfg.name!r}"
    print(f"  lazy.name = {lazy.name!r}  (safe outside context)")

    img_4d     = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    transcript = "Hello"

    with lazy as model:
        assert isinstance(model, Qwen2_5VL), f"[FAIL] context returned {type(model)}"
        loss = model.ce_loss(img_4d[0], transcript)
        assert loss.shape == (), f"[FAIL] lazy ce_loss shape {loss.shape}"
        print(f"  ce_loss inside context = {loss.item():.6f}")

    # After exit: model should be unloaded
    assert lazy._model is None, "[FAIL] lazy._model is not None after __exit__"
    print("  [PASS] LazySurrogate loaded, ran ce_loss, and unloaded cleanly")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient-flow smoke test for Qwen2_5VL surrogate"
    )
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--image_size", type=int, default=336,
                        help="Square image side (pixels)")
    parser.add_argument("--skip_lazy", action="store_true",
                        help="Skip the lazy-loading test (saves one full load)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Qwen2_5VL smoke test — {args.model_id}")
    print(f"image_size = {args.image_size}")
    print("=" * 60)

    cfg   = _make_cfg(args.model_id)
    model = Qwen2_5VL(cfg)
    print(f"\nLoaded model on {model.device}")

    test_ce_loss(model, args.image_size)
    test_align_loss(model, args.image_size)
    test_transcribe(model, args.image_size)
    test_salience_map(model, args.image_size)

    if not args.skip_lazy:
        # Unload the eager model first so the lazy test doesn't double-load
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_lazy_loading(cfg)

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
