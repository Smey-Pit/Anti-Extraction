"""
decoy_alpha_sweep.py — Find the perceptibility threshold for decoy text injection.

Generates a grid of images with decoy text at varying alpha levels and color
channel modes. Use these to find the crossover point where:
  - Human: cannot read the decoy text
  - VLM:   still returns the decoy values

Then run compare_ba.py on the crossover images to confirm VLM readability.

Usage:
    uv run python decoy_alpha_sweep.py \
        --out outputs/alpha_sweep/

Output structure:
    outputs/alpha_sweep/
        rgb_alpha_0.03.png
        rgb_alpha_0.05.png
        ...
        blue_alpha_0.03.png
        ...
        chroma_alpha_0.03.png
        ...
        sweep_grid.png          # all images in one visual grid for easy comparison
        sweep_manifest.csv      # image path + alpha + channel for feeding to models
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",    default="outputs/alpha_sweep/")
    p.add_argument("--width",  type=int, default=800)
    p.add_argument("--height", type=int, default=300)
    p.add_argument("--font-size", type=int, default=18)
    return p.parse_args()


# ── Decoy content ─────────────────────────────────────────────────────────────

DECOY_LINES = [
    "Account holder: Brian Johnson",
    "Account number: 1121 2312 9392 9090",
    "Account type: Checking Account",
    "Opening balance: $5,000.00",
    "Closing balance: $3,212.01",
]

BACKGROUND_COLOR = (245, 245, 245)   # light grey — neutral background
TEXT_COLOR_RGB   = (30, 30, 30)      # dark text

# Alpha levels to sweep
ALPHAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Channel modes
CHANNEL_MODES = ["rgb", "blue", "chroma", "hue_shift", "checkerboard"]


# ── Render decoy text onto white background ───────────────────────────────────

def render_decoy_text(
    width: int,
    height: int,
    font_size: int,
) -> np.ndarray:
    """
    Render decoy lines as black text on white background.
    Returns (H, W, 3) uint8 array — text is dark, background is white.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    y = 20
    line_height = font_size + 8
    for line in DECOY_LINES:
        draw.text((20, y), line, fill=TEXT_COLOR_RGB, font=font)
        y += line_height

    return np.array(img).astype(np.float32)


# ── Injection modes ───────────────────────────────────────────────────────────

def inject_rgb(background: np.ndarray, decoy: np.ndarray, alpha: float) -> np.ndarray:
    """
    Standard alpha blend: all three channels equally.
    background and decoy are (H, W, 3) float32 [0, 255].
    """
    # Text mask: pixels where decoy is dark (text)
    text_mask = (decoy.mean(axis=2) < 200).astype(np.float32)[:, :, np.newaxis]
    result = background.copy()
    # Blend text pixels toward decoy color
    result = result * (1 - text_mask * alpha) + decoy * text_mask * alpha
    return np.clip(result, 0, 255)


def inject_blue_only(background: np.ndarray, decoy: np.ndarray, alpha: float) -> np.ndarray:
    """
    Inject decoy signal into blue channel only.
    Humans are least sensitive to blue channel perturbations.
    VLMs process all channels with equal weight.
    Text appears as a slight blue tint — humans may not perceive it as text.
    """
    text_mask = (decoy.mean(axis=2) < 200).astype(np.float32)
    result = background.copy()
    # Darken blue channel at text pixels
    result[:, :, 2] = np.clip(
        result[:, :, 2] - text_mask * alpha * 255,
        0, 255
    )
    return result


def inject_chroma(background: np.ndarray, decoy: np.ndarray, alpha: float) -> np.ndarray:
    """
    Inject decoy into chrominance channels (Cb, Cr) in YCbCr space.
    Human luminance sensitivity >> chrominance sensitivity.
    Luminance (Y) is preserved — only color shifts, not brightness.
    VLMs process raw RGB and see the full signal.
    """
    from PIL import Image as PILImage

    text_mask = (decoy.mean(axis=2) < 200).astype(np.float32)

    # Convert background to YCbCr
    bg_pil  = PILImage.fromarray(background.astype(np.uint8)).convert("YCbCr")
    bg_ycbcr = np.array(bg_pil).astype(np.float32)

    # Decoy signal: shift Cb and Cr channels at text positions
    # Shift toward blue (Cb up, Cr down) — a subtle color cast
    bg_ycbcr[:, :, 1] = np.clip(
        bg_ycbcr[:, :, 1] + text_mask * alpha * 60,   # Cb shift
        0, 255
    )
    bg_ycbcr[:, :, 2] = np.clip(
        bg_ycbcr[:, :, 2] - text_mask * alpha * 40,   # Cr shift
        0, 255
    )

    # Convert back to RGB
    result_pil = PILImage.fromarray(bg_ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")
    return np.array(result_pil).astype(np.float32)


def inject_hue_shift(background: np.ndarray, decoy: np.ndarray, alpha: float) -> np.ndarray:
    """
    Shift hue at text pixels while preserving luminance exactly.
    Human vision is ~10x less sensitive to chromatic contrast than luminance.
    VLMs process all channels equally — hue shift is fully visible to them.
    alpha controls hue rotation magnitude (alpha=1.0 → 30 degree shift).
    Uses vectorized matplotlib colorsys for speed.
    """
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

    text_mask = (decoy.mean(axis=2) < 200).astype(np.float32)
    result = background.copy() / 255.0
    hue_delta = alpha * 0.3   # max 30 degree (0.083 turn) hue shift at alpha=1.0

    # Vectorized HSV conversion
    hsv = rgb_to_hsv(result)                          # (H, W, 3) in [0,1]
    hsv_shifted = hsv.copy()
    hsv_shifted[:, :, 0] = (hsv[:, :, 0] + hue_delta * text_mask) % 1.0
    # Boost saturation at text pixels so hue shift is more detectable by model
    hsv_shifted[:, :, 1] = np.clip(
        hsv[:, :, 1] + alpha * 0.4 * text_mask, 0, 1
    )
    result_shifted = hsv_to_rgb(hsv_shifted)

    # Blend: only modify text pixels
    result = result * (1 - text_mask[:, :, np.newaxis]) + \
             result_shifted * text_mask[:, :, np.newaxis]

    return (result * 255).astype(np.float32)


def inject_checkerboard(background: np.ndarray, decoy: np.ndarray, alpha: float) -> np.ndarray:
    """
    Encode text as alternating pixel checkerboard pattern.
    Human eyes spatially average fine checkerboard → perceive uniform grey.
    ViT patch tokens see alternating pixel values directly — potentially decodable.
    alpha controls perturbation magnitude (±alpha*255 per pixel).
    """
    text_mask = (decoy.mean(axis=2) < 200).astype(np.float32)
    result = background.copy()

    # Checkerboard: +1 at even pixels, -1 at odd pixels
    y_idx, x_idx = np.indices(result.shape[:2])
    checker = ((y_idx + x_idx) % 2).astype(np.float32) * 2 - 1  # ±1

    perturbation = checker * alpha * 255   # (H, W)

    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c] + text_mask * perturbation,
            0, 255
        )
    return result


INJECTORS = {
    "rgb":          inject_rgb,
    "blue":         inject_blue_only,
    "chroma":       inject_chroma,
    "hue_shift":    inject_hue_shift,
    "checkerboard": inject_checkerboard,
}


# ── Generate sweep ────────────────────────────────────────────────────────────

def generate_sweep(
    out_dir: Path,
    width: int,
    height: int,
    font_size: int,
) -> list[dict]:
    """
    Generate all sweep images. Returns manifest rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render decoy text once
    decoy = render_decoy_text(width, height, font_size)

    # Background: neutral light grey (not pure white — more like a document)
    background = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.float32)

    manifest = []

    for mode in CHANNEL_MODES:
        injector = INJECTORS[mode]
        for alpha in ALPHAS:
            result = injector(background.copy(), decoy, alpha)
            result_uint8 = result.astype(np.uint8)
            img = Image.fromarray(result_uint8)

            fname = f"{mode}_alpha_{alpha:.2f}.png"
            fpath = out_dir / fname
            img.save(fpath)

            manifest.append({
                "path":    str(fpath),
                "mode":    mode,
                "alpha":   alpha,
                "fname":   fname,
            })
            print(f"  saved: {fname}")

    # Also save the pure decoy at full opacity for reference
    ref = Image.fromarray(
        inject_rgb(background.copy(), decoy, 1.0).astype(np.uint8)
    )
    ref.save(out_dir / "reference_full_opacity.png")
    print(f"  saved: reference_full_opacity.png")

    return manifest


# ── Summary grid ──────────────────────────────────────────────────────────────

def save_grid(manifest: list[dict], out_dir: Path):
    """
    One row per channel mode, one column per alpha level.
    Makes it easy to visually scan the perceptibility threshold.
    """
    n_cols = len(ALPHAS)
    n_rows = len(CHANNEL_MODES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 2.0),
        squeeze=False,
    )
    fig.suptitle(
        "Decoy Alpha Sweep — Find where human can't read but VLM can\n"
        "Rows: channel mode | Columns: alpha level",
        fontsize=12, y=1.01,
    )

    # Group manifest
    by_mode: dict[str, dict[float, str]] = {m: {} for m in CHANNEL_MODES}
    for row in manifest:
        by_mode[row["mode"]][row["alpha"]] = row["path"]

    for r, mode in enumerate(CHANNEL_MODES):
        for c, alpha in enumerate(ALPHAS):
            ax = axes[r][c]
            path = by_mode[mode].get(alpha)
            if path:
                img = Image.open(path)
                ax.imshow(img)
            ax.set_title(f"α={alpha:.2f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(mode, fontsize=9, rotation=90, labelpad=4)
            ax.axis("off")

    plt.tight_layout()
    grid_path = out_dir / "sweep_grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: sweep_grid.png")


# ── Manifest CSV ──────────────────────────────────────────────────────────────

def save_manifest(manifest: list[dict], out_dir: Path):
    path = out_dir / "sweep_manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "mode", "alpha", "fname"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"  saved: sweep_manifest.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)

    print(f"Output dir : {out_dir}")
    print(f"Alphas     : {ALPHAS}")
    print(f"Modes      : {CHANNEL_MODES}")
    print(f"Images     : {len(ALPHAS) * len(CHANNEL_MODES)} + 1 reference")
    print()

    manifest = generate_sweep(out_dir, args.width, args.height, args.font_size)
    print()
    save_grid(manifest, out_dir)
    save_manifest(manifest, out_dir)

    print()
    print("═" * 60)
    print("  NEXT STEPS")
    print("═" * 60)
    print("  1. Open sweep_grid.png — visually identify lowest alpha")
    print("     per channel mode where you can still read the text")
    print("  2. Note the crossover alpha per mode (readable → unreadable)")
    print("  3. Run VLM on images just below your threshold:")
    print()
    print("     uv run python compare_ba.py \\")
    print("         --model qwen2_5vl \\")
    print("         --image-a <your_document.png> \\")
    print("         --image-b outputs/alpha_sweep/<mode>_alpha_<X>.png \\")
    print("         --gt <gt.json>")
    print()
    print("  4. If VLM returns decoy values at your imperceptible threshold")
    print("     → imperceptibility gap exists → proceed to learned attack")
    print("  5. If VLM also fails to read at your threshold")
    print("     → no gap with rendered text → must use adversarial noise")
    print("═" * 60)


if __name__ == "__main__":
    main()