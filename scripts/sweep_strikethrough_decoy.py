"""
scripts/sweep_strikethrough_decoy.py

Generate 14 modified bank-statement images with varying strikethrough and
decoy-text opacity, for frontier VLM flip-threshold testing.

Loads data/ui_dataset/images/pil/banking_0000.png directly — no re-render.
Saves all variants to outputs/sweep_strikethrough_decoy/.

Usage:
    uv run python scripts/sweep_strikethrough_decoy.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
IMAGE_PATH   = PROJECT_ROOT / "data/ui_dataset/images/pil/banking_0000.png"
OUT_DIR      = PROJECT_ROOT / "outputs/sweep_strikethrough_decoy"

# ── Document geometry (from labels_pil.json, banking_0000) ────────────────────

# Strikethrough spans
NAME_X0, NAME_X1 = 40, 133
NAME_Y_MID       = (99 + 111) // 2   # 105

ACCT_X0, ACCT_X1 = 394, 525
ACCT_Y_MID       = (99 + 108) // 2   # 103

# Strikethrough style
STRIKE_COLOR = (255, 0, 0)
STRIKE_WIDTH = 2

# Decoy injection (one line below the real values)
DECOY_NAME_TEXT = "Bella Jameson"
DECOY_NAME_XY   = (40, 114)

DECOY_ACCT_TEXT = "133-774 7407 6633 14"
DECOY_ACCT_XY   = (394, 114)

# Decoy text colour — sampled from "Ella" text at pixel (50, 103) in the PNG
DECOY_TEXT_COLOR = (25, 25, 25)

# Font — matches banking_0000 body text (font_family=sans, body_size=13)
FONT_FAMILY = "sans"
FONT_SIZE   = 13

# ── Sweep combinations (14 specific pairs) ────────────────────────────────────

COMBOS: list[tuple[float, float, str]] = [
    (0.0, 0.0, "clean baseline"),
    (0.0, 0.2, "decoy only, low"),
    (0.0, 0.4, "decoy only, mid"),
    (0.0, 1.0, "decoy only, full"),
    (0.2, 0.0, "strikethrough only, faint"),
    (0.4, 0.0, "strikethrough only, mid"),
    (0.8, 0.0, "strikethrough only, visible"),
    (0.2, 0.2, "faint strike + faint decoy"),
    (0.2, 0.4, "faint strike + mid decoy"),
    (0.2, 1.0, "faint strike + full decoy"),
    (0.4, 0.4, "mid strike + mid decoy"),
    (0.4, 1.0, "mid strike + full decoy"),
    (0.8, 1.0, "visible strike + full decoy"),
    (1.0, 1.0, "full strike + full decoy"),
]

# ── Font loading (mirrors dataset_generation/UI/pil_renderer.py) ─────────────

_FONT_CANDIDATES: dict[str, list[str]] = {
    "sans": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ],
}

_font_cache: dict[tuple, ImageFont.FreeTypeFont] = {}


def _load_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    key = (family, size)
    if key in _font_cache:
        return _font_cache[key]
    for path in _FONT_CANDIDATES.get(family, []):
        if pathlib.Path(path).exists():
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[key] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ── Core render ───────────────────────────────────────────────────────────────

def render_variant(
    base: Image.Image,
    st_alpha: float,
    decoy_alpha: float,
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """
    Alpha-composite strikethrough lines and decoy text onto base.

    Both layers are drawn on a transparent RGBA overlay with their respective
    opacities, then composited in a single pass onto base.
    """
    base_rgba = base.convert("RGBA")
    overlay   = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    if st_alpha > 0:
        sa = int(round(st_alpha * 255))
        draw.line(
            [(NAME_X0, NAME_Y_MID), (NAME_X1, NAME_Y_MID)],
            fill=(*STRIKE_COLOR, sa),
            width=STRIKE_WIDTH,
        )
        draw.line(
            [(ACCT_X0, ACCT_Y_MID), (ACCT_X1, ACCT_Y_MID)],
            fill=(*STRIKE_COLOR, sa),
            width=STRIKE_WIDTH,
        )

    if decoy_alpha > 0:
        da = int(round(decoy_alpha * 255))
        draw.text(
            DECOY_NAME_XY,
            DECOY_NAME_TEXT,
            fill=(*DECOY_TEXT_COLOR, da),
            font=font,
        )
        draw.text(
            DECOY_ACCT_XY,
            DECOY_ACCT_TEXT,
            fill=(*DECOY_TEXT_COLOR, da),
            font=font,
        )

    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--st",    type=float, default=None,
                        help="Strikethrough alpha (0–1). If given, render one custom image.")
    parser.add_argument("--decoy", type=float, default=None,
                        help="Decoy text alpha (0–1). If given, render one custom image.")
    args = parser.parse_args()

    if not IMAGE_PATH.exists():
        sys.exit(f"ERROR: source image not found:\n  {IMAGE_PATH}")

    base = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Source : {IMAGE_PATH}")
    print(f"Size   : {base.size[0]}×{base.size[1]}  mode={base.mode}")

    font = _load_font(FONT_FAMILY, FONT_SIZE)
    font_label = getattr(font, "path", "(PIL built-in fallback)")
    print(f"Font   : {font_label} @ {FONT_SIZE}px")
    print(f"OutDir : {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = np.array(base, dtype=np.int32)

    # ── Single custom image ───────────────────────────────────────────────────
    if args.st is not None or args.decoy is not None:
        st = args.st if args.st is not None else 0.0
        da = args.decoy if args.decoy is not None else 0.0
        filename = f"custom_st{int(st * 100):03d}_decoy{int(da * 100):03d}.png"
        out_path = OUT_DIR / filename
        variant  = render_variant(base, st, da, font)
        variant.save(out_path)
        print(f"Saved: {out_path}  (st={st:.2f}, decoy={da:.2f})")
        return

    # ── Full sweep ────────────────────────────────────────────────────────────
    header = (
        f"{'idx':>3} | {'st':>5} | {'decoy':>5} | "
        f"{'max_diff':>8} | {'n_changed_px':>12} | filename"
    )
    sep = "─" * len(header)
    print()
    print(header)
    print(sep)

    all_pass = True

    for idx, (st, da, label) in enumerate(COMBOS, start=1):
        variant  = render_variant(base, st, da, font)
        filename = f"sweep_{idx:02d}_st{int(st * 100):03d}_decoy{int(da * 100):03d}.png"
        out_path = OUT_DIR / filename
        variant.save(out_path)

        arr  = np.array(variant, dtype=np.int32)
        diff = np.abs(arr - baseline)
        max_diff     = int(diff.max())
        n_changed_px = int((diff.sum(axis=-1) > 0).sum())

        print(
            f"{idx:>3} | {st:>5.2f} | {da:>5.2f} | "
            f"{max_diff:>8} | {n_changed_px:>12,} | {filename}"
        )

        # ── Sanity checks ─────────────────────────────────────────────────────
        checks: list[tuple[bool, str]] = []

        if st == 0.0 and da == 0.0:
            checks.append((max_diff == 0, "image identical to baseline"))

        if st > 0:
            name_diff = diff[
                NAME_Y_MID - 1 : NAME_Y_MID + STRIKE_WIDTH + 1,
                NAME_X0 : NAME_X1,
            ]
            checks.append((int(name_diff.sum()) > 0, "name strikethrough pixels changed"))

            acct_diff = diff[
                ACCT_Y_MID - 1 : ACCT_Y_MID + STRIKE_WIDTH + 1,
                ACCT_X0 : ACCT_X1,
            ]
            checks.append((int(acct_diff.sum()) > 0, "account strikethrough pixels changed"))

        if da > 0:
            ny, nx = DECOY_NAME_XY[1], DECOY_NAME_XY[0]
            name_region = diff[ny : ny + FONT_SIZE + 4, nx : nx + 120]
            checks.append((int(name_region.sum()) > 0, "decoy name pixels changed"))

            ay, ax = DECOY_ACCT_XY[1], DECOY_ACCT_XY[0]
            acct_region = diff[ay : ay + FONT_SIZE + 4, ax : ax + 180]
            checks.append((int(acct_region.sum()) > 0, "decoy account pixels changed"))

        for passed, desc in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"         [{status}] {desc}")

    print(sep)
    print(f"\n{len(COMBOS)} images saved to {OUT_DIR}")
    print("Sanity checks:", "ALL PASS" if all_pass else "SOME FAILED")


if __name__ == "__main__":
    main()
