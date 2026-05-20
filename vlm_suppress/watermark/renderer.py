"""
vlm_suppress/watermark/renderer.py

Ghost watermark renderer.

Given a PIL image, a source word, its bounding box, and a replacement string,
renders the replacement as a semi-transparent overlay at the same position.
Opacity is set at JND scale (α ≈ 0.10–0.15) so the ghost is sub-perceptual
to casual inspection but provides a visual anchor for targeted PGD.

Usage
-----
    from vlm_suppress.watermark.renderer import render_ghost_watermark

    out_img, record = render_ghost_watermark(
        image       = pil_image,
        word        = "Thompson",
        box         = [120, 84, 210, 100],   # [x0, y0, x1, y1]
        replacement = "Henderson",
        alpha       = 0.12,
        font_family = "sans",
    )
    out_img.save("preview.png")
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font discovery — same candidate chains as dataset_generation/UI/pil_renderer.py
# (copied, not imported, because that module lives outside the vlm_suppress pkg)
# ---------------------------------------------------------------------------

_FONT_CANDIDATES: dict[str, list[str]] = {
    "sans": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ],
    "sans_bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "serif": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/System/Library/Fonts/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
    ],
    "serif_bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    ],
    "mono": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ],
}

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


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


# ---------------------------------------------------------------------------
# Font size estimation
# ---------------------------------------------------------------------------

def _estimate_font_size(
    box: list[int],
    text: str,
    font_family: str,
) -> tuple[int, ImageFont.FreeTypeFont]:
    """
    Find the largest font size where `text` fits within the box.

    Allows up to 1.25× the box width so that replacements slightly longer
    than the original word still receive a reasonable size rather than
    being shrunk to illegibility.  Falls back to box_height × 0.55 if
    nothing fits.
    """
    x0, y0, x1, y1 = box
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)

    hi = int(box_h * 0.90)
    lo = max(6, int(box_h * 0.45))

    for size in range(hi, lo - 1, -1):
        font = _load_font(font_family, size)
        bbox = font.getbbox(text)           # (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]
        if text_w <= box_w * 1.25:
            return size, font

    fallback = max(6, int(box_h * 0.55))
    return fallback, _load_font(font_family, fallback)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class WatermarkRecord:
    word_string: str
    replacement: str
    box: list[int]          # [x0, y0, x1, y1]
    alpha: float
    font_size_px: int
    font_family: str


def render_ghost_watermark(
    image: Image.Image,
    word: str,
    box: list[int],
    replacement: str,
    alpha: float = 0.12,
    font_family: str = "sans",
    text_color: tuple[int, int, int] = (30, 30, 30),
) -> tuple[Image.Image, WatermarkRecord]:
    """
    Render `replacement` as a ghost overlay over `word`'s bounding box.

    Parameters
    ----------
    image       : source PIL image (any mode; converted to RGB on return)
    word        : original word string (for the record only)
    box         : [x0, y0, x1, y1] pixel bounding box of `word`
    replacement : text to ghost-render at that position
    alpha       : opacity in [0, 1]; JND range is 0.08–0.15
    font_family : "sans" | "serif" | "mono" | "sans_bold" | "serif_bold"
    text_color  : RGB tuple for the ghost text; default near-black matches
                  most banking/medical document text colours

    Returns
    -------
    (modified_image, WatermarkRecord)
    """
    img_rgba = image.convert("RGBA")
    overlay  = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    x0, y0, *_ = box
    font_size, font = _estimate_font_size(box, replacement, font_family)

    r, g, b = text_color
    alpha_int = int(alpha * 255)
    draw.text((x0, y0), replacement, fill=(r, g, b, alpha_int), font=font)

    composited = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    record = WatermarkRecord(
        word_string  = word,
        replacement  = replacement,
        box          = list(box),
        alpha        = alpha,
        font_size_px = font_size,
        font_family  = font_family,
    )
    return composited, record
