"""
render_from_content.py
======================
Step 2 of 2: Render images from a saved content bank JSON.

Reads the JSON produced by generate_content.py and renders each content item
as a PIL image. Because all text content is fixed (loaded from JSON), the same
bank can be rendered multiple times under different conditions to produce a
paired dataset — all from identical text.

PIL renderer features
---------------------
- 8 font families (sans, serif, mono variants), sampled per image
- Font size variation: small (11–14pt), medium (15–19pt), large (20–26pt)
  sampled per-section, not per-image, so different text blocks have
  different sizes within the same image (realistic for UI layouts)
- Left margin jitter: each text block has an independent x offset
- Inter-line spacing variation: 1.1–1.6× line height multiplier
- Background: white / light-grey / very light tint, with optional
  low-amplitude Gaussian noise (simulates paper / screen texture)
- Header bar: varied height and colour brightness within category palette
- Text block positioning: top-anchor jitter ±20px so the same content
  does not always start at the same y coordinate
- Long lines are word-wrapped to a column width sampled per image (60–100 chars)
- All variation is seeded per (content_id, seed) so renders are fully
  reproducible: re-running with the same --seed always gives identical images

Usage
-----
    # Render all items in the default bank
    python render_from_content.py --content content_bank.json

    # Custom output directory, specific seed, specific categories
    python render_from_content.py \\
        --content content_bank.json \\
        --out-dir renders/condition_pil_seed99 \\
        --seed 99 \\
        --categories banking medical

    # Playwright mode (requires playwright + chromium installed)
    python render_from_content.py \\
        --content content_bank.json \\
        --mode playwright \\
        --out-dir renders/condition_playwright
"""
import os
import argparse
import asyncio
import json
import pathlib
import random
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# PIL imports — guarded so syntax check passes even if Pillow not installed
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

CATEGORIES = ["banking", "medical", "news", "copyright"]

# Canvas size
W, H = 1280, 900

# ---------------------------------------------------------------------------
# FONT CATALOGUE
# ---------------------------------------------------------------------------
# Each entry: (path_pattern, family_tag)
# We try each in order and fall back to PIL default if nothing loads.

FONT_CANDIDATES = [
    # DejaVu (almost always present on Linux)
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          "sans"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",     "sans-bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",         "serif"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",    "serif-bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",      "mono"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "mono-bold"),
    # Liberation (common on CentOS/RHEL/HPC)
    ("/usr/share/fonts/liberation/LiberationSans-Regular.ttf",   "sans"),
    ("/usr/share/fonts/liberation/LiberationSans-Bold.ttf",      "sans-bold"),
    ("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",  "serif"),
    ("/usr/share/fonts/liberation/LiberationMono-Regular.ttf",   "mono"),
    # Ubuntu fonts
    ("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",            "sans"),
    ("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",            "sans-bold"),
    ("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",        "mono"),
    # Noto (broad Unicode coverage)
    ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",      "sans"),
    ("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",     "serif"),
    # macOS system fonts (for local dev)
    ("/System/Library/Fonts/Helvetica.ttc",                      "sans"),
    ("/System/Library/Fonts/Times New Roman.ttf",                "serif"),
    ("/System/Library/Fonts/Courier New.ttf",                    "mono"),
]

# Size buckets in points
SIZE_BUCKETS = {
    "small":  (11, 14),
    "medium": (15, 19),
    "large":  (20, 26),
    "xlarge": (27, 34),
}

# Category header colour palettes: (dark, light) variants
HEADER_PALETTES = {
    "banking":   [(26, 58, 92),   (18, 44, 74),   (38, 78, 118)],
    "medical":   [(43, 108, 176), (30, 85, 148),  (60, 130, 195)],
    "news":      [(192, 57, 43),  (160, 40, 30),  (210, 75, 60)],
    "copyright": [(45, 55, 72),   (30, 40, 58),   (65, 78, 98)],
}

# Background styles
BG_STYLES = ["white", "light_grey", "very_light_tint", "noisy_white"]


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    content_path: str  = "content_bank.json"
    out_dir:      str  = "renders/pil_default"
    seed:         int  = 42
    categories:   list = field(default_factory=lambda: list(CATEGORIES))
    mode:         str  = "pil"          # "pil" | "playwright"
    split:        str  = "train"


# ---------------------------------------------------------------------------
# FONT LOADER
# ---------------------------------------------------------------------------

def _load_font_catalogue() -> dict:
    """
    Returns dict: family_tag -> list of (path, ImageFont.FreeTypeFont-loader).
    Only includes fonts that actually exist on this system.
    """
    if not _PIL_OK:
        return {}
    catalogue: dict = {}
    for path, tag in FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            catalogue.setdefault(tag, []).append(path)
    return catalogue


_FONT_CACHE: dict = {}   # (path, size) -> ImageFont


def get_font(path: Optional[str], size: int) -> "ImageFont.FreeTypeFont":
    """Cached font loader. Falls back to PIL default on any error."""
    if not _PIL_OK:
        return None
    key = (path, size)
    if key not in _FONT_CACHE:
        if path and pathlib.Path(path).exists():
            try:
                _FONT_CACHE[key] = ImageFont.truetype(path, size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# PER-IMAGE LAYOUT SAMPLER
# ---------------------------------------------------------------------------

@dataclass
class LayoutSpec:
    """All randomised layout choices for one image. Seeded per content_id."""
    # Fonts (paths, may be None → PIL default)
    font_body_path:    Optional[str]
    font_header_path:  Optional[str]
    font_mono_path:    Optional[str]
    # Sizes
    size_header:  int   # title / institution name
    size_label:   int   # field labels, section headings
    size_body:    int   # main body text
    size_small:   int   # footnotes, metadata
    # Layout
    left_margin:      int    # base left margin (px)
    header_height:    int    # colour strip height (px)
    line_spacing_mul: float  # multiplier on line height (1.1–1.6)
    wrap_width:       int    # chars per line for word-wrap (60–100)
    top_jitter:       int    # vertical offset of first text block (px)
    # Background
    bg_style:         str
    bg_tint:          tuple  # RGB tint for non-white backgrounds
    # Header colour
    header_color:     tuple  # RGB


def sample_layout(rng: random.Random, catalogue: dict, category: str) -> LayoutSpec:
    """Sample one LayoutSpec from the given seeded RNG."""

    # Font paths
    sans_paths   = catalogue.get("sans",       []) or catalogue.get("sans-bold",  [])
    serif_paths  = catalogue.get("serif",      []) or sans_paths
    mono_paths   = catalogue.get("mono",       []) or sans_paths
    bold_paths   = catalogue.get("sans-bold",  []) or sans_paths

    def _pick(paths):
        return rng.choice(paths) if paths else None

    # Body font: sans or serif, sampled per image
    body_family  = rng.choice(["sans", "serif"])
    body_paths   = catalogue.get(body_family, sans_paths)
    font_body    = _pick(body_paths)
    font_header  = _pick(bold_paths or sans_paths)
    font_mono    = _pick(mono_paths)

    # Sizes: sample independently within each bucket
    def _sz(bucket): return rng.randint(*SIZE_BUCKETS[bucket])

    size_header = _sz("large")
    size_label  = _sz("medium")
    size_body   = _sz("small")
    size_small  = rng.randint(9, 12)

    # Layout
    left_margin      = rng.randint(24, 60)
    header_height    = rng.randint(42, 68)
    line_spacing_mul = round(rng.uniform(1.1, 1.6), 2)
    wrap_width       = rng.randint(60, 100)
    top_jitter       = rng.randint(0, 20)

    # Background
    bg_style = rng.choice(BG_STYLES)
    tints = [(248,248,248),(245,247,252),(252,248,245),(248,252,248),(250,250,255)]
    bg_tint = rng.choice(tints)

    # Header colour: pick one variant from the category palette
    palette = HEADER_PALETTES.get(category, [(60,60,60)])
    header_color = rng.choice(palette)

    return LayoutSpec(
        font_body_path   = font_body,
        font_header_path = font_header,
        font_mono_path   = font_mono,
        size_header      = size_header,
        size_label       = size_label,
        size_body        = size_body,
        size_small       = size_small,
        left_margin      = left_margin,
        header_height    = header_height,
        line_spacing_mul = line_spacing_mul,
        wrap_width       = wrap_width,
        top_jitter       = top_jitter,
        bg_style         = bg_style,
        bg_tint          = bg_tint,
        header_color     = header_color,
    )


# ---------------------------------------------------------------------------
# BACKGROUND GENERATOR
# ---------------------------------------------------------------------------

def make_background(layout: LayoutSpec, rng: random.Random) -> "Image.Image":
    img = Image.new("RGB", (W, H), (255, 255, 255))

    if layout.bg_style == "white":
        pass  # already white

    elif layout.bg_style == "light_grey":
        img = Image.new("RGB", (W, H), (242, 242, 242))

    elif layout.bg_style == "very_light_tint":
        img = Image.new("RGB", (W, H), layout.bg_tint)

    elif layout.bg_style == "noisy_white":
        arr = np.full((H, W, 3), 252, dtype=np.uint8)
        noise_amp = rng.randint(3, 10)
        rng_np = np.random.RandomState(rng.randint(0, 2**31 - 1))
        noise  = rng_np.randint(-noise_amp, noise_amp + 1, arr.shape, dtype=np.int16)
        arr    = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img    = Image.fromarray(arr, "RGB")
        img    = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    return img


# ---------------------------------------------------------------------------
# TEXT DRAWING PRIMITIVES
# ---------------------------------------------------------------------------

def draw_header_bar(draw, layout: LayoutSpec, title: str, subtitle: str = "") -> int:
    """Draw coloured header bar. Returns y-coordinate after bar."""
    hh = layout.header_height
    draw.rectangle([0, 0, W, hh], fill=layout.header_color)

    font_title = get_font(layout.font_header_path, layout.size_header)
    font_sub   = get_font(layout.font_body_path,   layout.size_small)

    title_y = (hh - layout.size_header) // 2 - 2
    draw.text((layout.left_margin, max(4, title_y)), title,
              font=font_title, fill=(255, 255, 255))

    if subtitle:
        sub_y = title_y + layout.size_header + 3
        if sub_y + layout.size_small < hh:
            draw.text((layout.left_margin, sub_y), subtitle,
                      font=font_sub, fill=(210, 225, 245))

    return hh + 4


def draw_section_label(draw, x: int, y: int, text: str, layout: LayoutSpec) -> int:
    """Draw a small ALL-CAPS section label. Returns new y."""
    font = get_font(layout.font_body_path, layout.size_small)
    label = text.upper()
    draw.text((x, y), label, font=font, fill=(130, 130, 140))
    return y + layout.size_small + 4


def draw_text_block(
    draw,
    x: int,
    y: int,
    text: str,
    layout: LayoutSpec,
    font_path: Optional[str] = None,
    size:      Optional[int] = None,
    color:     tuple = (30, 30, 30),
    wrap:      bool  = True,
    max_y:     int   = H - 16,
) -> int:
    """
    Draw a paragraph of text with word-wrap. Returns new y after last line.
    Independent x-margin jitter per block is handled by the caller passing x.
    """
    fp   = font_path if font_path is not None else layout.font_body_path
    sz   = size      if size      is not None else layout.size_body
    font = get_font(fp, sz)

    line_h = int(sz * layout.line_spacing_mul)

    if wrap:
        lines = textwrap.wrap(str(text), width=layout.wrap_width) or [str(text)]
    else:
        lines = str(text).splitlines() or [""]

    for line in lines:
        if y >= max_y:
            break
        draw.text((x, y), line, font=font, fill=color)
        y += line_h

    return y


def draw_key_value(
    draw,
    x: int,
    y: int,
    label: str,
    value: str,
    layout: LayoutSpec,
    value_color: tuple = (20, 20, 60),
) -> int:
    """Draw a label: value pair on one line. Returns new y."""
    font_lbl = get_font(layout.font_body_path,   layout.size_small)
    font_val = get_font(layout.font_header_path,  layout.size_body)
    line_h   = int(layout.size_body * layout.line_spacing_mul)

    draw.text((x, y),       label + ":",   font=font_lbl, fill=(120, 120, 130))
    draw.text((x + 160, y), str(value),    font=font_val, fill=value_color)
    return y + line_h


def draw_divider(draw, x: int, y: int, length: int, color=(210, 210, 220)) -> int:
    draw.line([(x, y), (x + length, y)], fill=color, width=1)
    return y + 6


# ---------------------------------------------------------------------------
# CATEGORY RENDERERS
# ---------------------------------------------------------------------------

def render_banking(data: dict, layout: LayoutSpec, rng: random.Random) -> "Image.Image":
    img  = make_background(layout, rng)
    draw = ImageDraw.Draw(img)

    y = draw_header_bar(
        draw, layout,
        title    = data.get("bank_name", ""),
        subtitle = "Online Banking Portal",
    ) + layout.top_jitter

    x = layout.left_margin

    # Account metadata block
    y = draw_section_label(draw, x, y + 4, "Account Details", layout)
    y = draw_key_value(draw, x, y, "Account Holder", data.get("account_holder", ""), layout)
    y = draw_key_value(draw, x, y, "Account Number", data.get("account_number", ""), layout)
    y = draw_key_value(draw, x, y, "Account Type",   data.get("account_type", ""),   layout)
    y = draw_key_value(draw, x, y, "Period",         data.get("statement_period", ""), layout)
    y = draw_divider(draw, x, y + 4, W - x * 2)

    # Balances
    y = draw_section_label(draw, x, y + 2, "Balances", layout)
    y = draw_key_value(draw, x, y, "Opening", data.get("opening_balance", ""), layout,
                       value_color=(39, 120, 60))
    y = draw_key_value(draw, x, y, "Closing", data.get("closing_balance", ""), layout,
                       value_color=(39, 120, 60))
    y = draw_divider(draw, x, y + 4, W - x * 2)

    # Transactions
    y = draw_section_label(draw, x, y + 2, "Transactions", layout)
    font_txn = get_font(layout.font_mono_path, layout.size_small)
    line_h   = int(layout.size_small * layout.line_spacing_mul)

    for txn in data.get("transactions", [])[:8]:
        if y >= H - 50:
            break
        amt   = str(txn.get("amount", ""))
        color = (180, 40, 40) if amt.startswith("-") else (30, 130, 60)
        desc  = txn.get("description", "")[:45]
        line  = f"{txn.get('date',''):<10}  {desc:<46}  {amt:>12}  {txn.get('running_balance',''):>12}"
        draw.text((x, y), line, font=font_txn, fill=color)
        y += line_h

    y = draw_divider(draw, x, y + 4, W - x * 2)

    # Summary note
    note = data.get("summary_note", "")
    if note and y < H - 40:
        draw_text_block(draw, x, y + 4, f"Note: {note}", layout,
                        color=(140, 90, 10), wrap=True)

    return img


def render_medical(data: dict, layout: LayoutSpec, rng: random.Random) -> "Image.Image":
    img  = make_background(layout, rng)
    draw = ImageDraw.Draw(img)

    y = draw_header_bar(
        draw, layout,
        title    = data.get("hospital_name", ""),
        subtitle = "Patient Health Portal — Confidential",
    ) + layout.top_jitter

    x = layout.left_margin

    # Patient info
    y = draw_section_label(draw, x, y + 4, "Patient Information", layout)
    y = draw_key_value(draw, x, y, "Name",       data.get("patient_name", ""),      layout)
    y = draw_key_value(draw, x, y, "DOB",        data.get("dob", ""),               layout)
    y = draw_key_value(draw, x, y, "Patient ID", data.get("patient_id", ""),        layout)
    y = draw_key_value(draw, x, y, "Visit Date", data.get("visit_date", ""),        layout)
    y = draw_key_value(draw, x, y, "Physician",  data.get("attending_physician", ""), layout)
    y = draw_divider(draw, x, y + 4, W - x * 2)

    # Diagnosis — prominent
    diag = data.get("diagnosis", "")
    if diag and y < H - 80:
        y = draw_section_label(draw, x, y + 4, "Diagnosis", layout)
        diag_font = get_font(layout.font_header_path, layout.size_label)
        draw.rectangle([x - 4, y - 2, W - x + 4, y + layout.size_label + 6],
                       fill=(255, 245, 245))
        draw.text((x, y), diag, font=diag_font, fill=(180, 35, 35))
        y += int(layout.size_label * layout.line_spacing_mul) + 6

    y = draw_divider(draw, x, y + 2, W - x * 2)

    # Complaint
    cc = data.get("chief_complaint", "")
    if cc and y < H - 60:
        y = draw_section_label(draw, x, y + 2, "Chief Complaint", layout)
        y = draw_text_block(draw, x, y, cc, layout, color=(50, 50, 60), wrap=True)

    # Medications
    meds = data.get("medications", [])
    if meds and y < H - 60:
        y = draw_divider(draw, x, y + 4, W - x * 2)
        y = draw_section_label(draw, x, y + 2, "Medications", layout)
        font_med = get_font(layout.font_body_path, layout.size_body)
        line_h   = int(layout.size_body * layout.line_spacing_mul)
        for m in meds[:5]:
            if y >= H - 30:
                break
            draw.text((x + 12, y), f"• {m}", font=font_med, fill=(40, 60, 80))
            y += line_h

    # Lab results
    labs = data.get("lab_results", [])
    if labs and y < H - 60:
        y = draw_divider(draw, x, y + 4, W - x * 2)
        y = draw_section_label(draw, x, y + 2, "Lab Results", layout)
        font_lab = get_font(layout.font_mono_path, layout.size_small)
        line_h   = int(layout.size_small * layout.line_spacing_mul)
        for lab in labs[:5]:
            if y >= H - 20:
                break
            flag  = lab.get("flag", "Normal")
            fc    = (180, 40, 40) if flag != "Normal" else (30, 130, 60)
            line  = (f"{lab.get('test',''):<28}  {lab.get('value',''):>10}"
                     f"  [{lab.get('reference_range','')}]  {flag}")
            draw.text((x, y), line, font=font_lab, fill=fc)
            y += line_h

    # Clinical notes
    notes = data.get("clinical_notes", "")
    if notes and y < H - 50:
        y = draw_divider(draw, x, y + 4, W - x * 2)
        y = draw_section_label(draw, x, y + 2, "Clinical Notes", layout)
        draw_text_block(draw, x, y, notes, layout, color=(55, 55, 65), wrap=True)

    return img


def render_news(data: dict, layout: LayoutSpec, rng: random.Random) -> "Image.Image":
    img  = make_background(layout, rng)
    draw = ImageDraw.Draw(img)

    y = draw_header_bar(
        draw, layout,
        title    = data.get("outlet_name", ""),
        subtitle = data.get("category_tag", ""),
    ) + layout.top_jitter

    x = layout.left_margin

    # Headline — largest text on the page
    headline = data.get("headline", "")
    if headline:
        font_hl = get_font(layout.font_header_path, layout.size_header + 4)
        hl_lines = textwrap.wrap(headline, width=max(30, layout.wrap_width - 10))
        line_h   = int((layout.size_header + 4) * 1.25)
        for hl in hl_lines[:3]:
            if y >= H - 80:
                break
            draw.text((x, y), hl, font=font_hl, fill=(10, 10, 20))
            y += line_h
        y += 6

    # Byline + dateline
    byline   = f"By {data.get('byline', '')}   ·   {data.get('dateline', '')}"
    font_byl = get_font(layout.font_body_path, layout.size_small)
    draw.text((x, y), byline, font=font_byl, fill=(110, 110, 120))
    y += int(layout.size_small * layout.line_spacing_mul) + 4

    y = draw_divider(draw, x, y, W - x * 2)

    # Lead paragraph
    lead = data.get("lead_paragraph", "")
    if lead and y < H - 60:
        font_lead = get_font(layout.font_body_path, layout.size_label - 1)
        y = draw_text_block(draw, x, y + 4, lead, layout,
                            font_path=layout.font_body_path,
                            size=layout.size_label - 1,
                            color=(20, 20, 30), wrap=True)
        y += 8

    # Body paragraphs
    for para in data.get("body_paragraphs", [])[:4]:
        if y >= H - 60:
            break
        y = draw_text_block(draw, x, y, para, layout, color=(40, 40, 50), wrap=True)
        y += 6

    # Pull quote
    pq = data.get("pull_quote", "")
    if pq and y < H - 60:
        draw.line([(x, y + 4), (x + 4, y + 4)], fill=layout.header_color, width=4)
        draw_text_block(draw, x + 14, y + 2, f'"{pq}"', layout,
                        font_path=layout.font_header_path,
                        size=layout.size_label,
                        color=(80, 30, 30), wrap=True)

    return img


def render_copyright(data: dict, layout: LayoutSpec, rng: random.Random) -> "Image.Image":
    content_type = data.get("content_type", "book_excerpt")

    # For screenplay use mono; for prose use serif if available
    if content_type == "screenplay":
        body_path = layout.font_mono_path or layout.font_body_path
        bg_tint   = (252, 251, 246)
    else:
        body_path = layout.font_body_path
        bg_tint   = (253, 252, 248)

    # Override background to paper-like for copyright content
    layout_copy           = LayoutSpec(**layout.__dict__)
    layout_copy.bg_style  = "very_light_tint"
    layout_copy.bg_tint   = bg_tint

    img  = make_background(layout_copy, rng)
    draw = ImageDraw.Draw(img)

    y = draw_header_bar(
        draw, layout,
        title    = data.get("title", ""),
        subtitle = data.get("author", ""),
    ) + layout.top_jitter

    x = layout.left_margin

    # Publisher + chapter
    pub_line = f"{data.get('publisher', '')}   |   {data.get('chapter_or_scene', '')}"
    font_pub = get_font(layout.font_body_path, layout.size_small)
    draw.text((x, y), pub_line, font=font_pub, fill=(140, 140, 150))
    y += int(layout.size_small * layout.line_spacing_mul) + 2

    # Copyright line
    cr_line = data.get("copyright_line", "")
    draw.text((x, y), cr_line, font=font_pub, fill=(160, 160, 160))
    y += int(layout.size_small * layout.line_spacing_mul) + 8

    y = draw_divider(draw, x, y, W - x * 2)
    y += 8

    # Main content
    content = data.get("content", "")
    if content:
        # Screenplay: preserve line structure; prose: wrap
        is_script = content_type == "screenplay"
        draw_text_block(
            draw, x, y, content, layout,
            font_path = body_path,
            size      = layout.size_body,
            color     = (28, 28, 28),
            wrap      = not is_script,
            max_y     = H - 28,
        )

    # Page number bottom-right
    pg = str(data.get("page_number", ""))
    if pg:
        font_pg = get_font(layout.font_body_path, layout.size_small)
        draw.text((W - layout.left_margin - 24, H - 22), pg,
                  font=font_pg, fill=(180, 180, 180))

    return img


CATEGORY_RENDERERS = {
    "banking":   render_banking,
    "medical":   render_medical,
    "news":      render_news,
    "copyright": render_copyright,
}


# ---------------------------------------------------------------------------
# HTML TEMPLATES (imported from dataset_generation_ui for Playwright mode)
# ---------------------------------------------------------------------------

def _get_html_renderer(category: str):
    try:
        from dataset_generation_ui import HTML_RENDERERS
        return HTML_RENDERERS.get(category)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# PLAYWRIGHT SCREENSHOT
# ---------------------------------------------------------------------------

async def playwright_screenshot(html: str, path: Path) -> bool:
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = await browser.new_page(viewport={"width": W, "height": H})
            await page.set_content(html, wait_until="networkidle")
            await page.screenshot(path=str(path), full_page=False)
            await browser.close()
        return True
    except Exception as e:
        print(f"  [playwright] {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# ANNOTATION BUILDER
# ---------------------------------------------------------------------------

def build_annotation(
    item:     dict,
    img_path: str,
    layout:   LayoutSpec,
    cfg:      RenderConfig,
) -> dict:
    from dataset_generation_ui import extract_full_text
    category = item.get("_category", "unknown")
    full_text = extract_full_text(category, item)

    return {
        "image_id":       item.get("_content_id", ""),
        "image_path":     img_path,
        "full_text":      full_text,
        "domain":         "ui_web",
        "category":       category,
        "render_mode":    cfg.mode,
        "render_seed":    cfg.seed,
        "raw_content":    {k: v for k, v in item.items() if not k.startswith("_")},
        "layout": {
            "font_body":        layout.font_body_path,
            "font_header":      layout.font_header_path,
            "font_mono":        layout.font_mono_path,
            "size_header":      layout.size_header,
            "size_label":       layout.size_label,
            "size_body":        layout.size_body,
            "size_small":       layout.size_small,
            "left_margin":      layout.left_margin,
            "header_height":    layout.header_height,
            "line_spacing_mul": layout.line_spacing_mul,
            "wrap_width":       layout.wrap_width,
            "top_jitter":       layout.top_jitter,
            "bg_style":         layout.bg_style,
            "header_color":     list(layout.header_color),
        },
        "img_width":            W,
        "img_height":           H,
        "split":                cfg.split,
        "text_category":        category,
        "layout_type":          "web_portal",
        "has_ambiguous_chars":  False,
    }


# ---------------------------------------------------------------------------
# MAIN RENDER LOOP
# ---------------------------------------------------------------------------

async def render_all(cfg: RenderConfig) -> None:
    if not _PIL_OK:
        print("[error] Pillow not installed. Run: pip install pillow numpy --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    content_path = pathlib.Path(cfg.content_path)
    if not content_path.exists():
        print(f"[error] Content bank not found: {content_path}", file=sys.stderr)
        sys.exit(1)

    all_items: list[dict] = json.loads(content_path.read_text(encoding="utf-8"))
    items = [it for it in all_items if it.get("_category") in cfg.categories]
    print(f"[render] Loaded {len(items)} items from {content_path} "
          f"(filtered to {cfg.categories})")

    out_dir = pathlib.Path(cfg.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    catalogue = _load_font_catalogue()
    if not any(catalogue.values()):
        print("[warn] No system fonts found — PIL will use its built-in bitmap default. "
              "Install dejavu-fonts or liberation-fonts for better visual variety.",
              file=sys.stderr)

    annotations: list[dict] = []
    global_rng = random.Random(cfg.seed)

    for idx, item in enumerate(items):
        content_id = item.get("_content_id", f"item_{idx:04d}")
        category   = item.get("_category",   "unknown")

        # Per-item seed derived from global seed + content_id (reproducible)
        item_seed = cfg.seed ^ hash(content_id) & 0xFFFFFFFF
        item_rng  = random.Random(item_seed)

        layout = sample_layout(item_rng, catalogue, category)

        img_path = img_dir / f"{content_id}.png"

        if cfg.mode == "pil":
            renderer = CATEGORY_RENDERERS.get(category)
            if renderer is None:
                print(f"  [skip] Unknown category '{category}'", file=sys.stderr)
                continue
            try:
                img = renderer(item, layout, item_rng)
                img.save(str(img_path))
            except Exception as e:
                print(f"  [error] {content_id}: {e}", file=sys.stderr)
                continue

        elif cfg.mode == "playwright":
            html_fn = _get_html_renderer(category)
            if html_fn is None:
                print(f"  [skip] No HTML renderer for '{category}'", file=sys.stderr)
                continue
            try:
                html = html_fn(item)
                ok   = await playwright_screenshot(html, img_path)
                if not ok:
                    print(f"  [!] Playwright failed for {content_id}", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"  [error] {content_id}: {e}", file=sys.stderr)
                continue

        ann = build_annotation(item, str(img_path), layout, cfg)
        annotations.append(ann)

        if (idx + 1) % 50 == 0 or idx == len(items) - 1:
            print(f"  [{cfg.mode}] {idx+1}/{len(items)} rendered …", flush=True)

    # Write labels
    (out_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in annotations) + "\n",
        encoding="utf-8",
    )
    (out_dir / "labels.json").write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_summary(annotations, cfg)


def _print_summary(annotations: list[dict], cfg: RenderConfig) -> None:
    from collections import Counter
    cats = Counter(a["category"] for a in annotations)
    print("\n" + "=" * 52)
    print(f"  Render Summary  ({len(annotations)} images, mode={cfg.mode})")
    print("=" * 52)
    for cat in CATEGORIES:
        print(f"  {cat:<20} {cats.get(cat, 0):>5}")
    print(f"\n  Output: {cfg.out_dir}/")
    print(f"  Labels: {cfg.out_dir}/labels.jsonl")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the base data directory
base_data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data"))
def parse_args() -> RenderConfig:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--content",    type=str, default="content_bank.json",
                   help="Path to content bank JSON from generate_content.py")
    p.add_argument("--out-dir",    type=str, default=os.path.join(base_data_dir, "ui_dataset"),
                   help="Output directory for images and labels")
    p.add_argument("--seed",       type=int, default=42,
                   help="Render seed — controls all layout variation. "
                        "Same seed always produces identical images.")
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                   choices=CATEGORIES)
    p.add_argument("--mode",       type=str, default="pil",
                   choices=["pil", "playwright"],
                   help="pil: fast PIL renderer with layout variation. "
                        "playwright: HTML→Chromium screenshot (needs playwright).")
    p.add_argument("--split",      type=str, default="train",
                   help="Split label written into annotations (train/val/test)")

    args = p.parse_args()
    return RenderConfig(
        content_path = args.content,
        out_dir      = args.out_dir,
        seed         = args.seed,
        categories   = args.categories,
        mode         = args.mode,
        split        = args.split,
    )


def main() -> None:
    cfg = parse_args()
    asyncio.run(render_all(cfg))


if __name__ == "__main__":
    main()