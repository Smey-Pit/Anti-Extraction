"""
pil_renderer.py
===============
Varied PIL renderer for UI dataset images.

Replaces the static fixed-position renderer in dataset_generation_ui.py.
Each call produces a visually distinct image for the same content by sampling:

  - Canvas size        : width in [1024, 1280, 1440], height in [800, 900, 1024]
  - Font family        : sans-serif, serif, monospace (from system fonts + fallbacks)
  - Font sizes         : body size drawn from [12, 13, 14, 15, 16]pt with
                         title/heading scaled proportionally
  - Left margin        : [24, 32, 40, 48]px
  - Header height      : [44, 52, 60, 68]px
  - Line spacing       : body_size * factor in [1.3, 1.45, 1.6, 1.75]
  - Background         : white, off-white, very light tint (category-tinted)
  - Header colour      : sampled from a per-category palette of 4 variants
  - Text colour        : near-black with slight variation [15..35, 15..35, 15..35]
  - Paragraph gap      : [8, 12, 16, 20]px between logical sections
  - Word wrap width    : derived from canvas width and font size

All randomness is driven by a seeded rng passed in from outside, so the same
(content, seed) pair always produces the same image — enabling paired renders.
"""

from __future__ import annotations

import pathlib
import random
import textwrap
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# FONT DISCOVERY
# ---------------------------------------------------------------------------

# Ordered candidate lists — first one that loads wins.
# Covers Ubuntu/Debian HPC, macOS, and common conda envs.
_FONT_CANDIDATES = {
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
    # Hard fallback: PIL built-in (ugly but never crashes)
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ---------------------------------------------------------------------------
# LAYOUT SPEC
# ---------------------------------------------------------------------------

@dataclass
class LayoutSpec:
    """All varied parameters for one render. Fully determined by the rng."""
    W: int
    H: int
    margin_left: int
    margin_right: int
    header_h: int
    body_size: int        # base font size for body text
    line_spacing: float   # multiplier: line advance = body_size * line_spacing
    para_gap: int         # extra vertical gap between logical sections
    font_family: str      # "sans" | "serif" | "mono"
    header_color: tuple   # RGB
    subheader_color: tuple
    bg_color: tuple       # canvas background RGB
    text_color: tuple     # main body text RGB
    label_color: tuple    # secondary labels RGB
    accent_color: tuple   # highlights (amounts, diagnosis, etc.)
    good_color: tuple     # positive values (credits, normal labs)
    bad_color: tuple      # negative values (debits, abnormal labs)

    @property
    def title_size(self) -> int:
        return self.body_size + 6

    @property
    def heading_size(self) -> int:
        return self.body_size + 2

    @property
    def small_size(self) -> int:
        return max(self.body_size - 2, 10)

    @property
    def line_h(self) -> int:
        """Pixel height of one body line."""
        return int(self.body_size * self.line_spacing)

    @property
    def small_line_h(self) -> int:
        return int(self.small_size * self.line_spacing)

    @property
    def content_width(self) -> int:
        return self.W - self.margin_left - self.margin_right


# Per-category header colour palettes (4 variants each)
_HEADER_PALETTES = {
    "banking": [
        ((26, 58, 92),   (44, 82, 130)),
        ((15, 40, 80),   (30, 64, 120)),
        ((35, 70, 110),  (55, 95, 150)),
        ((20, 50, 100),  (40, 75, 140)),
    ],
    "medical": [
        ((43, 108, 176), (67, 136, 204)),
        ((30, 90, 160),  (55, 120, 190)),
        ((55, 120, 190), (80, 150, 220)),
        ((25, 80, 150),  (50, 110, 180)),
    ],
    "news": [
        ((192, 57, 43),  (220, 80, 60)),
        ((160, 30, 20),  (190, 55, 40)),
        ((180, 50, 35),  (210, 75, 55)),
        ((140, 20, 10),  (170, 45, 30)),
    ],
    "copyright": [
        ((45, 55, 72),   (70, 82, 100)),
        ((30, 40, 58),   (55, 68, 88)),
        ((55, 65, 82),   (80, 95, 115)),
        ((38, 48, 65),   (62, 76, 96)),
    ],
    "legal": [
        ((60, 45, 20),   (90, 68, 35)),
        ((45, 32, 10),   (75, 55, 25)),
        ((72, 55, 28),   (105, 80, 42)),
        ((38, 28, 8),    (65, 48, 18)),
    ],
    "identity": [
        ((20, 60, 100),  (35, 90, 140)),
        ((10, 45, 85),   (25, 72, 122)),
        ((30, 72, 115),  (48, 102, 155)),
        ((15, 52, 92),   (28, 80, 132)),
    ],
    "communications": [
        ((28, 142, 84),  (45, 170, 105)),
        ((20, 120, 70),  (35, 148, 88)),
        ((38, 155, 95),  (58, 185, 118)),
        ((15, 105, 60),  (28, 132, 78)),
    ],
}

# Background colour options: white, off-white, very light category tint
_BG_OPTIONS = {
    "banking":        [(255,255,255), (252,253,255), (245,248,255)],
    "medical":        [(255,255,255), (253,254,255), (245,250,255)],
    "news":           [(255,255,255), (255,253,252), (255,249,247)],
    "copyright":      [(255,255,255), (255,254,250), (252,250,245)],
    "legal":          [(255,255,255), (255,253,248), (252,249,242)],
    "identity":       [(255,255,255), (248,251,255), (242,247,255)],
    "communications": [(255,255,255), (248,255,251), (242,252,247)],
}


def make_layout(category: str, rng: random.Random) -> LayoutSpec:
    """Sample a fresh LayoutSpec from the rng."""
    W = rng.choice([1024, 1152, 1280, 1440])
    H = rng.choice([800, 900, 1024, 1100])
    margin_left  = rng.choice([24, 32, 40, 48, 56])
    margin_right = rng.choice([24, 32, 40, 48])
    header_h     = rng.choice([44, 52, 60, 68])
    body_size    = rng.choice([12, 13, 14, 15, 16])
    line_spacing = rng.choice([1.30, 1.40, 1.50, 1.60, 1.75])
    para_gap     = rng.choice([8, 10, 12, 16, 20])
    font_family  = rng.choice(["sans", "serif", "mono"])

    palette_idx  = rng.randrange(4)
    hdr, subhdr  = _HEADER_PALETTES[category][palette_idx]
    bg           = rng.choice(_BG_OPTIONS[category])

    # Text colour: near-black with slight warmth/coolness variation
    grey = rng.randint(15, 35)
    text_color   = (grey, grey, grey)
    label_color  = (rng.randint(90, 130),) * 3

    accent_color = hdr                          # category accent = header colour
    good_color   = (rng.randint(25,50), rng.randint(120,160), rng.randint(60,100))
    bad_color    = (rng.randint(160,210), rng.randint(30,60), rng.randint(20,50))

    return LayoutSpec(
        W=W, H=H,
        margin_left=margin_left, margin_right=margin_right,
        header_h=header_h,
        body_size=body_size,
        line_spacing=line_spacing,
        para_gap=para_gap,
        font_family=font_family,
        header_color=hdr,
        subheader_color=subhdr,
        bg_color=bg,
        text_color=text_color,
        label_color=label_color,
        accent_color=accent_color,
        good_color=good_color,
        bad_color=bad_color,
    )


# ---------------------------------------------------------------------------
# DRAWING PRIMITIVES
# ---------------------------------------------------------------------------

class Composer:
    """
    Stateful vertical-flow text compositor.
    Tracks the current y cursor and clips at the canvas bottom.
    """

    def __init__(self, img: Image.Image, draw: ImageDraw.ImageDraw, lay: LayoutSpec):
        self.img  = img
        self.draw = draw
        self.lay  = lay
        self.y    = lay.header_h + lay.para_gap   # start below header

    @property
    def x(self) -> int:
        return self.lay.margin_left

    def _fits(self, needed: int = 0) -> bool:
        return self.y + needed < self.lay.H - 8

    def skip(self, px: int) -> None:
        self.y += px

    def draw_header(self, main_text: str, sub_text: str) -> None:
        """Coloured header strip with main label and sub-label."""
        lay = self.lay
        self.draw.rectangle([0, 0, lay.W, lay.header_h], fill=lay.header_color)

        # Main label (title size, white, bold if available)
        font_title = _load_font(lay.font_family + "_bold", lay.title_size)
        font_sub   = _load_font(lay.font_family, lay.small_size)

        title_y = (lay.header_h - lay.title_size) // 2 - 2
        self.draw.text((lay.margin_left, title_y), main_text,
                       font=font_title, fill=(255, 255, 255))

        # Sub-label right-aligned
        if sub_text:
            sub_w = self.draw.textlength(sub_text, font=font_sub)
            sub_x = lay.W - lay.margin_right - int(sub_w)
            sub_y = (lay.header_h - lay.small_size) // 2
            self.draw.text((sub_x, sub_y), sub_text,
                           font=font_sub, fill=(220, 220, 220))

        self.y = lay.header_h + lay.para_gap

    def draw_subheader(self, text: str) -> None:
        """Coloured secondary bar (account bar, patient bar, etc.)."""
        lay = self.lay
        if not self._fits(lay.heading_size + 8):
            return
        bar_h = lay.heading_size + 14
        self.draw.rectangle(
            [0, self.y - 4, lay.W, self.y + bar_h],
            fill=lay.subheader_color,
        )
        font = _load_font(lay.font_family, lay.heading_size)
        self.draw.text((lay.margin_left, self.y + 2), text,
                       font=font, fill=(240, 240, 240))
        self.y += bar_h + lay.para_gap

    def draw_label_value(self, label: str, value: str,
                         value_color: Optional[tuple] = None) -> None:
        """One label: value line in two font sizes."""
        lay = self.lay
        if not self._fits(lay.line_h + lay.small_line_h):
            return
        font_lbl = _load_font(lay.font_family, lay.small_size)
        font_val = _load_font(lay.font_family + "_bold", lay.body_size)
        vc = value_color or lay.text_color

        self.draw.text((self.x, self.y), label.upper(),
                       font=font_lbl, fill=lay.label_color)
        self.y += lay.small_line_h
        self.draw.text((self.x, self.y), value,
                       font=font_val, fill=vc)
        self.y += lay.line_h + 4

    def draw_inline_fields(self, fields: list[tuple[str, str]]) -> None:
        """
        Render multiple (label, value) pairs side by side on one row.
        Falls back to stacked if canvas too narrow.
        """
        lay = self.lay
        if not self._fits(lay.line_h * 2 + 8):
            return
        n = len(fields)
        col_w = lay.content_width // max(n, 1)
        font_lbl = _load_font(lay.font_family, lay.small_size)
        font_val = _load_font(lay.font_family + "_bold", lay.body_size)

        for i, (label, value) in enumerate(fields):
            cx = self.x + i * col_w
            self.draw.text((cx, self.y), label.upper(),
                           font=font_lbl, fill=lay.label_color)
            self.draw.text((cx, self.y + lay.small_line_h), value,
                           font=font_val, fill=lay.text_color)

        self.y += lay.small_line_h + lay.line_h + lay.para_gap

    def draw_section_title(self, text: str) -> None:
        """Bold section heading."""
        lay = self.lay
        if not self._fits(lay.heading_size + 6):
            return
        font = _load_font(lay.font_family + "_bold", lay.heading_size)
        self.draw.text((self.x, self.y), text,
                       font=font, fill=lay.accent_color)
        # Underline
        line_y = self.y + lay.heading_size + 2
        self.draw.line(
            [(self.x, line_y), (self.x + lay.content_width, line_y)],
            fill=lay.accent_color, width=1,
        )
        self.y += lay.heading_size + 8 + lay.para_gap // 2

    def draw_body_text(self, text: str,
                       color: Optional[tuple] = None,
                       bold: bool = False) -> None:
        """Wrapped body paragraph."""
        lay = self.lay
        family = lay.font_family + ("_bold" if bold else "")
        font = _load_font(family, lay.body_size)
        col = color or lay.text_color

        # Estimate wrap width in characters from pixel width
        avg_char_w = lay.body_size * 0.55
        wrap_w = max(20, int(lay.content_width / avg_char_w))

        for line in textwrap.wrap(text, width=wrap_w):
            if not self._fits(lay.line_h):
                break
            self.draw.text((self.x, self.y), line, font=font, fill=col)
            self.y += lay.line_h

    def draw_small_text(self, text: str, color: Optional[tuple] = None) -> None:
        """Small single line (no wrap)."""
        lay = self.lay
        if not self._fits(lay.small_line_h):
            return
        font = _load_font(lay.font_family, lay.small_size)
        col = color or lay.label_color
        self.draw.text((self.x, self.y), str(text), font=font, fill=col)
        self.y += lay.small_line_h

    def draw_table_row(self, cols: list[str],
                       col_weights: list[float],
                       colors: Optional[list] = None,
                       bold: bool = False) -> None:
        """
        Draw one table row with proportional column widths.
        col_weights: relative widths, e.g. [2, 4, 1, 1] sums to 8 parts.
        """
        lay = self.lay
        if not self._fits(lay.small_line_h + 4):
            return
        family = lay.font_family + ("_bold" if bold else "")
        font   = _load_font(family, lay.small_size)
        total_w = sum(col_weights)
        x = self.x
        colors = colors or [lay.text_color] * len(cols)

        for col_text, weight, col_color in zip(cols, col_weights, colors):
            col_px = int(lay.content_width * weight / total_w)
            # Right-align last column (usually amounts)
            if col_text == cols[-1] and col_text:
                tw = self.draw.textlength(str(col_text), font=font)
                tx = x + col_px - int(tw) - 4
            else:
                tx = x
            self.draw.text((tx, self.y + 2), str(col_text), font=font, fill=col_color)
            x += col_px

        self.y += lay.small_line_h + 4

    def draw_divider(self) -> None:
        lay = self.lay
        if not self._fits(2):
            return
        self.draw.line(
            [(self.x, self.y), (self.x + lay.content_width, self.y)],
            fill=(220, 220, 220), width=1,
        )
        self.y += lay.para_gap

    def draw_highlight_box(self, text: str, color: tuple) -> None:
        """Coloured left-border callout box."""
        lay = self.lay
        avg_char_w = lay.body_size * 0.55
        wrap_w = max(20, int((lay.content_width - 20) / avg_char_w))
        lines = textwrap.wrap(text, width=wrap_w)
        box_h = len(lines) * lay.line_h + 12

        if not self._fits(box_h):
            return

        self.draw.rectangle(
            [self.x, self.y, self.x + lay.content_width, self.y + box_h],
            fill=(*color[:3], 20) if len(color) == 4 else (245, 245, 245),
        )
        self.draw.rectangle(
            [self.x, self.y, self.x + 4, self.y + box_h],
            fill=color,
        )
        font = _load_font(lay.font_family + "_bold", lay.body_size)
        ty = self.y + 6
        for line in lines:
            self.draw.text((self.x + 12, ty), line, font=font, fill=color)
            ty += lay.line_h

        self.y += box_h + lay.para_gap


# ---------------------------------------------------------------------------
# CATEGORY RENDERERS
# ---------------------------------------------------------------------------

def render_banking(data: dict, lay: LayoutSpec, rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    # Header
    c.draw_header(
        data.get("bank_name", ""),
        f"Period: {data.get('statement_period', '')}",
    )

    # Account info bar
    c.draw_inline_fields([
        ("Account Holder", data.get("account_holder", "")),
        ("Account Number", data.get("account_number", "")),
        ("Account Type",   data.get("account_type", "")),
    ])

    # Balances
    c.draw_inline_fields([
        ("Opening Balance", data.get("opening_balance", "")),
        ("Closing Balance", data.get("closing_balance", "")),
    ])

    c.draw_section_title("Transaction History")

    # Table header
    c.draw_table_row(
        ["Date", "Description", "Amount", "Balance"],
        [1.2, 3.5, 1.2, 1.2],
        colors=[lay.label_color] * 4,
        bold=True,
    )
    c.draw_divider()

    for txn in data.get("transactions", []):
        amt = str(txn.get("amount", ""))
        amt_color = lay.bad_color if amt.startswith("-") else lay.good_color
        c.draw_table_row(
            [
                txn.get("date", ""),
                txn.get("description", ""),
                amt,
                txn.get("running_balance", ""),
            ],
            [1.2, 3.5, 1.2, 1.2],
            colors=[lay.text_color, lay.text_color, amt_color, lay.text_color],
        )

    c.skip(lay.para_gap)
    note = data.get("summary_note", "")
    if note:
        c.draw_small_text(f"⚠  {note}", color=(130, 90, 10))

    return img


def render_medical(data: dict, lay: LayoutSpec, rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    c.draw_header(
        data.get("hospital_name", ""),
        f"Visit: {data.get('visit_date', '')}",
    )

    c.draw_inline_fields([
        ("Patient",    data.get("patient_name", "")),
        ("DOB",        data.get("dob", "")),
        ("Patient ID", data.get("patient_id", "")),
        ("Physician",  data.get("attending_physician", "")),
    ])

    c.draw_label_value("Chief Complaint", data.get("chief_complaint", ""))

    c.draw_section_title("Diagnosis")
    c.draw_highlight_box(data.get("diagnosis", ""), color=lay.bad_color)

    c.draw_section_title("Medications")
    for med in data.get("medications", []):
        c.draw_small_text(f"  •  {med}")

    c.skip(lay.para_gap)
    c.draw_section_title("Lab Results")

    c.draw_table_row(
        ["Test", "Result", "Reference Range", "Flag"],
        [2.5, 1.5, 2.0, 1.0],
        colors=[lay.label_color] * 4,
        bold=True,
    )
    c.draw_divider()

    for lab in data.get("lab_results", []):
        flag = lab.get("flag", "Normal")
        flag_color = lay.bad_color if flag != "Normal" else lay.good_color
        c.draw_table_row(
            [lab.get("test",""), lab.get("value",""),
             lab.get("reference_range",""), flag],
            [2.5, 1.5, 2.0, 1.0],
            colors=[lay.text_color, lay.text_color, lay.label_color, flag_color],
        )

    c.skip(lay.para_gap)
    c.draw_section_title("Clinical Notes")
    c.draw_body_text(data.get("clinical_notes", ""))

    c.skip(lay.para_gap)
    fu = data.get("follow_up", "")
    if fu:
        c.draw_small_text(f"Follow-up: {fu}", color=lay.good_color)

    return img


def render_news(data: dict, lay: LayoutSpec, rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    c.draw_header(
        data.get("outlet_name", ""),
        data.get("dateline", ""),
    )

    # Category tag
    cat_tag = data.get("category_tag", "")
    if cat_tag:
        font_tag = _load_font(lay.font_family + "_bold", lay.small_size)
        draw.text((c.x, c.y), cat_tag.upper(), font=font_tag, fill=lay.accent_color)
        c.y += lay.small_line_h + 4

    # Headline — large bold
    headline = data.get("headline", "")
    avg_char_w = (lay.title_size + 2) * 0.55
    wrap_w = max(20, int(lay.content_width / avg_char_w))
    font_hl = _load_font(lay.font_family + "_bold", lay.title_size + 2)
    for line in textwrap.wrap(headline, width=wrap_w):
        if not c._fits(lay.title_size + 4):
            break
        draw.text((c.x, c.y), line, font=font_hl, fill=lay.text_color)
        c.y += lay.title_size + 6

    c.skip(4)
    c.draw_small_text(f"By {data.get('byline', '')}", color=lay.label_color)
    c.skip(lay.para_gap)
    c.draw_divider()

    # Lead paragraph
    c.draw_body_text(data.get("lead_paragraph", ""), bold=True)
    c.skip(lay.para_gap // 2)

    for para in data.get("body_paragraphs", []):
        c.draw_body_text(para)
        c.skip(lay.para_gap // 2)

    # Pull quote
    pq = data.get("pull_quote", "")
    if pq and c._fits(lay.line_h * 2):
        c.skip(lay.para_gap)
        c.draw_highlight_box(f'"{pq}"', color=lay.accent_color)

    # Tags
    tags = data.get("tags", [])
    if tags and c._fits(lay.small_line_h + 4):
        c.draw_small_text("Tags: " + "  ·  ".join(tags), color=lay.label_color)

    return img


def render_copyright(data: dict, lay: LayoutSpec, rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    content_type = data.get("content_type", "book_excerpt")

    c.draw_header(
        data.get("title", ""),
        data.get("author", ""),
    )

    # Publisher / chapter / scene info
    c.draw_inline_fields([
        ("Publisher",        data.get("publisher", "")),
        ("Chapter / Scene",  data.get("chapter_or_scene", "")),
        ("Page",             str(data.get("page_number", ""))),
    ])

    c.draw_divider()

    # Copyright line
    c.draw_small_text(data.get("copyright_line", ""), color=lay.label_color)
    c.skip(lay.para_gap)

    # Main content
    content = data.get("content", "")
    if content_type == "screenplay":
        # Monospace feel for screenplay — indent action lines
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                c.skip(lay.small_line_h // 2)
                continue
            # Slug lines (all caps, short)
            if stripped.isupper() and len(stripped) < 60:
                c.draw_body_text(stripped, bold=True)
            else:
                # Indent dialogue / action
                indent = 40 if line.startswith("    ") or line.startswith("\t") else 0
                font = _load_font(
                    "mono" if content_type == "screenplay" else lay.font_family,
                    lay.body_size,
                )
                if c._fits(lay.line_h):
                    draw.text((c.x + indent, c.y), stripped, font=font, fill=lay.text_color)
                    c.y += lay.line_h
    else:
        # Book excerpt or newspaper feature — wrapped prose
        c.draw_body_text(content)

    return img


def render_legal(data: dict, lay: LayoutSpec, rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    doc_type   = data.get("document_type", "contract")
    type_label = {
        "contract": "CONTRACT", "nda": "NON-DISCLOSURE AGREEMENT",
        "will": "LAST WILL AND TESTAMENT",
        "eviction_notice": "EVICTION NOTICE", "court_filing": "COURT FILING",
    }.get(doc_type, "LEGAL DOCUMENT")

    c.draw_header(data.get("title", type_label), data.get("case_or_ref_number", ""))

    c.draw_inline_fields([
        ("Jurisdiction", data.get("jurisdiction", "")),
        ("Date",         data.get("date", "")),
    ])

    c.draw_section_title("Parties")
    for party in data.get("parties", []):
        c.draw_table_row(
            [party.get("role", ""), party.get("name", "")],
            [1.5, 3.5],
            colors=[lay.label_color, lay.text_color],
        )

    c.skip(lay.para_gap)
    c.draw_small_text(
        "WHEREAS the parties hereto agree to the following terms and conditions:",
        color=lay.label_color,
    )
    c.skip(lay.para_gap // 2)

    c.draw_section_title("Terms and Conditions")
    for clause in data.get("clauses", []):
        heading = f"{clause.get('number','')}  {clause.get('heading','')}"
        c.draw_body_text(heading, bold=True)
        c.draw_body_text(clause.get("text", ""))
        c.skip(lay.para_gap // 2)

    c.skip(lay.para_gap)
    c.draw_section_title("Signatures")
    for sig in data.get("signature_block", []):
        signed = sig.get("date_signed", "") or "____________________"
        c.draw_table_row(
            [sig.get("role",""), sig.get("name",""), f"Date: {signed}"],
            [2, 2.5, 2],
            colors=[lay.label_color, lay.text_color, lay.label_color],
        )

    notary = data.get("notary_note", "")
    if notary and c._fits(lay.small_line_h + 8):
        c.skip(lay.para_gap)
        c.draw_small_text(notary, color=lay.label_color)

    return img


def render_identity(data: dict, lay: LayoutSpec, rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    doc_type   = data.get("document_type", "passport")
    type_label = {
        "passport": "PASSPORT", "drivers_licence": "DRIVER'S LICENCE",
        "national_id": "NATIONAL IDENTITY CARD",
        "employee_id": "EMPLOYEE IDENTIFICATION",
        "insurance_card": "INSURANCE CARD",
    }.get(doc_type, "IDENTITY DOCUMENT")

    c.draw_header(type_label, data.get("issuing_authority", ""))

    c.draw_inline_fields([
        ("Surname",     data.get("surname", "")),
        ("Given Names", data.get("given_names", "")),
    ])
    c.draw_inline_fields([
        ("Date of Birth",        data.get("dob", "")),
        ("Nationality / State",  data.get("nationality_or_state", "")),
        ("Document Number",      data.get("document_number", "")),
    ])
    c.draw_inline_fields([
        ("Issue Date",  data.get("issue_date", "")),
        ("Expiry Date", data.get("expiry_date", "")),
    ])

    additional = data.get("additional_fields", {})
    if additional:
        c.draw_section_title("Additional Information")
        for k, v in additional.items():
            if v and k not in ("mrz_line1", "mrz_line2"):
                label = k.replace("_", " ").title()
                c.draw_label_value(label, str(v))

    mrz1 = additional.get("mrz_line1", "")
    mrz2 = additional.get("mrz_line2", "")
    if mrz1 or mrz2:
        c.skip(lay.para_gap)
        c.draw_section_title("Machine Readable Zone")
        font_mono = _load_font("mono", lay.small_size)
        for line in [mrz1, mrz2]:
            if line and c._fits(lay.small_line_h):
                draw.text((c.x, c.y), str(line), font=font_mono, fill=lay.text_color)
                c.y += lay.small_line_h

    sec = data.get("security_features", [])
    if sec:
        c.skip(lay.para_gap)
        c.draw_small_text("Security Features: " + "  ·  ".join(sec),
                          color=lay.label_color)

    return img


def render_communications(data: dict, lay: LayoutSpec,
                          rng: random.Random) -> Image.Image:
    img  = Image.new("RGB", (lay.W, lay.H), lay.bg_color)
    draw = ImageDraw.Draw(img)
    c    = Composer(img, draw, lay)

    comm_type = data.get("comm_type", "sms_thread")
    platform  = data.get("platform", "Messages")

    participants = data.get("participants", [])
    self_name = next(
        (p["name"] for p in participants if p.get("role") == "self"),
        "Me",
    )
    other_name = next(
        (p["name"] for p in participants if p.get("role") != "self"),
        "Contact",
    )

    header_sub = f"{platform}  ·  {data.get('timestamp','')}"
    if comm_type == "email" and data.get("subject"):
        header_sub = data.get("subject","")

    c.draw_header(other_name if comm_type != "email" else platform, header_sub)

    if comm_type == "email" and data.get("subject"):
        c.draw_label_value("Subject", data.get("subject",""))

    c.draw_small_text(
        "  ·  ".join(p.get("name","") for p in participants),
        color=lay.label_color,
    )
    c.skip(lay.para_gap)
    c.draw_divider()

    for msg in data.get("messages", []):
        is_self   = msg.get("sender") == self_name
        color     = lay.accent_color if is_self else lay.text_color
        sender    = msg.get("sender","")
        time_str  = msg.get("time","")
        text      = msg.get("text","")

        c.draw_small_text(f"{sender}  {time_str}", color=lay.label_color)
        c.draw_body_text(text, color=color)
        c.skip(lay.para_gap // 2)

    ctx = data.get("thread_context","")
    if ctx and c._fits(lay.small_line_h + 8):
        c.skip(lay.para_gap)
        c.draw_small_text(f"[Context: {ctx}]", color=lay.label_color)

    return img


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

CATEGORY_RENDERERS = {
    "banking":        render_banking,
    "medical":        render_medical,
    "news":           render_news,
    "copyright":      render_copyright,
    "legal":          render_legal,
    "identity":       render_identity,
    "communications": render_communications,
}


def render_pil(
    data: dict,
    category: str,
    rng: random.Random,
) -> Image.Image:
    """
    Main entry point. Renders `data` for `category` using layout parameters
    sampled from `rng`. Reproducible for the same rng state.

    Parameters
    ----------
    data     : LLM-generated content dict (from content_bank.json)
    category : one of "banking", "medical", "news", "copyright"
    rng      : seeded random.Random instance

    Returns
    -------
    PIL Image (RGB)
    """
    lay      = make_layout(category, rng)
    renderer = CATEGORY_RENDERERS[category]
    return renderer(data, lay, rng)