"""
dataset_generation.py
====================
Synthetic text-in-image dataset generator for text-extraction robustness research.

Domain 1: controlled, clean, fully-annotated rendered-text images.
All ground truth is derived from rendering time — no OCR used.

Usage:
    python dataset_generation.py
    python dataset_generation.py --num_samples 30 --seed 99 --out_dir my_dataset
    python dataset_generation.py --font_dir /path/to/fonts --num_samples 50

Dependencies:
    pip install Pillow numpy
"""

import argparse
import json
import math
import pathlib
import random
import re
import string
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """All tuneable parameters for dataset generation."""

    # Output
    out_dir: str = "domain1_dataset"

    # Scale
    num_samples: int = 40          # 30–50
    seed: int = 42

    # Image geometry
    img_width: int = 768
    img_height: int = 192          # bumped up so long text fits

    # Fonts
    font_dir: Optional[str] = None  # folder of .ttf/.otf; None → use PIL default

    # Preview grid
    grid_cols: int = 5
    grid_thumb_w: int = 256
    grid_thumb_h: int = 64

    # Split tag (all samples go here for this first-stage dataset)
    default_split: str = "debug"

    # Padding (px) from image edge before text is placed
    padding: int = 12

    # Max retries if text overflows
    max_retries: int = 8

    # Font size buckets (pt)
    font_sizes: dict = field(default_factory=lambda: {
        "small":  18,
        "medium": 26,
        "large":  36,
    })


# ---------------------------------------------------------------------------
# TEXT GENERATORS
# ---------------------------------------------------------------------------

def _rand_upper(n: int, rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=n))


def _rand_lower(n: int, rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=n))


def _rand_digits(n: int, rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=n))


def _rand_alnum(n: int, rng: random.Random) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(rng.choices(chars, k=n))


def _rand_hex(n: int, rng: random.Random) -> str:
    return "".join(rng.choices("0123456789ABCDEF", k=n))


def _rand_from(parts, rng: random.Random) -> str:
    return rng.choice(parts)

def gen_short_structured(rng: random.Random) -> str:
    """Category 1 – short structured strings (roughly 5–20 chars)."""
    templates = [
        lambda: f"{_rand_upper(2, rng)}{rng.randint(1,9)}{_rand_upper(1, rng)}{rng.randint(10,99)}",
        lambda: f"INV-{rng.randint(10000, 99999)}",
        lambda: f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/{rng.randint(2023,2027)}",
        lambda: f"{_rand_upper(2, rng)}{rng.randint(1,9)}-{rng.randint(10,99)}{_rand_upper(1, rng)}",
        lambda: f"#{rng.randint(1000,9999)}",
        lambda: f"REF-{_rand_upper(3, rng)}{rng.randint(100,999)}",
        lambda: f"PO-{rng.randint(100000,999999)}",
        lambda: f"ID-{_rand_alnum(6, rng)}",
        lambda: f"LOT-{rng.randint(100,999)}-{_rand_upper(2, rng)}",
        lambda: f"SN{rng.randint(1000000,9999999)}",
        lambda: f"PKG-{_rand_upper(2, rng)}{rng.randint(1000,9999)}",
        lambda: f"SEC-{_rand_digits(4, rng)}",
        lambda: f"OTP {rng.randint(100000,999999)}",
        lambda: f"PIN {rng.randint(1000,9999)}",
        lambda: f"RM {rng.randint(100,999)}",
        lambda: f"GATE {rng.choice(['A','B','C','D'])}{rng.randint(1,30)}",
        lambda: f"FL-{rng.randint(10,99)}",
        lambda: f"BIN-{rng.randint(100,999)}",
        lambda: f"TRK-{_rand_alnum(8, rng)}",
        lambda: f"UID:{_rand_hex(8, rng)}",
        lambda: f"X{rng.randint(10,99)}-{_rand_upper(2, rng)}-{rng.randint(100,999)}",
        lambda: f"{_rand_upper(3, rng)}-{_rand_digits(3, rng)}",
        lambda: f"{_rand_upper(1, rng)}{_rand_digits(2, rng)}{_rand_upper(2, rng)}{_rand_digits(2, rng)}",
        lambda: f"RMA-{rng.randint(10000,99999)}",
        lambda: f"ORD-{rng.randint(100000,999999)}",
        lambda: f"BATCH {_rand_upper(2, rng)}{rng.randint(10,99)}",
        lambda: f"{rng.randint(1,12):02d}-{rng.randint(1,28):02d}-{str(rng.randint(2023,2027))[2:]}",
        lambda: f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}",
        lambda: f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}",
        lambda: f"V{rng.randint(1,9)}.{rng.randint(0,9)}.{rng.randint(0,9)}",
        lambda: f"MAC-{_rand_hex(2, rng)}:{_rand_hex(2, rng)}:{_rand_hex(2, rng)}",
        lambda: f"ERR_{rng.randint(100,599)}",
        lambda: f"Z-{_rand_alnum(5, rng)}",
        lambda: f"QTY {rng.randint(1,99)}",
        lambda: f"AISLE {rng.choice(['A','B','C','D'])}{rng.randint(1,20)}",
        lambda: f"{_rand_upper(2, rng)}.{_rand_digits(3, rng)}",
        lambda: f"CHK-{_rand_hex(6, rng)}",
        lambda: f"NODE-{rng.randint(1,64)}",
        lambda: f"KEY-{_rand_upper(4, rng)}",
        lambda: f"TXN{rng.randint(100000,999999)}",
    ]
    return rng.choice(templates)()


def gen_name_entity(rng: random.Random) -> str:
    """Category 2 – names / entities (roughly 8–30 chars)."""
    first_names = [
        "Jane", "Marcus", "Olivia", "James", "Priya", "Lena", "Noah", "Aiko",
        "Carlos", "Sofia", "Mina", "Arjun", "Lucas", "Fatima", "Amira",
        "Daniel", "Eva", "Nadia", "Hugo", "Rina", "Maya", "Ethan", "Iris",
        "Leila", "Jonah", "Tariq", "Anya", "Mila", "Owen", "Zara"
    ]
    last_names = [
        "Patel", "Lee", "Chen", "Smith", "Okafor", "Kim", "Torres", "Nkosi",
        "Russo", "Nguyen", "Ibrahim", "Costa", "Santos", "Khan", "Park",
        "Bennett", "Wright", "Morris", "Ali", "Singh", "Rivera", "Lopez",
        "Diaz", "Sharma", "Brooks", "Fischer", "Ivanov", "Tanaka"
    ]
    prefixes = ["Dr", "Prof", "Ms", "Mr", "Mx", ""]
    suffixes = ["", "Jr", "Sr", "PhD", "MD"]
    org_prefixes = [
        "North", "South", "West", "East", "Central", "Metro", "Prime",
        "Summit", "Blue", "Green", "Oak", "Silver", "Grand"
    ]
    org_roots = [
        "Clinic", "Labs", "Group", "Systems", "Logistics", "Pharmacy",
        "Support", "Holdings", "Health", "Partners", "Works", "Dynamics",
        "Services", "Center"
    ]
    departments = [
        "Support", "Billing", "Operations", "Admissions", "Records",
        "Customer Care", "Security", "Research", "IT", "Compliance"
    ]
    titles = [
        "Nurse", "Manager", "Director", "Coordinator", "Assistant",
        "Consultant", "Analyst", "Engineer", "Supervisor", "Specialist"
    ]

    templates = [
        lambda: f"{_rand_from(prefixes, rng)} {rng.choice(first_names)} {rng.choice(last_names)}".strip(),
        lambda: f"{rng.choice(first_names)} {rng.choice(last_names)}",
        lambda: f"{rng.choice(first_names)} {rng.choice(last_names)} {_rand_from(suffixes, rng)}".strip(),
        lambda: f"{_rand_from(prefixes, rng)} {rng.choice(first_names)} {rng.choice(last_names)} {_rand_from(suffixes, rng)}".strip(),
        lambda: f"{rng.choice(first_names)} {rng.choice(last_names[0:10])}-{rng.choice(last_names[10:])}",
        lambda: f"{rng.choice(first_names)} {rng.choice(string.ascii_uppercase)}. {rng.choice(last_names)}",
        lambda: f"{rng.choice(last_names)}, {rng.choice(first_names)}",
        lambda: f"{rng.choice(titles)} {rng.choice(first_names)} {rng.choice(last_names)}",
        lambda: f"{rng.choice(org_prefixes)} {rng.choice(org_roots)}",
        lambda: f"{rng.choice(org_prefixes)} {rng.choice(org_roots)} {rng.choice(['Inc', 'LLC', 'Ltd', 'Group'])}",
        lambda: f"{rng.choice(departments)} Team",
        lambda: f"{rng.choice(departments)} Department",
        lambda: f"{rng.choice(departments)} Desk",
        lambda: f"Unit {rng.choice(string.ascii_uppercase)}{rng.randint(1,9)}",
        lambda: f"Ward {rng.randint(1,20)}",
        lambda: f"Room {rng.randint(100,999)}",
        lambda: f"Suite {rng.randint(100,999)}",
        lambda: f"Building {rng.choice(string.ascii_uppercase)}",
        lambda: f"Station {rng.randint(1,12)}",
        lambda: f"{rng.choice(first_names)} {rng.choice(last_names)} Clinic",
    ]
    return rng.choice(templates)()


def gen_natural_phrase(rng: random.Random) -> str:
    """Category 3 – natural language phrases (roughly 15–60 chars)."""
    fixed_phrases = [
        "Your order has shipped",
        "Payment received successfully",
        "Appointment confirmed for Monday",
        "Please verify your account",
        "Delivery expected by Friday",
        "Your session has expired",
        "New message from support",
        "Invoice due in 7 days",
        "Verification code sent",
        "Thank you for your purchase",
        "Reset link sent to your email",
        "Two-factor authentication enabled",
        "Please update your profile",
        "Your request is being processed",
        "Backup completed successfully",
        "System maintenance starts tonight",
        "Password changed successfully",
        "New device sign-in detected",
        "Subscription renews next week",
        "Connection restored successfully",
        "Shipment delayed due to weather",
        "Download completed successfully",
        "Please review the attached file",
        "Your report is ready to view",
    ]

    subjects = [
        "Your order", "Your payment", "Your account", "The system", "Your session",
        "The package", "Your request", "Your booking", "Your report", "The update",
        "Your transfer", "The meeting", "Your password", "The backup"
    ]
    verbs = [
        "has shipped", "was received", "is confirmed", "needs attention",
        "is now ready", "has expired", "was updated", "is delayed",
        "has been approved", "is complete", "is scheduled", "has failed",
        "has started", "was canceled"
    ]
    tails = [
        "today", "successfully", "for review", "for tomorrow", "this morning",
        "without errors", "at 10:30 AM", "for pickup", "by email", "in your inbox",
        "for processing", "for your records", "within 24 hours", "for the selected date"
    ]
    prompts = [
        "Please sign in again",
        "Please check your inbox",
        "Please contact support",
        "Please try again later",
        "Please confirm your details",
        "Please review the information",
        "Please keep this reference number",
        "Please verify the attached document"
    ]
    status_openers = [
        "Status updated:",
        "Notice:",
        "Reminder:",
        "Alert:",
        "Confirmation:",
        "Update:",
    ]

    templates = [
        lambda: rng.choice(fixed_phrases),
        lambda: f"{rng.choice(subjects)} {rng.choice(verbs)}",
        lambda: f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(tails)}",
        lambda: f"{rng.choice(status_openers)} {rng.choice(fixed_phrases).lower()}",
        lambda: f"{rng.choice(prompts)}",
        lambda: f"{rng.choice(subjects)} {rng.choice(verbs)}. {rng.choice(prompts)}",
        lambda: f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(tails)}.",
        lambda: f"{rng.choice(['Reminder', 'Notice', 'Update'])}: {rng.choice(subjects).lower()} {rng.choice(verbs)}",
    ]
    return rng.choice(templates)()


def gen_semi_structured(rng: random.Random) -> str:
    """Category 4 – mixed semi-structured strings (roughly 10–40 chars)."""
    domains = [
        "gmail.com", "outlook.com", "acme.io", "corp.net", "mailhub.org",
        "northlabs.ai", "sample.co", "service.app"
    ]
    streets = [
        "King St", "Park Ave", "Oak Rd", "Main Blvd", "Cedar Ln", "Maple Dr",
        "Lakeview Rd", "Station St", "Hillcrest Ave", "River Way"
    ]
    first_handles = [
        "john", "alice", "m.lee", "r.patel", "s.kim", "olivia", "marcus",
        "team.ops", "nora", "dchan", "support", "eva"
    ]
    cities = [
        "Melbourne", "Sydney", "Perth", "Brisbane", "Auckland", "Dublin",
        "Toronto", "Seattle", "Austin", "London"
    ]
    products = ["RX", "MD", "DX", "AX", "PRO", "LITE", "CORE"]
    priorities = ["Low", "Normal", "High", "Urgent"]
    labels = ["Ref", "Case", "Ticket", "Order", "Batch", "Auth", "Claim"]

    templates = [
        lambda: f"{rng.choice(first_handles)}.{_rand_lower(3, rng)}@{rng.choice(domains)}",
        lambda: f"{rng.choice(first_handles)}@{rng.choice(domains)}",
        lambda: f"Apt {rng.randint(1,20)}{rng.choice('ABCDE')}, {rng.randint(1,200)} {rng.choice(streets)}",
        lambda: f"{rng.randint(1,999)} {rng.choice(streets)}, {rng.choice(cities)}",
        lambda: f"Ref: {_rand_upper(2, rng)}-{rng.randint(10000,99999)}",
        lambda: f"Ticket ID {_rand_upper(2, rng)}{rng.randint(10,99)}{_rand_upper(2, rng)}",
        lambda: f"SKU-{rng.randint(1000,9999)}-{_rand_upper(2, rng)}",
        lambda: f"Case #{rng.randint(100000,999999)}",
        lambda: f"Order #{rng.randint(100000,999999)}",
        lambda: f"Tracking {_rand_upper(3, rng)}{rng.randint(100000,999999)}",
        lambda: f"Acct ending in {rng.randint(1000,9999)}",
        lambda: f"Card ending in {rng.randint(1000,9999)}",
        lambda: f"Auth code: {_rand_upper(2, rng)}{rng.randint(1000,9999)}",
        lambda: f"Room {rng.randint(100,999)}, Floor {rng.randint(1,20)}",
        lambda: f"Seat {rng.choice(string.ascii_uppercase)}{rng.randint(1,40)}",
        lambda: f"Gate {rng.choice(string.ascii_uppercase)}{rng.randint(1,20)}, Zone {rng.randint(1,6)}",
        lambda: f"{rng.choice(labels)} {_rand_upper(2, rng)}-{rng.randint(1000,9999)}-{_rand_upper(1, rng)}",
        lambda: f"Batch {_rand_upper(3, rng)}-{rng.randint(10,99)}",
        lambda: f"{rng.choice(products)}-{rng.randint(100,999)}-{_rand_upper(2, rng)}",
        lambda: f"Priority: {rng.choice(priorities)}",
        lambda: f"ETA {rng.randint(1,12):02d}/{rng.randint(1,28):02d} {rng.randint(0,23):02d}:{rng.randint(0,59):02d}",
        lambda: f"Meet at {rng.randint(1,12)}:{rng.randint(0,59):02d} PM",
        lambda: f"Desk {rng.choice(string.ascii_uppercase)}-{rng.randint(10,99)}",
        lambda: f"Locker {_rand_upper(1, rng)}{rng.randint(100,999)}",
        lambda: f"Serial: {_rand_hex(4, rng)}-{_rand_hex(4, rng)}",
        lambda: f"IP 10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        lambda: f"Host node-{rng.randint(1,64):02d}",
        lambda: f"User: {rng.choice(first_handles)}",
        lambda: f"Dept: {rng.choice(['Billing', 'Support', 'Ops', 'Records', 'IT'])}",
        lambda: f"Ext. {rng.randint(1000,9999)}",
        lambda: f"Tel: +61 4{rng.randint(10,99)} {rng.randint(100,999)} {rng.randint(100,999)}",
        lambda: f"Fax: +1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}",
        lambda: f"URL: /reset/{_rand_lower(6, rng)}",
        lambda: f"Code {_rand_upper(3, rng)} {_rand_digits(4, rng)}",
        lambda: f"Claim {_rand_upper(2, rng)}{rng.randint(10000,99999)}",
        lambda: f"Patient ID {rng.choice(string.ascii_uppercase)}{rng.randint(100000,999999)}",
        lambda: f"Lab sample {_rand_upper(2, rng)}-{rng.randint(1000,9999)}",
    ]
    return rng.choice(templates)()


TEXT_GENERATORS = {
    "short_structured": gen_short_structured,
    "name_entity":      gen_name_entity,
    "natural_phrase":   gen_natural_phrase,
    "semi_structured":  gen_semi_structured,
}

CATEGORY_LABELS = list(TEXT_GENERATORS.keys())


# ---------------------------------------------------------------------------
# MULTI-LINE LAYOUTS
# ---------------------------------------------------------------------------

def gen_multiline_kv(rng: random.Random) -> list[str]:
    """Key-value layout, 2–3 lines."""
    keys = ["Name", "ID", "Date", "Ref", "Status", "Code", "Order", "Ticket"]
    vals = [gen_name_entity, gen_short_structured, gen_semi_structured]
    n = rng.randint(2, 3)
    used_keys = rng.sample(keys, n)
    lines = [f"{k}: {rng.choice(vals)(rng)}" for k in used_keys]
    return lines


def gen_multiline_plain(rng: random.Random) -> list[str]:
    """2–3 independent phrase lines."""
    cats = [gen_natural_phrase, gen_semi_structured, gen_short_structured]
    n = rng.randint(2, 3)
    return [rng.choice(cats)(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# TYPOGRAPHY / VISUAL SETTINGS
# ---------------------------------------------------------------------------

# Named color palettes: (text_rgb, background_rgb_or_None, contrast_level)
PALETTES = [
    # high contrast
    ((10, 10, 10),     (250, 250, 250), "high"),   # near-black on white
    ((20, 20, 120),    (245, 245, 255), "high"),   # dark blue on near-white
    ((100, 10, 10),    (255, 248, 245), "high"),   # dark red on near-white
    ((10, 80, 10),     (240, 255, 240), "high"),   # dark green on near-white
    # medium contrast
    ((60, 60, 60),     (220, 220, 220), "medium"), # dark gray on light gray
    ((80, 50, 10),     (240, 230, 210), "medium"), # dark brown on cream
    ((30, 30, 100),    (210, 215, 240), "medium"), # dark indigo on light blue
]


@dataclass
class TypographySpec:
    font_family: str     # "sans", "serif", "mono"
    font_size_label: str # "small", "medium", "large"
    font_size_pt: int
    bold: bool
    text_color: tuple
    bg_color: tuple
    contrast_level: str
    background_type: str  # "plain" | "noisy"


def load_fonts(font_dir: Optional[str]) -> dict[str, list[pathlib.Path]]:
    """
    Load font paths from a directory, categorised by family.
    Falls back to empty lists (PIL default font used downstream).
    """
    categorised: dict[str, list[pathlib.Path]] = {"sans": [], "serif": [], "mono": []}
    if font_dir is None:
        return categorised
    d = pathlib.Path(font_dir)
    if not d.exists():
        print(f"[warn] Font dir '{font_dir}' not found; using PIL defaults.", file=sys.stderr)
        return categorised
    for p in sorted(d.glob("**/*.[ot]tf")):
        low = p.stem.lower()
        if any(kw in low for kw in ("mono", "courier", "consol", "inconsolata", "hack", "jetbrains")):
            categorised["mono"].append(p)
        elif any(kw in low for kw in ("serif", "georgia", "times", "garamond", "merriweather", "playfair")):
            categorised["serif"].append(p)
        else:
            categorised["sans"].append(p)
    total = sum(len(v) for v in categorised.values())
    print(f"[fonts] Loaded {total} fonts: "
          f"sans={len(categorised['sans'])} serif={len(categorised['serif'])} "
          f"mono={len(categorised['mono'])}")
    return categorised


def pick_font(family: str, size_pt: int, bold: bool,
              font_paths: dict[str, list[pathlib.Path]],
              rng: random.Random) -> ImageFont.ImageFont:
    """Return a PIL font, falling back gracefully."""
    candidates = font_paths.get(family, [])
    # Prefer bold variants if requested
    if bold:
        bold_cands = [p for p in candidates if "bold" in p.stem.lower()]
        if bold_cands:
            candidates = bold_cands
    if candidates:
        chosen = rng.choice(candidates)
        try:
            return ImageFont.truetype(str(chosen), size_pt)
        except Exception as e:
            print(f"[warn] Could not load {chosen}: {e}", file=sys.stderr)
    # Fallback: try any loaded font across families
    for fam in ("sans", "serif", "mono"):
        if font_paths.get(fam):
            try:
                return ImageFont.truetype(str(rng.choice(font_paths[fam])), size_pt)
            except Exception:
                pass
    # Final fallback: PIL built-in (no size control)
    return ImageFont.load_default()


def make_noisy_background(width: int, height: int,
                           bg_color: tuple, rng: random.Random) -> Image.Image:
    """Create a lightly noise-textured background."""
    arr = np.full((height, width, 3), bg_color, dtype=np.uint8)
    noise = rng.randint(0, 12)
    arr = np.clip(arr.astype(np.int16) +
                  np.random.RandomState(rng.randint(0, 2**31)).randint(-noise, noise + 1, arr.shape),
                  0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def make_background(width: int, height: int, bg_color: tuple,
                    bg_type: str, rng: random.Random) -> Image.Image:
    if bg_type == "noisy":
        return make_noisy_background(width, height, bg_color, rng)
    return Image.new("RGB", (width, height), bg_color)


# ---------------------------------------------------------------------------
# BOUNDING BOX HELPERS
# ---------------------------------------------------------------------------

def get_text_bbox(draw: ImageDraw.ImageDraw, text: str,
                  font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) via textbbox with anchor='lt'."""
    bb = draw.textbbox((0, 0), text, font=font, anchor="lt")
    return bb  # already (left, top, right, bottom)


def get_word_boxes(draw: ImageDraw.ImageDraw, line: str,
                   font: ImageFont.ImageFont,
                   line_x: int, line_y: int) -> list[dict]:
    """Compute per-word bounding boxes for a single rendered line."""
    boxes = []
    words = line.split()
    cursor_x = line_x
    for word in words:
        wbb = draw.textbbox((cursor_x, line_y), word, font=font, anchor="lt")
        boxes.append({"word": word, "box": list(wbb)})
        # Advance cursor: measure word + space
        space_w = draw.textbbox((0, 0), word + " ", font=font, anchor="lt")[2]
        cursor_x += space_w
    return boxes


def get_char_boxes(draw: ImageDraw.ImageDraw, line: str,
                   font: ImageFont.ImageFont,
                   line_x: int, line_y: int) -> list[dict]:
    """Compute per-character bounding boxes."""
    boxes = []
    cursor_x = line_x
    for ch in line:
        cbb = draw.textbbox((cursor_x, line_y), ch, font=font, anchor="lt")
        boxes.append({"char": ch, "box": list(cbb)})
        cursor_x = cbb[2]  # advance to right edge of this char
    return boxes


# ---------------------------------------------------------------------------
# SAMPLE GENERATION
# ---------------------------------------------------------------------------

@dataclass
class SampleSpec:
    """Fully describes one sample before rendering."""
    sample_id: int
    lines: list[str]
    layout_type: str          # "single_line" | "multi_line" | "key_value"
    text_category: str
    typo: TypographySpec


def build_sample_spec(sample_id: int, rng: random.Random,
                      font_paths: dict, cfg: DatasetConfig) -> SampleSpec:
    """Choose text content + typography for one sample."""
    # --- Layout type distribution ---
    layout_roll = rng.random()
    if layout_roll < 0.60:
        layout_type = "single_line"
    elif layout_roll < 0.80:
        layout_type = "multi_line"
    else:
        layout_type = "key_value"

    # --- Text category ---
    category = rng.choice(CATEGORY_LABELS)

    # --- Generate lines ---
    if layout_type == "single_line":
        text = TEXT_GENERATORS[category](rng)
        lines = [text]
    elif layout_type == "multi_line":
        lines = gen_multiline_plain(rng)
    else:  # key_value
        lines = gen_multiline_kv(rng)
        category = "semi_structured"

    # --- Typography ---
    family = rng.choice(["sans", "serif", "mono"])
    size_label = rng.choice(["small", "medium", "large"])
    size_pt = cfg.font_sizes[size_label]
    bold = rng.random() < 0.35
    palette = rng.choice(PALETTES)
    text_color, bg_color, contrast = palette
    bg_type = rng.choice(["plain", "plain", "noisy"])  # 2:1 plain bias

    typo = TypographySpec(
        font_family=family,
        font_size_label=size_label,
        font_size_pt=size_pt,
        bold=bold,
        text_color=text_color,
        bg_color=bg_color,
        contrast_level=contrast,
        background_type=bg_type,
    )

    return SampleSpec(
        sample_id=sample_id,
        lines=lines,
        layout_type=layout_type,
        text_category=category,
        typo=typo,
    )


def render_sample(spec: SampleSpec, cfg: DatasetConfig,
                  font_paths: dict, rng: random.Random
                  ) -> Optional[tuple[Image.Image, dict]]:
    """
    Render one sample to an image and produce its annotation dict.
    Returns None if text overflows the image boundary.
    """
    W, H = cfg.img_width, cfg.img_height
    pad = cfg.padding
    typo = spec.typo

    font = pick_font(typo.font_family, typo.font_size_pt, typo.bold, font_paths, rng)

    img = make_background(W, H, typo.bg_color, typo.background_type, rng)
    draw = ImageDraw.Draw(img)

    # ---- Measure all lines ----
    line_metrics: list[tuple[int, int, int, int]] = []
    for ln in spec.lines:
        bb = get_text_bbox(draw, ln, font)
        # bb = (left, top, right, bottom) relative to anchor (0,0)
        line_metrics.append(bb)

    line_height  = max(bb[3] - bb[1] for bb in line_metrics)
    line_spacing = int(line_height * 0.25)
    n_lines      = len(spec.lines)
    total_height = n_lines * line_height + (n_lines - 1) * line_spacing
    max_width    = max(bb[2] - bb[0] for bb in line_metrics)

    # ---- Validate fits in image ----
    if total_height > H - 2 * pad or max_width > W - 2 * pad:
        return None  # caller retries

    # ---- Choose top-left origin (centered, with jitter) ----
    x_free = W - 2 * pad - max_width
    y_free = H - 2 * pad - total_height

    x0 = pad + (x_free // 2) + rng.randint(-max(0, x_free // 4), max(0, x_free // 4))
    y0 = pad + (y_free // 2) + rng.randint(-max(0, y_free // 4), max(0, y_free // 4))

    # ---- Draw lines and collect boxes ----
    line_boxes_out   = []
    word_boxes_out   = []
    char_boxes_out   = []

    cursor_y = y0
    for i, (ln, bb) in enumerate(zip(spec.lines, line_metrics)):
        lx = x0
        ly = cursor_y

        draw.text((lx, ly), ln, font=font, fill=typo.text_color, anchor="lt")

        # Line bounding box in image coords
        lbb = [lx + bb[0], ly + bb[1], lx + bb[2], ly + bb[3]]
        line_boxes_out.append({"line": ln, "box": lbb})

        word_boxes_out.append(get_word_boxes(draw, ln, font, lx, ly))
        char_boxes_out.append(get_char_boxes(draw, ln, font, lx, ly))

        cursor_y += line_height + line_spacing

    # ---- Build annotation dict ----
    full_text = "\n".join(spec.lines)
    num_chars = sum(len(ln) for ln in spec.lines)
    fname = f"sample_{spec.sample_id:04d}.png"

    annotation = {
        "image_id":        f"sample_{spec.sample_id:04d}",
        "image_path":      f"images/{fname}",
        "full_text":       full_text,
        "lines":           spec.lines,
        "line_boxes":      line_boxes_out,
        "word_boxes":      word_boxes_out,
        "char_boxes":      char_boxes_out,
        "text_category":   spec.text_category,
        "num_lines":       n_lines,
        "num_chars":       num_chars,
        "font_family":     typo.font_family,
        "font_size_pt":    typo.font_size_pt,
        "font_size_label": typo.font_size_label,
        "font_bold":       typo.bold,
        "text_color":      list(typo.text_color),
        "bg_color":        list(typo.bg_color),
        "background_type": typo.background_type,
        "contrast_level":  typo.contrast_level,
        "layout_type":     spec.layout_type,
        "img_width":       W,
        "img_height":      H,
        "split":           cfg.default_split,
    }

    return img, annotation


# ---------------------------------------------------------------------------
# PREVIEW GRID
# ---------------------------------------------------------------------------

def make_preview_grid(imgs: list[Image.Image], cfg: DatasetConfig) -> Image.Image:
    """Create a contact-sheet grid from all generated images (thumbnailed)."""
    tw, th = cfg.grid_thumb_w, cfg.grid_thumb_h
    cols    = cfg.grid_cols
    rows    = math.ceil(len(imgs) / cols)
    grid    = Image.new("RGB", (cols * tw, rows * th), (200, 200, 200))
    for idx, im in enumerate(imgs):
        thumb = im.copy()
        thumb.thumbnail((tw, th), Image.LANCZOS)
        # Paste centred in cell
        cx = (idx % cols) * tw + (tw - thumb.width)  // 2
        cy = (idx // cols) * th + (th - thumb.height) // 2
        grid.paste(thumb, (cx, cy))
    return grid


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(annotations: list[dict]) -> None:
    """Print a console summary of dataset composition."""
    def count_by(key):
        counts: dict = {}
        for a in annotations:
            v = str(a[key])
            counts[v] = counts.get(v, 0) + 1
        return counts

    print("\n" + "=" * 52)
    print(f"  Domain 1 Dataset Summary  ({len(annotations)} samples)")
    print("=" * 52)
    for attr in ("text_category", "font_family", "font_size_label",
                 "contrast_level", "background_type", "layout_type", "split"):
        print(f"\n  {attr}:")
        for k, v in sorted(count_by(attr).items()):
            print(f"    {k:<28} {v:>4}")
    print()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def generate_dataset(cfg: DatasetConfig) -> None:
    """
    Top-level pipeline: generate samples, save images, write manifests,
    build preview grid.
    """

    # ---- Setup output dirs ----
    out  = pathlib.Path(cfg.out_dir)
    imgs_dir = out / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # ---- RNG ----
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)  # for noise generation

    # ---- Fonts ----
    font_paths = load_fonts(cfg.font_dir)

    # ---- Generate samples ----
    annotations: list[dict] = []
    rendered_imgs: list[Image.Image] = []
    sample_id = 1
    attempts  = 0

    print(f"[gen] Generating {cfg.num_samples} samples …")
    while len(annotations) < cfg.num_samples:
        attempts += 1
        if attempts > cfg.num_samples * cfg.max_retries:
            print(f"[warn] Could not generate enough valid samples after "
                  f"{attempts} attempts. Got {len(annotations)}.", file=sys.stderr)
            break

        spec   = build_sample_spec(sample_id, rng, font_paths, cfg)
        result = render_sample(spec, cfg, font_paths, rng)

        if result is None:
            # text overflowed; retry with same sample_id, different spec
            continue

        img, ann = result

        # Save image
        img_fname = imgs_dir / f"sample_{sample_id:04d}.png"
        img.save(img_fname, format="PNG")

        annotations.append(ann)
        rendered_imgs.append(img)
        sample_id += 1

        if len(annotations) % 10 == 0:
            print(f"  … {len(annotations)}/{cfg.num_samples}")

    print(f"[gen] Done. {len(annotations)} samples in {attempts} attempts.")

    # ---- Write manifests ----
    jsonl_path = out / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    json_path = out / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"[io]  Manifests written: {jsonl_path}, {json_path}")

    # ---- Preview grid ----
    grid = make_preview_grid(rendered_imgs, cfg)
    grid_path = out / "preview_grid.png"
    grid.save(grid_path)
    print(f"[io]  Preview grid: {grid_path} ({grid.width}×{grid.height})")

    # ---- Console summary ----
    print_summary(annotations)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def parse_args() -> DatasetConfig:
    current_dir = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Generate Domain 1 synthetic text-in-image dataset."
    )
    p.add_argument("--num_samples", type=int, default=40,
                   help="Number of samples to generate (30–50).")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--out_dir",     type=str, default= current_dir / "domain1_dataset",
                   help="Output directory.")
    p.add_argument("--font_dir",    type=str, default=None,
                   help="Path to directory of .ttf/.otf fonts.")
    p.add_argument("--img_width",   type=int, default=768)
    p.add_argument("--img_height",  type=int, default=192)
    args = p.parse_args()

    cfg = DatasetConfig(
        num_samples=args.num_samples,
        seed=args.seed,
        out_dir=args.out_dir,
        font_dir=args.font_dir,
        img_width=args.img_width,
        img_height=args.img_height,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    generate_dataset(cfg)


if __name__ == "__main__":
    main()