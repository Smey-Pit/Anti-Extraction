# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/data/dataset.py
# ══════════════════════════════════════════════════════════════════════════════
"""
TextImageDataset — loads your actual 50-image synthetic dataset.

Your labels.jsonl schema (per sample):
  image_id, image_path, full_text, lines[], line_boxes[], word_boxes[][],
  char_boxes[][], text_category, num_lines, num_chars, font_family,
  font_size_pt, font_size_label, font_bold, text_color, bg_color,
  background_type, contrast_level, layout_type, img_width, img_height, split

This loader is schema-aware: it extracts every field you have and makes it
available on the sample object for downstream use in:
  - L_shape  (word_boxes)
  - L_conf   (char_boxes)
  - Analysis (font metadata, contrast_level, text_category)
  - Logging  (all metadata fields logged with every result record)

Backwards compatible: if a field is absent, it defaults to None gracefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ── Box type aliases ──────────────────────────────────────────────────────────
# All boxes are [x0, y0, x1, y1] in pixel coordinates.
Box  = list[int]
WordBox  = dict   # {"word": str, "box": Box}
CharBox  = dict   # {"char": str, "box": Box}
LineBox  = dict   # {"line": str, "box": Box}


@dataclass
class TextImageSample:
    # ── Identity ──────────────────────────────────────────────────────────────
    image_id:   str
    image_path: Path

    # ── Image ─────────────────────────────────────────────────────────────────
    image:        Image.Image    # original PIL RGB (resized to image_size)
    image_tensor: torch.Tensor   # (3, H, W) float32 [0, 1]

    # ── Ground-truth text ─────────────────────────────────────────────────────
    transcript: str              # full_text (multi-line, \n separated)
    lines:      list[str]        # individual lines

    # ── Spatial annotations ───────────────────────────────────────────────────
    # word_boxes: flat list of [x0,y0,x1,y1] — used by L_shape
    word_boxes:  list[Box]
    # char_boxes: flat list of {"char": str, "box": Box} — used by L_conf (stage 3)
    char_boxes:  list[CharBox]
    # line_boxes: list of {"line": str, "box": Box}
    line_boxes:  list[LineBox]

    # ── Text metadata — logged with every result record ───────────────────────
    text_category:   str          # e.g. "short_structured"
    num_lines:       int
    num_chars:       int
    font_family:     str
    font_size_pt:    int
    font_size_label: str          # "small" | "medium" | "large"
    font_bold:       bool
    text_color:      list[int]    # [R, G, B]
    bg_color:        list[int]    # [R, G, B]
    background_type: str
    contrast_level:  str          # "high" | "medium" | "low"
    layout_type:     str          # "single_line" | "multi_line"
    img_width:       int
    img_height:      int
    split:           str          # "debug" | "train" | "test"

    # ── Scale factors for box coordinates after resize ────────────────────────
    # If the image was resized, these allow rescaling boxes back to pixel coords.
    # scale_x = new_W / orig_W,  scale_y = new_H / orig_H
    scale_x: float = 1.0
    scale_y: float = 1.0

    def scaled_word_boxes(self) -> list[Box]:
        """
        Returns word_boxes rescaled to match the loaded image_tensor dimensions.
        Always use this when passing boxes to loss functions, not raw word_boxes.
        """
        return _scale_boxes(self.word_boxes, self.scale_x, self.scale_y)

    def scaled_char_boxes(self) -> list[CharBox]:
        """Char boxes rescaled to image_tensor dimensions."""
        return [
            {"char": cb["char"], "box": _scale_box(cb["box"], self.scale_x, self.scale_y)}
            for cb in self.char_boxes
        ]

    def metadata_dict(self) -> dict:
        """
        Returns all metadata fields as a flat dict for logging.
        Call this when building a result record in run_probe.py / run_attack.py.
        """
        return {
            "text_category":   self.text_category,
            "num_lines":       self.num_lines,
            "num_chars":       self.num_chars,
            "font_family":     self.font_family,
            "font_size_pt":    self.font_size_pt,
            "font_size_label": self.font_size_label,
            "font_bold":       self.font_bold,
            "background_type": self.background_type,
            "contrast_level":  self.contrast_level,
            "layout_type":     self.layout_type,
            "split":           self.split,
        }


class TextImageDataset(Dataset):
    """
    Loads your 50-image synthetic dataset.

    Args:
        data_dir:         root directory — must contain labels.jsonl and images/
        image_size:       (H, W) to resize to. Set to None to keep original size.
        split_filter:     if set, only load samples with this split label
                          e.g. split_filter="debug" for your current 50-image set
        max_samples:      cap on number of samples
        category_filter:  if set, only load samples with this text_category
        contrast_filter:  if set, only load samples with this contrast_level
    """

    def __init__(
        self,
        data_dir:        Path,
        image_size:      Optional[tuple[int, int]] = (512, 512),
        split_filter:    Optional[str] = None,
        max_samples:     Optional[int] = None,
        category_filter: Optional[str] = None,
        contrast_filter: Optional[str] = None,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.image_size = image_size

        labels_path = self.data_dir / "labels.jsonl"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.jsonl not found in {self.data_dir}\n"
                f"Expected: {labels_path}"
            )

        records: list[dict] = []
        with labels_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                if split_filter    and rec.get("split")          != split_filter:
                    continue
                if category_filter and rec.get("text_category")  != category_filter:
                    continue
                if contrast_filter and rec.get("contrast_level") != contrast_filter:
                    continue

                records.append(rec)

        if max_samples is not None:
            records = records[:max_samples]

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TextImageSample:
        rec = self.records[idx]

        # ── Load image ────────────────────────────────────────────────────────
        # image_path in your schema is relative to data_dir
        img_path = self.data_dir / rec["image_path"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size   # PIL: (W, H)

        if self.image_size is not None:
            target_h, target_w = self.image_size
            img = img.resize((target_w, target_h), Image.LANCZOS)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # ── Parse spatial annotations ─────────────────────────────────────────
        # word_boxes: your schema is [[{"word":..,"box":..}, ...], [...]]
        # Flatten to a single list of [x0,y0,x1,y1]
        word_boxes_raw: list[list[WordBox]] = rec.get("word_boxes", [])
        flat_word_boxes: list[Box] = [
            wb["box"]
            for line_words in word_boxes_raw
            for wb in line_words
        ]

        # char_boxes: [[{"char":.., "box":..}, ...], [...]]
        # Flatten to single list of {"char": str, "box": Box}
        char_boxes_raw: list[list[CharBox]] = rec.get("char_boxes", [])
        flat_char_boxes: list[CharBox] = [
            cb
            for line_chars in char_boxes_raw
            for cb in line_chars
        ]

        # line_boxes: [{"line":.., "box": [x0,y0,x1,y1]}, ...]
        line_boxes: list[LineBox] = rec.get("line_boxes", [])

        return TextImageSample(
            image_id=rec["image_id"],
            image_path=img_path,
            image=img,
            image_tensor=tensor,
            transcript=rec["full_text"],
            lines=rec.get("lines", [rec["full_text"]]),
            word_boxes=flat_word_boxes,
            char_boxes=flat_char_boxes,
            line_boxes=line_boxes,
            text_category=rec.get("text_category", "unknown"),
            num_lines=rec.get("num_lines", 1),
            num_chars=rec.get("num_chars", 0),
            font_family=rec.get("font_family", "unknown"),
            font_size_pt=rec.get("font_size_pt", 0),
            font_size_label=rec.get("font_size_label", "unknown"),
            font_bold=rec.get("font_bold", False),
            text_color=rec.get("text_color", [0, 0, 0]),
            bg_color=rec.get("bg_color", [255, 255, 255]),
            background_type=rec.get("background_type", "unknown"),
            contrast_level=rec.get("contrast_level", "unknown"),
            layout_type=rec.get("layout_type", "unknown"),
            img_width=rec.get("img_width", orig_w),
            img_height=rec.get("img_height", orig_h),
            split=rec.get("split", "unknown"),
            scale_x=scale_x,
            scale_y=scale_y,
        )

    def summary(self) -> dict:
        """
        Quick dataset summary — call this before running experiments to
        verify the distribution of your 50-image set.
        """
        from collections import Counter
        cats      = Counter(r.get("text_category",  "?") for r in self.records)
        contrasts = Counter(r.get("contrast_level", "?") for r in self.records)
        layouts   = Counter(r.get("layout_type",    "?") for r in self.records)
        splits    = Counter(r.get("split",          "?") for r in self.records)
        sizes     = Counter(r.get("font_size_label","?") for r in self.records)
        return {
            "n_samples":    len(self.records),
            "categories":   dict(cats),
            "contrast":     dict(contrasts),
            "layout":       dict(layouts),
            "splits":       dict(splits),
            "font_sizes":   dict(sizes),
            "num_chars_min":  min((r.get("num_chars",0) for r in self.records), default=0),
            "num_chars_max":  max((r.get("num_chars",0) for r in self.records), default=0),
            "num_chars_mean": sum(r.get("num_chars",0) for r in self.records) / max(len(self.records),1),
        }


# ── Box scaling helpers ────────────────────────────────────────────────────────

def _scale_box(box: Box, sx: float, sy: float) -> Box:
    x0, y0, x1, y1 = box
    return [int(x0*sx), int(y0*sy), int(x1*sx), int(y1*sy)]


def _scale_boxes(boxes: list[Box], sx: float, sy: float) -> list[Box]:
    return [_scale_box(b, sx, sy) for b in boxes]