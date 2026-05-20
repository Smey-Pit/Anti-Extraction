"""
scripts/verify_watermark_renderer.py

Step 1 verification — ghost watermark renderer.
No GPU required; runs on the login node.

Loads banking_0000.png from the UI dataset, finds "Thompson"'s bounding box,
renders "Henderson" as a ghost overlay at α = 0.10, 0.12, 0.15, then saves
three PNGs and reports pixel-level diff stats so you can confirm the ghost
is rendering at the right scale.

Usage (from project root):
    python scripts/verify_watermark_renderer.py
    python scripts/verify_watermark_renderer.py --out /tmp/wm_verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.watermark.renderer import render_ghost_watermark

DATA_DIR  = Path("data/ui_dataset")
IMAGE_ID  = "banking_0000"
SOURCE    = "Thompson"
TARGET    = "Henderson"
ALPHAS    = [0.10, 0.12, 0.15]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/wm_verify"),
                        help="Directory to save output PNGs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels_path = DATA_DIR / "labels_pil.json"
    print(f"Loading labels from {labels_path} …")
    with labels_path.open() as f:
        labels = json.load(f)

    sample = next((s for s in labels if s["image_id"] == IMAGE_ID), None)
    if sample is None:
        sys.exit(f"ERROR: {IMAGE_ID} not found in labels_pil.json")

    # Flatten nested word_boxes: [lines][words] → flat list
    flat = [(w["word"], w["box"])
            for line in sample["word_boxes"]
            for w in line]

    match = next(((w, b) for w, b in flat if w == SOURCE), None)
    if match is None:
        sys.exit(f"ERROR: word '{SOURCE}' not found in {IMAGE_ID}")
    word, box = match

    image_path = DATA_DIR / sample["image_path"]
    image      = Image.open(image_path)
    orig_arr   = np.array(image.convert("RGB"))

    font_family = sample["layout"].get("font_family", "sans")
    print(f"Image : {image_path}  ({image.size[0]}×{image.size[1]})")
    print(f"Word  : '{SOURCE}' at box {box}  "
          f"(h={box[3]-box[1]:.0f}px  w={box[2]-box[0]:.0f}px)")
    print(f"Font  : {font_family}")
    print()

    all_pass = True
    for alpha in ALPHAS:
        out_img, rec = render_ghost_watermark(
            image, word, box, TARGET,
            alpha=alpha, font_family=font_family,
        )
        out_arr  = np.array(out_img)
        diff     = np.abs(orig_arr.astype(int) - out_arr.astype(int))
        max_diff = int(diff.max())
        n_px     = int(np.count_nonzero(diff.max(axis=2)))

        out_path = args.out / f"wm_{IMAGE_ID}_alpha{int(alpha*100):02d}.png"
        out_img.save(out_path)

        # Sanity checks
        ok = True
        notes = []
        if n_px == 0:
            notes.append("FAIL: no pixels changed — watermark not rendered")
            ok = False
        if max_diff > int(alpha * 255 * 1.5):
            notes.append(f"WARN: max_diff={max_diff} higher than expected "
                         f"(~{int(alpha*255)})")
        if n_px < 20:
            notes.append(f"WARN: only {n_px} pixels changed — box may be too small")

        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  α={alpha:.2f}  font_size={rec.font_size_px}px  "
              f"changed_pixels={n_px:4d}  max_pixel_diff={max_diff:3d}  "
              f"[{status}]  → {out_path}")
        for note in notes:
            print(f"          {note}")

    print()
    if all_pass:
        print("All checks passed.")
        print(f"Inspect the PNGs in {args.out}/ — 'Henderson' should be faintly")
        print("visible behind 'Thompson' at each opacity level.")
    else:
        print("Some checks failed — see notes above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
