"""
render_images.py
================
Step 2 of 2: Render images from a saved content bank.

Reads the JSON file produced by generate_content.py and renders each item
using the varied PIL renderer. The same (content_id, global seed) pair always
produces an identical image — enabling fully reproducible paired renders.

Re-run with a different --seed to get different layout variation on the same
text content (different canvas size, font, margins, spacing, etc.).

Usage
-----
# Render everything in the content bank (default seed 42)
python render_images.py --content content_bank.json --out domain2_ui_dataset/

# Different render seed — same content, different layouts
python render_images.py --content content_bank.json --seed 99 --out renders_seed99/

# Specific categories only
python render_images.py --content content_bank.json --categories banking medical

# Dry run — print plan without writing files
python render_images.py --content content_bank.json --dry-run
"""

import argparse
import json
import pathlib
import random
import sys
from dataclasses import dataclass, field

from pil_renderer import render_pil, make_layout

CATEGORIES = ["banking", "medical", "news", "copyright"]


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    content_path: str  = "content_bank.json"
    out_dir:      str  = "domain2_ui_dataset"
    seed:         int  = 42
    categories:   list = field(default_factory=lambda: list(CATEGORIES))
    dry_run:      bool = False
    mode:         str  = "pil"   # label written into annotations


# ---------------------------------------------------------------------------
# SEED STRATEGY
# ---------------------------------------------------------------------------

def make_item_rng(global_seed: int, content_id: str) -> random.Random:
    """
    Deterministic per-item RNG from (global_seed, content_id).
      - Same seed + content_id  →  identical layout every time
      - Different seed           →  different layouts, same text content
      - Item order irrelevant    →  each item is independent
    """
    item_seed = hash((global_seed, content_id)) & 0xFFFFFFFF
    return random.Random(item_seed)


# ---------------------------------------------------------------------------
# CONTENT BANK LOADING
# ---------------------------------------------------------------------------

def load_content_bank(path: pathlib.Path) -> list[dict]:
    """
    Load and normalise a content bank from generate_content.py.

    Supports two schemas:
      flat list  — list of dicts with _category / _content_id keys  (current)
      nested     — {"meta": ..., "items": [...]}                     (legacy)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict) and "items" in raw:
        items = []
        for it in raw["items"]:
            flat = dict(it.get("data", {}))
            flat["_category"]   = it.get("category", "")
            flat["_content_id"] = it.get("content_id", "")
            items.append(flat)
        return items

    raise ValueError(f"Unrecognised content bank format in {path}")


# ---------------------------------------------------------------------------
# ANNOTATION HELPERS
# ---------------------------------------------------------------------------

def extract_full_text(category: str, item: dict) -> str:
    """Flatten item dict into ordered ground-truth text for CER evaluation."""
    parts = []

    if category == "banking":
        parts += [
            item.get("bank_name", ""),
            item.get("account_holder", ""),
            item.get("account_number", ""),
            item.get("account_type", ""),
            item.get("statement_period", ""),
            f"Opening: {item.get('opening_balance','')} "
            f"Closing: {item.get('closing_balance','')}",
        ]
        for t in item.get("transactions", []):
            parts.append(
                f"{t.get('date','')} {t.get('description','')} "
                f"{t.get('amount','')} {t.get('running_balance','')}"
            )
        parts.append(item.get("summary_note", ""))

    elif category == "medical":
        parts += [
            item.get("hospital_name",""),   item.get("patient_name",""),
            item.get("dob",""),             item.get("patient_id",""),
            item.get("visit_date",""),      item.get("attending_physician",""),
            item.get("chief_complaint",""), item.get("diagnosis",""),
        ]
        parts += item.get("medications", [])
        for lab in item.get("lab_results", []):
            parts.append(
                f"{lab.get('test','')} {lab.get('value','')} "
                f"{lab.get('reference_range','')} {lab.get('flag','')}"
            )
        parts.append(item.get("clinical_notes", ""))
        parts.append(item.get("follow_up", ""))

    elif category == "news":
        parts += [
            item.get("outlet_name",""),   item.get("headline",""),
            item.get("byline",""),        item.get("dateline",""),
            item.get("category_tag",""), item.get("lead_paragraph",""),
        ]
        parts += item.get("body_paragraphs", [])
        parts.append(item.get("pull_quote",""))

    elif category == "copyright":
        parts += [
            item.get("title",""),            item.get("author",""),
            item.get("publisher",""),        item.get("copyright_line",""),
            item.get("chapter_or_scene",""), item.get("content",""),
        ]

    return "\n".join(str(p).strip() for p in parts if str(p).strip())


def build_annotation(item: dict, img_path: str,
                     cfg: RenderConfig, layout_meta: dict) -> dict:
    category   = item["_category"]
    content_id = item["_content_id"]
    raw        = {k: v for k, v in item.items() if not k.startswith("_")}

    return {
        "image_id":            content_id,
        "image_path":          img_path,
        "full_text":           extract_full_text(category, item),
        "domain":              "ui_web",
        "category":            category,
        "raw_content":         raw,
        "render_mode":         cfg.mode,
        "render_seed":         cfg.seed,
        "layout":              layout_meta,
        "split":               "train",
        "has_ambiguous_chars": False,
        "layout_type":         "web_portal",
        "text_category":       category,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = RenderConfig(
        content_path = args.content,
        out_dir      = args.out,
        seed         = args.seed,
        categories   = args.categories,
        dry_run      = args.dry_run,
        mode         = args.mode,
    )

    bank_path = pathlib.Path(cfg.content_path)
    if not bank_path.exists():
        print(f"[error] Content bank not found: {bank_path}", file=sys.stderr)
        sys.exit(1)

    all_items = load_content_bank(bank_path)
    items     = [it for it in all_items if it.get("_category") in cfg.categories]

    from collections import Counter
    cat_counts = Counter(it["_category"] for it in items)

    print("=" * 56)
    print("  Render plan")
    print("=" * 56)
    print(f"  Content bank : {bank_path}  ({len(all_items)} total items)")
    print(f"  Mode         : {cfg.mode}")
    print(f"  Seed         : {cfg.seed}")
    print(f"  Rendering    : {len(items)} items")
    for cat in CATEGORIES:
        if cat in cat_counts:
            print(f"    {cat:<14}: {cat_counts[cat]}")
    print(f"  Output       : {cfg.out_dir}/")
    print("=" * 56)

    if cfg.dry_run:
        print("[dry-run] No files written.")
        return

    out     = pathlib.Path(cfg.out_dir)
    img_dir = out / "images" / cfg.mode
    img_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    n_ok = 0
    n_err = 0

    for idx, item in enumerate(items):
        content_id = item["_content_id"]
        category   = item["_category"]
        img_path   = img_dir / f"{content_id}.png"

        item_rng = make_item_rng(cfg.seed, content_id)

        try:
            img = render_pil(item, category, item_rng)
            img.save(str(img_path))
            n_ok += 1

            # Re-derive layout metadata for annotation record
            meta_rng = make_item_rng(cfg.seed, content_id)
            lay = make_layout(category, meta_rng)
            layout_meta = {
                "W":            lay.W,
                "H":            lay.H,
                "font_family":  lay.font_family,
                "body_size":    lay.body_size,
                "line_spacing": lay.line_spacing,
                "margin_left":  lay.margin_left,
            }

            ann = build_annotation(
                item,
                str(img_path.relative_to(out)),
                cfg,
                layout_meta,
            )
            annotations.append(ann)

        except Exception as e:
            n_err += 1
            print(f"  [err] {content_id}: {e}", file=sys.stderr)
            continue

        if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
            print(f"  [{idx+1}/{len(items)}]  ok={n_ok}  err={n_err}", flush=True)

    labels_jsonl = out / f"labels_{cfg.mode}.jsonl"
    labels_json  = out / f"labels_{cfg.mode}.json"

    labels_jsonl.write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in annotations) + "\n",
        encoding="utf-8",
    )
    labels_json.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 56)
    print("  Done")
    print("=" * 56)
    print(f"  Rendered OK  : {n_ok}")
    print(f"  Errors       : {n_err}")
    print(f"  Labels       : {labels_jsonl.name}  +  {labels_json.name}")
    print(f"  Images       : {img_dir}/")
    print("=" * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--content",    type=str, default="content_bank.json",
                   help="Path to content_bank.json (from generate_content.py)")
    p.add_argument("--out",        type=str, default="domain2_ui_dataset",
                   help="Output directory (default: domain2_ui_dataset)")
    p.add_argument("--seed",       type=int, default=42,
                   help="Layout seed — same content + different seed = different layouts")
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                   choices=CATEGORIES)
    p.add_argument("--mode",       type=str, default="pil",
                   help="Render mode label written into annotations (default: pil)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print plan without writing any files")
    return p.parse_args()


if __name__ == "__main__":
    main()