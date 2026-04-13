"""
render_images.py
================
Step 2 of 2: Render images from a saved content bank.

Supports two render modes selectable via --mode:

  pil         Varied PIL renderer. CPU-only. No extra dependencies.
              Font family, size, canvas, margins all vary per item.

  playwright  Chromium browser screenshots via Playwright.
              Produces realistic anti-aliased renders with CSS layout,
              web fonts, shadows, and sub-pixel rendering.
              Requires: pip install playwright && playwright install chromium
              Colab/HPC: add --no-sandbox (enabled by default here)

Both modes use the same content bank and the same per-item seed strategy,
so (content_id, seed) always maps to the same image — enabling paired renders
where only the rendering pipeline changes, not the text content.

Usage
-----
    # PIL (default)
    python render_images.py --content content_bank.json --out out/

    # Playwright
    python render_images.py --content content_bank.json --out out/ --mode playwright

    # Both in one go (runs PIL then Playwright sequentially)
    python render_images.py --content content_bank.json --out out/ --mode both

    # Different layout seed
    python render_images.py --content content_bank.json --seed 99 --out out_s99/

    # Dry run
    python render_images.py --content content_bank.json --dry-run
"""

import argparse
import asyncio
import json
import pathlib
import random
import sys
from dataclasses import dataclass, field

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
    mode:         str  = "pil"    # "pil" | "playwright" | "both"
    dry_run:      bool = False
    no_sandbox:   bool = True     # required on Colab / HPC


# ---------------------------------------------------------------------------
# SEED STRATEGY
# ---------------------------------------------------------------------------

def make_item_rng(global_seed: int, content_id: str) -> random.Random:
    """
    Deterministic per-item RNG from (global_seed, content_id).
      - Same seed + content_id  →  identical layout every time
      - Different seed           →  different layouts, same text
      - Item order doesn't matter
    """
    item_seed = hash((global_seed, content_id)) & 0xFFFFFFFF
    return random.Random(item_seed)


# ---------------------------------------------------------------------------
# CONTENT BANK LOADING
# ---------------------------------------------------------------------------

def load_content_bank(path: pathlib.Path) -> list[dict]:
    """
    Load content bank produced by generate_content.py.
    Supports flat list schema (_category / _content_id keys).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return raw

    # Legacy nested schema
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
    """Flatten item into ordered ground-truth text for CER evaluation."""
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
                     render_mode: str, seed: int,
                     layout_meta: dict) -> dict:
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
        "render_mode":         render_mode,
        "render_seed":         seed,
        "layout":              layout_meta,
        "split":               "train",
        "has_ambiguous_chars": False,
        "layout_type":         "web_portal",
        "text_category":       category,
    }


# ---------------------------------------------------------------------------
# PIL RENDER PASS
# ---------------------------------------------------------------------------

def render_pass_pil(items: list[dict], cfg: RenderConfig,
                    img_dir: pathlib.Path, out: pathlib.Path) -> list[dict]:
    from pil_renderer import render_pil, make_layout

    annotations = []
    n_ok = n_err = 0

    for idx, item in enumerate(items):
        content_id = item["_content_id"]
        category   = item["_category"]
        img_path   = img_dir / f"{content_id}.png"
        item_rng   = make_item_rng(cfg.seed, content_id)

        try:
            img = render_pil(item, category, item_rng)
            img.save(str(img_path))
            n_ok += 1

            # Re-derive layout metadata (make_layout consumes rng first)
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
                "pil", cfg.seed, layout_meta,
            )
            annotations.append(ann)

        except Exception as e:
            n_err += 1
            print(f"  [pil err] {content_id}: {e}", file=sys.stderr)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
            print(f"  [pil] [{idx+1}/{len(items)}]  ok={n_ok}  err={n_err}",
                  flush=True)

    return annotations


# ---------------------------------------------------------------------------
# PLAYWRIGHT RENDER PASS
# ---------------------------------------------------------------------------

async def render_pass_playwright_async(
    items:   list[dict],
    cfg:     RenderConfig,
    img_dir: pathlib.Path,
    out:     pathlib.Path,
) -> list[dict]:
    from playwright_renderer import PlaywrightRenderer

    annotations = []
    n_ok = n_err = 0

    async with PlaywrightRenderer(no_sandbox=cfg.no_sandbox) as renderer:
        for idx, item in enumerate(items):
            content_id = item["_content_id"]
            category   = item["_category"]
            img_path   = img_dir / f"{content_id}.png"
            item_rng   = make_item_rng(cfg.seed, content_id)

            try:
                png_bytes = await renderer.render(item, category, rng=item_rng)
                img_path.write_bytes(png_bytes)
                n_ok += 1

                # Read back image size for annotation
                from PIL import Image
                import io
                img   = Image.open(io.BytesIO(png_bytes))
                W, H  = img.size

                layout_meta = {
                    "W":           W,
                    "H":           H,
                    "font_family": "css-system",   # browser picks from CSS stack
                    "body_size":   None,            # CSS-controlled
                    "line_spacing": None,
                    "margin_left": None,
                }

                ann = build_annotation(
                    item,
                    str(img_path.relative_to(out)),
                    "playwright", cfg.seed, layout_meta,
                )
                annotations.append(ann)

            except Exception as e:
                n_err += 1
                print(f"  [pw err] {content_id}: {e}", file=sys.stderr)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
                print(f"  [playwright] [{idx+1}/{len(items)}]  ok={n_ok}  err={n_err}",
                      flush=True)

    return annotations


def render_pass_playwright(items, cfg, img_dir, out):
    return asyncio.run(render_pass_playwright_async(items, cfg, img_dir, out))


# ---------------------------------------------------------------------------
# WRITE LABELS
# ---------------------------------------------------------------------------

def write_labels(annotations: list[dict], out: pathlib.Path,
                 mode: str) -> None:
    jsonl = out / f"labels_{mode}.jsonl"
    jsn   = out / f"labels_{mode}.json"
    jsonl.write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in annotations) + "\n",
        encoding="utf-8",
    )
    jsn.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Labels: {jsonl.name}  +  {jsn.name}")


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
        mode         = args.mode,
        dry_run      = args.dry_run,
        no_sandbox   = not args.allow_sandbox,
    )

    bank_path = pathlib.Path(cfg.content_path)
    if not bank_path.exists():
        print(f"[error] Content bank not found: {bank_path}", file=sys.stderr)
        sys.exit(1)

    all_items = load_content_bank(bank_path)
    items     = [it for it in all_items
                 if it.get("_category") in cfg.categories]

    from collections import Counter
    cat_counts = Counter(it["_category"] for it in items)
    modes      = ["pil", "playwright"] if cfg.mode == "both" else [cfg.mode]

    print("=" * 58)
    print("  Render plan")
    print("=" * 58)
    print(f"  Content bank : {bank_path}  ({len(all_items)} total items)")
    print(f"  Mode(s)      : {', '.join(modes)}")
    print(f"  Seed         : {cfg.seed}")
    print(f"  Rendering    : {len(items)} items")
    for cat in CATEGORIES:
        if cat in cat_counts:
            print(f"    {cat:<14}: {cat_counts[cat]}")
    print(f"  Output       : {cfg.out_dir}/")
    print("=" * 58)

    if cfg.dry_run:
        print("[dry-run] No files written.")
        return

    out = pathlib.Path(cfg.out_dir)

    for mode in modes:
        print(f"\n--- {mode.upper()} pass ---")
        img_dir = out / "images" / mode
        img_dir.mkdir(parents=True, exist_ok=True)

        if mode == "pil":
            annotations = render_pass_pil(items, cfg, img_dir, out)
        else:
            annotations = render_pass_playwright(items, cfg, img_dir, out)

        write_labels(annotations, out, mode)

        n_ok = sum(1 for a in annotations if a.get("image_path"))
        print(f"  Rendered: {n_ok}/{len(items)}")
        print(f"  Images:   {img_dir}/")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--content",    type=str, default="content_bank.json",
                   help="Path to content_bank.json")
    p.add_argument("--out",        type=str, default="domain2_ui_dataset",
                   help="Output directory")
    p.add_argument("--seed",       type=int, default=42,
                   help="Layout seed (default: 42)")
    p.add_argument("--mode",       type=str, default="pil",
                   choices=["pil", "playwright", "both"],
                   help="Render mode: pil | playwright | both (default: pil)")
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                   choices=CATEGORIES)
    p.add_argument("--dry-run",    action="store_true",
                   help="Print plan without writing files")
    p.add_argument("--allow-sandbox", action="store_true",
                   help="Disable --no-sandbox flag (not recommended on Colab/HPC)")
    return p.parse_args()


if __name__ == "__main__":
    main()