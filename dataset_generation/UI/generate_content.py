"""
generate_content.py
===================
Step 1 of 2: Generate LLM content and save to a reusable content bank JSON.

This script calls the LLM backend (Anthropic, qwen-api, or qwen-local) and
saves structured content dicts to a JSON file. No images are produced here.

The saved content bank can then be passed to render_from_content.py multiple
times under different render conditions (PIL, Playwright, different backgrounds)
to produce paired image datasets from identical text content.

Usage
-----
    # 1000 total samples (250 per category), Anthropic backend
    python generate_content.py --total 1000 --backend anthropic

    # Same but with Qwen via Ollama
    python generate_content.py --total 1000 --backend qwen-api \\
        --api-base-url http://localhost:11434/v1 --api-model qwen2.5:7b

    # Custom split across categories
    python generate_content.py \\
        --per-category banking=300 medical=300 news=200 copyright=200

    # Resume / append to an existing bank (safe to re-run after crash)
    python generate_content.py --total 1000 --out content_bank.json --resume
"""

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field

from backends import BackendConfig, build_backend

CATEGORIES     = ["banking", "medical", "news", "copyright", "legal", "identity", "communications"]
LLM_BATCH_SIZE = 5
DEFAULT_OUT    = "content_bank.json"


def compute_per_category_counts(total: int, categories: list[str]) -> dict[str, int]:
    """
    Distribute `total` evenly across categories; remainder goes to first N.
    Example: total=1000, 4 cats → {banking:250, medical:250, news:250, copyright:250}
    Example: total=7,    3 cats → {banking:3,   medical:2,   news:2}
    """
    n = len(categories)
    base, remainder = divmod(total, n)
    return {cat: base + (1 if i < remainder else 0)
            for i, cat in enumerate(categories)}


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class ContentGenConfig:
    out_path:       str           = DEFAULT_OUT
    per_category:   dict          = field(default_factory=lambda: {c: 250 for c in CATEGORIES})
    resume:         bool          = False
    backend_config: BackendConfig = field(default_factory=BackendConfig)


# ---------------------------------------------------------------------------
# GENERATION LOOP
# ---------------------------------------------------------------------------

def generate_category(
    category:  str,
    n_needed:  int,
    backend,
    already:   int = 0,
) -> list[dict]:
    """
    Collect exactly n_needed new content dicts for category.
    Stops early after 3 consecutive empty batches.
    already: count already in bank (for display only).
    """
    results: list[dict] = []
    empty_streak = 0

    while len(results) < n_needed:
        batch_size = min(LLM_BATCH_SIZE, n_needed - len(results))
        print(
            f"  [{category}] {already + len(results)}/{already + n_needed} — "
            f"requesting batch of {batch_size} …",
            flush=True,
        )

        batch = backend.generate(category, batch_size)

        if not batch:
            empty_streak += 1
            print(f"  [{category}] Empty batch ({empty_streak}/3)", file=sys.stderr)
            if empty_streak >= 3:
                print(f"  [{category}] Stopping early after 3 empty batches.", file=sys.stderr)
                break
            time.sleep(2 ** empty_streak)
            continue

        empty_streak = 0

        for item in batch:
            item["_category"]   = category
            item["_content_id"] = f"{category}_{already + len(results):04d}"

        results.extend(batch[: n_needed - len(results)])
        print(f"  [{category}] {len(results)}/{n_needed} collected.", flush=True)

    return results


# ---------------------------------------------------------------------------
# RESUME HELPERS
# ---------------------------------------------------------------------------

def load_existing(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        print(f"[resume] Loaded {len(items)} existing items from {path}")
        return items
    except Exception as e:
        print(f"[resume] Could not load {path}: {e}", file=sys.stderr)
        return []


def counts_by_category(items: list[dict]) -> dict:
    c = {cat: 0 for cat in CATEGORIES}
    for item in items:
        cat = item.get("_category", "")
        if cat in c:
            c[cat] += 1
    return c


# ---------------------------------------------------------------------------
# SAVE / SUMMARY
# ---------------------------------------------------------------------------

def save(items: list[dict], path: pathlib.Path) -> None:
    path.write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  [saved] {len(items)} items → {path}")


def print_summary(items: list[dict], cfg: ContentGenConfig) -> None:
    counts = counts_by_category(items)
    print("\n" + "=" * 52)
    print(f"  Content Bank Summary  ({len(items)} total items)")
    print("=" * 52)
    for cat in CATEGORIES:
        target = cfg.per_category.get(cat, 0)
        have   = counts.get(cat, 0)
        bar    = "✓" if have >= target else f"INCOMPLETE ({have}/{target})"
        print(f"  {cat:<20} {have:>4}   {bar}")
    print(f"\n  Output: {cfg.out_path}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(cfg: ContentGenConfig) -> None:
    out_path = pathlib.Path(cfg.out_path)

    all_items: list[dict] = []
    have: dict = {c: 0 for c in CATEGORIES}

    if cfg.resume:
        all_items = load_existing(out_path)
        have = counts_by_category(all_items)

    backend = build_backend(cfg.backend_config)

    print(f"\n[gen] Output:  {cfg.out_path}")
    print(f"[gen] Backend: {cfg.backend_config.backend}")
    print(f"[gen] Targets: { {c: cfg.per_category[c] for c in CATEGORIES} }")
    print()

    for category in CATEGORIES:
        target     = cfg.per_category.get(category, 0)
        still_need = max(0, target - have.get(category, 0))

        if target == 0:
            continue
        if still_need == 0:
            print(f"[{category}] Already complete ({have[category]}/{target}), skipping.\n")
            continue

        print(f"[{category}] Generating {still_need} items …")
        new_items = generate_category(
            category = category,
            n_needed = still_need,
            backend  = backend,
            already  = have.get(category, 0),
        )
        all_items.extend(new_items)
        print(f"[{category}] Done — {have.get(category,0) + len(new_items)}/{target} total.\n")

        # Incremental save — safe against job preemption on HPC
        save(all_items, out_path)

    save(all_items, out_path)
    print_summary(all_items, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ContentGenConfig:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--out", dest="out_path", type=str, default=DEFAULT_OUT,
                   help=f"Output JSON path (default: {DEFAULT_OUT})")
    p.add_argument("--resume", action="store_true",
                   help="Append to existing bank instead of overwriting")

    cg = p.add_mutually_exclusive_group()
    cg.add_argument("--total", type=int, default=None,
                    help="Total items split evenly across all 7 categories")
    cg.add_argument("--per-category", nargs="+", metavar="CAT=N", default=None,
                    help="e.g. --per-category banking=150 medical=150 legal=150 identity=150 communications=150 news=100 copyright=150")

    p.add_argument("--backend", type=str, default="anthropic",
                   choices=["anthropic", "qwen-local", "qwen-api"])

    ag = p.add_argument_group("Anthropic options")
    ag.add_argument("--anthropic-model", type=str, default="claude-sonnet-4-20250514")

    qg = p.add_argument_group("qwen-api options")
    qg.add_argument("--api-base-url", type=str, default="http://localhost:11434/v1")
    qg.add_argument("--api-model",    type=str, default="Qwen/Qwen2.5-7B-Instruct")
    qg.add_argument("--api-key",      type=str, default="ollama")

    lg = p.add_argument_group("qwen-local options")
    lg.add_argument("--local-model",  type=str, default="Qwen/Qwen2.5-7B-Instruct")
    lg.add_argument("--load-in-4bit", action="store_true")
    lg.add_argument("--load-in-8bit", action="store_true", help="Load in 8-bit (~9 GB VRAM, better quality than 4-bit)")
    lg.add_argument("--device-map",   type=str, default="auto")

    sg = p.add_argument_group("Shared generation options")
    sg.add_argument("--max-tokens",  type=int,   default=4000)
    sg.add_argument("--temperature", type=float, default=0.7)
    sg.add_argument("--max-retries", type=int,   default=3)

    args = p.parse_args()

    # Resolve per-category counts
    if args.total is not None:
        base, rem = divmod(args.total, len(CATEGORIES))
        per_category = {c: base for c in CATEGORIES}
        for i in range(rem):
            per_category[CATEGORIES[i]] += 1
    elif args.per_category is not None:
        per_category = {c: 0 for c in CATEGORIES}
        for token in args.per_category:
            try:
                cat, n = token.split("=")
            except ValueError:
                p.error(f"Invalid --per-category format: '{token}'. Use CAT=N, e.g. banking=300")
            if cat not in CATEGORIES:
                p.error(f"Unknown category '{cat}'. Choose from {CATEGORIES}")
            per_category[cat] = int(n)
    else:
        per_category = {c: 143 for c in CATEGORIES}   # ~1000 total across 7

    backend_cfg = BackendConfig(
        backend         = args.backend,
        anthropic_model = args.anthropic_model,
        api_base_url    = args.api_base_url,
        api_model       = args.api_model,
        api_key         = args.api_key,
        local_model     = args.local_model,
        load_in_4bit    = args.load_in_4bit,
        device_map      = args.device_map,
        max_tokens      = args.max_tokens,
        temperature     = args.temperature,
        max_retries     = args.max_retries,
    )

    return ContentGenConfig(
        out_path       = args.out_path,
        per_category   = per_category,
        resume         = args.resume,
        backend_config = backend_cfg,
    )


if __name__ == "__main__":
    main(parse_args())