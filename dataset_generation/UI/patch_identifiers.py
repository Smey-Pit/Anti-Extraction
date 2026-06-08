"""
patch_identifiers.py
====================
Patches synthetic identifiers in a content_bank.json so each record has
a unique, realistic-looking value instead of the LLM's repeated defaults.

What gets patched
-----------------
banking   : account_number  →  BSB + account  e.g. "062-001  2310 4821 03"
            patient_id is left alone (not a banking field)

medical   : patient_id      →  e.g. "P-448821"  (unique per record)

news      : no numeric IDs to patch

copyright : page_number     →  unique sequential page per record

All other fields are left untouched.

Usage
-----
    python patch_identifiers.py --input content_bank.json
    python patch_identifiers.py --input content_bank.json --output content_bank_patched.json
    python patch_identifiers.py --input content_bank.json --seed 7   # different number set
    python patch_identifiers.py --input content_bank.json --dry-run  # preview, no write
"""

import argparse
import json
import pathlib
import random
import sys


# ---------------------------------------------------------------------------
# SYNTHETIC IDENTIFIER GENERATORS
# ---------------------------------------------------------------------------

def gen_bank_account(rng: random.Random) -> str:
    """
    Australian-style BSB + account number.
    Format: "NNN-NNN  NNNN NNNN NN"
    e.g.    "062-001  2310 4821 03"
    """
    bsb_prefix = rng.choice(["012", "032", "062", "082", "112", "182",
                              "232", "313", "334", "484", "633", "734"])
    bsb_suffix  = rng.randint(0, 999)
    part1       = rng.randint(1000, 9999)
    part2       = rng.randint(1000, 9999)
    part3       = rng.randint(10, 99)
    return f"{bsb_prefix}-{bsb_suffix:03d}  {part1} {part2} {part3}"


def gen_masked_account(rng: random.Random) -> str:
    """
    Masked 4-digit suffix style as a fallback / alt format.
    e.g. "****3847"
    """
    return f"****{rng.randint(1000, 9999)}"


def gen_patient_id(rng: random.Random) -> str:
    """
    Hospital patient ID.  e.g. "P-448821" or "MR-2031847"
    """
    style = rng.choice(["P", "MR", "PT", "HC"])
    number = rng.randint(100000, 9999999)
    return f"{style}-{number}"


def gen_page_number(base: int, rng: random.Random) -> int:
    """
    Realistic page number: starts from a random chapter offset.
    Keeps pages increasing across records so they feel like different books.
    """
    return base + rng.randint(1, 40)


# ---------------------------------------------------------------------------
# PATCH FUNCTIONS (one per category)
# ---------------------------------------------------------------------------

def patch_banking(item: dict, rng: random.Random) -> dict:
    item = dict(item)
    item["account_number"] = gen_bank_account(rng)
    return item


def patch_medical(item: dict, rng: random.Random) -> dict:
    item = dict(item)
    item["patient_id"] = gen_patient_id(rng)
    return item


def patch_news(item: dict, rng: random.Random) -> dict:
    return item   # nothing numeric to patch


def patch_copyright(item: dict, rng: random.Random, page_counter: list) -> dict:
    item = dict(item)
    page_counter[0] = gen_page_number(page_counter[0], rng)
    item["page_number"] = page_counter[0]
    return item


PATCHERS = {
    "banking":   patch_banking,
    "medical":   patch_medical,
    "news":      patch_news,
    "copyright": None,   # handled separately (needs page_counter)
}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def patch_bank(items: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    page_counter = [1]   # mutable so patch_copyright can increment it

    patched = []
    counts  = {}

    for item in items:
        cat = item.get("_category", "")

        if cat == "banking":
            item = patch_banking(item, rng)
        elif cat == "medical":
            item = patch_medical(item, rng)
        elif cat == "copyright":
            item = patch_copyright(item, rng, page_counter)
        # news: no-op

        patched.append(item)
        counts[cat] = counts.get(cat, 0) + 1

    return patched, counts


def main() -> None:
    args = parse_args()

    in_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output) if args.output else in_path

    if not in_path.exists():
        print(f"[error] File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    items = json.loads(in_path.read_text(encoding="utf-8"))

    if not isinstance(items, list):
        print("[error] Expected a flat JSON array (content_bank.json format).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(items)} items from {in_path}")

    # Preview a before/after for each category
    if args.dry_run or args.verbose:
        print("\nBefore patch (first of each category):")
        seen = set()
        for item in items:
            cat = item.get("_category","")
            if cat not in seen:
                seen.add(cat)
                _preview(cat, item)

    patched, counts = patch_bank(items, seed=args.seed)

    if args.dry_run or args.verbose:
        print("\nAfter patch (first of each category):")
        seen = set()
        for item in patched:
            cat = item.get("_category","")
            if cat not in seen:
                seen.add(cat)
                _preview(cat, item)

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(patched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nPatched and saved → {out_path}")
    print(f"Records patched:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<14}: {n}")


def _preview(category: str, item: dict) -> None:
    """Print the relevant identifier field for a category."""
    fields = {
        "banking":   "account_number",
        "medical":   "patient_id",
        "news":      "(no numeric ID)",
        "copyright": "page_number",
    }
    field = fields.get(category)
    if field and field in item:
        print(f"  {category:<14} {field}: {item[field]}")
    else:
        print(f"  {category:<14} {field}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",   "-i", required=True,
                   help="Path to content_bank.json")
    p.add_argument("--output",  "-o", default=None,
                   help="Output path (default: overwrite input)")
    p.add_argument("--seed",    type=int, default=42,
                   help="RNG seed for identifier generation (default: 42)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview before/after without writing")
    p.add_argument("--verbose", action="store_true",
                   help="Show before/after preview AND write the file")
    return p.parse_args()


if __name__ == "__main__":
    main()