"""
sample_manifest.py
==================
Creates a stratified random sample of N images per category from
the full manifest.csv, writing a new manifest file for use with
phase0_eval.py.

Usage
-----
    # 50 per category (350 total for 7 domains)
    python sample_manifest.py \\
        --manifest data/ui_dataset/manifest.csv \\
        --out      data/ui_dataset/manifest_50.csv \\
        --n 50 \\
        --seed 42

    # Then run eval on the sample:
    uv run -m vlm_suppress.tests.phase0_eval \\
        --model qwen2_5vl \\
        --manifest data/ui_dataset/manifest_50.csv \\
        ...
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="Input manifest.csv")
    p.add_argument("--out",      required=True, help="Output sampled manifest CSV")
    p.add_argument("--n",        type=int, default=50,
                   help="Images per category (default: 50)")
    p.add_argument("--seed",     type=int, default=42,
                   help="RNG seed for reproducibility (default: 42)")
    args = p.parse_args()

    manifest = Path(args.manifest)
    out_path  = Path(args.out)

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    # Load and group by domain
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["doc_type"]].append(row)

    rng = random.Random(args.seed)

    sampled: list[dict] = []
    print(f"Sampling {args.n} per category (seed={args.seed}):")
    for domain in sorted(by_domain):
        domain_rows = by_domain[domain]
        n_available = len(domain_rows)
        n_sample    = min(args.n, n_available)
        chosen = rng.sample(domain_rows, n_sample)
        sampled.extend(chosen)
        shortfall = "" if n_sample == args.n else f"  ← only {n_available} available"
        print(f"  {domain:<20} {n_sample:>4} / {n_available}{shortfall}")

    print(f"  {'TOTAL':<20} {len(sampled):>4}")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sampled)

    print(f"\nSampled manifest written: {out_path}")


if __name__ == "__main__":
    main()