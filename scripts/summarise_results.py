# /// script
# dependencies = ["rich"]
# ///
"""
Summarise K=2 attack results from results.jsonl.

Usage:
    uv run python scripts/summarise_results.py outputs/<run_id>/results.jsonl

Pass criterion (matches terminal output):
    constraint_satisfied AND cer_delta > 0

Results are broken down per (epsilon, kappa) configuration, then by
text_category and contrast_level, with union/intersection pass rates.
"""

import json
import argparse
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Skip event records (run_started, run_finished, etc.)
            if "image_id" not in d or "model_name" not in d:
                continue
            # Skip held-out surrogates — only count optimisation surrogates
            if d.get("is_held_out"):
                continue
            records.append(d)
    return records


def passes(record: dict) -> bool:
    """True iff this result counts as a valid suppression pass."""
    return bool(record.get("constraint_satisfied")) and record.get("cer_delta", 0) > 0


def summarise(records: list[dict]) -> None:
    # ── Collect all models ────────────────────────────────────────────────
    all_models = sorted({r["model_name"] for r in records})

    # ── Group by (epsilon, kappa, image_id) ──────────────────────────────
    # Each cell: {model_name: pass_bool}
    # Also store metadata for category/contrast breakdown
    groups: dict[tuple, dict] = defaultdict(dict)
    meta:   dict[tuple, dict] = {}

    for r in records:
        cfg_key    = (round(r["epsilon"], 6), round(r["kappa"], 4))
        sample_key = (cfg_key, r["image_id"])
        groups[sample_key][r["model_name"]] = passes(r)
        meta[sample_key] = {
            "text_category":  r.get("text_category",  "?"),
            "contrast_level": r.get("contrast_level", "?"),
            "cer_clean":      {r["model_name"]: r.get("cer_clean", 0)},
        }

    # Merge cer_clean entries across models for same sample_key
    for r in records:
        cfg_key    = (round(r["epsilon"], 6), round(r["kappa"], 4))
        sample_key = (cfg_key, r["image_id"])
        meta[sample_key]["cer_clean"][r["model_name"]] = r.get("cer_clean", 0)

    # ── All unique configs ────────────────────────────────────────────────
    all_configs = sorted({k[0] for k in groups})

    # ── Per-config summary table ──────────────────────────────────────────
    cfg_table = Table(
        title="Pass rates per (ε, κ) — pass = constraint_satisfied AND cer_delta > 0",
        show_lines=True,
    )
    cfg_table.add_column("ε",        style="cyan",    justify="right")
    cfg_table.add_column("κ",        style="cyan",    justify="right")
    cfg_table.add_column("N",        justify="right")
    for m in all_models:
        cfg_table.add_column(m,      style="yellow",  justify="right")
    cfg_table.add_column("Union",    style="green",   justify="right")
    cfg_table.add_column("Joint",    style="magenta", justify="right")

    cfg_stats: dict[tuple, dict] = {}

    for cfg in all_configs:
        eps, kap = cfg
        cfg_samples = {k: v for k, v in groups.items() if k[0] == cfg}
        n = len(cfg_samples)

        model_pass  = {m: 0 for m in all_models}
        union_pass  = 0
        joint_pass  = 0

        for sample_key, model_results in cfg_samples.items():
            any_pass  = any(model_results.get(m, False) for m in all_models)
            all_pass  = all(model_results.get(m, False) for m in all_models)
            if any_pass:  union_pass  += 1
            if all_pass:  joint_pass  += 1
            for m in all_models:
                if model_results.get(m, False):
                    model_pass[m] += 1

        cfg_stats[cfg] = dict(
            n=n, model_pass=model_pass,
            union=union_pass, joint=joint_pass,
        )

        row = [f"{eps:.5f}", f"{kap:.4f}", str(n)]
        for m in all_models:
            p = model_pass[m]
            row.append(f"{p}/{n} ({100*p/n:.0f}%)")
        row.append(f"{union_pass}/{n} ({100*union_pass/n:.0f}%)")
        row.append(f"{joint_pass}/{n} ({100*joint_pass/n:.0f}%)")
        cfg_table.add_row(*row)

    console.print(cfg_table)

    # ── Category breakdown at primary config (mid ε, κ=0.05) ─────────────
    # Find the config closest to eps=8/255=0.03137, kappa=0.05
    primary_cfg = min(all_configs, key=lambda c: (abs(c[0] - 0.03137) + abs(c[1] - 0.05)))
    eps_p, kap_p = primary_cfg
    console.print(f"\n[bold]Category breakdown at ε={eps_p:.5f}, κ={kap_p:.4f}[/bold]")

    cat_table = Table(show_lines=True)
    cat_table.add_column("Category",  style="cyan")
    cat_table.add_column("Contrast",  style="white")
    cat_table.add_column("N",         justify="right")
    for m in all_models:
        cat_table.add_column(m,       style="yellow", justify="right")
    cat_table.add_column("Union",     style="green",  justify="right")
    cat_table.add_column("Joint",     style="magenta",justify="right")

    # Group by (category, contrast)
    cat_groups: dict[tuple, list] = defaultdict(list)
    for sample_key, model_results in groups.items():
        if sample_key[0] != primary_cfg:
            continue
        m_info = meta[sample_key]
        cat_key = (m_info["text_category"], m_info["contrast_level"])
        cat_groups[cat_key].append(model_results)

    for cat_key in sorted(cat_groups):
        cat, contrast = cat_key
        rows = cat_groups[cat_key]
        n = len(rows)
        model_pass = {m: sum(1 for r in rows if r.get(m, False)) for m in all_models}
        union  = sum(1 for r in rows if any(r.get(m, False) for m in all_models))
        joint  = sum(1 for r in rows if all(r.get(m, False) for m in all_models))

        row = [cat, contrast, str(n)]
        for m in all_models:
            p = model_pass[m]
            row.append(f"{p}/{n}")
        row.append(f"{union}/{n}")
        row.append(f"{joint}/{n}")
        cat_table.add_row(*row)

    console.print(cat_table)

    # ── Dirty baseline summary ────────────────────────────────────────────
    console.print("\n[bold]Dirty baseline count (CER_clean > 0.20 per model)[/bold]")
    dirty_table = Table(show_lines=True)
    dirty_table.add_column("Model",   style="cyan")
    dirty_table.add_column("Config",  style="white")
    dirty_table.add_column("Dirty N", justify="right")
    dirty_table.add_column("% of N",  justify="right")

    for cfg in all_configs:
        eps, kap = cfg
        cfg_samples = {k: v for k, v in groups.items() if k[0] == cfg}
        n = len(cfg_samples)
        for m in all_models:
            dirty = sum(
                1 for sk in cfg_samples
                if meta[sk]["cer_clean"].get(m, 0) > 0.20
            )
            if dirty > 0:
                dirty_table.add_row(
                    m, f"ε={eps:.5f} κ={kap:.4f}",
                    str(dirty), f"{100*dirty/n:.0f}%",
                )

    console.print(dirty_table)

    # ── Complementarity: one-but-not-both at primary config ───────────────
    # Shown twice: raw (all samples) and filtered (dirty baselines excluded).
    # A sample is "dirty" for a given model if its CER_clean > DIRTY_THRESHOLD.
    DIRTY_THRESHOLD = 0.20

    def _comp_table(label: str, cfg_samples: dict, dirty_mask: dict) -> None:
        """Print a complementarity table, optionally masking dirty baselines."""
        if len(all_models) != 2:
            return
        m0, m1 = all_models
        both = only_m0 = only_m1 = neither = 0
        n_excluded = 0

        for sample_key, model_results in cfg_samples.items():
            # In filtered mode: if ALL models are dirty for this sample, skip it
            # entirely from the denominator.  If only one is dirty, treat that
            # model as failing (conservative).
            r = dict(model_results)  # copy so we don't mutate
            if dirty_mask:
                m0_dirty = dirty_mask.get((sample_key, m0), False)
                m1_dirty = dirty_mask.get((sample_key, m1), False)
                if m0_dirty and m1_dirty:
                    n_excluded += 1
                    continue
                if m0_dirty:
                    r[m0] = False
                if m1_dirty:
                    r[m1] = False

            p0 = r.get(m0, False)
            p1 = r.get(m1, False)
            if   p0 and p1:       both    += 1
            elif p0 and not p1:   only_m0 += 1
            elif p1 and not p0:   only_m1 += 1
            else:                  neither += 1

        n = both + only_m0 + only_m1 + neither
        if n == 0:
            return

        t = Table(title=label, show_lines=True)
        t.add_column("Outcome",   style="cyan")
        t.add_column("Count",     justify="right")
        t.add_column("%",         justify="right")
        t.add_row("Both pass",    str(both),    f"{100*both/n:.0f}%")
        t.add_row(f"Only {m0}",   str(only_m0), f"{100*only_m0/n:.0f}%")
        t.add_row(f"Only {m1}",   str(only_m1), f"{100*only_m1/n:.0f}%")
        t.add_row("Neither",      str(neither), f"{100*neither/n:.0f}%")
        t.add_row("Union",        str(both+only_m0+only_m1),
                  f"{100*(both+only_m0+only_m1)/n:.0f}%")
        if n_excluded:
            t.add_row(f"[dim]Excluded (both dirty)[/dim]",
                      f"[dim]{n_excluded}[/dim]", "")
        console.print(t)

    cfg_samples_primary = {k: v for k, v in groups.items() if k[0] == primary_cfg}

    # Build dirty mask: (sample_key, model_name) -> bool
    dirty_mask: dict[tuple, bool] = {}
    for sample_key in cfg_samples_primary:
        for m in all_models:
            cer_clean = meta[sample_key]["cer_clean"].get(m, 0)
            dirty_mask[(sample_key, m)] = cer_clean > DIRTY_THRESHOLD

    console.print(f"\n[bold]Complementarity at primary config "
                  f"ε={eps_p:.5f}, κ={kap_p:.4f}[/bold]")
    _comp_table("Raw (all 50 samples)", cfg_samples_primary, {})
    _comp_table(
        f"Filtered (CER_clean > {DIRTY_THRESHOLD} treated as fail / excluded if both dirty)",
        cfg_samples_primary, dirty_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to results.jsonl")
    args = parser.parse_args()
    records = load_records(args.file)
    if not records:
        console.print("[red]No valid records found.[/red]")
        return
    console.print(f"Loaded {len(records)} result records from {args.file}\n")
    summarise(records)


if __name__ == "__main__":
    main()