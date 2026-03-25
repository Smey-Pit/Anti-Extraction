"""
Unconstrained probe attack entry point.

Purpose: verify F_M^ens is a real optimisable signal BEFORE introducing
the readability constraint. This is the first test to run.

Usage:
    python scripts/run_probe.py --config configs/probe.yaml --norm l2
    python scripts/run_probe.py --config configs/probe.yaml --norm linf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import dacite
import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.attack.probe import run_probe
from vlm_suppress.config import (
    Domain,
    EnsembleWeighting,
    ExperimentConfig,
    ObjectiveConfig,
    ProxyStage,
)
from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.eval.metrics import ModelMetrics
from vlm_suppress.eval.visualise import save_diff_heatmap
from vlm_suppress.logging.logger import RunLogger

console = Console()
app = typer.Typer()


def _load_cfg(config: Path) -> ExperimentConfig:
    with config.open() as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(
        data_class=ExperimentConfig,
        data=raw,
        config=dacite.Config(
            cast=[Path, Domain, ProxyStage, ObjectiveConfig, EnsembleWeighting],
            type_hooks={
                Optional[tuple[int, int]]: lambda v: tuple(v) if v is not None else None,
            },
        ),
    )


def _load_opt_surrogates(cfg: ExperimentConfig) -> list:
    from vlm_suppress.models.internvl import InternVL2
    from vlm_suppress.models.llava import LLaVA16
    from vlm_suppress.models.qwen2vl import Qwen2VL
    from vlm_suppress.models.qwenvl import QwenVL

    _REG = {
        "internvl2": InternVL2,
        "qwenvl":    QwenVL,
        "qwen2vl":   Qwen2VL,
        "llava16":   LLaVA16,
    }
    models = []
    for i, s in enumerate(cfg.surrogates):
        if i in cfg.held_out_indices:
            continue
        cls = _REG.get(s.name)
        if cls is None:
            raise ValueError(f"Unknown surrogate: {s.name}")
        console.log(f"Loading {s.name} ({s.model_id}) ...")
        models.append(cls(s))
    return models


def _save_probe_trajectory(trajectory: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps    = [s.step     for s in trajectory]
    fm       = [s.fm_ens   for s in trajectory]
    fm_ce    = [s.fm_ce    for s in trajectory]
    fm_align = [s.fm_align for s in trajectory]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, fm,       label="F_M^ens (total)",   color="#1f77b4", linewidth=2)
    ax.plot(steps, fm_ce,    label="F_ce component",    color="#2ca02c", linestyle="--")
    ax.plot(steps, fm_align, label="F_align component", color="#ff7f0e", linestyle="--")
    ax.set_xlabel("PGD Step")
    ax.set_ylabel("F_M value  (↑ = more suppression)")
    ax.set_title("Probe: F_M^ens over PGD steps  [no readability constraint]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    config: Path = typer.Option(...,   help="Path to YAML config"),
    norm:   str  = typer.Option("l2",  help="Norm constraint: 'l2' or 'linf'"),
    steps:  Optional[int] = typer.Option(None, help="Override PGD steps"),
) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    cfg = _load_cfg(config)
    cfg.run_id = f"{cfg.run_id}_probe_{norm}"

    console.rule(f"[bold yellow]PROBE RUN — unconstrained {norm.upper()} attack")
    console.log("Purpose: verify F_M^ens is optimisable BEFORE readability constraint.")
    console.log(f"Run ID:  {cfg.run_id}")

    with RunLogger(cfg) as logger:

        # ── Load dataset ───────────────────────────────────────────────────
        dataset = TextImageDataset(
            cfg.data.data_dir,
            image_size=cfg.data.image_size,
            max_samples=cfg.data.n_images,
            split_filter=cfg.data.split_filter,
            category_filter=cfg.data.category_filter,
            contrast_filter=cfg.data.contrast_filter,
        )
        console.log(f"Dataset summary:\n{json.dumps(dataset.summary(), indent=2)}")

        if len(dataset) == 0:
            console.print("[red]No samples loaded — check data_dir and split_filter.[/red]")
            raise SystemExit(1)

        # ── Load surrogates ────────────────────────────────────────────────
        surrogates = _load_opt_surrogates(cfg)

        # ── Results table ──────────────────────────────────────────────────
        table = Table(
            title=f"Probe Results — {norm.upper()} norm",
            show_lines=True,
        )
        table.add_column("Image ID",   style="cyan", no_wrap=True)
        table.add_column("Category",   style="white")
        table.add_column("Contrast",   style="white")
        table.add_column("F_M clean",  justify="right")
        table.add_column("F_M final",  justify="right")
        table.add_column("ΔF_M",       justify="right")
        table.add_column("CER clean",  justify="right")
        table.add_column("CER adv",    justify="right")
        table.add_column("ΔCER",       justify="right")
        table.add_column("Pass?",      justify="center")

        n_pass  = 0
        n_skip  = 0
        n_total = 0
        epsilons = cfg.epsilon_sweep or [cfg.attack.epsilon]

        for sample in dataset:
            for eps in epsilons:
                n_total += 1
                console.log(
                    f"  [{sample.image_id}] norm={norm} eps={eps:.5f} "
                    f"cat={sample.text_category} contrast={sample.contrast_level}"
                )

                # ── Clean transcriptions ───────────────────────────────────
                clean_preds: dict[str, str] = {}
                for m in surrogates:
                    with torch.no_grad():
                        clean_preds[m.name] = m.transcribe(sample.image_tensor)

                # ── Run probe (no constraint) ──────────────────────────────
                result = run_probe(
                    image_id=sample.image_id,
                    x_orig=sample.image_tensor,
                    transcript=sample.transcript,
                    surrogates=surrogates,
                    cfg=cfg.attack,
                    norm=norm,
                    epsilon=eps,
                    pgd_steps=steps,
                    word_boxes=sample.scaled_word_boxes(),
                )

                # ── Adversarial transcriptions ─────────────────────────────
                adv_preds: dict[str, str] = {}
                for m in surrogates:
                    with torch.no_grad():
                        adv_preds[m.name] = m.transcribe(result.x_adv)

                # ── Metrics ────────────────────────────────────────────────
                all_mets = [
                    ModelMetrics.compute(
                        image_id=sample.image_id,
                        model_name=m.name,
                        transcript_clean=clean_preds[m.name],
                        transcript_adv=adv_preds[m.name],
                        reference=sample.transcript,
                    )
                    for m in surrogates
                ]
                pm = all_mets[0]   # primary model (first surrogate)
                # ── Dirty baseline filter ──────────────────────────────────
                dirty_baseline = pm.cer_clean > cfg.cer_clean_threshold
                if dirty_baseline:
                    console.log(
                        f"  [skip] {sample.image_id} eps={eps:.5f} — "
                        f"CER_clean={pm.cer_clean:.4f} > threshold "
                        f"{cfg.cer_clean_threshold} (dirty baseline)"
                    )
                    n_skip += 1
                    # Still log to results.jsonl for record-keeping,
                    # but mark as excluded so it's easy to filter later.
                    logger.log_result({
                        **_build_result_record(sample, pm, result, norm, eps, cfg),
                        "excluded_dirty_baseline": True,
                    })
                    continue   # skip table row and pass/fail count

                passed = result.passed and pm.cer_delta > 0
                if passed:
                    n_pass += 1

                # ── Table row ──────────────────────────────────────────────
                fm_delta_str  = (
                    f"[green]+{result.fm_delta:.4f}[/green]"
                    if result.fm_delta > 0
                    else f"[red]{result.fm_delta:.4f}[/red]"
                )
                cer_delta_str = (
                    f"[green]+{pm.cer_delta:.4f}[/green]"
                    if pm.cer_delta > 0
                    else f"[red]{pm.cer_delta:.4f}[/red]"
                )
                pass_str = (
                    "[green]✓ PASS[/green]" if passed
                    else "[red]✗ FAIL[/red]"
                )
                table.add_row(
                    sample.image_id,
                    sample.text_category,
                    sample.contrast_level,
                    f"{result.fm_clean:.4f}",
                    f"{result.fm_final:.4f}",
                    fm_delta_str,
                    f"{pm.cer_clean:.4f}",
                    f"{pm.cer_adv:.4f}",
                    cer_delta_str,
                    pass_str,
                )

                # ── JSON log ───────────────────────────────────────────────
                logger.log_result({
                    "image_id":         sample.image_id,
                    "model_name":       pm.model_name,
                    "norm":             norm,
                    "epsilon":          eps,
                    "objective_config": cfg.attack.objective.value,
                    "probe_mode":       True,
                    "fm_clean":         result.fm_clean,
                    "fm_final":         result.fm_final,
                    "fm_delta":         result.fm_delta,
                    "cer_clean":        pm.cer_clean,
                    "cer_adv":          pm.cer_adv,
                    "cer_delta":        pm.cer_delta,
                    "wer_clean":        pm.wer_clean,
                    "wer_adv":          pm.wer_adv,
                    "wer_delta":        pm.wer_delta,
                    "exact_clean":      pm.exact_clean,
                    "exact_adv":        pm.exact_adv,
                    "passed":           passed,
                    "transcript_ref":   sample.transcript,
                    "transcript_clean": pm.transcript_clean,
                    "transcript_adv":   pm.transcript_adv,
                    **sample.metadata_dict(),
                })

                # ── Trajectory log ─────────────────────────────────────────
                if cfg.log.save_trajectories:
                    logger.log_trajectory(
                        sample.image_id, eps, kappa=0.0,
                        trajectory=result.trajectory,   # type: ignore[arg-type]
                    )

                # ── Visual outputs ─────────────────────────────────────────
                if cfg.log.save_images:
                    stem = f"{sample.image_id}_{norm}_eps{eps:.5f}"
                    save_diff_heatmap(
                        sample.image_tensor, result.x_adv,
                        logger.image_path(f"{stem}_heatmap.png"),
                    )
                    _save_probe_trajectory(
                        result.trajectory,
                        logger.image_path(f"{stem}_trajectory.png"),
                    )

        # ── Final summary ──────────────────────────────────────────────────
        console.print(table)
        console.rule()
        rate  = n_pass / max(n_total, 1)
        color = "green" if rate >= 0.7 else "red"
        console.print(
            f"[bold {color}]Pass rate: {n_pass}/{n_total} "
            f"({rate*100:.0f}%)  |  skipped (dirty baseline): {n_skip}"
            f"[/bold {color}]"
        )
        if rate >= 0.7:
            console.print(
                "[green]✓ Objective signal confirmed. "
                "Safe to proceed to constrained attack (run_attack.py).[/green]"
            )
        else:
            console.print(
                "[red]✗ Objective not moving reliably.[/red]\n"
                "[yellow]Debug checklist:\n"
                "  1. F_M_clean ≈ 0  → model already failing on clean images\n"
                "     → check model loading and extraction prompt\n"
                "  2. ΔF_M ≈ 0       → gradient not flowing through vision encoder\n"
                "     → check pixel_values.requires_grad in ce_loss()\n"
                "  3. ΔF_M < 0       → sign error in ce_loss or align_loss\n"
                "  4. ΔCER ≈ 0 but ΔF_M > 0 → loss moves but inference unaffected\n"
                "     → check transcribe() uses same preprocessing as ce_loss()\n"
                "  5. Try --norm linf and --steps 50 for a faster signal check[/yellow]"
            )

    console.rule("[bold]Probe complete")


if __name__ == "__main__":
    app()