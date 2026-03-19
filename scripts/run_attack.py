"""
Main constrained attack entry point.
Run ONLY after run_probe.py confirms the objective signal.

Usage:
    python scripts/run_attack.py --config configs/week1_sanity.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import dacite
import torch
import typer
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.attack.pgd import run_attack
from vlm_suppress.config import ExperimentConfig
from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.eval.metrics import ModelMetrics, compute_transfer_ratio
from vlm_suppress.eval.visualise import (
    save_comparison_strip,
    save_diff_heatmap,
    save_trajectory_plot,
)
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
            cast=[Path],
            type_hooks={
                Optional[tuple[int, int]]: lambda v: tuple(v) if v is not None else None,
            },
        ),
    )


def _load_surrogates(cfg: ExperimentConfig, held_out: bool = False) -> list:
    from vlm_suppress.models.internvl import InternVL2
    from vlm_suppress.models.llava import LLaVA16
    from vlm_suppress.models.qwenvl import QwenVL

    _REG = {"internvl2": InternVL2, "qwenvl": QwenVL, "llava16": LLaVA16}
    models = []
    for i, s_cfg in enumerate(cfg.surrogates):
        is_held_out = i in cfg.held_out_indices
        if held_out != is_held_out:
            continue
        cls = _REG.get(s_cfg.name)
        if cls is None:
            raise ValueError(f"Unknown surrogate: {s_cfg.name}")
        console.log(f"Loading {'[held-out] ' if is_held_out else ''}{s_cfg.name} ...")
        models.append(cls(s_cfg))
    return models


def _run_single_config(
    cfg: ExperimentConfig,
    epsilon: float,
    kappa: float,
    dataset: TextImageDataset,
    opt_surrogates: list,
    heldout_surrogates: list,
    logger: RunLogger,
) -> None:
    cfg.attack.epsilon = epsilon
    cfg.attack.kappa   = kappa

    for sample in dataset:
        console.log(
            f"  [{sample.image_id}] eps={epsilon:.5f} kappa={kappa:.4f} "
            f"cat={sample.text_category} contrast={sample.contrast_level}"
        )

        # ── Clean transcriptions ───────────────────────────────────────────
        clean_preds: dict[str, str] = {}
        for m in opt_surrogates + heldout_surrogates:
            with torch.no_grad():
                clean_preds[m.name] = m.transcribe(sample.image_tensor)

        # ── Run attack (with readability constraint) ───────────────────────
        result = run_attack(
            image_id=sample.image_id,
            x_orig=sample.image_tensor,
            transcript=sample.transcript,
            surrogates=opt_surrogates,
            cfg=cfg.attack,
            word_boxes=sample.scaled_word_boxes(),   # always use scaled boxes
        )

        # ── Adversarial transcriptions ─────────────────────────────────────
        adv_preds: dict[str, str] = {}
        for m in opt_surrogates + heldout_surrogates:
            with torch.no_grad():
                adv_preds[m.name] = m.transcribe(result.x_adv)

        # ── Metrics ────────────────────────────────────────────────────────
        all_metrics = [
            ModelMetrics.compute(
                image_id=sample.image_id,
                model_name=m.name,
                transcript_clean=clean_preds[m.name],
                transcript_adv=adv_preds[m.name],
                reference=sample.transcript,
            )
            for m in opt_surrogates + heldout_surrogates
        ]

        opt_names     = {m.name for m in opt_surrogates}
        heldout_names = {m.name for m in heldout_surrogates}

        wb_deltas  = [m.cer_delta for m in all_metrics if m.model_name in opt_names]
        hol_deltas = [m.cer_delta for m in all_metrics if m.model_name in heldout_names]
        wb_mean    = sum(wb_deltas)  / max(len(wb_deltas),  1)
        hol_mean   = sum(hol_deltas) / max(len(hol_deltas), 1)
        t_ratio    = compute_transfer_ratio(wb_mean, hol_mean)

        # ── Log results ────────────────────────────────────────────────────
        for mets in all_metrics:
            logger.log_result({
                "image_id":             sample.image_id,
                "model_name":           mets.model_name,
                "is_held_out":          mets.model_name in heldout_names,
                "epsilon":              epsilon,
                "kappa":                kappa,
                "objective_config":     cfg.attack.objective.value,
                "proxy_stage":          cfg.attack.proxy_stage.value,
                "lambda_ce":            cfg.attack.lambda_ce,
                "lambda_align":         cfg.attack.lambda_align,
                "cer_clean":            mets.cer_clean,
                "cer_adv":              mets.cer_adv,
                "cer_delta":            mets.cer_delta,
                "wer_clean":            mets.wer_clean,
                "wer_adv":              mets.wer_adv,
                "wer_delta":            mets.wer_delta,
                "exact_clean":          mets.exact_clean,
                "exact_adv":            mets.exact_adv,
                "constraint_satisfied": result.constraint_satisfied,
                "fm_ens_final":         result.fm_ens_final,
                "lh_final":             result.lh_final,
                "transfer_ratio": t_ratio if mets.model_name in heldout_names else None,
                "transcript_ref":       sample.transcript,
                "transcript_clean":     mets.transcript_clean,
                "transcript_adv":       mets.transcript_adv,
                **sample.metadata_dict(),
            })

        # ── Trajectory + visuals ───────────────────────────────────────────
        if cfg.log.save_trajectories:
            logger.log_trajectory(
                sample.image_id, epsilon, kappa, result.trajectory
            )

        if cfg.log.save_images:
            stem = f"{sample.image_id}_eps{epsilon:.5f}_kappa{kappa:.4f}"
            if cfg.log.save_visual_diffs:
                save_diff_heatmap(
                    sample.image_tensor, result.x_adv,
                    logger.image_path(f"{stem}_heatmap.png"),
                )
            save_trajectory_plot(
                result.trajectory,
                logger.image_path(f"{stem}_trajectory.png"),
                kappa=kappa,
            )
            save_comparison_strip(
                sample.image_tensor, result.x_adv,
                image_id=sample.image_id,
                transcript_ref=sample.transcript,
                transcripts={
                    m.model_name: (clean_preds[m.model_name], adv_preds[m.model_name])
                    for m in all_metrics
                },
                out_path=logger.image_path(f"{stem}_strip.png"),
            )

        # ── Terminal feedback ──────────────────────────────────────────────
        for mets in [m for m in all_metrics if m.model_name in opt_names]:
            status = (
                "✓ PASS" if mets.cer_delta > 0 and result.constraint_satisfied
                else "✗ FAIL"
            )
            console.log(
                f"    {mets.model_name}: CER {mets.cer_clean:.3f}→{mets.cer_adv:.3f} "
                f"(Δ={mets.cer_delta:+.3f}) | L_H={result.lh_final:.5f} "
                f"{'≤' if result.constraint_satisfied else '>'} κ={kappa:.4f} | {status}"
            )


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to YAML config"),
) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    cfg = _load_cfg(config)
    console.rule(f"[bold blue]VLM-Suppress | Run: {cfg.run_id}")
    console.log(f"Phase: {cfg.phase}")

    with RunLogger(cfg) as logger:
        dataset = TextImageDataset(
            cfg.data.data_dir,
            image_size=cfg.data.image_size,
            max_samples=cfg.data.n_images,
            split_filter=cfg.data.split_filter,
            category_filter=cfg.data.category_filter,
            contrast_filter=cfg.data.contrast_filter,
        )
        console.log(f"Loaded {len(dataset)} samples")

        opt_surrogates     = _load_surrogates(cfg, held_out=False)
        heldout_surrogates = _load_surrogates(cfg, held_out=True)

        epsilons = cfg.epsilon_sweep or [cfg.attack.epsilon]
        kappas   = cfg.kappa_sweep   or [cfg.attack.kappa]

        for eps in epsilons:
            for kap in kappas:
                console.rule(f"ε={eps:.5f}  κ={kap:.4f}")
                _run_single_config(
                    cfg, eps, kap, dataset,
                    opt_surrogates, heldout_surrogates, logger,
                )

    console.rule("[bold green]Done")


if __name__ == "__main__":
    app()