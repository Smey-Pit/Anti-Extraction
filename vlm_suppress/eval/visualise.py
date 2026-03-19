"""
Visualisation utilities.

Produces:
  - Difference heatmaps (original vs perturbed)
  - PGD trajectory plots (F_M and L_H over steps)
  - Side-by-side comparison images

All saved to disk as PNG. No display — runs headlessly on HPC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) float [0,1] -> PIL RGB."""
    arr = (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_diff_heatmap(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    out_path: Path,
    amplify: float = 10.0,
) -> None:
    """
    Saves a 3-panel figure: original | perturbed | amplified diff heatmap.
    amplify: multiplier on the diff for visibility.
    """
    orig_np = x_orig.permute(1, 2, 0).clamp(0, 1).numpy()
    adv_np  = x_adv.permute(1, 2, 0).clamp(0, 1).numpy()
    diff_np = np.abs(adv_np - orig_np).mean(axis=2)  # mean over channels

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(adv_np);  axes[1].set_title("Perturbed"); axes[1].axis("off")
    im = axes[2].imshow(diff_np * amplify, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"Diff ×{amplify}"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_plot(
    trajectory: list,    # list[StepLog]
    out_path: Path,
    kappa: float,
) -> None:
    """
    Plots F_M^ens and L_H over PGD steps.
    A horizontal dashed line marks kappa.
    This is the primary diagnostic for checking the optimizer is alive.
    """
    steps  = [s.step      for s in trajectory]
    fm     = [s.fm_ens    for s in trajectory]
    lh     = [s.lh        for s in trajectory]
    ok     = [s.constraint_ok for s in trajectory]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    color_fm = "#1f77b4"
    color_lh = "#d62728"

    ax1.plot(steps, fm, color=color_fm, label="F_M^ens (↑ = more suppression)")
    ax1.set_xlabel("PGD Step")
    ax1.set_ylabel("F_M^ens", color=color_fm)
    ax1.tick_params(axis="y", labelcolor=color_fm)

    ax2 = ax1.twinx()
    ax2.plot(steps, lh, color=color_lh, linestyle="--", label="L_H (↓ = better readability)")
    ax2.axhline(y=kappa, color="gray", linestyle=":", linewidth=1.5, label=f"κ = {kappa:.4f}")
    ax2.set_ylabel("L_H", color=color_lh)
    ax2.tick_params(axis="y", labelcolor=color_lh)

    # Shade steps where constraint is violated
    for i, (s, is_ok) in enumerate(zip(steps, ok)):
        if not is_ok and i > 0:
            ax1.axvspan(steps[i-1], s, alpha=0.05, color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title("PGD Optimisation Trajectory")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_comparison_strip(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    image_id: str,
    transcript_ref: str,
    transcripts: dict[str, tuple[str, str]],  # model_name -> (clean_pred, adv_pred)
    out_path: Path,
) -> None:
    """
    Side-by-side original | perturbed with transcription annotations.
    Used for the qualitative visual inspection in week 1.
    """
    orig_pil = tensor_to_pil(x_orig)
    adv_pil  = tensor_to_pil(x_adv)

    W, H = orig_pil.size
    padding = 20
    text_height = 20 * (len(transcripts) + 2) + padding
    total_h = H + text_height

    canvas = Image.new("RGB", (W * 2 + padding, total_h), (255, 255, 255))
    canvas.paste(orig_pil, (0, 0))
    canvas.paste(adv_pil,  (W + padding, 0))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    y = H + 5
    draw.text((5, y), f"ID: {image_id}  |  GT: {transcript_ref[:60]}", fill=(0, 0, 0))
    y += 20
    for mname, (pred_c, pred_a) in transcripts.items():
        draw.text((5, y), f"[{mname}] clean: {pred_c[:40]}", fill=(30, 100, 30))
        draw.text((W + padding + 5, y), f"[{mname}]  adv: {pred_a[:40]}", fill=(180, 30, 30))
        y += 20

    canvas.save(out_path)