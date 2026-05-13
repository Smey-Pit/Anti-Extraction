"""
scripts/test_salience_spartan.py

Standalone diagnostic for build_salience_budget_map with real surrogates.

Loads one sample from the configured dataset, runs the salience budget map
computation, prints statistics, and saves a three-panel visualisation.

Usage:
    python scripts/test_salience_spartan.py --config configs/attack.yaml

The script forces cfg.attack.salience_budget = True as a safety net
(attack.yaml already has salience_budget: true).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import dacite
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Make vlm_suppress importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.attack.masks import build_text_mask
from vlm_suppress.attack.salience import build_salience_budget_map
from vlm_suppress.config import (
    Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
)
from vlm_suppress.data.dataset import TextImageDataset


# ── Config loading (mirrors run_attack.py exactly) ────────────────────────────

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


# ── Device assignment (mirrors run_attack.py exactly) ─────────────────────────

def _assign_devices(n_models: int) -> list[torch.device]:
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return [torch.device("cpu")] * n_models
    assignments = [torch.device(f"cuda:{i % n_gpus}") for i in range(n_models)]
    summary = ", ".join(str(d) for d in assignments)
    print(f"  GPU assignment ({n_models} model(s), {n_gpus} GPU(s)): [{summary}]")
    return assignments


# ── Surrogate loading — opt split only ───────────────────────────────────────

def _load_opt_surrogates(cfg: ExperimentConfig, lazy: bool = False) -> list:
    from vlm_suppress.models.internvl2 import InternVL2
    from vlm_suppress.models.internvl3_5 import InternVL35
    from vlm_suppress.models.llava import LLaVA16
    from vlm_suppress.models.llama3_2 import LlamaVision
    from vlm_suppress.models.paligemma2 import PaliGemma2
    from vlm_suppress.models.qwenvl import QwenVL
    from vlm_suppress.models.qwen2vl import Qwen2VL
    from vlm_suppress.models.qwen2_5vl import Qwen2_5VL
    from vlm_suppress.models.lazy import LazySurrogate

    _REG = {
        "internvl3_5": InternVL35,
        "internvl2":   InternVL2,
        "paligemma2":  PaliGemma2,
        "llama3_2":    LlamaVision,
        "qwenvl":      QwenVL,
        "qwen2vl":     Qwen2VL,
        "qwen2_5vl":   Qwen2_5VL,
        "llava16":     LLaVA16,
    }

    selected = [
        (i, s_cfg)
        for i, s_cfg in enumerate(cfg.surrogates)
        if i not in cfg.held_out_indices
    ]

    devices = _assign_devices(len(selected))
    models  = []

    for (i, s_cfg), device in zip(selected, devices):
        cls = _REG.get(s_cfg.name)
        if cls is None:
            raise ValueError(f"Unknown surrogate name: {s_cfg.name!r}")
        s_cfg.device = str(device)
        if lazy:
            print(f"  Registering [lazy] {s_cfg.name} → {device}")
            models.append(LazySurrogate(s_cfg, cls))
        else:
            print(f"  Loading {s_cfg.name} → {device} ...")
            models.append(cls(s_cfg))

    return models


# ── Visualisation ─────────────────────────────────────────────────────────────

def _save_visualisation(
    image_tensor: torch.Tensor,   # (3, H, W) float32 [0, 1]
    budget_map:   torch.Tensor,   # (1, H, W) float32, on CPU
    out_path:     Path,
    vmin:         float,
    vmax:         float,
) -> None:
    """
    Three-panel figure:
      Left   – original image
      Middle – salience budget map as a viridis heatmap
      Right  – budget map overlaid on original at 50 % opacity + colourbar
    """
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
    bmap   = budget_map.squeeze(0).cpu().numpy()            # (H, W)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Left: original image ──────────────────────────────────────────────────
    axes[0].imshow(img_np.clip(0, 1))
    axes[0].set_title("Original image", fontsize=11)
    axes[0].axis("off")

    # ── Middle: raw budget map heatmap ────────────────────────────────────────
    axes[1].imshow(bmap, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(
        f"Salience budget map\n"
        f"[{vmin:.4f}, {vmax:.4f}]",
        fontsize=11,
    )
    axes[1].axis("off")

    # ── Right: overlay + colourbar ────────────────────────────────────────────
    axes[2].imshow(img_np.clip(0, 1))
    im = axes[2].imshow(bmap, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.5)
    axes[2].set_title("Overlay (50 % opacity)", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="ε budget")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salience budget map diagnostic — runs with real surrogates on Spartan."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/attack.yaml"),
        help="Path to experiment YAML config (default: configs/attack.yaml)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = _load_cfg(args.config)

    # Force salience mode on — no YAML edit required
    cfg.attack.salience_budget = True

    atk = cfg.attack
    print("=" * 60)
    print(f"Config:         {args.config}")
    print(f"salience_budget = {atk.salience_budget}")
    print(f"epsilon_min     = {atk.epsilon_min:.6f}")
    print(f"epsilon (max)   = {atk.epsilon:.6f}   (ceiling for text pixels)")
    print(f"epsilon_bg      = {atk.epsilon_bg:.6f}  (background budget)")
    print(f"mask_dilation   = {atk.mask_dilation}")
    print("=" * 60)

    # ── Load opt surrogates ───────────────────────────────────────────────────
    use_lazy = cfg.attack.salience_lazy
    print(f"\n{'Registering lazy' if use_lazy else 'Loading'} opt surrogates "
          f"(salience_lazy={use_lazy}) ...")
    surrogates = _load_opt_surrogates(cfg, lazy=use_lazy)
    print(f"{'Registered' if use_lazy else 'Loaded'} {len(surrogates)} "
          f"surrogate(s): {[m.name for m in surrogates]}")

    # ── Load first dataset sample ─────────────────────────────────────────────
    print("\nLoading dataset (first sample only) ...")
    dataset = TextImageDataset(
        data_dir             = cfg.data.data_dir,
        data_dir_additional  = cfg.data.data_dir_additional,
        image_size           = cfg.data.image_size,
        max_samples          = 1,
        split_filter         = cfg.data.split_filter,
        category_filter      = cfg.data.category_filter,
        contrast_filter      = cfg.data.contrast_filter,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty — check data_dir, split_filter, and other filters in config."
        )
    sample     = dataset[0]
    word_boxes = sample.scaled_word_boxes()
    H, W       = sample.image_tensor.shape[-2], sample.image_tensor.shape[-1]

    if not word_boxes:
        raise RuntimeError(
            f"Sample '{sample.image_id}' has no word boxes. "
            "build_salience_budget_map requires bounding box annotations. "
            "Check that labels.jsonl contains word_boxes for this sample."
        )

    print(f"\nSample:          {sample.image_id}")
    print(f"Image shape:     {tuple(sample.image_tensor.shape)}  (C={3}, H={H}, W={W})")
    print(f"Image device:    {sample.image_tensor.device}")
    print(f"Word boxes:      {len(word_boxes)}")
    print(f"Transcript:      {sample.transcript[:80]!r}{'...' if len(sample.transcript) > 80 else ''}")

    # ── Run salience budget map (timed) ───────────────────────────────────────
    device        = surrogates[0].device
    alpha_weights = [1.0 / len(surrogates)] * len(surrogates)
    image_4d      = sample.image_tensor.unsqueeze(0)   # (1, 3, H, W), CPU float32

    print(f"\nRunning build_salience_budget_map on {device} ...")
    t0 = time.perf_counter()

    budget_map = build_salience_budget_map(
        image_tensor  = image_4d,
        transcript    = sample.transcript,
        word_boxes    = word_boxes,
        surrogates    = surrogates,
        alpha_weights = alpha_weights,
        epsilon_min   = atk.epsilon_min,
        epsilon_max   = atk.epsilon,          # cfg.epsilon reused as text ceiling
        epsilon_bg    = atk.epsilon_bg,
        dilation      = atk.mask_dilation,
        device        = device,
    )

    elapsed = time.perf_counter() - t0
    print(f"  Wall-clock time (salience only): {elapsed:.2f} s")

    # ── Statistics ────────────────────────────────────────────────────────────
    bmap_cpu = budget_map.squeeze(0).cpu()   # (H, W)

    print(f"\nBudget map shape:  {tuple(budget_map.shape)}")
    print(
        f"Overall   — "
        f"min: {bmap_cpu.min():.6f}  "
        f"mean: {bmap_cpu.mean():.6f}  "
        f"max: {bmap_cpu.max():.6f}"
    )

    # Split stats by text / background mask
    text_mask  = build_text_mask(
        H, W, word_boxes, dilation=atk.mask_dilation, device=torch.device("cpu")
    )                                                   # (1, H, W)
    text_flag  = text_mask.squeeze(0) > 0              # (H, W) bool
    text_vals  = bmap_cpu[text_flag]
    bg_vals    = bmap_cpu[~text_flag]

    if text_vals.numel() > 0:
        print(
            f"Text region — "
            f"min: {text_vals.min():.6f}  "
            f"mean: {text_vals.mean():.6f}  "
            f"max: {text_vals.max():.6f}  "
            f"({text_vals.numel()} px)"
        )
    else:
        print("Text region — no text pixels (word_boxes empty or all outside image)")

    if bg_vals.numel() > 0:
        print(
            f"Background  — "
            f"min: {bg_vals.min():.6f}  "
            f"mean: {bg_vals.mean():.6f}  "
            f"max: {bg_vals.max():.6f}  "
            f"({bg_vals.numel()} px)"
        )
        expected = torch.full_like(bg_vals, atk.epsilon_bg)
        if torch.allclose(bg_vals, expected):
            print(f"  ✓ All background pixels exactly equal epsilon_bg={atk.epsilon_bg:.6f}")
        else:
            print(
                f"  ✗ Background pixels deviate from epsilon_bg={atk.epsilon_bg:.6f}  "
                f"(max deviation: {(bg_vals - expected).abs().max():.2e})"
            )

    # ── Visualisation ─────────────────────────────────────────────────────────
    # Colourbar spans the true data range of the budget map.
    vmin = float(bmap_cpu.min().item())
    vmax = float(bmap_cpu.max().item())

    out_path = Path("outputs/salience_debug.png")
    _save_visualisation(
        image_tensor = sample.image_tensor,
        budget_map   = budget_map.cpu(),
        out_path     = out_path,
        vmin         = vmin,
        vmax         = vmax,
    )

    print(f"\nVisualisation saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
