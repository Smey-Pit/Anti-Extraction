"""
scripts/test_importance_spartan.py

Stage 1 importance map diagnostic.

Computes the token-surprise × visual-KL-reweighted salience map and
compares it to the plain gradient salience.  Produces a five-panel
visualisation and a quantitative sanity check (value vs label region ratio).

Usage (from project root, on Spartan):
    uv run python scripts/test_importance_spartan.py --config configs/attack.yaml
    uv run python scripts/test_importance_spartan.py --config configs/attack.yaml \\
        --no_kl          # skip visual KL (much faster)
    uv run python scripts/test_importance_spartan.py --config configs/attack.yaml \\
        --out_dir outputs/importance_debug

Output
------
  outputs/importance_debug/importance_<image_id>.png
        Five panels: original | salience | surprise | kl | importance (product)
  stdout: per-component stats, top/bottom word ranking, label vs value ratio
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.attack.importance import build_importance_map
from vlm_suppress.attack.masks import build_text_mask
from vlm_suppress.config import (
    Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
)
from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.models.lazy import LazySurrogate, OffloadSurrogate


# ── Config + surrogate loading (mirrors run_attack.py) ────────────────────────

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
    from vlm_suppress.models.internvl2 import InternVL2
    from vlm_suppress.models.internvl3_5 import InternVL35
    from vlm_suppress.models.llava import LLaVA16
    from vlm_suppress.models.llama3_2 import LlamaVision
    from vlm_suppress.models.paligemma2 import PaliGemma2
    from vlm_suppress.models.qwenvl import QwenVL
    from vlm_suppress.models.qwen2vl import Qwen2VL
    from vlm_suppress.models.qwen2_5vl import Qwen2_5VL

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

    n_gpus = torch.cuda.device_count()
    selected = [
        (i, s_cfg)
        for i, s_cfg in enumerate(cfg.surrogates)
        if i not in cfg.held_out_indices
    ]

    models = []
    for k, (i, s_cfg) in enumerate(selected):
        device = torch.device(f"cuda:{k % n_gpus}" if n_gpus > 0 else "cpu")
        s_cfg.device = str(device)
        cls = _REG.get(s_cfg.name)
        if cls is None:
            raise ValueError(f"Unknown surrogate: {s_cfg.name!r}")
        print(f"  Loading {s_cfg.name} → {device} ...")
        models.append(cls(s_cfg))

    return models


# ── Visualisation ─────────────────────────────────────────────────────────────

def _save_five_panel(
    image_tensor: torch.Tensor,   # (3, H, W) CPU float32 [0,1]
    components:   dict,           # {'salience', 'surprise', 'kl', 'importance'} — (H,W) CPU
    out_path: Path,
) -> None:
    """
    Five-panel figure:
      [0] Original image
      [1] Gradient salience
      [2] Token surprise (-log p | blank)
      [3] Visual KL
      [4] Importance = product of 1-3
    """
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    panels = [
        ("Original",         img_np,                               False),
        ("Gradient salience", components["salience"].numpy(),       True),
        ("Token surprise\n(-log p | blank)", components["surprise"].numpy(), True),
        ("Visual KL\n(log p_orig - log p_masked)", components["kl"].numpy(), True),
        ("Importance\n(product, normalised)", components["importance"].numpy(), True),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    for ax, (title, data, is_heatmap) in zip(axes, panels):
        if is_heatmap:
            im = ax.imshow(data, cmap="plasma", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Sanity check: label vs value region ratio ─────────────────────────────────

_LABEL_SUFFIXES = (":", "：", "#", "-")
_LABEL_KEYWORDS = {
    "account", "holder", "name", "date", "description",
    "transaction", "history", "balance", "type", "reference",
    "details", "information", "address", "phone", "email",
    "amount", "total", "from", "to", "subject",
}


def _is_label_word(word: str) -> bool:
    w = word.strip().rstrip(":：").lower()
    if word.endswith(_LABEL_SUFFIXES):
        return True
    if w in _LABEL_KEYWORDS:
        return True
    return False


def _sanity_check(
    importance_map: torch.Tensor,   # (H, W) normalized [0,1] CPU
    word_boxes:     list[list[int]],
    transcript:     str,
    H: int, W: int,
) -> None:
    words = transcript.split()
    n_words = min(len(words), len(word_boxes))

    word_scores: list[tuple[float, str, bool]] = []
    for i in range(n_words):
        word = words[i]
        x0, y0, x1, y1 = (int(v) for v in word_boxes[i])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            region = importance_map[y0:y1, x0:x1]
            score  = float(region.mean()) if region.numel() > 0 else 0.0
        else:
            score = 0.0
        word_scores.append((score, word, _is_label_word(word)))

    word_scores.sort(key=lambda x: x[0], reverse=True)

    print("\n── Top 10 words by importance: ─────────────────────────────────")
    for score, word, is_label in word_scores[:10]:
        tag = "[label]" if is_label else "[value]"
        print(f"  {score:.4f}  {tag:<8}  {word!r}")

    print("\n── Bottom 10 words by importance: ──────────────────────────────")
    for score, word, is_label in word_scores[-10:]:
        tag = "[label]" if is_label else "[value]"
        print(f"  {score:.4f}  {tag:<8}  {word!r}")

    label_scores = [s for s, _, is_label in word_scores if is_label]
    value_scores = [s for s, _, is_label in word_scores if not is_label]

    print("\n── Label vs value region sanity check: ─────────────────────────")
    if label_scores:
        label_mean = sum(label_scores) / len(label_scores)
        print(f"  Label words  ({len(label_scores):3d}):  mean importance = {label_mean:.4f}")
    else:
        label_mean = 0.0
        print("  Label words: none found by heuristic")

    if value_scores:
        value_mean = sum(value_scores) / len(value_scores)
        print(f"  Value words  ({len(value_scores):3d}):  mean importance = {value_mean:.4f}")
    else:
        value_mean = 0.0
        print("  Value words: none found by heuristic")

    if label_scores and value_scores:
        ratio = value_mean / (label_mean + 1e-9)
        print(f"  Value/Label ratio: {ratio:.2f}x")
        if ratio > 2.0:
            print("  ✓ PASS: value regions are >2× more salient than label regions")
        else:
            print("  ✗ NOTE: ratio < 2× — KL/surprise weighting may not dominate gradient")
            print("         Consider increasing epsilon_max/epsilon_min spread or")
            print("         inspecting per-component maps for which signal is flat.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 importance map diagnostic — runs on Spartan with real surrogates."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/attack.yaml"),
    )
    parser.add_argument(
        "--no_kl", action="store_true",
        help="Skip visual KL (much faster; shows surprise + salience only)",
    )
    parser.add_argument(
        "--no_surprise", action="store_true",
        help="Skip token surprise (shows KL + salience only)",
    )
    parser.add_argument(
        "--context_radius", type=float, default=50.0,
        help="Context masking radius in pixels for visual KL (default: 50)",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("outputs/importance_debug"),
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    atk = cfg.attack

    print("=" * 65)
    print(f"Config:        {args.config}")
    print(f"use_surprise:  {not args.no_surprise}")
    print(f"use_visual_kl: {not args.no_kl}")
    print(f"epsilon_min:   {atk.epsilon_min:.5f}")
    print(f"epsilon_max:   {atk.epsilon:.5f}")
    print(f"epsilon_bg:    {atk.epsilon_bg:.5f}")
    print(f"mask_dilation: {atk.mask_dilation}")
    print("=" * 65)

    # ── Load surrogates ───────────────────────────────────────────────────────
    print("\nLoading opt surrogates ...")
    surrogates = _load_opt_surrogates(cfg)
    print(f"Loaded: {[m.name for m in surrogates]}")

    # ── Load sample ───────────────────────────────────────────────────────────
    print("\nLoading dataset (first sample) ...")
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        split_filter        = cfg.data.split_filter,
        category_filter     = cfg.data.category_filter,
        contrast_filter     = cfg.data.contrast_filter,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — check config filters.")

    sample     = dataset[0]
    word_boxes = sample.scaled_word_boxes()
    H, W       = sample.image_tensor.shape[-2], sample.image_tensor.shape[-1]

    if not word_boxes:
        raise RuntimeError(
            f"Sample '{sample.image_id}' has no word boxes. "
            "Importance mapping requires bounding box annotations."
        )

    print(f"\nSample:       {sample.image_id}")
    print(f"Image shape:  {tuple(sample.image_tensor.shape)}")
    print(f"Word boxes:   {len(word_boxes)}")
    print(f"Transcript:   {sample.transcript[:100]!r}"
          f"{'...' if len(sample.transcript) > 100 else ''}")

    # ── Run importance map ────────────────────────────────────────────────────
    device        = surrogates[0].device
    alpha_weights = [1.0 / len(surrogates)] * len(surrogates)

    # Salience surrogate index filtering (mirrors pgd.py logic)
    _pool = surrogates   # opt-only for diagnostic
    if atk.salience_surrogate_indices is not None:
        sal_surrogates = [
            _pool[i] for i in atk.salience_surrogate_indices if i < len(_pool)
        ]
        if not sal_surrogates:
            sal_surrogates = surrogates
        sal_weights = [1.0 / len(sal_surrogates)] * len(sal_surrogates)
    else:
        sal_surrogates = surrogates
        sal_weights    = alpha_weights

    # Wrap eager surrogates in OffloadSurrogate so only one model occupies VRAM
    # at a time during the salience/surprise/KL passes.  Each __enter__ moves
    # weights to GPU; __exit__ moves them back to CPU.
    eager_sal = [s for s in sal_surrogates if not isinstance(s, LazySurrogate)]
    if eager_sal:
        print(
            f"  Offloading {len(eager_sal)} eager surrogate(s) to CPU "
            "(one-at-a-time VRAM during importance pass) ..."
        )
        for s in eager_sal:
            s.model.to("cpu")
        torch.cuda.empty_cache()
        sal_surrogates = [
            OffloadSurrogate(s) if not isinstance(s, LazySurrogate) else s
            for s in sal_surrogates
        ]

    print(f"\nSalience surrogates: {[m.name for m in sal_surrogates]}")
    print(
        f"use_visual_kl={not args.no_kl}   use_surprise={not args.no_surprise}   "
        f"context_radius={args.context_radius:.0f}px"
    )

    t0 = time.perf_counter()
    eps_map, components = build_importance_map(
        image_tensor      = sample.image_tensor,
        transcript        = sample.transcript,
        word_boxes        = word_boxes,
        surrogates        = sal_surrogates,
        alpha_weights     = sal_weights,
        epsilon_min       = atk.epsilon_min,
        epsilon_max       = atk.epsilon,
        epsilon_bg        = atk.epsilon_bg,
        dilation          = atk.mask_dilation,
        device            = device,
        use_surprise      = not args.no_surprise,
        use_visual_kl     = not args.no_kl,
        context_radius_px = args.context_radius,
    )
    elapsed = time.perf_counter() - t0

    # Restore offloaded models to GPU (so they're available for any post-pass use)
    for s in eager_sal:
        s.model.to(s.device)
    print(f"\nWall-clock time: {elapsed:.1f} s")

    # ── Component statistics ──────────────────────────────────────────────────
    text_mask  = build_text_mask(H, W, word_boxes, atk.mask_dilation, torch.device("cpu"))
    text_flag  = text_mask.squeeze(0) > 0

    print("\n── Component statistics (text region only): ─────────────────────")
    for key in ("salience", "surprise", "kl", "importance"):
        m = components[key]
        vals = m[text_flag]
        if vals.numel() == 0:
            print(f"  {key:<12}: no text pixels")
        else:
            print(
                f"  {key:<12}: "
                f"min={vals.min():.4f}  mean={vals.mean():.4f}  max={vals.max():.4f}  "
                f"nonzero={int((vals > 0).sum())} / {vals.numel()}"
            )

    # ── Sanity check ──────────────────────────────────────────────────────────
    _sanity_check(components["importance"], word_boxes, sample.transcript, H, W)

    # ── Visualise ─────────────────────────────────────────────────────────────
    out_path = args.out_dir / f"importance_{sample.image_id}.png"
    _save_five_panel(
        image_tensor = sample.image_tensor,
        components   = components,
        out_path     = out_path,
    )

    # Also save the eps_map for reference
    eps_np = eps_map.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(eps_np, cmap="viridis",
                   vmin=float(eps_np.min()), vmax=float(eps_np.max()))
    ax.set_title(f"Importance-weighted ε budget map\n"
                 f"[{eps_np.min():.4f}, {eps_np.max():.4f}]")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ε budget")
    eps_path = args.out_dir / f"eps_importance_{sample.image_id}.png"
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(eps_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {eps_path}")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
