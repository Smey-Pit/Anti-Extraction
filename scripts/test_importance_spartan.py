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

    use_lazy = (
        getattr(cfg.attack, "salience_lazy", False)
        or getattr(cfg.attack, "salience_offload", False)
    )

    models = []
    for k, (i, s_cfg) in enumerate(selected):
        device = torch.device(f"cuda:{k % n_gpus}" if n_gpus > 0 else "cpu")
        s_cfg.device = str(device)
        cls = _REG.get(s_cfg.name)
        if cls is None:
            raise ValueError(f"Unknown surrogate: {s_cfg.name!r}")
        if use_lazy:
            print(f"  Registering {s_cfg.name} → {device} (lazy, no eager load) ...")
            models.append(LazySurrogate(s_cfg, cls))
        else:
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
    log_path:     Optional[Path] = None,
    word_strings: Optional[list] = None,
) -> None:
    lines: list[str] = []

    def _emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    words = word_strings if word_strings is not None else transcript.split()
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

    _emit("\n── Top 10 words by importance: ─────────────────────────────────")
    for score, word, is_label in word_scores[:10]:
        tag = "[label]" if is_label else "[value]"
        _emit(f"  {score:.4f}  {tag:<8}  {word!r}")

    _emit("\n── Bottom 10 words by importance: ──────────────────────────────")
    for score, word, is_label in word_scores[-10:]:
        tag = "[label]" if is_label else "[value]"
        _emit(f"  {score:.4f}  {tag:<8}  {word!r}")

    label_scores = [s for s, _, is_label in word_scores if is_label]
    value_scores = [s for s, _, is_label in word_scores if not is_label]

    _emit("\n── Label vs value region sanity check: ─────────────────────────")
    if label_scores:
        label_mean = sum(label_scores) / len(label_scores)
        _emit(f"  Label words  ({len(label_scores):3d}):  mean importance = {label_mean:.4f}")
    else:
        label_mean = 0.0
        _emit("  Label words: none found by heuristic")

    if value_scores:
        value_mean = sum(value_scores) / len(value_scores)
        _emit(f"  Value words  ({len(value_scores):3d}):  mean importance = {value_mean:.4f}")
    else:
        value_mean = 0.0
        _emit("  Value words: none found by heuristic")

    if label_scores and value_scores:
        ratio = value_mean / (label_mean + 1e-9)
        _emit(f"  Value/Label ratio: {ratio:.2f}x")
        if ratio > 2.0:
            _emit("  ✓ PASS: value regions are >2× more salient than label regions")
        else:
            _emit("  ✗ NOTE: ratio < 2× — KL/surprise weighting may not dominate gradient")
            _emit("         Consider increasing epsilon_max/epsilon_min spread or")
            _emit("         inspecting per-component maps for which signal is flat.")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Saved → {log_path}")


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
    parser.add_argument(
        "--max_samples", type=int, default=1,
        help="Number of samples per category (default: 1)",
    )
    parser.add_argument(
        "--confidence_drop", action="store_true",
        help="Also run compute_confidence_drop and add sixth panel.",
    )
    parser.add_argument(
        "--entropy", action="store_true",
        help="Run compute_blank_entropy on all surrogates individually "
             "and show per-model contributions.",
    )
    parser.add_argument(
        "--entropy_only", action="store_true",
        help="Skip importance pipeline; run only compute_blank_entropy.",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help=(
            "Categories to test (default: use config category_filter, "
            "or all UI categories if unset)"
        ),
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

    _UI_CATEGORIES = [
        "banking", "medical", "legal", "identity",
        "communications", "news", "copyright",
    ]

    # Which categories to iterate over
    if args.categories:
        categories = args.categories
    elif cfg.data.category_filter:
        categories = [cfg.data.category_filter]
    else:
        categories = _UI_CATEGORIES

    # ── Load surrogates (once, shared across all categories/samples) ──────────
    print("\nLoading opt surrogates ...")
    surrogates = _load_opt_surrogates(cfg)
    print(f"Loaded: {[m.name for m in surrogates]}")

    # ── Surrogate selection + offload wrapping (done once) ────────────────────
    device        = surrogates[0].device
    alpha_weights = [1.0 / len(surrogates)] * len(surrogates)

    _pool = surrogates
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
    print(f"Categories: {categories}   max_samples: {args.max_samples}")

    # ── Category × sample loop ────────────────────────────────────────────────
    total_processed = 0
    skipped         = 0

    for category in categories:
        print(f"\n{'━' * 65}")
        print(f"Category: {category}")
        print(f"{'━' * 65}")

        dataset = TextImageDataset(
            data_dir            = cfg.data.data_dir,
            data_dir_additional = cfg.data.data_dir_additional,
            image_size          = cfg.data.image_size,
            max_samples         = args.max_samples,
            split_filter        = cfg.data.split_filter,
            category_filter     = category,
            contrast_filter     = cfg.data.contrast_filter,
        )

        if len(dataset) == 0:
            print(f"  No samples found for category={category!r} — skipping.")
            skipped += 1
            continue

        print(f"  {len(dataset)} sample(s) found.")

        for sample_idx, sample in enumerate(dataset):
            word_boxes   = sample.scaled_word_boxes()
            word_strings = sample.scaled_word_strings()
            print(f"Word boxes:   {len(word_boxes)}")
            print(f"Word strings: {len(word_strings)}")
            assert len(word_boxes) == len(word_strings), (
                f"MISMATCH: {len(word_boxes)} boxes vs "
                f"{len(word_strings)} strings — "
                "scaled_word_strings() iteration does not match "
                "scaled_word_boxes()"
            )
            print("PASS: word_boxes and word_strings are aligned")
            H, W = sample.image_tensor.shape[-2], sample.image_tensor.shape[-1]

            print(f"\n  [{sample_idx + 1}/{len(dataset)}] {sample.image_id}")
            print(f"  Image shape: {tuple(sample.image_tensor.shape)}  "
                  f"Word boxes: {len(word_boxes)}")
            print(f"  Transcript:  {sample.transcript[:80]!r}"
                  f"{'...' if len(sample.transcript) > 80 else ''}")

            if not word_boxes:
                print(f"  SKIP — no word boxes for {sample.image_id}.")
                skipped += 1
                continue

            out_path = args.out_dir / category / f"importance_{sample.image_id}.png"

            if not args.entropy_only:

                # ── Importance map ────────────────────────────────────────────────
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
                    word_strings      = word_strings,
                )
                elapsed = time.perf_counter() - t0
                print(f"  Wall-clock: {elapsed:.1f} s")

                # ── Component statistics ──────────────────────────────────────────
                text_mask = build_text_mask(
                    H, W, word_boxes, atk.mask_dilation, torch.device("cpu")
                )
                text_flag = text_mask.squeeze(0) > 0

                print("  ── Component statistics (text region): ──────────────────")
                for key in ("salience", "surprise", "kl", "importance"):
                    m    = components[key]
                    vals = m[text_flag]
                    if vals.numel() == 0:
                        print(f"    {key:<12}: no text pixels")
                    else:
                        print(
                            f"    {key:<12}: "
                            f"min={vals.min():.4f}  mean={vals.mean():.4f}  "
                            f"max={vals.max():.4f}  "
                            f"nonzero={int((vals > 0).sum())}/{vals.numel()}"
                        )

                # ── Sanity check + per-sample log ────────────────────────────────
                log_path = args.out_dir / category / f"importance_{sample.image_id}.txt"
                _sanity_check(
                    components["importance"], word_boxes, sample.transcript, H, W,
                    log_path=log_path,
                    word_strings=word_strings,
                )

                # ── Visualise ─────────────────────────────────────────────────────
                _save_five_panel(
                    image_tensor = sample.image_tensor,
                    components   = components,
                    out_path     = out_path,
                )

                eps_np  = eps_map.squeeze(0).cpu().numpy()
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                im = ax.imshow(eps_np, cmap="viridis",
                               vmin=float(eps_np.min()), vmax=float(eps_np.max()))
                ax.set_title(
                    f"Importance-weighted ε budget map\n"
                    f"[{eps_np.min():.4f}, {eps_np.max():.4f}]"
                )
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ε budget")
                eps_path = args.out_dir / category / f"eps_importance_{sample.image_id}.png"
                eps_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(eps_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved → {eps_path}")

                # ── Confidence drop (optional) ────────────────────────────────────
                if args.confidence_drop:
                    from vlm_suppress.attack.importance import (
                        compute_confidence_drop, _normalize_01,
                    )

                    cd_maps = []
                    for surrogate, alpha in zip(sal_surrogates, sal_weights):
                        print(f"  [confidence_drop] running on {surrogate.name} ...")
                        if isinstance(surrogate, LazySurrogate):
                            with surrogate as model:
                                cd_k = compute_confidence_drop(
                                    model, sample.image_tensor,
                                    sample.transcript, word_boxes,
                                    word_strings=word_strings,
                                    context_radius_px=atk.mask_dilation * 10,
                                )
                        else:
                            cd_k = compute_confidence_drop(
                                surrogate, sample.image_tensor,
                                sample.transcript, word_boxes,
                                word_strings=word_strings,
                                context_radius_px=atk.mask_dilation * 10,
                            )
                        cd_maps.append(alpha * cd_k)

                    cd_map = torch.stack(cd_maps).sum(dim=0)   # weighted average

                    cd_norm = _normalize_01(cd_map)

                    # ── Top/bottom 10 by confidence drop ─────────────────────────
                    words   = sample.transcript.split()
                    n_words = min(len(words), len(word_boxes))
                    cd_word_scores = []
                    for i in range(n_words):
                        x0, y0, x1, y1 = (int(v) for v in word_boxes[i])
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(W, x1), min(H, y1)
                        region = cd_norm[y0:y1, x0:x1]
                        score  = float(region.mean()) if region.numel() > 0 else 0.0
                        cd_word_scores.append((score, words[i]))

                    cd_sorted = sorted(cd_word_scores, key=lambda x: x[0], reverse=True)
                    print("\n── Top 10 by confidence drop: ──────────────────────────")
                    for score, word in cd_sorted[:10]:
                        print(f"  {score:.4f}  {word!r}")
                    print("\n── Bottom 10 by confidence drop: ───────────────────────")
                    for score, word in cd_sorted[-10:]:
                        print(f"  {score:.4f}  {word!r}")

                    # ── Cross-compare with existing importance top-10 ─────────────
                    imp_sorted = sorted(
                        [(float(components["importance"][
                              max(0, int(word_boxes[i][1])):min(H, int(word_boxes[i][3])),
                              max(0, int(word_boxes[i][0])):min(W, int(word_boxes[i][2]))
                          ].mean()), words[i])
                         for i in range(n_words)
                         if int(word_boxes[i][2]) > int(word_boxes[i][0])
                         and int(word_boxes[i][3]) > int(word_boxes[i][1])],
                        key=lambda x: x[0], reverse=True,
                    )
                    imp_top10 = {w for _, w in imp_sorted[:10]}
                    cd_top10  = {w for _, w in cd_sorted[:10]}
                    cap_words = {w for w in (imp_top10 | cd_top10) if w[0].isupper()}

                    if cap_words:
                        print("\n── Capitalised token cross-comparison: ─────────────────")
                        for w in sorted(cap_words):
                            in_imp = w in imp_top10
                            in_cd  = w in cd_top10
                            tag = "[both]   " if in_imp and in_cd \
                                  else "[cd_only] " if in_cd \
                                  else "[imp_only]"
                            print(f"  {tag}  {w!r}")

                    # ── Six-panel visualisation ───────────────────────────────────
                    img_np = sample.image_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    panels = [
                        ("Original",           img_np,                           False),
                        ("Gradient salience",  components["salience"].numpy(),   True),
                        ("Token surprise",     components["surprise"].numpy(),   True),
                        ("Visual KL",          components["kl"].numpy(),         True),
                        ("Importance\n(current)", components["importance"].numpy(), True),
                        ("Confidence Drop\n(new)", cd_norm.numpy(),              True),
                    ]
                    fig, axes = plt.subplots(1, 6, figsize=(34, 5))
                    for ax, (title, data, is_heatmap) in zip(axes, panels):
                        if is_heatmap:
                            im = ax.imshow(data, cmap="plasma", vmin=0, vmax=1)
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        else:
                            ax.imshow(data)
                        ax.set_title(title, fontsize=9)
                        ax.axis("off")
                    plt.tight_layout()
                    cd_out = out_path.parent / f"cd_{sample.image_id}.png"
                    fig.savefig(cd_out, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"\n  Saved six-panel → {cd_out}")

            # ── Blank-image entropy (optional) ────────────────────────────────
            if args.entropy or args.entropy_only:
                from vlm_suppress.attack.importance import (
                    compute_blank_entropy, _normalize_01,
                )

                entropy_maps   = {}   # name -> (H, W) normalised tensor
                entropy_scores = {}   # name -> list[float] per word

                for surrogate, alpha in zip(sal_surrogates, sal_weights):
                    print(f"\n  [entropy] running on {surrogate.name} ...")
                    if isinstance(surrogate, LazySurrogate):
                        with surrogate as model:
                            ent_k = compute_blank_entropy(
                                model, sample.image_tensor,
                                sample.transcript, word_boxes,
                                word_strings=word_strings,
                            )
                    else:
                        ent_k = compute_blank_entropy(
                            surrogate, sample.image_tensor,
                            sample.transcript, word_boxes,
                            word_strings=word_strings,
                        )

                    ent_norm = _normalize_01(ent_k)
                    entropy_maps[surrogate.name] = ent_norm

                    # Per-word scores for this surrogate
                    _words = word_strings if word_strings else sample.transcript.split()
                    _n     = min(len(_words), len(word_boxes))
                    _scores = []
                    for i in range(_n):
                        x0, y0, x1, y1 = (int(v) for v in word_boxes[i])
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(W, x1), min(H, y1)
                        region = ent_norm[y0:y1, x0:x1]
                        score  = float(region.mean()) if region.numel() > 0 else 0.0
                        _scores.append((score, _words[i]))
                    entropy_scores[surrogate.name] = _scores

                    # Print top/bottom 10 for this surrogate
                    _sorted = sorted(_scores, key=lambda x: x[0], reverse=True)
                    print(f"\n  ── Top 10 by entropy ({surrogate.name}): ──────────")
                    for score, word in _sorted[:10]:
                        print(f"    {score:.4f}  {word!r}")
                    print(f"\n  ── Bottom 10 by entropy ({surrogate.name}): ───────")
                    for score, word in _sorted[-10:]:
                        print(f"    {score:.4f}  {word!r}")

                    # Debug values for key words
                    _word_list = [w for _, w in _scores]
                    print(f"\n  ── Key word entropy ({surrogate.name}): ────────────")
                    for _kw in ['Emily', 'Hartley', 'MR-9149760',
                                '1980-03-15', 'presents', 'ratio', 'COPD.']:
                        if _kw in _word_list:
                            _kidx   = _word_list.index(_kw)
                            _kscore = _scores[_kidx][0]
                            print(f"    {_kscore:.4f}  {_kw!r}")

                # ── Ensemble average entropy map ──────────────────────────────
                if entropy_maps:
                    _ent_maps_filtered = {
                        k: v for k, v in entropy_maps.items()
                        if k != "llama3_2"
                    }
                    _maps_to_avg = _ent_maps_filtered if _ent_maps_filtered \
                                   else entropy_maps
                    _excluded = [k for k in entropy_maps if k not in _maps_to_avg]
                    print(f"\n  [entropy] ensemble average from: "
                          f"{list(_maps_to_avg.keys())}")
                    if _excluded:
                        print(f"  [entropy] excluded from ensemble: {_excluded} "
                              f"(architectural outlier — collapses on blank image)")
                    ent_avg      = torch.stack(list(_maps_to_avg.values())).mean(dim=0)
                    ent_avg_norm = _normalize_01(ent_avg)

                    _words = word_strings if word_strings else sample.transcript.split()
                    _n     = min(len(_words), len(word_boxes))
                    _avg_scores = []
                    for i in range(_n):
                        x0, y0, x1, y1 = (int(v) for v in word_boxes[i])
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(W, x1), min(H, y1)
                        region = ent_avg_norm[y0:y1, x0:x1]
                        score  = float(region.mean()) if region.numel() > 0 else 0.0
                        _avg_scores.append((score, _words[i]))

                    _avg_sorted = sorted(_avg_scores, key=lambda x: x[0], reverse=True)
                    print(f"\n  ── Top 10 by entropy (ENSEMBLE AVERAGE): ──────────")
                    for score, word in _avg_sorted[:10]:
                        print(f"    {score:.4f}  {word!r}")
                    print(f"\n  ── Bottom 10 by entropy (ENSEMBLE AVERAGE): ───────")
                    for score, word in _avg_sorted[-10:]:
                        print(f"    {score:.4f}  {word!r}")

                    if not args.entropy_only:
                        # Cross-compare ensemble entropy top-10 vs importance top-10
                        imp_sorted = sorted(
                            [(float(components["importance"][
                                max(0, int(word_boxes[i][1])):min(H, int(word_boxes[i][3])),
                                max(0, int(word_boxes[i][0])):min(W, int(word_boxes[i][2]))
                             ].mean()), _words[i])
                             for i in range(_n)
                             if int(word_boxes[i][2]) > int(word_boxes[i][0])
                             and int(word_boxes[i][3]) > int(word_boxes[i][1])],
                            key=lambda x: x[0], reverse=True,
                        )
                        imp_top10 = {w for _, w in imp_sorted[:10]}
                        ent_top10 = {w for _, w in _avg_sorted[:10]}
                        cap_words = {w for w in (imp_top10 | ent_top10) if w[0].isupper()}

                        if cap_words:
                            print(f"\n  ── Capitalised token cross-comparison "
                                  f"(importance vs entropy): ──")
                            for w in sorted(cap_words):
                                in_imp = w in imp_top10
                                in_ent = w in ent_top10
                                tag = "[both]    " if in_imp and in_ent \
                                      else "[ent_only] " if in_ent \
                                      else "[imp_only] "
                                print(f"    {tag}  {w!r}")

                    # Save visualisation — original + per-model + ensemble
                    n_models  = len(entropy_maps)
                    n_panels  = 2 + n_models   # original + per-model + ensemble
                    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

                    img_np = sample.image_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    axes[0].imshow(img_np)
                    axes[0].set_title("Original", fontsize=9)
                    axes[0].axis("off")

                    for ax, (name, emap) in zip(axes[1:], entropy_maps.items()):
                        im = ax.imshow(emap.numpy(), cmap="plasma", vmin=0, vmax=1)
                        ax.set_title(f"Entropy\n{name}", fontsize=9)
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    ax_last = axes[-1]
                    im = ax_last.imshow(ent_avg_norm.numpy(), cmap="plasma",
                                        vmin=0, vmax=1)
                    ax_last.set_title("Entropy\n(ensemble avg)", fontsize=9)
                    ax_last.axis("off")
                    fig.colorbar(im, ax=ax_last, fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    ent_out = out_path.parent / f"entropy_{sample.image_id}.png"
                    fig.savefig(ent_out, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    # ── Entropy × KL product validation ──────────────────────
                    # Rationale: entropy alone cannot separate structural labels
                    # (ID, PATIENT, PHYSICIAN) from PII values (MR-9149760,
                    # Emily, 1980-03-15) because both groups score high entropy
                    # for different reasons:
                    #   - Labels score high because they appear early in the
                    #     autoregressive sequence before context accumulates
                    #   - Values score high because any specific value could
                    #     go there without visual evidence
                    #
                    # KL (visual dependency) breaks this tie:
                    #   - Labels have LOW KL — masking label pixels doesn't
                    #     change the model's prediction (structure is inferred
                    #     from document type prior)
                    #   - Values have HIGH KL — masking value pixels causes
                    #     genuine prediction change
                    #
                    # The product entropy × KL suppresses labels (high entropy,
                    # low KL → moderate product) and amplifies values (high
                    # entropy, high KL → high product).
                    #
                    # This is the formulation for Task 4 / build_importance_map.

                    if not args.entropy_only and "kl" in components:
                        kl_map   = components["kl"]              # (H, W) normalised
                        ent_x_kl = _normalize_01(ent_avg_norm * kl_map)

                        _words  = word_strings if word_strings \
                                  else sample.transcript.split()
                        _n      = min(len(_words), len(word_boxes))

                        _exk_scores = []
                        for i in range(_n):
                            x0, y0, x1, y1 = (int(v) for v in word_boxes[i])
                            x0, y0 = max(0, x0), max(0, y0)
                            x1, y1 = min(W, x1), min(H, y1)
                            region = ent_x_kl[y0:y1, x0:x1]
                            score  = float(region.mean()) \
                                     if region.numel() > 0 else 0.0
                            _exk_scores.append((score, _words[i]))

                        _exk_sorted = sorted(
                            _exk_scores, key=lambda x: x[0], reverse=True
                        )

                        print("\n  ── Top 10 by entropy × KL (product): ──────────")
                        for score, word in _exk_sorted[:10]:
                            print(f"    {score:.4f}  {word!r}")

                        print("\n  ── Bottom 10 by entropy × KL (product): ───────")
                        for score, word in _exk_sorted[-10:]:
                            print(f"    {score:.4f}  {word!r}")

                        _exk_word_map = {w: s for s, w in _exk_scores}
                        print("\n  ── Key words entropy × KL: ─────────────────────")
                        for _kw in ['Emily', 'Hartley', 'MR-9149760',
                                    '1980-03-15', '2023-04-17',
                                    'presents', 'ratio', 'with', 'of',
                                    'ID', 'PATIENT', 'PHYSICIAN',
                                    'COPD.', '(J44.9)', 'Salmeterol/Xinafoate']:
                            if _kw in _exk_word_map:
                                print(f"    {_exk_word_map[_kw]:.4f}  {_kw!r}")

                        # Cross-compare product top-10 vs existing importance top-10
                        _imp_sorted = sorted(
                            [(float(components["importance"][
                                max(0, int(word_boxes[i][1])):
                                min(H, int(word_boxes[i][3])),
                                max(0, int(word_boxes[i][0])):
                                min(W, int(word_boxes[i][2]))
                             ].mean()), _words[i])
                             for i in range(_n)
                             if int(word_boxes[i][2]) > int(word_boxes[i][0])
                             and int(word_boxes[i][3]) > int(word_boxes[i][1])],
                            key=lambda x: x[0], reverse=True,
                        )
                        _imp_top10 = {w for _, w in _imp_sorted[:10]}
                        _exk_top10 = {w for _, w in _exk_sorted[:10]}
                        _cap = {w for w in (_imp_top10 | _exk_top10)
                                if w[0].isupper()}

                        if _cap:
                            print("\n  ── Capitalised token cross-comparison "
                                  "(importance vs entropy×KL): ───")
                            for w in sorted(_cap):
                                in_imp = w in _imp_top10
                                in_exk = w in _exk_top10
                                tag = "[both]     " if in_imp and in_exk \
                                      else "[exk_only]  " if in_exk \
                                      else "[imp_only]  "
                                print(f"    {tag}  {w!r}")
                    else:
                        print("\n  [entropy×KL] skipped — KL component not "
                              "available in components dict. Run without "
                              "--no_kl to enable.")
                    # ── End entropy × KL validation ───────────────────────────

                    print(f"\n  Saved entropy panels → {ent_out}")

            total_processed += 1

    # Restore any CPU-offloaded eager models
    for s in eager_sal:
        s.model.to(s.device)

    print(f"\n{'=' * 65}")
    print(f"Done.  Processed: {total_processed}   Skipped: {skipped}")
    print("=" * 65)


if __name__ == "__main__":
    main()
