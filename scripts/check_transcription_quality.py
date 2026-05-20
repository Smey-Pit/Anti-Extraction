"""
scripts/check_transcription_quality.py

Transcription quality check for surrogate models on a clean sample.
Loads each model, runs transcribe(), and prints the result alongside
the ground-truth transcript for side-by-side comparison.

Usage:
    uv run python scripts/check_transcription_quality.py --config configs/attack.yaml

Optional:
    --models   comma-separated list of model names (default: all in config)
    --category category filter for dataset (default: banking)
    --device   override device (default: cuda:0)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import traceback
from pathlib import Path
from typing import Optional

import dacite
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.config import (
    Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
)
from vlm_suppress.data.dataset import TextImageDataset

_MODEL_REGISTRY = {
    "qwen2_5vl":   ("vlm_suppress.models.qwen2_5vl",   "Qwen2_5VL"),
    "paligemma2":  ("vlm_suppress.models.paligemma2",   "PaliGemma2"),
    "internvl3_5": ("vlm_suppress.models.internvl3_5",  "InternVL35"),
    "llama3_2":    ("vlm_suppress.models.llama3_2",     "LlamaVision"),
    "llava1_6":    ("vlm_suppress.models.llava",        "LLaVA16"),
}


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


def _cer(ref: str, hyp: str) -> float:
    """Character error rate (edit distance / len(ref))."""
    r, h = list(ref), list(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument("--models",   type=str,  default=None,
                        help="Comma-separated model names (default: all in config)")
    parser.add_argument("--category", type=str,  default="banking")
    parser.add_argument("--device",   type=str,  default=None)
    args = parser.parse_args()

    cfg    = _load_cfg(args.config)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        category_filter     = args.category,
    )
    if len(dataset) == 0:
        dataset = TextImageDataset(
            data_dir            = cfg.data.data_dir,
            data_dir_additional = cfg.data.data_dir_additional,
            image_size          = cfg.data.image_size,
            max_samples         = 1,
        )
    if len(dataset) == 0:
        sys.exit("ERROR: dataset is empty.")

    sample   = dataset[0]
    gt       = sample.transcript
    img      = sample.image_tensor.to(device)

    print(f"Sample        : {sample.image_id}")
    print(f"Ground truth  : {gt[:120].replace(chr(10), ' ')!r}{'...' if len(gt) > 120 else ''}")
    print(f"GT length     : {len(gt)} chars")
    print()

    # ── Model selection ───────────────────────────────────────────────────────
    if args.models:
        names = [n.strip() for n in args.models.split(",")]
        surrogates = [s for s in cfg.surrogates if s.name in names]
        missing = [n for n in names if not any(s.name == n for s in cfg.surrogates)]
        if missing:
            print(f"WARNING: {missing} not in config surrogates — skipping.")
    else:
        surrogates = [s for s in cfg.surrogates if s.name in _MODEL_REGISTRY]

    results: list[dict] = []

    for s_cfg in surrogates:
        if s_cfg.name not in _MODEL_REGISTRY:
            print(f"{'─'*60}")
            print(f"SKIP  {s_cfg.name}  (not in registry)")
            continue

        print(f"{'─'*60}")
        print(f"Loading {s_cfg.name} …")
        s_cfg.device = device

        mod_path, cls_name = _MODEL_REGISTRY[s_cfg.name]
        try:
            mod   = importlib.import_module(mod_path)
            model = getattr(mod, cls_name)(s_cfg)
        except Exception:
            traceback.print_exc()
            print(f"FAIL  {s_cfg.name}  (could not load)")
            continue

        try:
            with torch.no_grad():
                hyp = model.transcribe(img)
        except Exception:
            traceback.print_exc()
            hyp = ""
            print(f"FAIL  {s_cfg.name}  (transcribe() raised)")

        cer = _cer(gt, hyp)
        src_present = "Thompson" in hyp
        results.append({"name": s_cfg.name, "cer": cer, "thompson": src_present, "hyp": hyp})

        print(f"CER           : {cer:.3f}")
        print(f"'Thompson' in output: {src_present}")
        print(f"Output        :")
        for line in hyp.splitlines():
            print(f"  {line}")
        print()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"{'Model':<15} {'CER':>6}  {'Thompson?':>10}  Quality")
    print("-" * 60)
    for r in results:
        quality = "good" if r["cer"] < 0.10 else ("ok" if r["cer"] < 0.25 else "poor")
        print(f"{r['name']:<15} {r['cer']:>6.3f}  {str(r['thompson']):>10}  {quality}")


if __name__ == "__main__":
    main()
