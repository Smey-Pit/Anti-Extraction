"""
scripts/test_token_logprobs.py

Smoke test for token_logprobs() on all surrogate wrappers that implement it.

For each model:
  - Load the model
  - Load the first sample from the banking category
  - Call token_logprobs(image_tensor, transcript)
  - Assert shape/dtype/value invariants
  - Print T, mean/min log_prob, first 5 decoded tokens
  - Unload the model and clear CUDA cache

Usage (from project root):
    uv run python scripts/test_token_logprobs.py --config configs/attack.yaml
    uv run python scripts/test_token_logprobs.py --config configs/attack.yaml \\
        --models paligemma2 internvl3_5   # test a subset
"""

from __future__ import annotations

import argparse
import gc
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
    "paligemma2":  ("vlm_suppress.models.paligemma2",  "PaliGemma2"),
    "internvl3_5": ("vlm_suppress.models.internvl3_5", "InternVL35"),
    "qwen2_5vl":   ("vlm_suppress.models.qwen2_5vl",   "Qwen2_5VL"),
    "llava16":     ("vlm_suppress.models.llava",        "LLaVA16"),
    "llama3_2":    ("vlm_suppress.models.llama3_2",     "LlamaVision"),
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


def _find_surrogate_cfg(cfg: ExperimentConfig, name: str):
    for s_cfg in cfg.surrogates:
        if s_cfg.name == name:
            return s_cfg
    return None


def _load_model(name: str, s_cfg):
    import importlib
    mod_path, cls_name = _MODEL_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(s_cfg)


def _run_one(model, image_tensor: torch.Tensor, transcript: str, name: str) -> bool:
    print(f"\n{'─'*55}")
    print(f"  Model: {name}")
    print(f"  Transcript (first 80 chars): {transcript[:80]!r}")

    try:
        log_probs, tok_ids = model.token_logprobs(image_tensor, transcript)
    except NotImplementedError:
        print(f"  SKIP — token_logprobs not implemented for {name}")
        return True
    except Exception as e:
        print(f"  FAIL — exception during token_logprobs:")
        traceback.print_exc()
        return False

    T = log_probs.shape[0]

    # ── Invariant checks ──────────────────────────────────────────────────────
    errors = []

    if T == 0:
        errors.append("T == 0 (no transcript tokens)")

    if log_probs.dtype != torch.float32:
        errors.append(f"log_probs.dtype={log_probs.dtype}, expected float32")

    if tok_ids.dtype not in (torch.int64, torch.long):
        errors.append(f"tok_ids.dtype={tok_ids.dtype}, expected int64")

    if log_probs.shape != (T,):
        errors.append(f"log_probs.shape={log_probs.shape}, expected ({T},)")

    if tok_ids.shape != (T,):
        errors.append(f"tok_ids.shape={tok_ids.shape}, expected ({T},)")

    if T > 0 and float(log_probs.max()) > 0.0:
        errors.append(f"max log_prob={float(log_probs.max()):.4f} > 0 (not log-probabilities)")

    if T > 0 and float(log_probs.min()) < -100.0:
        errors.append(f"min log_prob={float(log_probs.min()):.4f} < -100 (suspiciously low)")

    dev = model.device
    if log_probs.device.type != dev.type:
        errors.append(f"log_probs on {log_probs.device}, expected {dev}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"  T (transcript tokens): {T}")
    print(f"  mean log_prob: {float(log_probs.mean()):.4f}")
    print(f"  min  log_prob: {float(log_probs.min()):.4f}")
    print(f"  max  log_prob: {float(log_probs.max()):.4f}")

    # Decode first 5 tokens — use whatever tokenizer is available
    tok = getattr(model, "tokenizer", None) or getattr(
        getattr(model, "processor", None), "tokenizer", None
    )
    if tok is not None:
        first5 = tok.convert_ids_to_tokens(tok_ids[:5].tolist())
        first5_lp = [f"{float(log_probs[i]):.3f}" for i in range(min(5, T))]
        pairs = [f"{t!r}({lp})" for t, lp in zip(first5, first5_lp)]
        print(f"  First 5 tokens: {', '.join(pairs)}")

    print("  PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: token_logprobs for all implemented surrogate wrappers."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/attack.yaml"),
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(_MODEL_REGISTRY.keys()),
        default=list(_MODEL_REGISTRY.keys()),
        help="Which models to test (default: all)",
    )
    parser.add_argument(
        "--category", default="banking",
        help="Dataset category filter for sample selection (default: banking)",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    # ── Load one sample ───────────────────────────────────────────────────────
    print(f"\nLoading dataset (first sample, category={args.category!r}) ...")
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        category_filter     = args.category,
    )
    if len(dataset) == 0:
        print(f"Dataset empty with category_filter={args.category!r}. Retrying without filter.")
        dataset = TextImageDataset(
            data_dir            = cfg.data.data_dir,
            data_dir_additional = cfg.data.data_dir_additional,
            image_size          = cfg.data.image_size,
            max_samples         = 1,
        )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — check config data_dir.")

    sample     = dataset[0]
    image_tensor = sample.image_tensor
    transcript   = sample.transcript

    print(f"Sample:      {sample.image_id}")
    print(f"Image shape: {tuple(image_tensor.shape)}")
    print(f"Transcript:  {transcript[:100]!r}{'...' if len(transcript) > 100 else ''}")

    # ── Test each model ───────────────────────────────────────────────────────
    results: dict[str, bool] = {}

    for name in args.models:
        s_cfg = _find_surrogate_cfg(cfg, name)
        if s_cfg is None:
            print(f"\n  SKIP {name} — not found in config surrogates list")
            results[name] = True
            continue

        # Assign device
        n_gpus = torch.cuda.device_count()
        s_cfg.device = "cuda:0" if n_gpus > 0 else "cpu"

        print(f"\nLoading {name} → {s_cfg.device} ...")
        try:
            model = _load_model(name, s_cfg)
        except Exception:
            print(f"  FAIL — could not load {name}:")
            traceback.print_exc()
            results[name] = False
            continue

        ok = _run_one(model, image_tensor.to(model.device), transcript, name)
        results[name] = ok

        # Unload
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Summary:")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<15} {status}")
        if not ok:
            all_pass = False
    print("=" * 55)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
