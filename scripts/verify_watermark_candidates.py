"""
scripts/verify_watermark_candidates.py

Step 2 verification — surrogate candidate aggregator.
Requires GPU (A100/V100 on Spartan).

Loads the first banking sample, then queries every config surrogate that is
in the model registry (skipping any that do not support return_top_k) for
top-K alternative tokens at the first-token position of each target word.
Prints the top-3 candidates per word.

Usage (from project root):
    uv run python scripts/verify_watermark_candidates.py --config configs/attack.yaml
    uv run python scripts/verify_watermark_candidates.py --config configs/attack.yaml \
        --top-k 30 --n-words 5 --require-cap
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
from vlm_suppress.watermark.candidate import aggregate_surrogate_candidates

_MODEL_REGISTRY = {
    "qwen2_5vl":   ("vlm_suppress.models.qwen2_5vl",   "Qwen2_5VL"),
    "paligemma2":  ("vlm_suppress.models.paligemma2",   "PaliGemma2"),
    "internvl3_5": ("vlm_suppress.models.internvl3_5",  "InternVL35"),
    "llama3_2":    ("vlm_suppress.models.llama3_2",     "LlamaVision"),
}

# Common structural/layout words to exclude from target selection
_SKIP = {
    "the", "a", "an", "of", "in", "at", "on", "to", "by", "for",
    "dr", "mr", "ms", "mrs", "and", "or", "not",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify surrogate candidate aggregator (Step 2)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument("--top-k", type=int, default=20,
                        help="Vocab alternatives to request per token position")
    parser.add_argument("--n-candidates", type=int, default=3,
                        help="Candidates to print per word")
    parser.add_argument("--n-words", type=int, default=4,
                        help="Number of target words to query")
    parser.add_argument("--require-cap", action="store_true",
                        help="Only accept candidates with an initial capital letter "
                             "(useful for named-entity / person-name targets)")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    # Determine which config surrogates are available in the registry
    available = [s for s in cfg.surrogates if s.name in _MODEL_REGISTRY]
    if not available:
        sys.exit(
            "ERROR: none of the config surrogates are in the model registry.\n"
            f"Config has: {[s.name for s in cfg.surrogates]}\n"
            f"Registry has: {list(_MODEL_REGISTRY)}"
        )

    print("Loading dataset (banking category) …")
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        category_filter     = "banking",
    )
    if len(dataset) == 0:
        print("No banking samples — retrying without category filter.")
        dataset = TextImageDataset(
            data_dir            = cfg.data.data_dir,
            data_dir_additional = cfg.data.data_dir_additional,
            image_size          = cfg.data.image_size,
            max_samples         = 1,
        )
    if len(dataset) == 0:
        sys.exit("ERROR: dataset is empty.")

    sample       = dataset[0]
    transcript   = sample.transcript
    word_strings = sample.word_strings or transcript.split()

    print(f"Sample     : {sample.image_id}")
    print(f"Transcript : {transcript[:120]!r}{'...' if len(transcript) > 120 else ''}")
    print(f"Words      : {len(word_strings)}")

    # Target word heuristic:
    #   - pure alpha (no punctuation like "Period:", colons, dashes)
    #   - initial capital
    #   - length >= 4 (skips month abbreviations "Oct", "Jan" and short tokens)
    #   - not in structural skip list
    target_indices = [
        i for i, w in enumerate(word_strings)
        if w.isalpha()
        and w[0].isupper()
        and len(w) >= 4
        and w.lower() not in _SKIP
    ][:args.n_words]

    if not target_indices:
        sys.exit("ERROR: no suitable target words found in sample.")

    print(f"\nTarget words ({len(target_indices)}):")
    for i in target_indices:
        print(f"  [{i:3d}] {word_strings[i]!r}")

    n_gpus = torch.cuda.device_count()
    device = "cuda:0" if n_gpus > 0 else "cpu"
    if n_gpus == 0:
        print("\nWARNING: no CUDA GPUs detected — running on CPU (will be slow).")

    # Load each available surrogate, query, accumulate scores across all
    # models that support return_top_k.
    loaded_surrogates = []
    loaded_alpha      = []

    for s_cfg in available:
        s_cfg.device = device
        print(f"\nLoading {s_cfg.name} on {device} …")
        mod_path, cls_name = _MODEL_REGISTRY[s_cfg.name]
        try:
            mod   = importlib.import_module(mod_path)
            model = getattr(mod, cls_name)(s_cfg)
            loaded_surrogates.append(model)
            loaded_alpha.append(1.0)        # raw weights; aggregator normalises
        except Exception:
            print(f"  WARN: could not load {s_cfg.name} — skipping.")
            traceback.print_exc()

    if not loaded_surrogates:
        sys.exit("ERROR: no surrogates loaded successfully.")

    img = sample.image_tensor.to(device)

    print(f"\nQuerying top-{args.top_k} alternatives per token position …")
    try:
        results = aggregate_surrogate_candidates(
            image_tensor        = img,
            transcript          = transcript,
            word_strings        = word_strings,
            target_word_indices = target_indices,
            surrogates          = loaded_surrogates,
            alpha_weights       = [w / sum(loaded_alpha) for w in loaded_alpha],
            top_k               = args.top_k,
            n_candidates        = args.n_candidates,
            require_initial_cap = args.require_cap,
        )
    except Exception:
        traceback.print_exc()
        sys.exit("ERROR: aggregate_surrogate_candidates failed.")
    finally:
        for m in loaded_surrogates:
            del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "─" * 60)
    print(f"{'Word':<20}  Candidates (descending score)")
    print("─" * 60)
    any_found = False
    for word, cands in results.items():
        if cands:
            cand_str  = "  |  ".join(f"{c!r} ({s:.3f})" for c, s in cands)
            any_found = True
        else:
            cand_str = "(empty — all top-K tokens filtered; try --top-k 50 or drop --require-cap)"
        print(f"{word!r:<20}  {cand_str}")
    print("─" * 60)

    if not any_found:
        print("\nFAIL: no candidates returned for any word.")
        sys.exit(1)
    print("\nPASS: candidates returned. Inspect the names above for plausibility.")


if __name__ == "__main__":
    main()
