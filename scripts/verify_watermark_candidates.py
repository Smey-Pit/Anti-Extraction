"""
scripts/verify_watermark_candidates.py

Step 2 verification — surrogate candidate aggregator.
Requires GPU (A100/V100 on Spartan).

Loads the first banking sample, then queries qwen2_5vl for top-K alternative
tokens at the first-token position of each of the top-N capitalised words.
Prints the top-3 candidates per word so you can visually confirm the model
is returning plausible name substitutions.

Usage (from project root):
    uv run python scripts/verify_watermark_candidates.py --config configs/attack.yaml
    uv run python scripts/verify_watermark_candidates.py --config configs/attack.yaml \
        --model qwen2_5vl --top-k 30 --n-words 5
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
    parser.add_argument("--model", default="qwen2_5vl",
                        choices=list(_MODEL_REGISTRY.keys()),
                        help="Surrogate to use (must support return_top_k)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of vocab alternatives to request per token position")
    parser.add_argument("--n-candidates", type=int, default=3,
                        help="Number of candidates to print per word")
    parser.add_argument("--n-words", type=int, default=4,
                        help="Number of capitalised target words to query")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    # Override to banking so we can compare against the renderer verification
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

    sample      = dataset[0]
    transcript  = sample.transcript
    word_strings = sample.word_strings or transcript.split()

    print(f"Sample     : {sample.image_id}")
    print(f"Transcript : {transcript[:120]!r}{'...' if len(transcript) > 120 else ''}")
    print(f"Words      : {len(word_strings)}")

    # Pick target words: first N capitalised, non-trivial words
    SKIP = {"the", "a", "an", "of", "in", "at", "on", "to", "dr", "mr", "ms", "mrs"}
    target_indices = [
        i for i, w in enumerate(word_strings)
        if len(w) > 2 and w[0].isupper() and w.strip(".,").lower() not in SKIP
    ][:args.n_words]

    print(f"\nTarget words ({len(target_indices)}):")
    for i in target_indices:
        print(f"  [{i:3d}] {word_strings[i]!r}")

    # Load surrogate
    s_cfg = next((s for s in cfg.surrogates if s.name == args.model), None)
    if s_cfg is None:
        sys.exit(f"ERROR: '{args.model}' not found in config surrogates list.")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("\nWARNING: no CUDA GPUs detected — running on CPU (will be slow).")
    s_cfg.device = f"cuda:0" if n_gpus > 0 else "cpu"

    print(f"\nLoading {args.model} on {s_cfg.device} …")
    mod_path, cls_name = _MODEL_REGISTRY[args.model]
    try:
        mod   = importlib.import_module(mod_path)
        model = getattr(mod, cls_name)(s_cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(f"ERROR: could not load {args.model}.")

    img = sample.image_tensor.to(model.device)

    print(f"\nQuerying top-{args.top_k} alternatives per token position …")
    try:
        results = aggregate_surrogate_candidates(
            image_tensor        = img,
            transcript          = transcript,
            word_strings        = word_strings,
            target_word_indices = target_indices,
            surrogates          = [model],
            top_k               = args.top_k,
            n_candidates        = args.n_candidates,
        )
    except Exception:
        traceback.print_exc()
        sys.exit("ERROR: aggregate_surrogate_candidates failed.")

    print("\n" + "─" * 60)
    print(f"{'Word':<20}  Candidates (descending score)")
    print("─" * 60)
    for word, cands in results.items():
        if cands:
            cand_str = "  |  ".join(f"{c!r} ({s:.3f})" for c, s in cands)
        else:
            cand_str = "(no candidates — model may not support return_top_k)"
        print(f"{word!r:<20}  {cand_str}")
    print("─" * 60)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not any(results.values()):
        print("\nFAIL: no candidates returned for any word.")
        sys.exit(1)
    print("\nPASS: candidates returned. Inspect the names above for plausibility.")


if __name__ == "__main__":
    main()
