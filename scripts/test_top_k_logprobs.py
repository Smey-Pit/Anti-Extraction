"""
scripts/test_top_k_logprobs.py

Smoke test for the return_top_k extension to token_logprobs().

For each surrogate in the active ensemble (load → test → del → empty_cache):

  STEP 1 — Backward compatibility
    Call with no third argument. Assert 2-tuple, correct shapes/dtypes/range.

  STEP 2 — Top-K contract
    Call with return_top_k=10. Assert 4-tuple, (T,10) shapes, sorted descending,
    and that gathered log-prob ≤ top-1 log-prob (soft check).

  STEP 3 — Human-readable alternatives
    Find the first capitalised token (prefers "Emily"). Decode top-10 alternatives
    with log probs. Visual sanity check only — no assertion.

Usage (from project root):
    uv run python scripts/test_top_k_logprobs.py --config configs/attack.yaml
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
    "paligemma2":  ("vlm_suppress.models.paligemma2",  "PaliGemma2"),
    "internvl3_5": ("vlm_suppress.models.internvl3_5", "InternVL35"),
    "qwen2_5vl":   ("vlm_suppress.models.qwen2_5vl",   "Qwen2_5VL"),
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


def _load_model(name: str, s_cfg):
    mod_path, cls_name = _MODEL_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(s_cfg)


def _get_tokenizer(model):
    if hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
        return model.processor.tokenizer
    if hasattr(model, "tokenizer"):
        return model.tokenizer
    return None


def _find_interesting_position(
    token_ids: torch.Tensor,   # (T,)
    tokenizer,
    transcript: str,
) -> tuple[int, str]:
    """
    Return (position, word) for the first token of "Emily" in the transcript,
    or the first capitalised token otherwise.
    """
    if tokenizer is None:
        return 0, "<unknown>"

    words = transcript.split()
    target_words = ["Emily"] + [w for w in words if w[0].isupper() and w != "Emily"]

    for target in target_words:
        try:
            target_ids = tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
            if len(target_ids) == 0:
                continue
            first_id = int(target_ids[0])
            matches = (token_ids == first_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                return int(matches[0]), target
        except Exception:
            continue

    return 0, words[0] if words else "<empty>"


def _run_one(model, image_tensor: torch.Tensor, transcript: str) -> bool:
    name = model.name
    print(f"\n{'─' * 60}")
    print(f"  Model: {name}")

    tokenizer = _get_tokenizer(model)
    ok = True

    # ── STEP 1: backward compatibility ───────────────────────────────────────
    try:
        result = model.token_logprobs(image_tensor, transcript)
    except Exception:
        print("  STEP 1 FAIL — exception:")
        traceback.print_exc()
        return False

    if not isinstance(result, tuple) or len(result) != 2:
        print(f"  STEP 1 FAIL — expected 2-tuple, got {type(result)} len={len(result) if isinstance(result, tuple) else 'n/a'}")
        return False

    log_probs, token_ids = result
    T = log_probs.shape[0]
    errors = []

    if T == 0:
        errors.append("T == 0")
    if log_probs.dtype != torch.float32:
        errors.append(f"log_probs.dtype={log_probs.dtype}, want float32")
    if token_ids.dtype not in (torch.int64, torch.long):
        errors.append(f"token_ids.dtype={token_ids.dtype}, want int64")
    if log_probs.shape != (T,):
        errors.append(f"log_probs.shape={log_probs.shape}")
    if token_ids.shape != (T,):
        errors.append(f"token_ids.shape={token_ids.shape}")
    if T > 0 and float(log_probs.max()) > 0.0:
        errors.append(f"max log_prob={float(log_probs.max()):.4f} > 0")
    if T > 0 and float(log_probs.min()) < -500.0:
        errors.append(f"min log_prob={float(log_probs.min()):.4f} < -500")

    if errors:
        for e in errors:
            print(f"  STEP 1 FAIL: {e}")
        ok = False
    else:
        print(f"  PASS: 2-tuple contract preserved for {name}  (T={T})")

    # ── STEP 2: top-K contract ────────────────────────────────────────────────
    try:
        result4 = model.token_logprobs(image_tensor, transcript, return_top_k=10)
    except Exception:
        print("  STEP 2 FAIL — exception:")
        traceback.print_exc()
        return False

    if not isinstance(result4, tuple) or len(result4) != 4:
        print(f"  STEP 2 FAIL — expected 4-tuple, got len={len(result4) if isinstance(result4, tuple) else 'n/a'}")
        return False

    lp2, ids2, top_k_lp, top_k_id = result4
    K = top_k_lp.shape[1] if top_k_lp.dim() == 2 else -1
    errors2 = []

    if top_k_lp.shape != (T, K):
        errors2.append(f"top_k_lp.shape={top_k_lp.shape}, want ({T}, {K})")
    if top_k_id.shape != (T, K):
        errors2.append(f"top_k_id.shape={top_k_id.shape}, want ({T}, {K})")
    if top_k_lp.dtype != torch.float32:
        errors2.append(f"top_k_lp.dtype={top_k_lp.dtype}, want float32")
    if top_k_id.dtype not in (torch.int64, torch.long):
        errors2.append(f"top_k_id.dtype={top_k_id.dtype}, want int64")
    if T > 0 and K > 0 and float(top_k_lp.max()) > 0.0:
        errors2.append(f"max top_k_lp={float(top_k_lp.max()):.4f} > 0")
    if T > 0 and K > 1:
        if not (top_k_lp[:, 0] >= top_k_lp[:, -1]).all():
            errors2.append("top_k_lp not sorted descending")
    if T > 0 and K > 0:
        if not (lp2 <= top_k_lp[:, 0] + 1e-4).all():
            worst = float((lp2 - top_k_lp[:, 0]).max())
            errors2.append(f"gathered lp > top-1 lp by up to {worst:.5f}")

    if errors2:
        for e in errors2:
            print(f"  STEP 2 FAIL: {e}")
        ok = False
    else:
        print(f"  PASS: 4-tuple contract verified for {name}  (T={T}, K={K})")

    # ── STEP 3: human-readable alternatives at correct token position ────────
    print("\n  Step 3 — token alignment debug + alternatives")

    words = transcript.split()

    # Print first 20 tokens so position can be verified visually
    enc = tokenizer(transcript, add_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(enc.input_ids)
    print("  First 20 tokens:")
    for i, tok in enumerate(token_strings[:20]):
        print(f"    {i:3d}  {tok!r}")

    # Find the first capitalised, non-structural, non-trivial word
    SKIP = {"the", "a", "an", "of", "in", "at", "on", "st", "dr", "mr", "ms", "mrs"}
    target_word = next(
        (w for w in words
         if len(w) > 2
         and w[0].isupper()
         and w.strip(".,").lower() not in SKIP),
        words[0],
    )
    word_idx = words.index(target_word)

    # Use _align_tokens_to_words for correct subword span lookup
    from vlm_suppress.attack.importance import _align_tokens_to_words
    spans = _align_tokens_to_words(tokenizer, transcript, len(words))
    span_s, span_e = spans[word_idx]

    print(f"\n  Target word: {target_word!r}")
    print(f"  Word index in transcript.split(): {word_idx}")
    print(f"  Token span: [{span_s}, {span_e})")
    print(f"  Tokens in span: "
          f"{[repr(token_strings[i]) for i in range(span_s, min(span_e, len(token_strings)))]}")

    # Get top-k at first token of the span
    t = span_s
    if t < top_k_lp.shape[0]:
        print(f"\n  Top-10 alternatives at span position {t} ({target_word!r}):")
        for rank in range(min(10, top_k_lp.shape[1])):
            tok_id  = int(top_k_id[t, rank].item())
            tok_str = tokenizer.decode([tok_id])
            lp      = float(top_k_lp[t, rank].item())
            marker  = " ◀" if rank == 0 else ""
            print(f"    {rank:2d}: {tok_str!r:<20} {lp:.4f}{marker}")
    else:
        print(f"  WARNING: span_s={t} is out of range "
              f"(T={top_k_lp.shape[0]}). "
              "Check _align_tokens_to_words output above.")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for token_logprobs return_top_k extension."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/attack.yaml"),
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(_MODEL_REGISTRY.keys()),
        default=list(_MODEL_REGISTRY.keys()),
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    # Load one medical sample
    print("\nLoading dataset (first medical sample) ...")
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        category_filter     = "medical",
    )
    if len(dataset) == 0:
        print("No medical samples found — retrying without category filter.")
        dataset = TextImageDataset(
            data_dir            = cfg.data.data_dir,
            data_dir_additional = cfg.data.data_dir_additional,
            image_size          = cfg.data.image_size,
            max_samples         = 1,
        )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    sample     = dataset[0]
    transcript = sample.transcript
    print(f"Sample:      {sample.image_id}")
    print(f"Transcript:  {transcript[:100]!r}{'...' if len(transcript) > 100 else ''}")

    results: dict[str, bool] = {}

    for name in args.models:
        s_cfg = next(
            (s for s in cfg.surrogates if s.name == name), None
        )
        if s_cfg is None:
            print(f"\nSKIP {name} — not in config surrogates list")
            results[name] = True
            continue

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

        img = sample.image_tensor.to(model.device)
        ok  = _run_one(model, img, transcript)
        results[name] = ok

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Summary:")
    all_pass = True
    for name, ok in results.items():
        print(f"  {name:<15} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
