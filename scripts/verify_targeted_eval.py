"""
scripts/verify_targeted_eval.py

Verify evaluate_targeted_substitution() on Spartan.

Two-part test:

  PART 1 — Logic check (no GPU needed, stub surrogate)
    Exercises all four outcome branches with canned transcripts.
    Runs on the login node.

  PART 2 — Real surrogate (GPU required)
    Loads banking_0000, runs the surrogate on:
      a) the clean image   → expected "clean_read"
      b) the ghost-watermarked image (α=0.12, no PGD yet)
         → likely still "clean_read"; confirms the eval plumbing works
         end-to-end before you attach the attack loop.

NOTE: "exact_sub" on the ghost-only image is not expected at this stage.
The ghost + PGD is what produces substitution.  This script only validates
that the eval function correctly classifies whatever the surrogate says.

Usage (from project root):
    # Part 1 only (login node):
    python scripts/verify_targeted_eval.py --no-model

    # Both parts (GPU node):
    uv run python scripts/verify_targeted_eval.py --config configs/attack.yaml
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vlm_suppress.eval.metrics import evaluate_targeted_substitution
except ImportError:
    # jiwer not installed (e.g. base conda on login node) — inline the function
    # so Part 1 (--no-model) still runs without the full project environment.
    def evaluate_targeted_substitution(image, source_word, target_word, surrogate):  # type: ignore[misc]
        transcript = surrogate.transcribe(image)
        t = transcript.lower()
        src = source_word.lower() in t
        tgt = target_word.lower() in t
        if src and not tgt:
            outcome = "clean_read"
        elif tgt and not src:
            outcome = "exact_sub"
        elif src and tgt:
            outcome = "hallucination"
        else:
            outcome = "garbled"
        return {"transcript": transcript, "source_present": src,
                "target_present": tgt, "outcome": outcome}

SOURCE = "Thompson"
TARGET = "Henderson"

_MODEL_REGISTRY = {
    "qwen2_5vl":   ("vlm_suppress.models.qwen2_5vl",   "Qwen2_5VL"),
    "paligemma2":  ("vlm_suppress.models.paligemma2",   "PaliGemma2"),
    "internvl3_5": ("vlm_suppress.models.internvl3_5",  "InternVL35"),
    "llama3_2":    ("vlm_suppress.models.llama3_2",     "LlamaVision"),
}


# ── Part 1: stub surrogate logic check ────────────────────────────────────────

class _StubSurrogate:
    """Surrogate that returns a fixed transcript. No model needed."""
    def __init__(self, text: str):
        self._text = text
    def transcribe(self, image) -> str:
        return self._text


def run_logic_check() -> bool:
    cases = [
        (f"Account holder: Ella {SOURCE}",         "clean_read"),
        (f"Account holder: Ella {TARGET}",          "exact_sub"),
        (f"{SOURCE} and {TARGET} both detected",    "hallucination"),
        ("Balance: $4,821.33  Date: 01/10/2024",    "garbled"),
        # case-insensitive
        (f"holder: {SOURCE.upper()}  ref: 001",     "clean_read"),
        (f"{TARGET.lower()} — primary account",     "exact_sub"),
    ]

    print("PART 1 — Logic check (stub surrogate, no GPU)")
    print("─" * 60)
    all_pass = True
    for text, expected in cases:
        r = evaluate_targeted_substitution(None, SOURCE, TARGET, _StubSurrogate(text))
        ok = r["outcome"] == expected
        if not ok:
            all_pass = False
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {r['outcome']:<15}  {text[:50]!r}")

    print("─" * 60)
    print(f"  Result: {'all passed' if all_pass else 'FAILURES above'}\n")
    return all_pass


# ── Part 2: real surrogate ─────────────────────────────────────────────────────

def _pil_to_tensor(pil_img):
    import torch
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


def _load_cfg(config: Path):
    import dacite, yaml
    from vlm_suppress.config import (
        Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
    )
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


def run_real_surrogate(config: Path) -> bool:
    import torch
    from PIL import Image
    from vlm_suppress.data.dataset import TextImageDataset
    from vlm_suppress.watermark.renderer import render_ghost_watermark

    cfg = _load_cfg(config)

    # ── Load sample ────────────────────────────────────────────────────────────
    print("PART 2 — Real surrogate (GPU required)")
    print("─" * 60)
    print("Loading dataset (banking) …")
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        category_filter     = "banking",
    )
    if len(dataset) == 0:
        dataset = TextImageDataset(
            data_dir            = cfg.data.data_dir,
            data_dir_additional = cfg.data.data_dir_additional,
            image_size          = cfg.data.image_size,
            max_samples         = 1,
        )
    if len(dataset) == 0:
        print("ERROR: dataset is empty.")
        return False

    sample = dataset[0]
    print(f"Sample: {sample.image_id}")

    # Find SOURCE word bounding box from labels
    labels_path = Path(cfg.data.data_dir) / "labels_pil.json"
    with labels_path.open() as f:
        labels = json.load(f)
    label = next((s for s in labels if s["image_id"] == sample.image_id), None)
    if label is None:
        print(f"WARNING: {sample.image_id} not in labels_pil.json — skipping ghost render.")
        source_box = None
    else:
        flat = [(w["word"], w["box"]) for line in label["word_boxes"] for w in line]
        match = next(((w, b) for w, b in flat if w == SOURCE), None)
        source_box = match[1] if match else None
        if source_box is None:
            print(f"WARNING: '{SOURCE}' not found in {sample.image_id} word_boxes.")

    # ── Ghost-watermark image ─────────────────────────────────────────────────
    clean_pil   = sample.image
    font_family = label["layout"].get("font_family", "sans") if label else "sans"

    if source_box is not None:
        ghost_pil, rec = render_ghost_watermark(
            clean_pil, SOURCE, source_box, TARGET,
            alpha=0.12, font_family=font_family,
        )
        print(f"Ghost watermark: '{SOURCE}' → '{TARGET}'  "
              f"box={[round(v) for v in source_box]}  "
              f"font_size={rec.font_size_px}px  α=0.12")
    else:
        ghost_pil = clean_pil
        print("Ghost watermark: skipped (box not found) — using clean image as stand-in.")

    # ── Pick first available surrogate ────────────────────────────────────────
    available = [s for s in cfg.surrogates if s.name in _MODEL_REGISTRY]
    if not available:
        print(f"ERROR: no config surrogates in registry {list(_MODEL_REGISTRY)}.")
        return False

    n_gpus  = torch.cuda.device_count()
    device  = "cuda:0" if n_gpus > 0 else "cpu"
    s_cfg   = available[0]
    s_cfg.device = device
    if n_gpus == 0:
        print("WARNING: no GPU detected — running on CPU (slow).")

    print(f"Loading {s_cfg.name} on {device} …")
    mod_path, cls_name = _MODEL_REGISTRY[s_cfg.name]
    try:
        mod   = importlib.import_module(mod_path)
        model = getattr(mod, cls_name)(s_cfg)
    except Exception:
        traceback.print_exc()
        return False

    # ── Convert PIL → tensor and evaluate ─────────────────────────────────────
    clean_tensor = sample.image_tensor.to(device)
    ghost_tensor = _pil_to_tensor(ghost_pil).to(device)

    print(f"\nEvaluating …  (source='{SOURCE}'  target='{TARGET}')")
    print("─" * 60)

    all_pass = True
    for label_str, tensor, expected in [
        ("clean image ",  clean_tensor, "clean_read"),
        ("ghost image ", ghost_tensor,  None),          # no fixed expectation pre-PGD
    ]:
        r = evaluate_targeted_substitution(tensor, SOURCE, TARGET, model)
        short = r["transcript"][:80].replace("\n", " ")
        outcome_ok = (expected is None) or (r["outcome"] == expected)
        if not outcome_ok:
            all_pass = False
        mark = ("PASS" if outcome_ok else "FAIL") if expected else "INFO"
        print(f"  [{mark}]  {label_str}  outcome={r['outcome']:<15}  "
              f"src={r['source_present']}  tgt={r['target_present']}")
        print(f"          transcript: {short!r}")
        print()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("─" * 60)
    note = ("NOTE: ghost-only 'clean_read' is expected — "
            "substitution requires ghost + PGD.")
    print(note)
    return all_pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument("--no-model", action="store_true",
                        help="Run Part 1 only (no GPU, no model loading)")
    args = parser.parse_args()

    p1 = run_logic_check()

    if args.no_model:
        sys.exit(0 if p1 else 1)

    p2 = run_real_surrogate(args.config)
    sys.exit(0 if (p1 and p2) else 1)


if __name__ == "__main__":
    main()
