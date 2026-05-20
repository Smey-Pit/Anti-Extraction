"""
scripts/verify_targeted_grad.py

Gradient-flow check for targeted_ce_loss on Qwen2.5-VL.

Loads banking_0000, renders the ghost watermark ("Thompson" → "Henderson"),
then asserts that δ.grad is non-zero after loss.backward().  Prints gradient
statistics for inspection.

Must PASS on Spartan before the PGD loop is implemented.

Usage (from project root, GPU node):
    uv run python scripts/verify_targeted_grad.py --config configs/attack.yaml

Optional flags:
    --alpha   float   ghost watermark opacity (default 0.12)
    --device  str     override device (default: cuda:0 if available)
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
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import dacite
import yaml

from vlm_suppress.config import (
    Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
)
from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.watermark.renderer import render_ghost_watermark

SOURCE = "Thompson"
TARGET = "Henderson"

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


def _pil_to_tensor(pil_img) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument("--alpha", type=float, default=0.12,
                        help="Ghost watermark opacity")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (default: cuda:0 if available)")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    # ── Pick surrogate: prefer qwen2_5vl, fall back to first available ────────
    available = [s for s in cfg.surrogates if s.name in _MODEL_REGISTRY]
    preferred = [s for s in available if s.name == "qwen2_5vl"]
    s_cfg     = (preferred or available)[0] if (preferred or available) else None

    if s_cfg is None:
        sys.exit(
            f"ERROR: no config surrogate in registry.\n"
            f"Config has: {[s.name for s in cfg.surrogates]}\n"
            f"Registry has: {list(_MODEL_REGISTRY)}"
        )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — running on CPU (very slow for gradient check).")
    s_cfg.device = device

    # ── Load sample ───────────────────────────────────────────────────────────
    print("Loading dataset (banking category) …")
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
        sys.exit("ERROR: dataset is empty.")

    sample = dataset[0]
    print(f"Sample: {sample.image_id}")

    # ── Find SOURCE bounding box ──────────────────────────────────────────────
    labels_path = Path(cfg.data.data_dir) / "labels_pil.json"
    with labels_path.open() as f:
        labels = json.load(f)
    label = next((s for s in labels if s["image_id"] == sample.image_id), None)

    source_box   = None
    font_family  = "sans"
    if label is not None:
        font_family = label["layout"].get("font_family", "sans")
        flat = [(w["word"], w["box"]) for line in label["word_boxes"] for w in line]
        match = next(((w, b) for w, b in flat if w == SOURCE), None)
        source_box = match[1] if match else None

    if source_box is None:
        print(f"WARNING: '{SOURCE}' not found in {sample.image_id} — "
              "using clean image as ghost stand-in.")
        wm_pil = sample.image
    else:
        wm_pil, rec = render_ghost_watermark(
            sample.image, SOURCE, source_box, TARGET,
            alpha=args.alpha, font_family=font_family,
        )
        print(f"Ghost watermark: '{SOURCE}' → '{TARGET}'  "
              f"box={[round(v) for v in source_box]}  "
              f"font_size={rec.font_size_px}px  α={args.alpha}")

    wm_tensor = _pil_to_tensor(wm_pil).to(device)   # (3, H, W) — no grad yet

    # ── Build transcripts ─────────────────────────────────────────────────────
    source_transcript = sample.transcript
    if not source_transcript:
        sys.exit("ERROR: sample.transcript is empty — cannot construct target transcript.")

    if SOURCE not in source_transcript:
        print(f"WARNING: '{SOURCE}' not in sample.transcript — "
              "target_transcript will equal source_transcript.")
    target_transcript = source_transcript.replace(SOURCE, TARGET)

    print(f"\nSource transcript ({len(source_transcript)} chars): "
          f"{source_transcript[:80].replace(chr(10), ' ')!r}{'...' if len(source_transcript) > 80 else ''}")
    print(f"Target transcript ({len(target_transcript)} chars): "
          f"{target_transcript[:80].replace(chr(10), ' ')!r}{'...' if len(target_transcript) > 80 else ''}")

    # ── Load surrogate ────────────────────────────────────────────────────────
    print(f"\nLoading {s_cfg.name} on {device} …")
    mod_path, cls_name = _MODEL_REGISTRY[s_cfg.name]
    try:
        mod   = importlib.import_module(mod_path)
        model = getattr(mod, cls_name)(s_cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(f"ERROR: failed to load {s_cfg.name}.")

    # ── Gradient check ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Gradient check — targeted_ce_loss")
    print("─" * 60)

    δ = torch.zeros_like(wm_tensor, requires_grad=True)

    try:
        loss = model.targeted_ce_loss(
            wm_tensor + δ,
            source_transcript,
            target_transcript,
            SOURCE,
            TARGET,
        )
    except Exception:
        traceback.print_exc()
        sys.exit("ERROR: targeted_ce_loss raised an exception.")

    print(f"  loss value : {loss.item():.6f}")
    print(f"  loss.grad_fn : {loss.grad_fn}")

    if loss.grad_fn is None:
        print("\nFAIL: loss has no grad_fn — computation graph is broken.")
        sys.exit(1)

    try:
        loss.backward()
    except Exception:
        traceback.print_exc()
        sys.exit("ERROR: loss.backward() raised an exception.")

    if δ.grad is None:
        print("\nFAIL: δ.grad is None after backward — gradients did not flow to δ.")
        sys.exit(1)

    g = δ.grad
    g_max  = g.abs().max().item()
    g_mean = g.abs().mean().item()
    g_nnz  = (g.abs() > 1e-9).sum().item()

    print(f"\n  δ.grad shape  : {list(g.shape)}")
    print(f"  δ.grad max    : {g_max:.6e}")
    print(f"  δ.grad mean   : {g_mean:.6e}")
    print(f"  nonzero entries: {g_nnz} / {g.numel()}")

    if g_max <= 0:
        print("\nFAIL: δ.grad is all-zero — no gradient signal.")
        sys.exit(1)

    print("\nPASS: gradients flow correctly through targeted_ce_loss.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, loss, δ
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
