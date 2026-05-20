"""
scripts/run_targeted_pgd.py

Run the ghost-watermark + targeted PGD substitution attack on a single sample.

Flow
----
1. Load config + dataset sample (banking category by default)
2. Render ghost watermark  ("Thompson" → "Henderson") onto the clean image
3. Build source_transcript and target_transcript from sample.transcript
4. Load the first available surrogate that implements targeted_ce_loss
5. Run targeted PGD (minimise targeted_ce_loss starting from ghost image)
6. Log each step to a JSONL file; eval every --eval-every steps
7. Save adversarial PNG and print final outcome

Usage (from project root, GPU node)
------------------------------------
    uv run python scripts/run_targeted_pgd.py --config configs/attack.yaml

Optional flags
--------------
    --source      STR   source word to suppress  (default: Thompson)
    --target      STR   target word to inject     (default: Henderson)
    --alpha       F     ghost watermark opacity   (default: 0.12)
    --epsilon     F     L-inf budget (0-1 scale)  (default: 8/255 ≈ 0.0314)
    --step-size   F     PGD step size             (default: 1/255 ≈ 0.0039)
    --n-steps     N     PGD iterations            (default: from config)
    --eval-every  N     eval checkpoint interval  (default: 25)
    --out         DIR   output directory           (default: runs/targeted_pgd)
    --device      STR   override device
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import dacite
import yaml

from vlm_suppress.config import (
    Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
)
from vlm_suppress.attack.targeted_pgd import make_word_mask, run_targeted_pgd
from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.watermark.renderer import render_ghost_watermark
from vlm_suppress.watermark.steplog import StepLogger

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


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=Path,  default=Path("configs/attack.yaml"))
    parser.add_argument("--source",     type=str,   default="Thompson")
    parser.add_argument("--target",     type=str,   default="Henderson")
    parser.add_argument("--alpha",      type=float, default=0.12,
                        help="Ghost watermark opacity")
    parser.add_argument("--epsilon",    type=float, default=8 / 255,
                        help="L-inf perturbation budget (0–1 scale, default 8/255)")
    parser.add_argument("--step-size",  type=float, default=1 / 255,
                        help="PGD step size (default 1/255)")
    parser.add_argument("--n-steps",    type=int,   default=None,
                        help="PGD iterations (default: attack.pgd_steps from config)")
    parser.add_argument("--eval-every", type=int,   default=25,
                        help="Eval checkpoint every N steps")
    parser.add_argument("--out",         type=Path,  default=Path("runs/targeted_pgd"),
                        help="Output directory for PNG and JSONL")
    parser.add_argument("--box-padding", type=int,   default=4,
                        help="Pixels to pad the source-word box on each side (default 4)")
    parser.add_argument("--no-mask",     action="store_true",
                        help="Disable region mask — perturb the full image (not recommended)")
    parser.add_argument("--span-weight", type=float, default=5.0,
                        help="Upweight on substitution token span (default 5.0)")
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    cfg     = _load_cfg(args.config)
    n_steps = args.n_steps or cfg.attack.pgd_steps
    device  = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU — running on CPU (very slow).")

    # ── Dataset ───────────────────────────────────────────────────────────────
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

    # ── Source bounding box ───────────────────────────────────────────────────
    labels_path = Path(cfg.data.data_dir) / "labels_pil.json"
    with labels_path.open() as f:
        labels = json.load(f)
    label = next((s for s in labels if s["image_id"] == sample.image_id), None)

    source_box  = None
    font_family = "sans"
    if label is not None:
        font_family = label["layout"].get("font_family", "sans")
        flat = [(w["word"], w["box"]) for line in label["word_boxes"] for w in line]
        match = next(((w, b) for w, b in flat if w == args.source), None)
        source_box = match[1] if match else None

    if source_box is None:
        print(f"WARNING: '{args.source}' not found in word_boxes — using clean image.")
        wm_pil = sample.image
    else:
        wm_pil, rec = render_ghost_watermark(
            sample.image, args.source, source_box, args.target,
            alpha=args.alpha, font_family=font_family,
        )
        print(f"Ghost: '{args.source}' → '{args.target}'  "
              f"box={[round(v) for v in source_box]}  "
              f"font={rec.font_size_px}px  α={args.alpha}")

    # ── Transcripts ───────────────────────────────────────────────────────────
    source_transcript = sample.transcript
    if not source_transcript:
        sys.exit("ERROR: sample.transcript is empty.")
    if args.source not in source_transcript:
        print(f"WARNING: '{args.source}' not in transcript — "
              "target_transcript == source_transcript.")
    target_transcript = source_transcript.replace(args.source, args.target)

    wm_tensor = _pil_to_tensor(wm_pil).to(device)

    # ── Word-region mask ──────────────────────────────────────────────────────
    # wm_tensor has the same H, W as wm_pil (no resize in _pil_to_tensor),
    # and labels_pil.json boxes are in the same PIL coordinate space.
    word_mask = None
    if not args.no_mask and source_box is not None:
        word_mask = make_word_mask(
            tensor_shape = tuple(wm_tensor.shape),
            box          = source_box,
            padding      = args.box_padding,
        )
        C, H, W = wm_tensor.shape
        x0, y0, x1, y1 = source_box
        px0 = max(0, int(x0) - args.box_padding)
        py0 = max(0, int(y0) - args.box_padding)
        px1 = min(W, int(x1) + args.box_padding)
        py1 = min(H, int(y1) + args.box_padding)
        active_pixels = int(word_mask.sum().item())
        print(f"Mask: word box [{px0},{py0},{px1},{py1}]  "
              f"active pixels={active_pixels} / {H * W}  "
              f"({100 * active_pixels / (H * W):.2f}%)")
    elif args.no_mask:
        print("Mask: disabled (full-image perturbation)")
    else:
        print("Mask: disabled (source box not found)")

    # ── Surrogate ─────────────────────────────────────────────────────────────
    available = [s for s in cfg.surrogates if s.name in _MODEL_REGISTRY]
    preferred = [s for s in available if s.name == "qwen2_5vl"]
    s_cfg     = (preferred or available)[0] if (preferred or available) else None
    if s_cfg is None:
        sys.exit(f"ERROR: no config surrogate in registry {list(_MODEL_REGISTRY)}.")

    s_cfg.device = device
    print(f"Loading {s_cfg.name} on {device} …")
    mod_path, cls_name = _MODEL_REGISTRY[s_cfg.name]
    try:
        mod   = importlib.import_module(mod_path)
        model = getattr(mod, cls_name)(s_cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(f"ERROR: failed to load {s_cfg.name}.")

    # ── Output paths ──────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    stem     = f"{sample.image_id}_{args.source.lower()}_{args.target.lower()}"
    log_path = args.out / f"{stem}.jsonl"
    png_path = args.out / f"{stem}_adv.png"

    # ── PGD ───────────────────────────────────────────────────────────────────
    print(f"\nTargeted PGD  n_steps={n_steps}  ε={args.epsilon:.4f}  "
          f"step_size={args.step_size:.4f}  eval_every={args.eval_every}  "
          f"span_weight={args.span_weight}")
    print("─" * 60)

    with StepLogger(
        path        = log_path,
        image_id    = sample.image_id,
        source_word = args.source,
        target_word = args.target,
        surrogate   = s_cfg.name,
        n_steps     = n_steps,
        epsilon     = args.epsilon,
        alpha_pgd   = args.step_size,
        ghost_alpha = args.alpha,
        masked      = word_mask is not None,
        source_box  = [round(v) for v in source_box] if source_box else None,
        box_padding = args.box_padding,
        span_weight = args.span_weight,
    ) as logger:
        adv_tensor, outcome = run_targeted_pgd(
            wm_tensor         = wm_tensor,
            source_transcript = source_transcript,
            target_transcript = target_transcript,
            source_word       = args.source,
            target_word       = args.target,
            surrogate         = model,
            n_steps           = n_steps,
            epsilon           = args.epsilon,
            step_size         = args.step_size,
            eval_every        = args.eval_every,
            span_weight       = args.span_weight,
            mask              = word_mask,
            logger            = logger,
            verbose           = True,
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    _tensor_to_pil(adv_tensor).save(png_path)
    print("─" * 60)
    print(f"Final outcome : {outcome}")
    print(f"Adv image     : {png_path}")
    print(f"Step log      : {log_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
