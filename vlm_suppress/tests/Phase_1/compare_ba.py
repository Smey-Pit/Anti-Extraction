"""
compare_ba.py — Compare Binding Accuracy between two images (e.g. clean vs shuffled).

Loads one model, runs answer_query() per entity on both images, prints BA scores
side by side. Uses the same comparators as phase0_eval.

Usage:
    uv run python compare_ba.py \
        --model    qwen2_5vl \
        --image-a  data/ui_dataset/images/pil/banking_0000.png \
        --image-b  data/ui_dataset/images/pil/banking_0000_shuffled.png \
        --gt       data/ui_dataset/ground_truth/pil/banking_0000.json \
        --label-a  "clean" \
        --label-b  "shuffled"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

import sys
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))
# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True,
                   choices=["qwen2_5vl","internvl3_5","llama3_2vl","deepseekvl2"])
    p.add_argument("--image-a",  required=True, help="First image (e.g. clean)")
    p.add_argument("--image-b",  required=True, help="Second image (e.g. shuffled)")
    p.add_argument("--gt",       required=True, help="GT JSON for this image")
    p.add_argument("--label-a",  default="image-a")
    p.add_argument("--label-b",  default="image-b")
    p.add_argument("--model-id", default=None)
    p.add_argument("--device",   default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    name: str
    model_id: str
    max_new_tokens: int = 512
    device: str | None = None


DEFAULT_REGISTRY = {
    "qwen2_5vl":   ModelCfg("qwen2_5vl",   "Qwen/Qwen2.5-VL-7B-Instruct"),
    "internvl3_5": ModelCfg("internvl3_5", "OpenGVLab/InternVL3_5-8B"),
    "llama3_2vl":  ModelCfg("llama3_2vl",  "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "deepseekvl2": ModelCfg("deepseekvl2", "deepseek-ai/deepseek-vl2-small"),
}


def load_wrapper(model_key: str, cfg: ModelCfg):
    if model_key == "qwen2_5vl":
        from vlm_suppress.models.qwen2_5vl import Qwen2_5VL
        return Qwen2_5VL(cfg)
    if model_key == "internvl3_5":
        from vlm_suppress.models.internvl3_5 import InternVL35
        return InternVL35(cfg)
    if model_key == "llama3_2vl":
        from vlm_suppress.models.llama3_2vl import LlamaVision
        return LlamaVision(cfg)
    if model_key == "deepseekvl2":
        from vlm_suppress.models.deepseekvl2 import DeepSeekVL2
        return DeepSeekVL2(cfg)
    raise ValueError(f"Unknown model: {model_key}")


# ── GT loading ────────────────────────────────────────────────────────────────

def load_entities(gt_path: str) -> list[dict]:
    with open(gt_path) as f:
        gt = json.load(f)
    return [e for e in gt["entities"] if e.get("question")]


# ── Per-entity scoring (mirrors phase0_eval exactly) ─────────────────────────

def score_entity(answer: str, entity: dict) -> float:
    from vlm_suppress.metrics.comparators import COMPARATORS
    matcher = COMPARATORS.get(entity["type"])
    if matcher is None:
        return 0.0
    return float(matcher(entity["value"], answer))


# ── Evaluate one image ────────────────────────────────────────────────────────

def evaluate_image(
    model,
    image_tensor: torch.Tensor,
    entities: list[dict],
    label: str,
) -> tuple[float, list[dict]]:
    """
    Run answer_query() for each entity, return (BA_score, per_entity_results).
    """
    results = []
    t0 = time.perf_counter()

    for ent in entities:
        try:
            answer = model.answer_query(image_tensor, ent["question"])
            score  = score_entity(answer, ent)
        except Exception as e:
            answer = ""
            score  = 0.0

        results.append({
            "label":    ent["label"],
            "type":     ent["type"],
            "question": ent["question"],
            "gt_value": ent["value"],
            "answer":   answer,
            "score":    score,
        })

    elapsed = time.perf_counter() - t0
    ba = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"  [{label}] BA={ba:.4f}  ({len(results)} entities, {elapsed:.1f}s)")
    return ba, results


# ── Print comparison ──────────────────────────────────────────────────────────

def print_comparison(
    entities: list[dict],
    results_a: list[dict],
    results_b: list[dict],
    ba_a: float,
    ba_b: float,
    label_a: str,
    label_b: str,
):
    col = 22
    print()
    print("═" * 80)
    print(f"  {'Entity':<20} {'Type':<12} {label_a:>10} {label_b:>10}  Delta")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}  {'-'*6}")

    for ra, rb in zip(results_a, results_b):
        delta = rb["score"] - ra["score"]
        delta_str = f"{delta:+.3f}" if delta != 0 else "  —  "
        flag = " ←" if abs(delta) > 0.3 else ""
        print(f"  {ra['label']:<20} {ra['type']:<12} "
              f"{ra['score']:>10.3f} {rb['score']:>10.3f}  {delta_str}{flag}")
        # Print answers if they differ
        
        print(f"    {label_a}: {ra['answer'][:70]!r}")
        print(f"    {label_b}: {rb['answer'][:70]!r}")

    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}  {'-'*6}")
    delta_total = ba_b - ba_a
    print(f"  {'OVERALL BA':<20} {'':12} "
          f"{ba_a:>10.4f} {ba_b:>10.4f}  {delta_total:+.4f}")
    print("═" * 80)
    print()

    # Verdict
    pct_drop = (ba_a - ba_b) / ba_a * 100 if ba_a > 0 else 0
    if pct_drop > 50:
        verdict = f"✓ STRONG — BA dropped {pct_drop:.1f}%. Spatial structure is a binding attack surface."
    elif pct_drop > 20:
        verdict = f"⚠ MODERATE — BA dropped {pct_drop:.1f}%. Spatial structure contributes to binding."
    elif pct_drop > 5:
        verdict = f"△ WEAK — BA dropped {pct_drop:.1f}%. Spatial structure is a minor factor."
    else:
        verdict = f"✗ NONE — BA dropped only {pct_drop:.1f}%. Model uses semantic matching, not spatial layout."

    print(f"  Verdict: {verdict}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = DEFAULT_REGISTRY[args.model]
    if args.model_id:
        cfg.model_id = args.model_id
    if args.device:
        cfg.device = args.device
    cfg.max_new_tokens = args.max_new_tokens

    print(f"Model   : {cfg.name} ({cfg.model_id})")
    print(f"Image A : {args.image_a}  ({args.label_a})")
    print(f"Image B : {args.image_b}  ({args.label_b})")
    print(f"GT      : {args.gt}")
    print()

    # Load entities
    entities = load_entities(args.gt)
    print(f"Entities: {len(entities)}")
    for e in entities:
        print(f"  {e['label']:<20} {e['type']:<12}  Q: {e['question']}")
    print()

    # Load model
    print("Loading model…")
    model = load_wrapper(args.model, cfg)
    print(f"Model loaded on {model.device}")
    print()

    # Load images
    img_a = to_tensor(Image.open(args.image_a).convert("RGB"))
    img_b = to_tensor(Image.open(args.image_b).convert("RGB"))
    print(f"Image A size: {img_a.shape}")
    print(f"Image B size: {img_b.shape}")
    print()

    # Evaluate both
    print(f"Evaluating {args.label_a}…")
    ba_a, results_a = evaluate_image(model, img_a, entities, args.label_a)

    print(f"Evaluating {args.label_b}…")
    ba_b, results_b = evaluate_image(model, img_b, entities, args.label_b)

    # Print comparison
    print_comparison(
        entities, results_a, results_b,
        ba_a, ba_b,
        args.label_a, args.label_b,
    )


if __name__ == "__main__":
    main()