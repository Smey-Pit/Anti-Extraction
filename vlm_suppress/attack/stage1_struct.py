"""
stage1_struct.py — Stage 1: structural disruption attack on Qwen2.5-VL.

Optimizes δ_struct to maximize transcript CE loss on a single image.
Measures effect on BA_amounts vs BA_names separately at each eval step.

The target pattern (reproducing the spatial shuffle effect):
  BA_amounts drops  — spatial binding broken for ambiguous-format fields
  BA_names stays    — semantic fallback still recovers unique values

If BA_names also drops, semantic fallback is broken too — this is fine but
means Stage 2 (decoy injection) may be redundant for name fields.

Usage (from Anti-Extraction-v2 root):
    uv run python attack/stage1_struct.py \\
        --image  data/ui_dataset/images/pil/banking_0000.png \\
        --labels data/ui_dataset/labels_pil.jsonl \\
        --gt     data/ui_dataset/ground_truth/pil/banking_0000.json \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --epsilon 16 \\
        --steps   100 \\
        --out     outputs/stage1/

Outputs
-------
  outputs/stage1/banking_0000_adv_eps16.png    perturbed image
  outputs/stage1/banking_0000_delta_eps16.npy  raw perturbation (float32)
  outputs/stage1/banking_0000_log_eps16.json   step-by-step BA log
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# ── make vlm_suppress importable when run from repo root ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm_suppress.models.qwen2_5vl import Qwen2_5VL
from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.attack.pgd import pgd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",     required=True)
    p.add_argument("--labels",    required=True,  help="labels_pil.jsonl")
    p.add_argument("--gt",        required=True,  help="per-image GT JSON")
    p.add_argument("--model_id",  default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--epsilon",   type=int,   default=16,
                   help="L∞ budget in units of /255")
    p.add_argument("--alpha",     type=float, default=None,
                   help="Step size. Default: epsilon/10/255")
    p.add_argument("--steps",     type=int,   default=100)
    p.add_argument("--eval-every",type=int,   default=10)
    p.add_argument("--out",       default="outputs/stage1/")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def get_full_text(labels_path: str, image_path: str) -> str:
    import csv
    target = Path(image_path).name
    with open(labels_path) as f:
        for line in f:
            import json as _json
            row = _json.loads(line)
            if Path(row["image_path"]).name == target:
                return row["full_text"]
    raise ValueError(f"{target} not found in {labels_path}")


def load_gt(gt_path: str) -> dict:
    with open(gt_path) as f:
        return json.load(f)


# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    name: str
    model_id: str
    max_new_tokens: int = 512
    device: str | None = None


# ── BA evaluation ─────────────────────────────────────────────────────────────

AMOUNT_TYPES  = {"amount"}
NAME_TYPES    = {"name", "short_phrase", "digit_seq", "date_range", "date"}


def evaluate_ba(
    model: Qwen2_5VL,
    image_tensor: torch.Tensor,
    entities: list[dict],
) -> dict:
    """
    Evaluate BA separately for amount-type and name-type entities.
    Uses the same comparators as phase0_eval.
    Returns dict with ba_amounts, ba_names, ba_overall and per-entity answers.
    """
    from vlm_suppress.metrics.comparators import COMPARATORS

    amount_scores, name_scores = [], []
    per_entity = []

    for ent in entities:
        if not ent.get("question"):
            continue
        with torch.no_grad():
            answer = model.answer_query(image_tensor, ent["question"])

        matcher = COMPARATORS.get(ent["type"])
        score   = float(matcher(ent["value"], answer)) if matcher else 0.0

        per_entity.append({
            "label":   ent["label"],
            "type":    ent["type"],
            "gt":      ent["value"],
            "answer":  answer,
            "score":   score,
        })

        if ent["type"] in AMOUNT_TYPES:
            amount_scores.append(score)
        else:
            name_scores.append(score)

    ba_amounts  = np.mean(amount_scores)  if amount_scores  else None
    ba_names    = np.mean(name_scores)    if name_scores    else None
    ba_overall  = np.mean([e["score"] for e in per_entity]) if per_entity else None

    return {
        "ba_amounts":  ba_amounts,
        "ba_names":    ba_names,
        "ba_overall":  ba_overall,
        "per_entity":  per_entity,
    }


def print_ba(step: int, result: dict, loss: float | None = None):
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "  n/a"

    loss_str = f"  loss={loss:.4f}" if loss is not None else ""
    print(
        f"  step {step:4d}{loss_str}"
        f"  BA_amounts={_fmt(result['ba_amounts'])}"
        f"  BA_names={_fmt(result['ba_names'])}"
        f"  BA_overall={_fmt(result['ba_overall'])}"
    )
    # Print any wrong answers
    for e in result["per_entity"]:
        if e["score"] < 1.0:
            print(f"    ✗ {e['label']:<20} gt={e['gt']!r:<25} got={e['answer'][:60]!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    eps    = args.epsilon / 255.0
    alpha  = args.alpha if args.alpha is not None else eps / 10.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_id = Path(args.image).stem
    tag      = f"eps{args.epsilon}"

    print(f"Image    : {args.image}  ({image_id})")
    print(f"Model    : {args.model_id}")
    print(f"Device   : {args.device}")
    print(f"Epsilon  : {args.epsilon}/255 = {eps:.4f}")
    print(f"Alpha    : {alpha:.4f}")
    print(f"Steps    : {args.steps}")
    print(f"Out dir  : {out_dir}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    full_text = get_full_text(args.labels, args.image)
    gt        = load_gt(args.gt)
    entities  = gt["entities"]
    pil_image = Image.open(args.image).convert("RGB")
    image_tensor = to_tensor(pil_image).to(args.device)   # (3, H, W) [0,1]

    print(f"Full text ({len(full_text)} chars), {len(entities)} entities")
    print(f"Amount entities: {[e['label'] for e in entities if e['type'] in AMOUNT_TYPES]}")
    print(f"Name entities  : {[e['label'] for e in entities if e['type'] in NAME_TYPES]}")
    print()

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading model…")
    cfg   = Cfg(name="qwen2_5vl", model_id=args.model_id, device=args.device)
    model = Qwen2_5VL(cfg)
    print(f"Model loaded on {model.device}")
    print()

    # ── Baseline BA on clean image ─────────────────────────────────────────
    print("━" * 70)
    print("Baseline (clean image):")
    baseline = evaluate_ba(model, image_tensor, entities)
    print_ba(0, baseline)
    print("━" * 70)
    print()

    # ── Define loss function for PGD ──────────────────────────────────────
    # Maximize transcript CE loss → disrupts spatial/structural encoding
    def loss_fn(x_adv: torch.Tensor) -> torch.Tensor:
        return model.transcript_loss(x_adv, full_text)

    # ── Eval function called every eval_every steps ───────────────────────
    step_log = []

    def eval_fn(x_adv: torch.Tensor, step: int) -> dict:
        result = evaluate_ba(model, x_adv.detach(), entities)
        entry  = {
            "step":        step,
            "ba_amounts":  result["ba_amounts"],
            "ba_names":    result["ba_names"],
            "ba_overall":  result["ba_overall"],
            "per_entity":  result["per_entity"],
        }
        step_log.append(entry)
        return {
            "ba_amounts": result["ba_amounts"] if result["ba_amounts"] is not None else -1.0,
            "ba_names":   result["ba_names"]   if result["ba_names"]   is not None else -1.0,
        }

    # ── Run PGD ──────────────────────────────────────────────────────────
    print("Running PGD (maximizing transcript CE loss)…")
    print("Target pattern: BA_amounts ↓  BA_names stays high")
    print()

    x_adv, pgd_log = pgd(
        image_tensor=image_tensor,
        loss_fn=loss_fn,
        epsilon=eps,
        alpha=alpha,
        n_steps=args.steps,
        targeted=False,      # maximize loss
        random_init=True,
        verbose=True,
        eval_fn=eval_fn,
        eval_every=args.eval_every,
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    print()
    print("━" * 70)
    print("Final (perturbed image):")
    final = evaluate_ba(model, x_adv, entities)
    print_ba(args.steps, final)
    print("━" * 70)
    print()

    # ── Verdict ──────────────────────────────────────────────────────────
    ba_amounts_clean = baseline["ba_amounts"] or 0.0
    ba_names_clean   = baseline["ba_names"]   or 0.0
    ba_amounts_adv   = final["ba_amounts"]    or 0.0
    ba_names_adv     = final["ba_names"]      or 0.0

    amounts_drop = ba_amounts_clean - ba_amounts_adv
    names_drop   = ba_names_clean   - ba_names_adv

    print("═" * 70)
    print("  STAGE 1 RESULT")
    print("═" * 70)
    print(f"  BA_amounts : {ba_amounts_clean:.4f} → {ba_amounts_adv:.4f}  "
          f"(drop={amounts_drop:.4f})")
    print(f"  BA_names   : {ba_names_clean:.4f} → {ba_names_adv:.4f}  "
          f"(drop={names_drop:.4f})")
    print()

    if amounts_drop > 0.5 and names_drop < 0.2:
        verdict = ("✓ SHUFFLE EFFECT REPRODUCED — structural encoding disrupted. "
                   "Stage 2 needed for names.")
    elif amounts_drop > 0.5 and names_drop > 0.5:
        verdict = ("✓ STRONG — both amounts and names disrupted. "
                   "Stage 2 may be redundant.")
    elif amounts_drop > 0.2:
        verdict = ("⚠ PARTIAL — amounts partially disrupted. "
                   "Try more steps or larger epsilon.")
    else:
        verdict = ("✗ WEAK — structural attack not working. "
                   "Check gradient flow or increase epsilon.")

    print(f"  Verdict: {verdict}")
    print("═" * 70)
    print()

    # ── Save outputs ──────────────────────────────────────────────────────
    # Perturbed image
    adv_pil  = tensor_to_pil(x_adv.cpu())
    adv_path = out_dir / f"{image_id}_adv_{tag}.png"
    adv_pil.save(adv_path)
    print(f"Perturbed image : {adv_path}")

    # Raw delta
    delta     = (x_adv - image_tensor).detach().cpu().numpy()
    delta_path = out_dir / f"{image_id}_delta_{tag}.npy"
    np.save(delta_path, delta)
    print(f"Delta (numpy)   : {delta_path}")

    # L∞ norm of actual perturbation
    linf = np.abs(delta).max()
    print(f"Actual L∞       : {linf:.4f} ({linf*255:.2f}/255)")

    # Step log
    log_data = {
        "image_id":  image_id,
        "model_id":  args.model_id,
        "epsilon":   args.epsilon,
        "alpha":     alpha,
        "steps":     args.steps,
        "baseline":  {
            "ba_amounts": baseline["ba_amounts"],
            "ba_names":   baseline["ba_names"],
            "ba_overall": baseline["ba_overall"],
        },
        "final": {
            "ba_amounts": final["ba_amounts"],
            "ba_names":   final["ba_names"],
            "ba_overall": final["ba_overall"],
        },
        "step_log": step_log,
    }
    log_path = out_dir / f"{image_id}_log_{tag}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Step log        : {log_path}")
    print()
    print("Run compare_ba.py on the perturbed image to verify with all surrogates:")
    print(f"  uv run python compare_ba.py \\")
    print(f"      --model qwen2_5vl \\")
    print(f"      --image-a {args.image} \\")
    print(f"      --image-b {adv_path} \\")
    print(f"      --gt {args.gt} \\")
    print(f"      --label-a clean --label-b adv_{tag}")


if __name__ == "__main__":
    main()