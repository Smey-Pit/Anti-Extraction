"""
verify_ce_loss_deepseek.py — Step 0a: sanity-check CE loss on a clean image.

DeepSeek-VL2 specific. The processor handles label masking automatically
when the assistant turn is filled in — no manual -100 masking required.

Key differences from Qwen:
  - No chat template; uses conversation list with role tokens
  - Vision tokens don't expand sequence length (1:1 with input_ids)
  - Labels produced by processor directly (prompt tokens already -100)
  - Forward pass: prepare_inputs_embeds -> model.language(inputs_embeds, labels)

Usage (must use .venv-deepseek):
    .venv-deepseek/bin/python verify_ce_loss_deepseek.py \
        --image  data/ui_dataset/images/pil/banking_0000.png \
        --labels data/ui_dataset/labels_pil.jsonl \
        --model_id deepseek-ai/deepseek-vl2-small \
        [--decoy data/ui_dataset/images/decoy.png]

Expected output on a clean image:
    CE loss : low (comparable to Qwen's ~0.18, may be slightly higher due
              to conversation-header tokens leaking into the label region)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--decoy",    default=None)
    p.add_argument("--model_id", default="deepseek-ai/deepseek-vl2-small")
    p.add_argument("--device",   default="cuda:0")
    return p.parse_args()


# ── Ground-truth lookup ───────────────────────────────────────────────────────

def get_full_text(labels_path: str, image_path: str) -> str:
    target = Path(image_path).name
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            if Path(row["image_path"]).name == target:
                return row["full_text"]
    raise ValueError(f"No entry found for {target} in {labels_path}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_id: str, device: str):
    from deepseek_vl2.models import DeepseekVLV2Processor
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    return model, processor, dtype


# ── Teacher-forced CE loss ────────────────────────────────────────────────────

TRANSCRIBE_PROMPT = (
    "<image>\n"
    "Read the text in this image and output it exactly as written. "
    "Output the text only, no coordinates, no descriptions, no explanations."
)


def compute_ce_loss(
    model,
    processor,
    device: str,
    pil_image: Image.Image,
    full_text: str,
) -> dict:
    """
    Teacher-forced forward pass for DeepSeek-VL2.

    The processor automatically masks prompt tokens to -100 when the
    assistant turn is filled in. We run prepare_inputs_embeds to splice
    vision features, then call model.language with inputs_embeds + labels.
    """
    conversation = [
        {
            "role": "<|User|>",
            "content": TRANSCRIBE_PROMPT,
            "images": [pil_image],
        },
        {
            "role": "<|Assistant|>",
            "content": full_text,   # filled in → processor masks prompt, keeps this
        },
    ]

    prepare_inputs = processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
        system_prompt="",
    ).to(device)

    labels = prepare_inputs.labels
    n_real   = (labels != -100).sum().item()
    n_masked = (labels == -100).sum().item()

    # Splice vision features into embedding sequence
    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            labels=labels,
        )

    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        "ce_loss":       loss,
        "perplexity":    perplexity,
        "total_tokens":  labels.shape[1],
        "target_tokens": n_real,
        "masked_tokens": n_masked,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def print_results(label: str, r: dict):
    print(f"═" * 60)
    print(f"  RESULTS — {label}")
    print(f"═" * 60)
    print(f"  CE loss          : {r['ce_loss']:.4f}")
    print(f"  Per-token PPL    : {r['perplexity']:.4f}")
    print(f"  Total tokens     : {r['total_tokens']}")
    print(f"  Target tokens    : {r['target_tokens']}  (loss computed over these)")
    print(f"  Masked tokens    : {r['masked_tokens']} (prompt + vision, ignored)")
    print(f"═" * 60)


def main():
    args = parse_args()

    print(f"Image  : {args.image}")
    print(f"Labels : {args.labels}")
    print(f"Model  : {args.model_id}")
    print(f"Device : {args.device}")
    print()

    full_text = get_full_text(args.labels, args.image)
    print(f"Ground-truth transcript ({len(full_text)} chars):")
    print("─" * 60)
    print(full_text)
    print("─" * 60)
    print()

    pil_image = Image.open(args.image).convert("RGB")
    print(f"Image size: {pil_image.size}")
    print()

    print("Loading model…")
    model, processor, dtype = load_model_and_processor(args.model_id, args.device)
    print("Model loaded.")
    print()

    # ── Correct image ─────────────────────────────────────────────────────
    print("Running teacher-forced forward pass (correct image)…")
    results = compute_ce_loss(model, processor, args.device, pil_image, full_text)
    print()
    print_results("correct image", results)

    # ── Decoy image ───────────────────────────────────────────────────────
    decoy_results = None
    if args.decoy:
        print()
        print(f"Running teacher-forced forward pass (decoy: {args.decoy})…")
        decoy_image = Image.open(args.decoy).convert("RGB")
        decoy_results = compute_ce_loss(model, processor, args.device, decoy_image, full_text)
        print()
        print_results("decoy image", decoy_results)

    # ── Comparison ────────────────────────────────────────────────────────
    if decoy_results is not None:
        delta = decoy_results["ce_loss"] - results["ce_loss"]
        ratio = decoy_results["ce_loss"] / results["ce_loss"]
        print()
        print(f"  Loss delta (decoy − correct) : +{delta:.4f}")
        print(f"  Loss ratio (decoy / correct) : {ratio:.2f}×")
        print()
        if delta > 1.0:
            verdict = "✓ GOOD — image is genuinely conditioning the output."
        elif delta > 0.3:
            verdict = "⚠ MODERATE — some image sensitivity but smaller delta than expected."
        else:
            verdict = "✗ FAIL — decoy loss barely differs; image may not be conditioning output."
        print(f"  Verdict: {verdict}")
    else:
        print()
        loss = results["ce_loss"]
        if loss < 0.8:
            # Slightly relaxed threshold vs Qwen — conversation headers
            # leak into label region, inflating loss modestly
            verdict = "✓ GOOD — model assigns high probability to the correct transcript."
        elif loss < 2.0:
            verdict = "⚠ MODERATE — check prompt format or label alignment."
        else:
            verdict = "✗ HIGH — something is wrong."
        print(f"  Verdict: {verdict}")

    print()
    if results["ce_loss"] > 2.0:
        print("Debugging info:")
        print(f"  first 200 chars of full_text: {full_text[:200]!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()