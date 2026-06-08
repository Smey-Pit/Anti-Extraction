"""
verify_ce_loss_qwen.py — Step 0a: sanity-check CE loss on a clean image.

Confirms that teacher-forced cross-entropy loss is LOW on a clean image
when the target is the ground-truth full transcript. This is the baseline
that Phase 1 perturbations must be measured against.

Usage (from Anti-Extraction-v2 root):
    uv run python verify_ce_loss_qwen.py \
        --image  data/ui_dataset/images/pil/banking_0000.png \
        --labels data/ui_dataset/labels_pil.jsonl \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct

Expected output on a clean image:
    CE loss : ~0.1 – 0.5   (low  = model assigns high prob to correct transcript)
    Per-token perplexity: ~1.1 – 1.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True,  help="Path to a clean .png image")
    p.add_argument("--labels",   required=True,  help="Path to labels_pil.jsonl")
    p.add_argument("--decoy",    default=None,   help="Path to a decoy image (optional)")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device",   default="cuda:0")
    return p.parse_args()


# ── Ground-truth lookup ───────────────────────────────────────────────────────

def get_full_text(labels_path: str, image_path: str) -> str:
    """
    Find the full_text for a given image from labels_pil.jsonl.
    Matches on the basename of image_path (e.g. 'banking_0000.png').
    """
    target = Path(image_path).name
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            if Path(row["image_path"]).name == target:
                return row["full_text"]
    raise ValueError(f"No entry found for {target} in {labels_path}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_id: str, device: str):
    from transformers import AutoProcessor

    # Try both naming conventions (newer / older transformers)
    try:
        from transformers import Qwen2_5VLForConditionalGeneration as ModelCls
    except ImportError:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration as ModelCls,
            )

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = (
        ModelCls.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )
    return model, processor, dtype


# ── Teacher-forced CE loss ────────────────────────────────────────────────────

def compute_ce_loss(
    model,
    processor,
    dtype,
    device: str,
    pil_image: Image.Image,
    full_text: str,
    transcribe_prompt: str,
) -> dict:
    """
    Teacher-forced forward pass.

    The input sequence is:
        [chat_prefix | transcribe_prompt] [full_text] [eos]

    Labels are -100 for the prompt portion (not included in loss),
    and the actual token ids for the target portion.

    Returns dict with loss, per-token perplexity, and token counts.
    """

    # ── Build prompt-only chat text (no target yet) ───────────────────────
    chat_prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": transcribe_prompt},
        ]}],
        tokenize=False,
        add_generation_prompt=True,   # adds the assistant turn opener
    )

    # ── Tokenise prompt + image ───────────────────────────────────────────
    prompt_inputs = processor(
        text=[chat_prompt],
        images=[pil_image],
        return_tensors="pt",
    )
    prompt_inputs = {
        k: (v.to(device).to(dtype) if k == "pixel_values"
            else v.to(device) if torch.is_tensor(v)
            else v)
        for k, v in prompt_inputs.items()
    }
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # ── Tokenise target (full_text + eos) ─────────────────────────────────
    # Tokenise without special tokens so we don't double-add BOS/EOS
    target_ids = processor.tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    eos_id = torch.tensor(
        [[processor.tokenizer.eos_token_id]], device=device
    )
    target_ids = torch.cat([target_ids, eos_id], dim=1)
    target_len = target_ids.shape[1]

    # ── Concatenate full sequence ──────────────────────────────────────────
    full_input_ids = torch.cat(
        [prompt_inputs["input_ids"], target_ids], dim=1
    )

    # Labels: -100 for prompt tokens (ignored in loss), real ids for target
    labels = torch.full_like(full_input_ids, -100)
    labels[:, prompt_len:] = target_ids

    # Build attention mask for the full sequence
    full_attention_mask = torch.ones_like(full_input_ids)

    # ── Forward pass ──────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=prompt_inputs.get("pixel_values"),
            image_grid_thw=prompt_inputs.get("image_grid_thw"),
            labels=labels,
        )

    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        "ce_loss":        loss,
        "perplexity":     perplexity,
        "prompt_tokens":  prompt_len,
        "target_tokens":  target_len,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

TRANSCRIBE_PROMPT = (
    "Read the text in this image and output it exactly as written. "
    "Output the text only, no coordinates, no descriptions, no explanations."
)


def main():
    args = parse_args()

    print(f"Image  : {args.image}")
    print(f"Labels : {args.labels}")
    print(f"Model  : {args.model_id}")
    print(f"Device : {args.device}")
    print()

    # ── Load ground truth ─────────────────────────────────────────────────
    full_text = get_full_text(args.labels, args.image)
    print(f"Ground-truth transcript ({len(full_text)} chars):")
    print("─" * 60)
    print(full_text)
    print("─" * 60)
    print()

    # ── Load image ────────────────────────────────────────────────────────
    pil_image = Image.open(args.image).convert("RGB")
    print(f"Image size: {pil_image.size}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model…")
    model, processor, dtype = load_model_and_processor(
        args.model_id, args.device
    )
    print("Model loaded.")
    print()

    # ── Compute CE loss on correct image ─────────────────────────────────
    print("Running teacher-forced forward pass (correct image)…")
    results = compute_ce_loss(
        model, processor, dtype, args.device,
        pil_image, full_text, TRANSCRIBE_PROMPT,
    )

    # ── Compute CE loss on decoy image (optional) ─────────────────────────
    decoy_results = None
    if args.decoy:
        print(f"Running teacher-forced forward pass (decoy: {args.decoy})…")
        decoy_image = Image.open(args.decoy).convert("RGB")
        print(f"Decoy image size: {decoy_image.size}")
        decoy_results = compute_ce_loss(
            model, processor, dtype, args.device,
            decoy_image, full_text, TRANSCRIBE_PROMPT,
        )

    # ── Print results ─────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  RESULTS — correct image")
    print("═" * 60)
    print(f"  CE loss          : {results['ce_loss']:.4f}")
    print(f"  Per-token PPL    : {results['perplexity']:.4f}")
    print(f"  Prompt tokens    : {results['prompt_tokens']}")
    print(f"  Target tokens    : {results['target_tokens']}")
    print("═" * 60)

    if decoy_results is not None:
        print()
        print("═" * 60)
        print("  RESULTS — decoy image")
        print("═" * 60)
        print(f"  CE loss          : {decoy_results['ce_loss']:.4f}")
        print(f"  Per-token PPL    : {decoy_results['perplexity']:.4f}")
        print(f"  Prompt tokens    : {decoy_results['prompt_tokens']}")
        print(f"  Target tokens    : {decoy_results['target_tokens']}")
        print("═" * 60)
        print()

        delta = decoy_results['ce_loss'] - results['ce_loss']
        ratio = decoy_results['ce_loss'] / results['ce_loss']
        print(f"  Loss delta (decoy − correct) : +{delta:.4f}")
        print(f"  Loss ratio (decoy / correct) : {ratio:.2f}×")
        print()

        if delta > 1.0:
            verdict = "✓ GOOD — decoy loss is substantially higher. Image is genuinely conditioning the output."
        elif delta > 0.3:
            verdict = "⚠ MODERATE — some sensitivity to image content, but delta is smaller than expected."
        else:
            verdict = "✗ FAIL — decoy loss barely differs from correct. Image may not be conditioning the output."
        print(f"  Verdict: {verdict}")
    else:
        print()
        loss = results["ce_loss"]
        if loss < 0.5:
            verdict = "✓ GOOD — model assigns high probability to the correct transcript."
        elif loss < 1.5:
            verdict = "⚠ MODERATE — loss higher than expected; check prompt format or tokenisation."
        else:
            verdict = "✗ HIGH — something is wrong; model doesn't recognise the transcript."
        print(f"  Verdict: {verdict}")

    print()
    if results['ce_loss'] > 1.5:
        print("Debugging info:")
        print(f"  First 200 chars of full_text: {full_text[:200]!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()