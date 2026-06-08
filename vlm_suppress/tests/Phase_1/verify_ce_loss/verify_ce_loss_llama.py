"""
verify_ce_loss_llama.py — Step 0a: sanity-check CE loss on a clean image.

Llama-3.2-11B-Vision-Instruct specific. Visual features are passed as
cross_attention_states — they don't expand input_ids. Teacher-forcing is
therefore straightforward: build prompt ids, append target ids, mask prompt
with -100, call model.forward() with cross_attention_states + labels.

Key details:
  - No image token expansion in input_ids (cross-attention adapter design)
  - cross_attention_states extracted via vision encoder + projector directly
  - model.forward() accepts both cross_attention_states and labels
  - cross_attention_mask must cover the full (prompt + target) sequence

Usage (main venv):
    uv run python verify_ce_loss_llama.py \
        --image  data/ui_dataset/images/pil/banking_0000.png \
        --labels data/ui_dataset/labels_pil.jsonl \
        --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
        [--decoy data/ui_dataset/images/decoy.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--decoy",    default=None)
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
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

TRANSCRIBE_PROMPT = (
    "Perform a complete OCR scan of this document. "
    "Output every single line of text exactly as it appears, "
    "reading from the top of the document to the very bottom. "
    "Include all headers, body text, numbers, dates, amounts, names, "
    "and table contents. Do not skip, summarize, or omit any line. "
    "Output only the raw text content, nothing else."
)


def load_model_and_processor(model_id: str, device: str):
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).eval().to(device)
    return model, processor, dtype


# ── Vision encoder (mirrors wrapper's _encode_image exactly) ─────────────────

@torch.no_grad()
def encode_image(model, inputs: dict, device: str, dtype) -> torch.Tensor:
    """
    Run vision encoder + projector, return cross_attention_states.
    Matches LlamaVision._encode_image() exactly.
    """
    vision_model = model.model.vision_model
    vis_out = vision_model(
        pixel_values=inputs["pixel_values"],
        aspect_ratio_ids=inputs.get("aspect_ratio_ids"),
        aspect_ratio_mask=inputs.get("aspect_ratio_mask"),
    )
    states = vis_out.last_hidden_state  # (B, n_img, n_tiles, seq_len, vision_dim)

    projector = getattr(model.model, "multi_modal_projector", None)
    if projector is not None:
        B, n_img, n_tiles, seq_len, D = states.shape
        projected = projector(states.reshape(-1, D).to(dtype))
        states = projected.reshape(B, n_img, n_tiles, seq_len, -1)

    return states  # (..., lm_dim)


# ── Teacher-forced CE loss ────────────────────────────────────────────────────

def compute_ce_loss(
    model,
    processor,
    dtype,
    device: str,
    pil_image: Image.Image,
    full_text: str,
) -> dict:
    """
    Teacher-forced forward pass for Llama-3.2-Vision.

    1. Build chat-templated prompt, encode image → cross_attention_states
    2. Tokenize full_text + eos → target_ids
    3. Concatenate prompt_ids + target_ids
    4. Labels: -100 for prompt, real ids for target
    5. Extend cross_attention_mask to cover full sequence length
    6. model.forward(input_ids, cross_attention_states, cross_attention_mask,
                     attention_mask, labels)
    """
    # ── Build prompt inputs ──────────────────────────────────────────────
    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=prompt_text,
        images=pil_image,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[1]

    # ── Encode image → cross_attention_states ────────────────────────────
    cross_attention_states = encode_image(model, inputs, device, dtype)

    # ── Tokenize target ──────────────────────────────────────────────────
    eos_token = processor.tokenizer.eos_token or "<|eot_id|>"
    target_str = full_text + eos_token
    target_ids = processor.tokenizer(
        target_str,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    target_len = target_ids.shape[1]

    # ── Concatenate full sequence ────────────────────────────────────────
    full_input_ids   = torch.cat([inputs["input_ids"], target_ids], dim=1)
    attention_mask   = torch.ones_like(full_input_ids)
    full_seq_len     = full_input_ids.shape[1]

    # Labels: -100 for prompt tokens, real ids for target
    labels = torch.full_like(full_input_ids, -100)
    labels[:, prompt_len:] = target_ids

    # ── Extend cross_attention_mask to full sequence length ──────────────
    # Processor returns cross_attention_mask of shape (1, prompt_len, n_img, n_tiles)
    # We need it to cover (1, full_seq_len, n_img, n_tiles).
    # The target tokens don't attend to image features — pad with zeros.
    ca_mask_prompt = inputs.get("cross_attention_mask")  # (1, prompt_len, n_img, n_tiles)
    if ca_mask_prompt is not None:
        pad_len = full_seq_len - prompt_len
        pad = torch.zeros(
            ca_mask_prompt.shape[0],
            pad_len,
            *ca_mask_prompt.shape[2:],
            dtype=ca_mask_prompt.dtype,
            device=device,
        )
        cross_attention_mask = torch.cat([ca_mask_prompt, pad], dim=1)
    else:
        cross_attention_mask = None

    # ── Forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            labels=labels,
        )

    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        "ce_loss":       loss,
        "perplexity":    perplexity,
        "prompt_tokens": prompt_len,
        "target_tokens": target_len,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_results(label: str, r: dict):
    print("═" * 60)
    print(f"  RESULTS — {label}")
    print("═" * 60)
    print(f"  CE loss          : {r['ce_loss']:.4f}")
    print(f"  Per-token PPL    : {r['perplexity']:.4f}")
    print(f"  Prompt tokens    : {r['prompt_tokens']}")
    print(f"  Target tokens    : {r['target_tokens']}")
    print("═" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

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
    results = compute_ce_loss(
        model, processor, dtype, args.device, pil_image, full_text
    )
    print()
    print_results("correct image", results)

    # ── Decoy image ───────────────────────────────────────────────────────
    decoy_results = None
    if args.decoy:
        print()
        print(f"Running teacher-forced forward pass (decoy: {args.decoy})…")
        decoy_image = Image.open(args.decoy).convert("RGB")
        decoy_results = compute_ce_loss(
            model, processor, dtype, args.device, decoy_image, full_text
        )
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
        loss = results["ce_loss"]
        print()
        if loss < 0.5:
            verdict = "✓ GOOD — model assigns high probability to the correct transcript."
        elif loss < 1.5:
            verdict = "⚠ MODERATE — check prompt format or label alignment."
        else:
            verdict = "✗ HIGH — something is wrong."
        print(f"  Verdict: {verdict}")

    print()
    if results["ce_loss"] > 2.0:
        print("Debugging info:")
        print(f"  first 200 chars of full_text : {full_text[:200]!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()