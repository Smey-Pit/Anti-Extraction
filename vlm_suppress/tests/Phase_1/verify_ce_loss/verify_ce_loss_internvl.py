"""
verify_ce_loss_internvl.py — Step 0a: sanity-check CE loss on a clean image.

InternVL3.5-8B specific. Replicates the prompt construction from model.chat()
exactly, appends the ground-truth transcript as the assistant turn, masks
prompt tokens to -100, and calls model.forward() with labels for CE loss.

Key details:
  - Image tokens: <img><IMG_CONTEXT> * 256 * num_tiles</img> replace <image>
  - image_flags: LongTensor of ones, shape (num_tiles, 1)
  - model.forward() accepts pixel_values + input_ids + labels directly
  - get_conv_template lives in the model's remote-code module

Usage (main venv):
    uv run python verify_ce_loss_internvl.py \
        --image  data/ui_dataset/images/pil/banking_0000.png \
        --labels data/ui_dataset/labels_pil.jsonl \
        --model_id OpenGVLab/InternVL3_5-8B \
        [--decoy data/ui_dataset/images/decoy.png]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--decoy",    default=None)
    p.add_argument("--model_id", default="OpenGVLab/InternVL3_5-8B")
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

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_TILE_SIZE = 448
_MAX_TILES = 12


def load_model_and_processor(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    with torch.device("cpu"):
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        ).eval()
    model = model.to(device)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    # get_conv_template lives in the remote-code module
    model_module = type(model).__module__
    mod = importlib.import_module(model_module)
    get_conv_template = mod.get_conv_template

    return model, tokenizer, get_conv_template


# ── Image preprocessing (matches wrapper's _dynamic_preprocess) ───────────────

def preprocess_image(pil: Image.Image, device: str) -> torch.Tensor:
    """
    Aspect-ratio-aware tiling — same logic as the Phase 0 wrapper.
    Returns (N_tiles, 3, 448, 448) bfloat16.
    """
    img = pil.convert("RGB")
    W, H = img.size

    grids = {
        (i, j)
        for n in range(1, _MAX_TILES + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= _MAX_TILES
    }
    ncols, nrows = min(grids, key=lambda rc: abs(rc[0] / rc[1] - W / H))

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    resized = img.resize((ncols * _TILE_SIZE, nrows * _TILE_SIZE), Image.BICUBIC)
    patches = [
        tfm(resized.crop((
            c * _TILE_SIZE, r * _TILE_SIZE,
            (c + 1) * _TILE_SIZE, (r + 1) * _TILE_SIZE,
        )))
        for r in range(nrows) for c in range(ncols)
    ]
    if ncols * nrows > 1:
        patches.append(tfm(img.resize((_TILE_SIZE, _TILE_SIZE), Image.BICUBIC)))

    return torch.stack(patches).to(device=device, dtype=torch.bfloat16)


# ── Prompt construction (replicates model.chat() exactly) ────────────────────

IMG_START = "<img>"
IMG_END   = "</img>"
IMG_CTX   = "<IMG_CONTEXT>"


def build_prompt(model, tokenizer, get_conv_template, num_tiles: int) -> str:
    """
    Build the full prompt string exactly as model.chat() does, with the
    <image> placeholder replaced by the expanded image token sequence.
    """
    question = "<image>\n" + InternVL35CE._TRANSCRIBE_PROMPT

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)   # assistant turn: empty for generation
    query = template.get_prompt()

    # Replace <image> with expanded image tokens
    image_tokens = IMG_START + IMG_CTX * model.num_image_token * num_tiles + IMG_END
    query = query.replace("<image>", image_tokens, 1)
    return query


class InternVL35CE:
    _TRANSCRIBE_PROMPT = (
        "Transcribe exactly all visible text in the image. "
        "Preserve line breaks. Output only the text."
    )


# ── Teacher-forced CE loss ────────────────────────────────────────────────────

def compute_ce_loss(
    model,
    tokenizer,
    get_conv_template,
    device: str,
    pil_image: Image.Image,
    full_text: str,
) -> dict:
    """
    Teacher-forced forward pass for InternVL3.5.

    1. Build prompt string (replicating chat() exactly)
    2. Tokenize prompt → get prompt_len
    3. Tokenize full_text + sep + eos → target_ids
    4. Concatenate, build labels (-100 for prompt, real ids for target)
    5. Build image_flags (ones, shape [num_tiles, 1])
    6. Call model.forward(pixel_values, input_ids, attention_mask,
                          image_flags, labels)
    """
    pixel_values = preprocess_image(pil_image, device)
    num_tiles = pixel_values.shape[0]

    # ── Build and tokenize prompt ────────────────────────────────────────
    prompt_str = build_prompt(model, tokenizer, get_conv_template, num_tiles)
    prompt_ids = tokenizer(
        prompt_str, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # ── Tokenize target (full_text + sep token) ──────────────────────────
    # chat() uses template.sep ('<|im_end|>\n') as the response terminator
    sep = "<|im_end|>\n"
    target_str = full_text + sep
    target_ids = tokenizer(
        target_str, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    target_len = target_ids.shape[1]

    # ── Concatenate full sequence ────────────────────────────────────────
    full_input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.ones_like(full_input_ids)

    # Labels: -100 for prompt, real token ids for target
    labels = torch.full_like(full_input_ids, -100)
    labels[:, prompt_len:] = target_ids

    # ── image_flags: ones tensor, shape (num_tiles, 1) ──────────────────
    image_flags = torch.ones(num_tiles, 1, dtype=torch.long, device=device)

    # ── Forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            labels=labels,
        )

    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        "ce_loss":       loss,
        "perplexity":    perplexity,
        "prompt_tokens": prompt_len,
        "target_tokens": target_len,
        "num_tiles":     num_tiles,
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
    print(f"  Num tiles        : {r['num_tiles']}")
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
    model, tokenizer, get_conv_template = load_model_and_processor(
        args.model_id, args.device
    )
    print("Model loaded.")
    print()

    # ── Correct image ─────────────────────────────────────────────────────
    print("Running teacher-forced forward pass (correct image)…")
    results = compute_ce_loss(
        model, tokenizer, get_conv_template, args.device, pil_image, full_text
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
            model, tokenizer, get_conv_template, args.device, decoy_image, full_text
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
        print(f"  prompt_str[:300]             :")
        sys.exit(1)


if __name__ == "__main__":
    main()