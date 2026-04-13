#!/usr/bin/env python3
"""
smoke_tests/run_qwen_smoke.py

Surrogate suitability screen for Qwen2-VL-7B-Instruct.
Answers the key questions before committing to wrapper development:

    1. Processor output shape (pixel_values are 2D patch-flattened)
    2. ce_proc — model's baseline confidence (target: > 1.0)
    3. Gradient flow to pixel_values
    4. Transcription quality (CER)
    5. Per-step timing

Qwen2-VL specifics handled here:
    - pixel_values shape: (N_patches, patch_dim) — 2D, not spatial
    - mm_token_type_ids must be zero-padded when transcript tokens are appended
    - image_grid_thw must be passed through unchanged
    - qwen_vl_utils.process_vision_info required for image encoding

Run:
    uv run python smoke_tests/run_qwen_smoke.py
    uv run python smoke_tests/run_qwen_smoke.py --max-samples 5 --split debug
    uv run python smoke_tests/run_qwen_smoke.py --quantization 8bit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

sys.path.insert(0, ".")
from vlm_suppress.data.dataset import TextImageDataset

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
PROMPT   = "Transcribe exactly all visible text. Output only the text, nothing else."

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "


def fmt(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.3e}"


def section(title: str):
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().cpu().clamp(0, 1) * 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), mode="RGB")


def cer(pred: str, gold: str) -> float:
    p, g = list(pred), list(gold)
    m, n = len(p), len(g)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if p[i-1] == g[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(len(g), 1)


# ── Shared encoding helper ────────────────────────────────────────────────────

def encode_prompt(processor, pil: Image.Image, device: torch.device) -> dict:
    """
    Run processor on a single PIL image and move all tensors to device.
    Returns the full encoding dict including image_grid_thw and mm_token_type_ids.
    """
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    enc = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in enc.items()}


def build_ce_inputs(
    prompt_enc: dict,
    transcript_ids: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Append transcript tokens to prompt encoding and build labels.
    Handles Qwen2-VL's mm_token_type_ids extension requirement.

    Labels mask everything except the transcript tail.
    """
    t_len = transcript_ids.size(1)

    full_ids  = torch.cat([prompt_enc["input_ids"], transcript_ids], dim=1)
    full_attn = torch.cat([
        prompt_enc["attention_mask"],
        torch.ones((1, t_len), device=device, dtype=torch.long),
    ], dim=1)
    total_len = full_ids.size(1)

    labels = torch.full((1, total_len), -100, device=device, dtype=torch.long)
    labels[0, -t_len:] = transcript_ids[0]

    # Qwen2-VL 3D-RoPE requires mm_token_type_ids to cover the full sequence.
    # Transcript tokens are plain text — pad with zeros.
    extra = {}
    if "mm_token_type_ids" in prompt_enc:
        mm  = prompt_enc["mm_token_type_ids"]
        pad = torch.zeros((1, t_len), device=device, dtype=mm.dtype)
        extra["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)

    # Pass all remaining prompt kwargs (image_grid_thw etc.) unchanged.
    skip = {"input_ids", "attention_mask", "mm_token_type_ids"}
    base = {k: v for k, v in prompt_enc.items() if k not in skip}

    return {
        **base,
        "input_ids":      full_ids,
        "attention_mask": full_attn,
        "labels":         labels,
        "return_dict":    True,
        "use_cache":      False,
        **extra,
    }


# ── CHECK 0 — model load ──────────────────────────────────────────────────────

def check_load(device, dtype, quantization):
    section("CHECK 0 — model load")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    load_kwargs = {}
    quant_label = "bfloat16"

    if quantization in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=(quantization == "8bit"),
            load_in_4bit=(quantization == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
        quant_label = quantization
    else:
        load_kwargs["torch_dtype"] = dtype

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, **load_kwargs
    ).eval()

    if quantization is None:
        model = model.to(device)

    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"    {PASS} loaded | quant={quant_label} | vram={mem_gb:.1f} GB")

    if quantization is not None:
        print(f"    {WARN} quantised model — ce_proc and CER are indicative only.")
        print(f"    {INFO} gradient check is not reliable in quantised mode.")
        print(f"    {INFO} if ce_proc > 1.0 here, rerun in bfloat16 before building wrapper.")

    return model, processor


# ── CHECK 1 — processor output shape ─────────────────────────────────────────

def check_processor_shape(processor, ds, device):
    section("CHECK 1 — processor output shape")
    s   = ds[0]
    pil = tensor_to_pil(s.image_tensor)
    enc = encode_prompt(processor, pil, device)

    for k, v in enc.items():
        if torch.is_tensor(v):
            print(f"    {INFO} {k}.shape = {tuple(v.shape)}  dtype={v.dtype}")
            if k == "pixel_values":
                print(f"    {INFO} pixel_values.min/max = {v.min():.3f} / {v.max():.3f}")

    pv = enc.get("pixel_values")
    if pv is not None and pv.ndim == 2:
        print(f"    {INFO} 2D patch-flattened format confirmed")
        print(f"    {INFO} N_patches={pv.shape[0]}  patch_dim={pv.shape[1]}")
        print(f"    {WARN} _preprocess will need patch-space delta injection, not spatial resize")

    return enc


# ── CHECK 2 — ce_proc ─────────────────────────────────────────────────────────

def check_ce_proc(model, processor, ds, device):
    section("CHECK 2 — ce_proc (ground truth confidence)")
    print(f"    {INFO} target: > 1.0  |  < 0.5 = model too confident, reject")

    ce_vals = []

    for s in ds:
        pil          = tensor_to_pil(s.image_tensor)
        prompt_enc   = encode_prompt(processor, pil, device)
        transcript_ids = processor.tokenizer(
            s.transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        model_inputs = build_ce_inputs(prompt_enc, transcript_ids, device)

        with torch.no_grad():
            out = model(**model_inputs)

        ce_val = float(out.loss.item())
        ce_vals.append(ce_val)

        tag = PASS if ce_val > 1.0 else (WARN if ce_val > 0.5 else FAIL)
        print(f"    {tag} [{s.image_id}]  ce_proc={fmt(ce_val)}  ref={s.transcript!r}")

    mean_ce = sum(ce_vals) / len(ce_vals)
    tag = PASS if mean_ce > 1.0 else (WARN if mean_ce > 0.5 else FAIL)
    print(f"\n    {tag} mean_ce_proc = {fmt(mean_ce)}")
    return ce_vals


# ── CHECK 3 — gradient flow ───────────────────────────────────────────────────

def check_gradient(model, processor, ds, device, dtype):
    section("CHECK 3 — gradient flow through pixel_values")

    s            = ds[0]
    pil          = tensor_to_pil(s.image_tensor)
    prompt_enc   = encode_prompt(processor, pil, device)
    transcript_ids = processor.tokenizer(
        s.transcript, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # Patch-space delta injection:
    # pv_proc is (N_patches, patch_dim) — no spatial structure.
    # We attach a gradient path via: pv_proc + (x - sg(x))
    # where x is a linear function of image_tensor.
    # At init delta=0, so ce_loss is identical to processor path.
    pv_proc = prompt_enc["pixel_values"].to(dtype)  # (N_patches, patch_dim)
    N, D    = pv_proc.shape

    x = s.image_tensor.to(device=device, dtype=dtype).unsqueeze(0)  # (1,3,H,W)
    x_flat = F.adaptive_avg_pool2d(x, (1, 1)).reshape(1, 3)         # (1, 3) — tiny proxy
    # Project to patch_dim via a fixed random matrix (no learned params needed —
    # we only need a non-zero gradient path, not a meaningful representation)
    torch.manual_seed(0)
    proj = torch.randn(3, D, device=device, dtype=dtype) * 0.001    # (3, D)
    x_proj = (x_flat @ proj).expand(N, -1)                          # (N, D)
    delta  = x_proj - x_proj.detach()                               # zero value, live grad

    pv_diff = pv_proc + delta

    model_inputs = build_ce_inputs(prompt_enc, transcript_ids, device)
    model_inputs["pixel_values"] = pv_diff

    try:
        loss = model(**model_inputs).loss
        loss.backward()

        if delta.grad is not None:
            g    = delta.grad.detach().float()
            gmean = g.abs().mean().item()
            gmax  = g.abs().max().item()
            tag   = PASS if gmean > 0 and math.isfinite(gmean) else FAIL
            print(f"    {tag} gradient reached pixel_values")
            print(f"    {INFO} grad_mean={fmt(gmean)}  grad_max={fmt(gmax)}")
            print(f"    {INFO} gradient is finite: {torch.isfinite(g).all().item()}")
        else:
            print(f"    {FAIL} delta.grad is None — gradient did not flow through pixel_values")
    except Exception as e:
        print(f"    {FAIL} backward failed: {type(e).__name__}: {e}")
        traceback.print_exc()


# ── CHECK 4 — transcription quality ──────────────────────────────────────────

def check_transcribe(model, processor, ds, device):
    section("CHECK 4 — transcribe quality (CER)")

    for s in ds:
        pil = tensor_to_pil(s.image_tensor)
        enc = encode_prompt(processor, pil, device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
            )

        prompt_len = enc["input_ids"].shape[1]
        pred = processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
        c    = cer(pred, s.transcript)
        tag  = PASS if c < 0.10 else (WARN if c < 0.30 else FAIL)
        print(f"    {tag} [{s.image_id}]  CER={c:.3f}")
        print(f"      pred : {pred!r}")
        print(f"      gold : {s.transcript!r}")


# ── CHECK 5 — per-step timing ─────────────────────────────────────────────────

def check_timing(model, processor, ds, device, dtype):
    section("CHECK 5 — forward+backward timing (3 steps)")

    s            = ds[0]
    pil          = tensor_to_pil(s.image_tensor)
    prompt_enc   = encode_prompt(processor, pil, device)
    transcript_ids = processor.tokenizer(
        s.transcript, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    pv = prompt_enc["pixel_values"].clone().to(dtype).requires_grad_(True)

    times = []
    for i in range(3):
        model_inputs = build_ce_inputs(prompt_enc, transcript_ids, device)
        model_inputs["pixel_values"] = pv

        torch.cuda.synchronize()
        t0   = time.time()
        loss = model(**model_inputs).loss
        loss.backward()
        torch.cuda.synchronize()

        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    {INFO} step {i+1}: {elapsed:.2f}s")

        pv = pv.detach().requires_grad_(True)

    mean_t = sum(times) / len(times)
    proj   = mean_t * 250 / 60
    tag    = PASS if mean_t < 2.0 else (WARN if mean_t < 4.0 else FAIL)
    print(f"\n    {INFO} mean per step    : {mean_t:.2f}s")
    print(f"    {INFO} projected 250 steps : {proj:.1f} min / sample")
    print(f"    {tag} {'acceptable' if mean_t < 2.0 else 'slow — consider vs K=2 runtime'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     default="data/synthetic")
    ap.add_argument("--split",        default="debug")
    ap.add_argument("--max-samples",  type=int, default=5)
    ap.add_argument("--image-h",      type=int, default=192)
    ap.add_argument("--image-w",      type=int, default=768)
    ap.add_argument("--quantization", choices=["8bit", "4bit"], default=None,
                    help="Indicative suitability screen only. Rerun in bfloat16 before wrapper dev.")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("\n" + "═" * 68)
    print("  Qwen2-VL-7B — Surrogate Suitability Smoke Test")
    print("═" * 68)

    try:
        model, processor = check_load(device, dtype, args.quantization)
    except Exception as e:
        print(f"  {FAIL} model load failed: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    try:
        ds = TextImageDataset(
            args.data_dir,
            image_size=(args.image_h, args.image_w),
            max_samples=args.max_samples,
            split_filter=args.split,
        )
        assert len(ds) > 0
        print(f"  {PASS} {len(ds)} samples loaded")
    except Exception as e:
        print(f"  {FAIL} dataset load failed: {e}")
        raise SystemExit(1)

    check_processor_shape(processor, ds, device)
    ce_vals = check_ce_proc(model, processor, ds, device)
    check_gradient(model, processor, ds, device, dtype)
    check_transcribe(model, processor, ds, device)
    check_timing(model, processor, ds, device, dtype)

    # ── Verdict ───────────────────────────────────────────────────────────────
    mean_ce = sum(ce_vals) / len(ce_vals)

    print("\n" + "═" * 68)
    print("  Suitability verdict")
    print("═" * 68)

    if mean_ce > 1.0:
        print(f"  {PASS} ce_proc={fmt(mean_ce)} — good attack surface. Proceed to wrapper.")
    elif mean_ce > 0.5:
        print(f"  {WARN} ce_proc={fmt(mean_ce)} — borderline. Run on more samples before deciding.")
    else:
        print(f"  {FAIL} ce_proc={fmt(mean_ce)} — model too confident. Reject as K=3 candidate.")

    if args.quantization:
        print(f"\n  {INFO} These results are from a {args.quantization} quantised model.")
        print(f"  {INFO} ce_proc will be slightly higher than bfloat16 (conservative estimate).")
        print(f"  {INFO} Rerun without --quantization before building the wrapper.")

    print()


if __name__ == "__main__":
    main()