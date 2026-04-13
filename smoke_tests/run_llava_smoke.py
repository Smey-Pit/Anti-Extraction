#!/usr/bin/env python3
"""
tools/smoke_llava16.py

Surrogate suitability screen for LLaVA-1.6-Mistral-7B (LLaVA-NeXT).

Architecture notes:
    Vision encoder : CLIP ViT-L/14@336
    Connector      : 2-layer MLP projection
    LM backbone    : Mistral-7B
    Tiling         : AnyRes — image split into tiles + global thumbnail
    pixel_values   : (1, N_tiles, 3, H_tile, W_tile) — 5D spatial

Key differences from Qwen2-VL:
    - No mm_token_type_ids
    - pixel_values are spatial (5D), not patch-flattened
    - image_sizes must be passed to model.forward()
    - processor: LlavaNextProcessor

Run:
    uv run python tools/smoke_llava16.py
    uv run python tools/smoke_llava16.py --max-samples 5 --split debug
    uv run python tools/smoke_llava16.py --quantization 8bit
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
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

sys.path.insert(0, ".")
from vlm_suppress.data.dataset import TextImageDataset

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
PROMPT   = "[INST] <image>\nTranscribe exactly all visible text. Output only the text, nothing else. [/INST]"

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


# ── Shared helpers ────────────────────────────────────────────────────────────

def encode_prompt(processor, pil: Image.Image, device: torch.device) -> dict:
    enc = processor(text=PROMPT, images=pil, return_tensors="pt")
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in enc.items()}


def build_ce_inputs(
    prompt_enc: dict,
    transcript_ids: torch.Tensor,
    device: torch.device,
    pixel_values: torch.Tensor | None = None,
) -> dict:
    """
    Append transcript tokens and build label-masked inputs.
    Optionally override pixel_values (for differentiable delta path).
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

    pv = pixel_values if pixel_values is not None else prompt_enc["pixel_values"]

    # image_sizes is required by LlavaNext for AnyRes tile layout
    out = {
        "input_ids":      full_ids,
        "attention_mask": full_attn,
        "pixel_values":   pv,
        "labels":         labels,
        "return_dict":    True,
        "use_cache":      False,
    }
    if "image_sizes" in prompt_enc:
        out["image_sizes"] = prompt_enc["image_sizes"]

    return out


# ── CHECK 0 — model load ──────────────────────────────────────────────────────

def check_load(device, dtype, quantization):
    section("CHECK 0 — model load")
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

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

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, **load_kwargs
    ).eval()

    if quantization is None:
        model = model.to(device)

    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"    {PASS} loaded | quant={quant_label} | vram={mem_gb:.1f} GB")

    if quantization is not None:
        print(f"    {WARN} quantised — ce_proc and CER are indicative only.")
        print(f"    {INFO} rerun in bfloat16 before building wrapper if ce_proc passes.")

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
    if pv is not None:
        if pv.ndim == 5:
            _, n_tiles, c, h, w = pv.shape
            print(f"    {INFO} 5D spatial format: n_tiles={n_tiles}  tile_size=({h},{w})")
            print(f"    {INFO} _preprocess: delta injection over spatial tiles (same as Llama approach)")
        elif pv.ndim == 2:
            print(f"    {WARN} 2D patch-flattened format — unexpected for LLaVA-1.6")

    return enc


# ── CHECK 2 — ce_proc ─────────────────────────────────────────────────────────

def check_ce_proc(model, processor, ds, device):
    section("CHECK 2 — ce_proc (ground truth confidence)")
    print(f"    {INFO} target: > 1.0  |  < 0.5 = model too confident, reject")

    ce_vals = []
    # (category, contrast, ce_val)
    records = []

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
        records.append((s.text_category, s.contrast_level, ce_val))

        tag = PASS if ce_val > 1.0 else (WARN if ce_val > 0.5 else FAIL)
        print(f"    {tag} [{s.image_id}] {s.text_category}/{s.contrast_level}"
              f"  ce_proc={fmt(ce_val)}")

    mean_ce = sum(ce_vals) / len(ce_vals)
    tag = PASS if mean_ce > 1.0 else (WARN if mean_ce > 0.5 else FAIL)
    print(f"\n    {tag} mean_ce_proc = {fmt(mean_ce)}")

    # Category breakdown
    print(f"\n    {'─' * 54}")
    print(f"    {'Category':<22} {'Contrast':<10} {'N':>3}  {'mean_ce':>8}  {'> 1.0':>6}")
    print(f"    {'─' * 54}")

    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for cat, con, ce in records:
        buckets[(cat, con)].append(ce)

    for (cat, con), vals in sorted(buckets.items()):
        m   = sum(vals) / len(vals)
        n_pass = sum(1 for v in vals if v > 1.0)
        tag = PASS if m > 1.0 else (WARN if m > 0.5 else FAIL)
        print(f"    {tag} {cat:<22} {con:<10} {len(vals):>3}  {fmt(m):>8}  {n_pass}/{len(vals):>3}")

    print(f"    {'─' * 54}")

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

    pv_proc = prompt_enc["pixel_values"].to(dtype)  # (1, n_tiles, 3, H, W)

    # Spatial delta injection — same pattern as LlamaVision wrapper.
    # Resize image_tensor to tile spatial size, normalise, compute zero delta.
    H_tile = pv_proc.shape[-2]
    W_tile = pv_proc.shape[-1]
    n_tiles = pv_proc.shape[1]

    x = s.image_tensor.to(device=device, dtype=dtype).unsqueeze(0)
    x = x.detach().requires_grad_(True)  # make image_tensor a leaf with grad
    x_r = F.interpolate(x, size=(H_tile, W_tile), mode="bilinear", align_corners=False)

    ip = processor.image_processor

    def _to3(v):
        return list(v) if isinstance(v, (list, tuple)) else [v, v, v]

    mean_t = torch.tensor(_to3(ip.image_mean), device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t  = torch.tensor(_to3(ip.image_std),  device=device, dtype=dtype).view(1, 3, 1, 1)
    x_norm = (x_r - mean_t) / std_t  # grad-connected to x (leaf)

    delta  = (x_norm - x_norm.detach()).unsqueeze(1).expand(-1, n_tiles, -1, -1, -1)
    delta.retain_grad()
    pv_diff = pv_proc + delta

    model_inputs = build_ce_inputs(prompt_enc, transcript_ids, device, pixel_values=pv_diff)

    try:
        loss = model(**model_inputs).loss
        loss.backward()

        if delta.grad is not None:
            g     = delta.grad.detach().float()
            gmean = g.abs().mean().item()
            gmax  = g.abs().max().item()
            tag   = PASS if gmean > 0 and math.isfinite(gmean) else FAIL
            print(f"    {tag} gradient reached pixel_values delta")
            print(f"    {INFO} grad_mean={fmt(gmean)}  grad_max={fmt(gmax)}")
            print(f"    {INFO} finite: {torch.isfinite(g).all().item()}")
        else:
            print(f"    {FAIL} delta.grad is None — gradient did not reach pixel_values")
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


# ── CHECK 5 — timing ──────────────────────────────────────────────────────────

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
        model_inputs = build_ce_inputs(prompt_enc, transcript_ids, device, pixel_values=pv)

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
    print(f"\n    {INFO} mean per step       : {mean_t:.2f}s")
    print(f"    {INFO} projected 250 steps : {proj:.1f} min / sample")
    print(f"    {tag} {'acceptable' if mean_t < 2.0 else 'slow'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     default="data/synthetic")
    ap.add_argument("--split",        default="debug")
    ap.add_argument("--max-samples",  type=int, default=5)
    ap.add_argument("--image-h",      type=int, default=192)
    ap.add_argument("--image-w",      type=int, default=768)
    ap.add_argument("--quantization", choices=["8bit", "4bit"], default=None,
                    help="Indicative screen only. Rerun bfloat16 before wrapper dev.")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("\n" + "═" * 68)
    print("  LLaVA-1.6-Mistral-7B — Surrogate Suitability Smoke Test")
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

    mean_ce = sum(ce_vals) / len(ce_vals)

    print("\n" + "═" * 68)
    print("  Suitability verdict")
    print("═" * 68)

    if mean_ce > 1.0:
        print(f"  {PASS} ce_proc={fmt(mean_ce)} — good attack surface. Proceed to wrapper.")
    elif mean_ce > 0.5:
        print(f"  {WARN} ce_proc={fmt(mean_ce)} — borderline. Run on more samples first.")
    else:
        print(f"  {FAIL} ce_proc={fmt(mean_ce)} — model too confident. Reject.")

    if args.quantization:
        print(f"\n  {INFO} {args.quantization} quantised run — ce_proc is slightly inflated vs bfloat16.")
        print(f"  {INFO} Rerun without --quantization before building the wrapper.")
    print()


if __name__ == "__main__":
    main()