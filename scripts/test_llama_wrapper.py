#!/usr/bin/env python3
"""
tools/diag_llama_wrapper.py

Full diagnostic for the LlamaVision wrapper.
Structured as independent numbered checks so failures are isolated.

Run:
    uv run python tools/diag_llama_wrapper.py
    uv run python tools/diag_llama_wrapper.py --max-samples 3 --split debug
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from vlm_suppress.models.llama3_2 import LlamaVision, _tensor_to_pil
from vlm_suppress.data.dataset import TextImageDataset

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
INFO  = "ℹ️ "

@dataclass
class Cfg:
    name: str = "llama3_2"
    model_id: str = MODEL_ID
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    alpha: float = 1.0


def fmt(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "nan"
    return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.3e}"


def section(title: str):
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def result(tag: str, label: str, detail: str = ""):
    suffix = f" | {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 0 — model load
# ──────────────────────────────────────────────────────────────────────────────

def check_load(cfg) -> LlamaVision:
    section("CHECK 0 — model load")
    model = LlamaVision(cfg)
    result(PASS, "wrapper loaded", f"device={model.device} dtype={model._dtype}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 1 — dataset
# ──────────────────────────────────────────────────────────────────────────────

def check_dataset(args) -> TextImageDataset:
    section("CHECK 1 — dataset")
    ds = TextImageDataset(
        args.data_dir,
        image_size=(args.image_h, args.image_w),
        max_samples=args.max_samples,
        split_filter=args.split,
    )
    assert len(ds) > 0, "No samples found"
    result(PASS, f"{len(ds)} samples loaded",
           f"image_size=({args.image_h}, {args.image_w})")
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 2 — processor pixel_values shape
# ──────────────────────────────────────────────────────────────────────────────

def check_processor_shape(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 2 — processor pixel_values shape")
    s = ds[0]
    pil = _tensor_to_pil(s.image_tensor)

    with torch.no_grad():
        enc = model.processor(
            text=model._prompt_text,
            images=pil,
            return_tensors="pt",
            add_special_tokens=False,
        )

    pv = enc["pixel_values"]
    result(PASS, f"pixel_values.shape = {tuple(pv.shape)}")
    result(INFO, f"pixel_values.dtype = {pv.dtype}")
    result(INFO, f"pixel_values.min/max = {pv.min():.3f} / {pv.max():.3f}")

    for k in ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"):
        if k in enc:
            result(INFO, f"{k}.shape = {tuple(enc[k].shape)}")
        else:
            result(INFO, f"{k} : not present")

    return enc


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 3 — _preprocess output vs processor output
# ──────────────────────────────────────────────────────────────────────────────

def check_preprocess(model: LlamaVision, ds: TextImageDataset, proc_enc: dict):
    section("CHECK 3 — _preprocess differentiable path vs processor path")
    s = ds[0]
    pil = _tensor_to_pil(s.image_tensor)

    pv_proc = proc_enc["pixel_values"]
    pv_diff = model._preprocess(s.image_tensor, pil).detach().cpu()

    result(INFO, f"processor  pv.shape = {tuple(pv_proc.shape)}")
    result(INFO, f"_preprocess pv.shape = {tuple(pv_diff.shape)}")

    shape_match = pv_proc.shape == pv_diff.shape
    result(
        PASS if shape_match else FAIL,
        "shapes match" if shape_match else f"SHAPE MISMATCH",
    )

    if shape_match:
        pv_proc_f = pv_proc.float()
        pv_diff_f = pv_diff.float()
        mae  = (pv_proc_f - pv_diff_f).abs().mean().item()
        rmax = (pv_proc_f - pv_diff_f).abs().max().item()
        cos  = F.cosine_similarity(
            pv_proc_f.reshape(1, -1),
            pv_diff_f.reshape(1, -1),
        ).item()
        result(INFO, f"MAE  = {mae:.5f}")
        result(INFO, f"max_abs_err = {rmax:.5f}")
        result(INFO, f"cosine_sim  = {cos:.6f}  (target: > 0.999)")

        if cos > 0.999:
            result(PASS, "pixel_values are numerically consistent")
        elif cos > 0.99:
            result(WARN, "pixel_values differ slightly — check tiling logic")
        else:
            result(FAIL, "pixel_values differ significantly — _preprocess is wrong")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 4 — CE with processor pixel_values vs wrapper pixel_values
#           This isolates whether the CE loss is high due to _preprocess
# ──────────────────────────────────────────────────────────────────────────────

def check_ce_paths(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 4 — CE loss: processor path vs _preprocess path")

    for s in ds:
        pil = _tensor_to_pil(s.image_tensor)
        prompt_text = model._prompt_text

        transcript_ids = model._transcript_ids(s.transcript)
        t_len = transcript_ids.size(1)

        prompt_enc  = model._processor_inputs(pil)
        prompt_ids  = prompt_enc["input_ids"]
        prompt_attn = prompt_enc["attention_mask"]

        full_ids  = torch.cat([prompt_ids, transcript_ids], dim=1)
        full_attn = torch.cat(
            [prompt_attn, torch.ones((1, t_len), device=model.device, dtype=torch.long)],
            dim=1,
        )
        total_len = full_ids.size(1)

        labels = torch.full((1, total_len), -100, device=model.device, dtype=torch.long)
        labels[0, -t_len:] = transcript_ids[0]
        if model._image_token_id is not None:
            labels[full_ids == model._image_token_id] = -100

        vision_kw = model._vision_kwargs(prompt_enc)
        if "cross_attention_mask" in vision_kw:
            cam = vision_kw["cross_attention_mask"]
            if cam.size(1) < total_len:
                pad = torch.zeros(
                    cam.size(0), total_len - cam.size(1), *cam.shape[2:],
                    device=model.device, dtype=cam.dtype,
                )
                vision_kw["cross_attention_mask"] = torch.cat([cam, pad], dim=1)

        base_inputs = dict(
            input_ids=full_ids,
            attention_mask=full_attn,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        base_inputs.update(vision_kw)

        # ── Path A: processor's own pixel_values (ground truth) ──
        with torch.no_grad():
            out_proc = model.model(**{
                **base_inputs,
                "pixel_values": prompt_enc["pixel_values"].to(model.device),
            })
        ce_proc = out_proc.loss.item()

        # ── Path B: wrapper _preprocess pixel_values ──
        pv_diff = model._preprocess(s.image_tensor, pil)
        with torch.no_grad():
            out_diff = model.model(**{**base_inputs, "pixel_values": pv_diff})
        ce_diff = out_diff.loss.item()

        ratio = ce_diff / ce_proc if ce_proc > 0 else float("inf")
        tag = PASS if ratio < 1.5 else (WARN if ratio < 3.0 else FAIL)

        print(
            f"    [{s.image_id}]  "
            f"ce_proc={fmt(ce_proc)}  "
            f"ce_diff={fmt(ce_diff)}  "
            f"ratio={ratio:.2f}  {tag}"
        )

    result(INFO, "ratio target: < 1.5  |  > 3.0 = _preprocess is wrong")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 5 — CE naïve baseline (no image — pixel_values all zeros)
#           Establishes how much of the CE is explained by prior alone
# ──────────────────────────────────────────────────────────────────────────────

def check_ce_baseline(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 5 — CE baseline: blank image (tests label masking + prior)")

    for s in [ds[i] for i in range(min(2, len(ds)))]:
        pil = _tensor_to_pil(s.image_tensor)
        prompt_enc = model._processor_inputs(pil)

        transcript_ids = model._transcript_ids(s.transcript)
        t_len = transcript_ids.size(1)
        prompt_ids  = prompt_enc["input_ids"]
        prompt_attn = prompt_enc["attention_mask"]

        full_ids  = torch.cat([prompt_ids, transcript_ids], dim=1)
        full_attn = torch.cat(
            [prompt_attn, torch.ones((1, t_len), device=model.device, dtype=torch.long)],
            dim=1,
        )
        total_len = full_ids.size(1)

        labels = torch.full((1, total_len), -100, device=model.device, dtype=torch.long)
        labels[0, -t_len:] = transcript_ids[0]
        if model._image_token_id is not None:
            labels[full_ids == model._image_token_id] = -100

        vision_kw = model._vision_kwargs(prompt_enc)
        if "cross_attention_mask" in vision_kw:
            cam = vision_kw["cross_attention_mask"]
            if cam.size(1) < total_len:
                pad = torch.zeros(
                    cam.size(0), total_len - cam.size(1), *cam.shape[2:],
                    device=model.device, dtype=cam.dtype,
                )
                vision_kw["cross_attention_mask"] = torch.cat([cam, pad], dim=1)

        # blank image — same shape as processor output
        blank_pv = torch.zeros_like(prompt_enc["pixel_values"]).to(model.device)

        base = dict(
            input_ids=full_ids, attention_mask=full_attn,
            labels=labels, return_dict=True, use_cache=False,
        )
        base.update(vision_kw)

        with torch.no_grad():
            ce_real  = model.model(**{**base, "pixel_values": prompt_enc["pixel_values"].to(model.device)}).loss.item()
            ce_blank = model.model(**{**base, "pixel_values": blank_pv}).loss.item()

        delta = ce_blank - ce_real
        tag = PASS if delta > 0.5 else (WARN if delta > 0.0 else FAIL)
        print(
            f"    [{s.image_id}]  "
            f"ce_real={fmt(ce_real)}  "
            f"ce_blank={fmt(ce_blank)}  "
            f"delta={fmt(delta)}  {tag}"
        )

    result(INFO, "ce_blank should be > ce_real — image must lower loss vs blank")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 6 — gradient flow through _preprocess
# ──────────────────────────────────────────────────────────────────────────────

def check_gradient_flow(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 6 — gradient flow: image_tensor → pixel_values → loss")

    s = ds[0]
    x = s.image_tensor.to(model.device).detach().clone().requires_grad_(True)

    loss = model.ce_loss(x, s.transcript)
    loss.backward()

    assert x.grad is not None, "gradient did not reach image_tensor"
    g = x.grad.detach().float()

    grad_mean = g.abs().mean().item()
    grad_max  = g.abs().max().item()
    grad_l2   = g.norm(p=2).item()

    tag = PASS if grad_mean > 1e-6 else FAIL
    result(tag, "gradient reached image_tensor",
           f"mean_abs={fmt(grad_mean)}  max={fmt(grad_max)}  l2={fmt(grad_l2)}")

    # Check gradient is not NaN/Inf
    finite_tag = PASS if torch.isfinite(g).all() else FAIL
    result(finite_tag, "gradient is finite everywhere")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 7 — align_loss
# ──────────────────────────────────────────────────────────────────────────────

def check_align(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 7 — align_loss")

    for s in ds:
        x = s.image_tensor.to(model.device).detach().clone().requires_grad_(True)
        align = model.align_loss(x, s.transcript)

        val = float(align.detach().float().item())
        is_finite = math.isfinite(val)
        is_nonzero = abs(val) > 1e-6

        tag = PASS if (is_finite and is_nonzero) else (WARN if is_finite else FAIL)
        detail = f"value={fmt(val)}"

        if align.requires_grad:
            align.backward()
            if x.grad is not None:
                gmean = x.grad.detach().float().abs().mean().item()
                detail += f"  grad_mean={fmt(gmean)}"
                result(tag, s.image_id, detail)
            else:
                result(FAIL, s.image_id, detail + "  [backward: no grad reached image]")
        else:
            result(WARN, s.image_id, detail + "  [fallback zero — align disabled]")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 8 — transcribe quality (CER)
# ──────────────────────────────────────────────────────────────────────────────

def cer(pred: str, gold: str) -> float:
    """Character Error Rate via edit distance."""
    p, g = list(pred), list(gold)
    m, n = len(p), len(g)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if p[i-1] == g[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(len(g), 1)


def check_transcribe(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 8 — transcribe quality (CER)")

    for s in ds:
        pred = model.transcribe(s.image_tensor)
        c = cer(pred, s.transcript)
        tag = PASS if c < 0.10 else (WARN if c < 0.30 else FAIL)
        print(f"    [{s.image_id}]  CER={c:.3f}  {tag}")
        print(f"      pred : {pred!r}")
        print(f"      gold : {s.transcript!r}")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 9 — single PGD step sanity
# ──────────────────────────────────────────────────────────────────────────────

def check_pgd_step(model: LlamaVision, ds: TextImageDataset):
    section("CHECK 9 — single PGD step (epsilon=4/255)")

    s = ds[0]
    eps = 4 / 255

    x = s.image_tensor.to(model.device).detach().clone().requires_grad_(True)
    loss = model.ce_loss(x, s.transcript)
    loss.backward()

    with torch.no_grad():
        x_adv = (s.image_tensor.to(model.device) + eps * x.grad.sign()).clamp(0, 1)

    pred_clean = model.transcribe(s.image_tensor)
    pred_adv   = model.transcribe(x_adv)
    cer_clean  = cer(pred_clean, s.transcript)
    cer_adv    = cer(pred_adv,   s.transcript)
    delta_cer  = cer_adv - cer_clean

    changed = pred_adv != pred_clean
    tag = PASS if changed else WARN

    result(tag, "adversarial output differs from clean" if changed else "output unchanged after PGD step")
    print(f"      clean pred : {pred_clean!r}  CER={cer_clean:.3f}")
    print(f"      adv   pred : {pred_adv!r}  CER={cer_adv:.3f}  ΔCER={delta_cer:+.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     default="data/synthetic")
    ap.add_argument("--split",        default="debug")
    ap.add_argument("--max-samples",  type=int, default=3)
    ap.add_argument("--image-h",      type=int, default=192)
    ap.add_argument("--image-w",      type=int, default=768)
    args = ap.parse_args()

    print("\n" + "═" * 72)
    print("  LlamaVision — Full Wrapper Diagnostic")
    print("═" * 72)

    cfg   = Cfg()
    model = check_load(cfg)
    ds    = check_dataset(args)
    enc   = check_processor_shape(model, ds)
    check_preprocess(model, ds, enc)
    check_ce_paths(model, ds)
    check_ce_baseline(model, ds)
    check_gradient_flow(model, ds)
    check_align(model, ds)
    check_transcribe(model, ds)
    check_pgd_step(model, ds)

    print("\n" + "═" * 72)
    print("  Diagnostic complete.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()