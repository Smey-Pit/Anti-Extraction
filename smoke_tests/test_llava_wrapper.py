#!/usr/bin/env python3
"""
tools/smoke_llava16.py

Smoke test for the LLaVA-1.6-Mistral-7B wrapper (vlm_suppress/models/llava16.py).
Tests the wrapper contract directly rather than the raw model, matching the
pattern established by tools/diag_llama_wrapper.py.

Checks:
    0. Wrapper load
    1. Dataset load
    2. Processor output shape (via wrapper internals)
    3. ce_proc — processor path vs wrapper _preprocess path (ratio target < 1.5)
    4. CE backward — gradient reaches image_tensor
    5. align_loss — finite, non-zero, backprops
    6. transcribe — CER per sample with category breakdown
    7. Single PGD step — output changes

Run:
    uv run python tools/smoke_llava16.py
    uv run python tools/smoke_llava16.py --max-samples 10 --split debug
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass

import torch

sys.path.insert(0, ".")

from vlm_suppress.models.llava import LLaVA16, _tensor_to_pil
from vlm_suppress.data.dataset import TextImageDataset

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "


@dataclass
class Cfg:
    name:           str   = "llava16"
    model_id:       str   = "llava-hf/llava-v1.6-mistral-7b-hf"
    dtype:          str   = "bfloat16"
    device_map:     str   = "auto"
    max_new_tokens: int   = 128
    alpha:          float = 1.0
    device:         str   = ""


def fmt(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.3e}"


def section(title: str):
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")


def cer(pred: str, gold: str) -> float:
    p, g = list(pred), list(gold)
    m, n = len(p), len(g)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if p[i-1] == g[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(len(g), 1)


# ── CHECK 0 — wrapper load ────────────────────────────────────────────────────

def check_load(cfg) -> LLaVA16:
    section("CHECK 0 — wrapper load")
    model = LLaVA16(cfg)
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"    {PASS} loaded | device={model.device} dtype={model._dtype} vram={mem:.1f}GB")
    return model


# ── CHECK 1 — dataset ─────────────────────────────────────────────────────────

def check_dataset(args) -> TextImageDataset:
    section("CHECK 1 — dataset")
    ds = TextImageDataset(
        args.data_dir,
        image_size=(args.image_h, args.image_w),
        max_samples=args.max_samples,
        split_filter=args.split,
    )
    assert len(ds) > 0, "No samples found"
    print(f"    {PASS} {len(ds)} samples | image_size=({args.image_h}, {args.image_w})")
    return ds


# ── CHECK 2 — processor output shape ─────────────────────────────────────────

def check_processor_shape(model: LLaVA16, ds: TextImageDataset):
    section("CHECK 2 — processor output shape")
    s   = ds[0]
    pil = _tensor_to_pil(s.image_tensor)

    with torch.no_grad():
        enc = model._processor_inputs(pil)

    for k, v in enc.items():
        if torch.is_tensor(v):
            print(f"    {INFO} {k}.shape = {tuple(v.shape)}  dtype={v.dtype}")
            if k == "pixel_values":
                print(f"    {INFO} pixel_values.min/max = {v.min():.3f} / {v.max():.3f}")

    pv = enc.get("pixel_values")
    if pv is not None and pv.ndim == 5:
        _, n_tiles, _, h, w = pv.shape
        print(f"    {INFO} n_tiles={n_tiles}  tile_size=({h},{w})")

    return enc


# ── CHECK 3 — ce_proc ratio ───────────────────────────────────────────────────

def check_ce_paths(model: LLaVA16, ds: TextImageDataset):
    section("CHECK 3 — CE loss: processor path vs wrapper _preprocess path")
    print(f"    {INFO} ratio target: < 1.5  |  > 3.0 = _preprocess is wrong")

    for s in ds:
        pil = _tensor_to_pil(s.image_tensor)
        model._pv_proc_cache = None

        transcript_ids = model._transcript_ids(s.transcript)
        t_len          = transcript_ids.size(1)

        # ── Processor path (ground truth) ─────────────────────────────────
        with torch.no_grad():
            prompt_enc = model._processor_inputs(pil)
            full_ids   = torch.cat([prompt_enc["input_ids"], transcript_ids], dim=1)
            full_attn  = torch.cat([
                prompt_enc["attention_mask"],
                torch.ones((1, t_len), device=model.device, dtype=torch.long),
            ], dim=1)
            total_len = full_ids.size(1)
            labels    = torch.full((1, total_len), -100, device=model.device, dtype=torch.long)
            labels[0, -t_len:] = transcript_ids[0]

            out_proc = model.model(
                input_ids=full_ids,
                attention_mask=full_attn,
                pixel_values=prompt_enc["pixel_values"].to(model._dtype),
                image_sizes=prompt_enc["image_sizes"],
                labels=labels,
                return_dict=True,
                use_cache=False,
            )
        ce_proc = out_proc.loss.item()

        # ── Wrapper path ───────────────────────────────────────────────────
        model._pv_proc_cache = None
        with torch.no_grad():
            ce_wrap = model.ce_loss(s.image_tensor, s.transcript).item()

        ratio = ce_wrap / ce_proc if ce_proc > 0 else float("inf")
        tag   = PASS if ratio < 1.5 else (WARN if ratio < 3.0 else FAIL)
        print(
            f"    {tag} [{s.image_id}]  "
            f"ce_proc={fmt(ce_proc)}  ce_wrap={fmt(ce_wrap)}  ratio={ratio:.2f}"
        )

        model._pv_proc_cache = None


# ── CHECK 4 — CE backward ─────────────────────────────────────────────────────

def check_gradient(model: LLaVA16, ds: TextImageDataset):
    section("CHECK 4 — CE backward: gradient reaches image_tensor")

    s = ds[0]
    model._pv_proc_cache = None

    x = s.image_tensor.to(model.device).detach().clone().requires_grad_(True)
    loss = model.ce_loss(x, s.transcript)
    loss.backward()

    assert x.grad is not None, "gradient did not reach image_tensor"
    g = x.grad.detach().float()

    gmean = g.abs().mean().item()
    gmax  = g.abs().max().item()
    gl2   = g.norm(p=2).item()

    finite_tag = PASS if torch.isfinite(g).all() else FAIL
    grad_tag   = PASS if gmean > 1e-8 else FAIL

    print(f"    {grad_tag} gradient reached image_tensor | mean={fmt(gmean)}  max={fmt(gmax)}  l2={fmt(gl2)}")
    print(f"    {finite_tag} gradient is finite everywhere")


# ── CHECK 5 — align_loss ──────────────────────────────────────────────────────

def check_align(model: LLaVA16, ds: TextImageDataset):
    section("CHECK 5 — align_loss")

    for s in [ds[i] for i in range(min(3, len(ds)))]:
        model._pv_proc_cache = None
        x = s.image_tensor.to(model.device).detach().clone().requires_grad_(True)
        align = model.align_loss(x, s.transcript)

        val      = float(align.detach().float().item())
        is_finite  = math.isfinite(val)
        is_nonzero = abs(val) > 1e-6

        tag    = PASS if (is_finite and is_nonzero) else (WARN if is_finite else FAIL)
        detail = f"value={fmt(val)}"

        if align.requires_grad:
            align.backward()
            if x.grad is not None:
                detail += f"  grad_mean={fmt(x.grad.detach().float().abs().mean().item())}"
                print(f"    {tag} [{s.image_id}] {detail}")
            else:
                print(f"    {FAIL} [{s.image_id}] {detail}  [no grad reached image_tensor]")
        else:
            print(f"    {WARN} [{s.image_id}] {detail}  [fallback zero — align disabled]")


# ── CHECK 6 — transcribe + CER ───────────────────────────────────────────────

def check_transcribe(model: LLaVA16, ds: TextImageDataset):
    section("CHECK 6 — transcribe quality (CER) with category breakdown")

    records = []
    for s in ds:
        pred = model.transcribe(s.image_tensor)
        c    = cer(pred, s.transcript)
        tag  = PASS if c < 0.10 else (WARN if c < 0.30 else FAIL)
        print(f"    {tag} [{s.image_id}] {s.text_category}/{s.contrast_level}  CER={c:.3f}")
        print(f"      pred : {pred!r}")
        print(f"      gold : {s.transcript!r}")
        records.append((s.text_category, s.contrast_level, c))

    # Category breakdown
    print(f"\n    {'─' * 54}")
    print(f"    {'Category':<22} {'Contrast':<10} {'N':>3}  {'mean_CER':>8}")
    print(f"    {'─' * 54}")
    buckets: dict[tuple, list] = defaultdict(list)
    for cat, con, c in records:
        buckets[(cat, con)].append(c)
    for (cat, con), vals in sorted(buckets.items()):
        m   = sum(vals) / len(vals)
        tag = PASS if m < 0.10 else (WARN if m < 0.30 else FAIL)
        print(f"    {tag} {cat:<22} {con:<10} {len(vals):>3}  {m:>8.3f}")
    print(f"    {'─' * 54}")


# ── CHECK 7 — single PGD step ─────────────────────────────────────────────────

def check_pgd_step(model: LLaVA16, ds: TextImageDataset):
    section("CHECK 7 — single PGD step (epsilon=4/255)")

    s   = ds[0]
    eps = 4 / 255

    model._pv_proc_cache = None
    x = s.image_tensor.to(model.device).detach().clone().requires_grad_(True)
    loss = model.ce_loss(x, s.transcript)
    loss.backward()

    with torch.no_grad():
        x_adv = (s.image_tensor.to(model.device) + eps * x.grad.sign()).clamp(0, 1)

    pred_clean = model.transcribe(s.image_tensor)
    pred_adv   = model.transcribe(x_adv)
    cer_clean  = cer(pred_clean, s.transcript)
    cer_adv    = cer(pred_adv,   s.transcript)

    changed = pred_adv != pred_clean
    tag     = PASS if changed else WARN
    print(f"    {tag} {'output changed' if changed else 'output unchanged after 1 step (expected at low ce_proc)'}")
    print(f"      clean : {pred_clean!r}  CER={cer_clean:.3f}")
    print(f"      adv   : {pred_adv!r}  CER={cer_adv:.3f}  ΔCER={cer_adv - cer_clean:+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",    default="data/synthetic")
    ap.add_argument("--split",       default="debug")
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument("--image-h",     type=int, default=192)
    ap.add_argument("--image-w",     type=int, default=768)
    args = ap.parse_args()

    print("\n" + "═" * 68)
    print("  LLaVA-1.6-Mistral-7B — Wrapper Smoke Test")
    print("═" * 68)

    try:
        model = check_load(Cfg())
    except Exception as e:
        print(f"  {FAIL} wrapper load failed: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    try:
        ds = check_dataset(args)
    except Exception as e:
        print(f"  {FAIL} dataset load failed: {e}")
        raise SystemExit(1)

    check_processor_shape(model, ds)
    check_ce_paths(model, ds)
    check_gradient(model, ds)
    check_align(model, ds)
    check_transcribe(model, ds)
    check_pgd_step(model, ds)

    print("\n" + "═" * 68)
    print("  Smoke test complete.")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    main()