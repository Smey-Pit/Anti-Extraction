#!/usr/bin/env python3
"""
run_llama_smoke.py

Stronger surrogate smoke test for:
    meta-llama/Llama-3.2-11B-Vision-Instruct

Checks:
1) model loading / device / dtype
2) clean OCR sanity on a focused sample slice
3) teacher-forced CE viability
4) explicit gradient flow to pixel_values
5) local CE-only PGD attackability on processed pixel_values
6) cheap feature-path probe for future alignment work

This is still a smoke test:
- batch size = 1
- minimal dependencies
- easy to run on cluster

Recommended default run:
    uv run python scripts/run_llama_smoke.py \
      --data-dir data/synthetic \
      --split debug \
      --categories semi_structured,natural_phrase \
      --max-eval 10
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

sys.path.insert(0, ".")

from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.eval.metrics import compute_cer


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_PROMPT = "Transcribe exactly all visible text. Output only the text, nothing else."


@dataclass
class SampleResult:
    image_id: str
    category: str
    contrast: str

    cer_clean: float
    ce_clean: float
    grad_mean_abs: float
    grad_max_abs: float
    grad_l2: float

    ce_adv: float
    ce_delta: float
    cer_adv: float
    cer_delta: float

    pred_clean: str
    pred_adv: str
    gold: str


def fmt(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "nan"
    if abs(x) >= 1e-2:
        return f"{x:.4f}"
    return f"{x:.3e}"


def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return statistics.median(xs)


def normalize_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*assistant\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^Transcribe exactly all visible text\. Output only the text, nothing else\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def to_pil(sample) -> Image.Image:
    arr = (sample.image_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def build_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def make_prompt_text(processor, prompt_text: str) -> str:
    return processor.apply_chat_template(
        build_messages(prompt_text),
        add_generation_prompt=True,
        tokenize=False,
    )


def get_image_token_id(processor):
    tok = processor.tokenizer
    for candidate in ["<|image|>", "<image>"]:
        try:
            tid = tok.convert_tokens_to_ids(candidate)
            if tid is not None and tid != tok.unk_token_id:
                return tid
        except Exception:
            pass
    return None


def move_inputs_to_device(d: dict[str, Any], device: str) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_processor_inputs(processor, sample, prompt_text: str):
    pil = to_pil(sample)
    prompt = make_prompt_text(processor, prompt_text)

    prompt_inputs = processor(
        text=prompt,
        images=pil,
        return_tensors="pt",
    )

    full_text = prompt + sample.transcript
    full_inputs = processor(
        text=full_text,
        images=pil,
        return_tensors="pt",
    )
    return prompt_inputs, full_inputs


@torch.no_grad()
def generate_from_inputs(
    model,
    processor,
    prompt_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> str:
    out = model.generate(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    gen_tokens = out[0, prompt_len:]
    pred = processor.decode(gen_tokens, skip_special_tokens=True)
    return normalize_prediction(pred)


def compute_ce_and_grad(
    model,
    processor,
    full_inputs: dict[str, torch.Tensor],
    prompt_inputs: dict[str, torch.Tensor],
    device: str,
    verbose_warnings: bool = True,
):
    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs["attention_mask"].to(device)

    pixel_values = full_inputs["pixel_values"].to(device)
    pixel_values = pixel_values.detach().clone().requires_grad_(True)

    labels = input_ids.clone()

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    labels[attention_mask == 0] = -100

    image_token_id = get_image_token_id(processor)
    if image_token_id is not None:
        labels[input_ids == image_token_id] = -100

    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
        "return_dict": True,
        "use_cache": False,
    }

    # Required Mllama image-side fields if present
    for k in ["aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"]:
        if k in full_inputs:
            model_kwargs[k] = full_inputs[k].to(device)

    outputs = model(**model_kwargs)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("Model forward returned loss=None")

    model.zero_grad(set_to_none=True)
    if pixel_values.grad is not None:
        pixel_values.grad.zero_()

    loss.backward()

    grad = pixel_values.grad
    if grad is None:
        if verbose_warnings:
            print("    WARNING: gradient did not reach pixel_values — possible broken vision/resampler path")
        return {
            "ce": float(loss.detach().float().item()),
            "grad_exists": False,
            "grad_mean_abs": 0.0,
            "grad_max_abs": 0.0,
            "grad_l2": 0.0,
            "pixel_values": pixel_values.detach(),
            "labels": labels.detach(),
            "forward_kwargs": {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in model_kwargs.items()
                if k not in {"labels", "pixel_values"}
            },
        }

    grad_f = grad.detach().float()
    return {
        "ce": float(loss.detach().float().item()),
        "grad_exists": True,
        "grad_mean_abs": float(grad_f.abs().mean().item()),
        "grad_max_abs": float(grad_f.abs().max().item()),
        "grad_l2": float(grad_f.norm(p=2).item()),
        "pixel_values": pixel_values.detach(),
        "labels": labels.detach(),
        "forward_kwargs": {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in model_kwargs.items()
            if k not in {"labels", "pixel_values"}
        },
    }


def pgd_attack_on_pixel_values(
    model,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    forward_kwargs: dict[str, Any],
    steps: int,
    step_size: float,
    epsilon: float,
):
    """
    Tiny CE-only PGD directly on processed pixel_values.
    This is a model-level exploitability probe, not the final surrogate wrapper.
    """
    x0 = pixel_values.detach().clone()
    x = x0.clone()

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        kwargs = dict(forward_kwargs)
        kwargs["pixel_values"] = x
        kwargs["labels"] = labels
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True

        outputs = model(**kwargs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Attack probe forward returned loss=None")

        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            g = x.grad
            if g is None:
                raise RuntimeError("Attack probe gradient vanished (pixel_values.grad is None)")
            x = x + step_size * g.sign()
            x = torch.max(torch.min(x, x0 + epsilon), x0 - epsilon)

    with torch.no_grad():
        kwargs = dict(forward_kwargs)
        kwargs["pixel_values"] = x
        kwargs["labels"] = labels
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        adv_outputs = model(**kwargs)
        adv_ce = float(adv_outputs.loss.detach().float().item())

    return x.detach(), adv_ce


def prompt_inputs_with_adv_pixels(
    prompt_inputs: dict[str, torch.Tensor],
    adv_pixel_values: torch.Tensor,
    device: str,
):
    out = {}
    for k, v in prompt_inputs.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    out["pixel_values"] = adv_pixel_values
    return out


def probe_feature_access(
    model,
    processor,
    sample,
    prompt_text: str,
    device: str,
    probe_hidden_states: bool = False,
):
    prompt_inputs, _ = build_processor_inputs(processor, sample, prompt_text)
    prompt_inputs = move_inputs_to_device(prompt_inputs, device)

    report = {}
    report["has_get_image_features"] = hasattr(model, "get_image_features")
    report["top_level_attrs"] = [
        name for name in ["model", "vision_model", "vision_tower", "vision_encoder"]
        if hasattr(model, name)
    ]

    nested = {}
    if hasattr(model, "model"):
        inner = model.model
        for name in ["vision_model", "vision_tower", "vision_encoder"]:
            nested[name] = hasattr(inner, name)
    report["nested_model_attrs"] = nested

    try:
        with torch.no_grad():
            outputs = model(
                **prompt_inputs,
                return_dict=True,
                use_cache=False,
                output_hidden_states=probe_hidden_states,
            )
        report["forward_ok"] = True
        report["output_type"] = type(outputs).__name__

        if probe_hidden_states:
            hs = getattr(outputs, "hidden_states", None)
            report["hidden_states_available"] = hs is not None
            if hs is not None:
                report["n_hidden_layers"] = len(hs)
                report["last_hidden_state_shape"] = tuple(hs[-1].shape)
        else:
            report["hidden_states_available"] = "not_probed"
    except Exception as e:
        report["forward_ok"] = False
        report["forward_error"] = repr(e)
        report["hidden_states_available"] = False

    return report


def select_dataset(args):
    ds = TextImageDataset(
        args.data_dir,
        image_size=(args.image_h, args.image_w),
        max_samples=args.max_samples,
        split_filter=args.split,
        category_filter=None,
        contrast_filter=args.contrast if args.contrast != "any" else None,
    )

    selected = []
    allowed_categories = None
    if args.categories:
        allowed_categories = {x.strip() for x in args.categories.split(",") if x.strip()}

    for s in ds:
        if allowed_categories is not None and s.text_category not in allowed_categories:
            continue
        selected.append(s)

    if args.max_eval is not None:
        selected = selected[: args.max_eval]
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--split", default="debug")
    ap.add_argument("--contrast", default="any", choices=["any", "low", "medium", "high"])
    ap.add_argument(
        "--categories",
        default="semi_structured,natural_phrase",
        help="comma-separated categories to include",
    )
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--max-eval", type=int, default=10)
    ap.add_argument("--image-h", type=int, default=192)
    ap.add_argument("--image-w", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=128)

    # Local attackability probe defaults: stronger than 1/255 smoke test
    ap.add_argument("--attack-steps", type=int, default=5)
    ap.add_argument("--attack-step-size", type=float, default=0.004)
    ap.add_argument("--attack-epsilon", type=float, default=0.03137255)  # 8/255

    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--probe-hidden-states", action="store_true")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "cuda" in device else torch.float32

    print("\n" + "═" * 72)
    print("  Llama 3.2 Vision Strong Surrogate Smoke Test")
    print("═" * 72 + "\n")

    print("── 0. Load model / processor ──")
    print(f"  model_id={MODEL_ID}")
    print(f"  device={device}")
    print(f"  dtype={dtype}")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            token=True,
            low_cpu_mem_usage=True,
        ).eval().to(device)
    except Exception as e:
        print(f"\n❌ LOAD FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    print(f"  loaded | model={type(model).__name__}")

    print("\n── 1. Dataset ──")
    dataset = select_dataset(args)
    print(f"  selected {len(dataset)} samples")
    print(f"  image_size=({args.image_h}, {args.image_w})")
    print(f"  categories={args.categories}")
    print(f"  contrast={args.contrast}")

    if len(dataset) == 0:
        print("\n❌ No samples selected.")
        raise SystemExit(1)

    print("\n── 2. Feature path probe ──")
    feat = probe_feature_access(
        model=model,
        processor=processor,
        sample=dataset[0],
        prompt_text=args.prompt,
        device=device,
        probe_hidden_states=args.probe_hidden_states,
    )
    for k, v in feat.items():
        print(f"  {k}: {v}")

    print("\n── 3. Per-sample evaluation ──")
    rows: list[SampleResult] = []
    n_sample_fail = 0

    for sample in dataset:
        try:
            prompt_inputs, full_inputs = build_processor_inputs(processor, sample, args.prompt)

            clean_prompt_inputs = move_inputs_to_device(prompt_inputs, device)
            pred_clean = generate_from_inputs(model, processor, clean_prompt_inputs, args.max_new_tokens)
            cer_clean = float(compute_cer(pred_clean, sample.transcript))

            ce_info = compute_ce_and_grad(
                model=model,
                processor=processor,
                full_inputs=full_inputs,
                prompt_inputs=prompt_inputs,
                device=device,
                verbose_warnings=True,
            )
            ce_clean = ce_info["ce"]

            adv_pixel_values, ce_adv = pgd_attack_on_pixel_values(
                model=model,
                pixel_values=ce_info["pixel_values"],
                labels=ce_info["labels"],
                forward_kwargs=ce_info["forward_kwargs"],
                steps=args.attack_steps,
                step_size=args.attack_step_size,
                epsilon=args.attack_epsilon,
            )

            adv_prompt_inputs = prompt_inputs_with_adv_pixels(prompt_inputs, adv_pixel_values, device)
            pred_adv = generate_from_inputs(model, processor, adv_prompt_inputs, args.max_new_tokens)
            cer_adv = float(compute_cer(pred_adv, sample.transcript))

            row = SampleResult(
                image_id=sample.image_id,
                category=sample.text_category,
                contrast=sample.contrast_level,
                cer_clean=cer_clean,
                ce_clean=float(ce_clean),
                grad_mean_abs=float(ce_info["grad_mean_abs"]),
                grad_max_abs=float(ce_info["grad_max_abs"]),
                grad_l2=float(ce_info["grad_l2"]),
                ce_adv=float(ce_adv),
                ce_delta=float(ce_adv - ce_clean),
                cer_adv=cer_adv,
                cer_delta=float(cer_adv - cer_clean),
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                gold=sample.transcript,
            )
            rows.append(row)

            print(
                f"  {row.image_id} | cat={row.category} | contrast={row.contrast} | "
                f"CER {fmt(row.cer_clean)} -> {fmt(row.cer_adv)} (Δ={fmt(row.cer_delta)}) | "
                f"CE {fmt(row.ce_clean)} -> {fmt(row.ce_adv)} (Δ={fmt(row.ce_delta)}) | "
                f"grad_mean={fmt(row.grad_mean_abs)} | grad_l2={fmt(row.grad_l2)}"
            )
            print(f"    clean={row.pred_clean!r}")
            print(f"    adv  ={row.pred_adv!r}")
            print(f"    gold ={row.gold!r}")

        except Exception as e:
            n_sample_fail += 1
            print(f"  {sample.image_id} | ❌ sample failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n── 4. Summary ──")
    if not rows:
        print("❌ No successful sample evaluations.")
        raise SystemExit(1)

    clean_cers = [r.cer_clean for r in rows]
    clean_ces = [r.ce_clean for r in rows]
    grad_means = [r.grad_mean_abs for r in rows]
    grad_l2s = [r.grad_l2 for r in rows]

    adv_ces = [r.ce_adv for r in rows]
    ce_deltas = [r.ce_delta for r in rows]
    adv_cers = [r.cer_adv for r in rows]
    cer_deltas = [r.cer_delta for r in rows]

    n_clean_bad050 = sum(x > 0.50 for x in clean_cers)
    n_ce_positive = sum(x > 0 for x in clean_ces)
    n_grad_nonzero = sum(x > 1e-8 for x in grad_means)

    n_ce_delta_pos = sum(x > 0 for x in ce_deltas)
    n_ce_delta_big = sum(x > 0.1 for x in ce_deltas)
    n_cer_delta_pos = sum(x > 0 for x in cer_deltas)
    n_cer_delta_big = sum(x > 0.05 for x in cer_deltas)

    print(f"  successful_samples={len(rows)}")
    print(f"  failed_samples={n_sample_fail}")
    print()
    print(f"  clean_mean_cer={fmt(mean(clean_cers))}")
    print(f"  clean_median_cer={fmt(median(clean_cers))}")
    print(f"  clean_>0.50={n_clean_bad050}/{len(rows)}")
    print()
    print(f"  clean_mean_ce={fmt(mean(clean_ces))}")
    print(f"  clean_median_ce={fmt(median(clean_ces))}")
    print(f"  ce_positive={n_ce_positive}/{len(rows)}")
    print()
    print(f"  mean_grad_abs={fmt(mean(grad_means))}")
    print(f"  median_grad_abs={fmt(median(grad_means))}")
    print(f"  mean_grad_l2={fmt(mean(grad_l2s))}")
    print(f"  grad_nonzero={n_grad_nonzero}/{len(rows)}")
    print()
    print(f"  adv_mean_ce={fmt(mean(adv_ces))}")
    print(f"  mean_ce_delta={fmt(mean(ce_deltas))}")
    print(f"  ce_delta_positive={n_ce_delta_pos}/{len(rows)}")
    print(f"  ce_delta_>0.1={n_ce_delta_big}/{len(rows)}")
    print()
    print(f"  adv_mean_cer={fmt(mean(adv_cers))}")
    print(f"  mean_cer_delta={fmt(mean(cer_deltas))}")
    print(f"  cer_delta_positive={n_cer_delta_pos}/{len(rows)}")
    print(f"  cer_delta_>0.05={n_cer_delta_big}/{len(rows)}")

    print("\n── 5. Decision logic ──")
    print("  Clean OCR target: not degenerate, ideally mean CER ~0.02-0.35")
    print("  CE clean target: non-degenerate and not dirty on clean data")
    print("  Gradient target: nonzero on almost all samples")
    print("  Attackability target: small CE-PGD should raise CE on most samples")
    print("  Strong-surrogate bonus: small CE-PGD should also degrade generated OCR on some samples")

    feature_ok = bool(
        feat.get("has_get_image_features", False)
        or feat.get("forward_ok", False)
        or any(feat.get("nested_model_attrs", {}).values())
        or feat.get("hidden_states_available", False) is True
    )

    clean_ok = (mean(clean_cers) <= 0.35) and (n_clean_bad050 <= 2)
    not_overperfect = mean(clean_cers) >= 0.02

    ce_clean_ok = (
        n_ce_positive == len(rows)
        and math.isfinite(mean(clean_ces))
        and 0.01 < mean(clean_ces) < 8.0
    )

    grad_ok = (
        n_grad_nonzero >= max(len(rows) - 1, 1)
        and mean(grad_means) > 1e-8
        and math.isfinite(mean(grad_means))
    )

    ce_attackable_ok = (
        n_ce_delta_pos >= math.ceil(0.8 * len(rows))
        and mean(ce_deltas) > 0.05
    )

    strong_attack_signal = (
        n_cer_delta_pos >= math.ceil(0.2 * len(rows))
        and mean(cer_deltas) > 0.01
    )

    print("\n── 6. Verdict ──")
    if clean_ok and ce_clean_ok and grad_ok and ce_attackable_ok and strong_attack_signal and feature_ok and not_overperfect:
        print("✅ STRONG PASS: strong surrogate candidate.")
        print("   Rationale: clean OCR is in-range, CE is usable, gradients are real, CE is attackable, and OCR degrades under small self-attack.")
    elif clean_ok and ce_clean_ok and grad_ok and ce_attackable_ok and feature_ok:
        print("✅ PASS: plausible surrogate candidate.")
        print("   Rationale: CE and gradients are good, and local CE-PGD works. Transfer value should be tested next.")
    elif ce_clean_ok and grad_ok and ce_attackable_ok:
        print("⚠️ PARTIAL PASS: usable CE surrogate, but complementarity still unclear.")
        print("   Rationale: optimization signal is real, but OCR may be too perfect or degradation is limited.")
    else:
        print("❌ FAIL: not a good surrogate under current setup.")
        print("   Rationale: CE/gradient/attackability signal is too weak or unstable.")

    print("\nRecommended next step:")
    if ce_clean_ok and grad_ok and ce_attackable_ok:
        print("  Run a tiny K=2 vs K=3 probe on hard categories and especially the K=2 joint-failure subset.")
    else:
        print("  Fix wrapper/gradient path before ensemble integration.")


if __name__ == "__main__":
    main()