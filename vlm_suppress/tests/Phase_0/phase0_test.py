"""
Phase 0 smoke + sanity test for surrogate wrappers.

Tests one surrogate at a time:

  1. SMOKE   : loads the model, runs transcribe() and answer_query() on a
               test image. Verifies no crashes, non-empty output, gradient
               machinery is absent (no requires_grad leakage), reasonable
               wall-clock time.

  2. SANITY  : prints the raw output of transcribe() (truncated in stdout,
               full to file) and answer_query() for 4 hardcoded questions
               against the banking test image, so you can read the
               outputs and judge clean-image baseline quality.

Run one model per invocation (loading all five at once OOMs single-GPU
allocations on Spartan). Use the SLURM wrapper to fan out across models.

Usage
-----
    uv run python -m vlm_suppress.tests.phase0_test \\
        --model llava1_6 \\
        --image /path/to/banking_0000.png \\
        --out-dir outputs/phase0_test/llava1_6

    # or, if not running as a module
    python tests/phase0_test.py --model qwen2_5vl --image ... --out-dir ...

Supported --model values: qwen2_5vl, llava1_6, internvl3_5, llama3_2, paligemma2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


# ── Model registry ────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    name: str
    model_id: str
    max_new_tokens: int = 1024
    device: str | None = None  # None → auto cuda:0 / cpu
    device_map: str | None = None  # only honoured by LLaVA wrapper


# Default checkpoints. Override on the command line if your HPC cache uses
# different paths or you want a smaller variant.
DEFAULT_REGISTRY: dict[str, ModelCfg] = {
    "qwen2_5vl":    ModelCfg("qwen2_5vl",    "Qwen/Qwen2.5-VL-7B-Instruct"),
    "llava1_6":        ModelCfg("llava1_6",        "llava-hf/llava-v1.6-mistral-7b-hf"),
    "internvl3_5":  ModelCfg("internvl3_5",  "OpenGVLab/InternVL3_5-8B"),
    "llama3_2vl":     ModelCfg("llama3_2vl",     "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "paligemma2":   ModelCfg("paligemma2",   "google/paligemma2-10b-pt-896"),
    "deepseekvl2": ModelCfg("deepseek_vl2", "deepseek-ai/deepseek-vl2-small"),
}


def load_wrapper(model_key: str, cfg: ModelCfg):
    """Import and instantiate the requested wrapper. Lazy import keeps each
    model's dependencies isolated — a broken paligemma install doesn't
    block testing of qwen, etc."""
    if model_key == "qwen2_5vl":
        from vlm_suppress.models.qwen2_5vl import Qwen2_5VL
        return Qwen2_5VL(cfg)
    if model_key == "llava1_6":
        from vlm_suppress.models.llava1_6 import LLaVA16
        return LLaVA16(cfg)
    if model_key == "internvl3_5":
        from vlm_suppress.models.internvl3_5 import InternVL35
        return InternVL35(cfg)
    if model_key == "llama3_2vl":
        from vlm_suppress.models.llama3_2vl import LlamaVision
        return LlamaVision(cfg)
    if model_key == "paligemma2":
        from vlm_suppress.models.paligemma2 import PaliGemma2
        return PaliGemma2(cfg)
    if model_key == "deepseekvl2":
        from vlm_suppress.models.deepseekvl2 import DeepSeekVL2
        return DeepSeekVL2(cfg)
    raise ValueError(f"Unknown model key: {model_key}")


# ── Test inputs ───────────────────────────────────────────────────────────

# Hardcoded field-extraction questions for the banking test image
# (banking_0000.png — FuturaBank statement for Ella Thompson).
# These are the "clean baseline" probes for field-level accuracy.
TEST_QUESTIONS: list[tuple[str, str]] = [
    ("account_holder",   "What is the account holder name? Answer with just the name."),
    ("account_number",   "What is the account number? Answer with just the number."),
    ("account_type",     "What is the account type? Answer with just the type."),
    ("opening_balance",  "What is the opening balance? Answer with just the amount."),
]


# ── Pretty-printing helpers ───────────────────────────────────────────────

def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def section(s: str) -> None:
    print()
    print("-" * 78)
    print(s)
    print("-" * 78)


def truncate(s: str, n: int = 500) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n...[truncated, total {len(s)} chars]"


# ── GPU memory & timing ───────────────────────────────────────────────────

def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── Image loading ─────────────────────────────────────────────────────────

def load_image_tensor(path: Path) -> torch.Tensor:
    """PNG/JPG → (3, H, W) float32 [0,1]. No augmentation."""
    img = Image.open(path).convert("RGB")
    return to_tensor(img)  # float32, [0,1], (C, H, W)


# ── Smoke + sanity tests ──────────────────────────────────────────────────

def run_smoke(model, image_tensor: torch.Tensor) -> dict:
    """Verify the wrapper runs end-to-end and produces non-trivial output."""
    results = {}

    section("SMOKE: transcribe() default prompt")
    reset_peak_mem()
    t0 = time.perf_counter()
    out_default = model.transcribe(image_tensor)
    dt = time.perf_counter() - t0
    print(f"  wall-clock: {dt:.2f}s   peak GPU: {gpu_mem_mb():.0f} MB")
    print(f"  output ({len(out_default)} chars):")
    print(truncate(out_default))
    assert isinstance(out_default, str), "transcribe must return str"
    assert len(out_default.strip()) > 0, "transcribe returned empty string"
    results["transcribe_default"] = {
        "output": out_default, "wall_clock_s": dt, "peak_gpu_mb": gpu_mem_mb(),
    }

    section("SMOKE: transcribe() with explicit prompt override")
    custom_prompt = (
        "Read every visible character in this image. "
        "Output the text only, with line breaks preserved."
    )
    reset_peak_mem()
    t0 = time.perf_counter()
    out_custom = model.transcribe(image_tensor, prompt=custom_prompt)
    dt = time.perf_counter() - t0
    print(f"  wall-clock: {dt:.2f}s   peak GPU: {gpu_mem_mb():.0f} MB")
    print(f"  output ({len(out_custom)} chars):")
    print(truncate(out_custom))
    assert isinstance(out_custom, str)
    assert len(out_custom.strip()) > 0
    results["transcribe_custom"] = {
        "output": out_custom, "wall_clock_s": dt, "peak_gpu_mb": gpu_mem_mb(),
    }

    return results


def run_sanity(model, image_tensor: torch.Tensor) -> dict:
    """Run the field-extraction probes and print raw outputs for manual
    inspection. No automated correctness check here — that's the job of
    the eval harness in the next step."""
    results = {}

    section("SANITY: answer_query() — field extraction probes")
    for field_name, question in TEST_QUESTIONS:
        reset_peak_mem()
        t0 = time.perf_counter()
        ans = model.answer_query(image_tensor, question)
        dt = time.perf_counter() - t0
        print(f"\n  Q [{field_name}]: {question}")
        print(f"  A: {ans!r}")
        print(f"  ({dt:.2f}s, peak {gpu_mem_mb():.0f} MB)")
        assert isinstance(ans, str), "answer_query must return str"
        results[field_name] = {
            "question": question, "answer": ans,
            "wall_clock_s": dt, "peak_gpu_mb": gpu_mem_mb(),
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Phase 0 surrogate smoke + sanity test")
    p.add_argument("--model", required=True, choices=list(DEFAULT_REGISTRY.keys()))
    p.add_argument("--data-path", required=True, help="Path to the jsonl that contains the image dataset")
    p.add_argument("--category", required=True, nargs="+", help="Category of dataset to run", choices=['banking', 'medical', 'news', 'copyright', 'legal', 'identity', 'communications'])
    p.add_argument("--num-sample", required=False, type=int, default=1, help="Sample size for image to test")
    p.add_argument("--image", required=False, type=Path, help="Path to test image (PNG/JPG)")
    p.add_argument("--out-dir", type=Path, default=Path("./phase0_results"),
                   help="Where to write per-model JSON + full transcribe outputs")
    p.add_argument("--model-id", default=None,
                   help="Override the HF model id (use this if your HPC has a non-standard cache)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--device", default=None, help="e.g. cuda:0, cpu — default auto")
    args = p.parse_args()
    
    
    script_dir = Path(__file__).resolve().parent
    
    #save directory
    args.out_dir = args.out_dir / args.model
    args.out_dir.mkdir(parents=True, exist_ok=True)

    #different cat to test on
    cats = args.category
    print(cats)
    #loading jsonl 
    df = pd.read_json(args.data_path, lines=True)
    
    #retrieving the images
    filtered_df = df[df['category'].isin(cats)]

    # 2. Group by category, then take the top 10 rows of EACH group
    dataset_image = filtered_df.groupby('category').head(args.num_sample)
    
    desired_columns = ['image_id', 'image_path', 'category', 'full_text']
    
    dataset_image = dataset_image[desired_columns]


    cfg = DEFAULT_REGISTRY[args.model]
    if args.model_id is not None:
        cfg.model_id = args.model_id
    cfg.max_new_tokens = args.max_new_tokens
    cfg.device = args.device
    
   
    banner(f"PHASE 0 TEST — {cfg.name}  ({cfg.model_id})")
    print(f"image dataset path:        {args.data_path}")
    print(f"device hint:  {cfg.device or 'auto'}")
    print(f"max_new_tok:  {cfg.max_new_tokens}")
    print(f"torch:        {torch.__version__}")
    if torch.cuda.is_available():
        print(f"cuda:         {torch.version.cuda}, gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda:         not available — running on CPU (will be SLOW)")

    section(f"LOADING {cfg.name}")
    t0 = time.perf_counter()
    reset_peak_mem()
    model = load_wrapper(args.model, cfg)
    load_dt = time.perf_counter() - t0
    print(f"  loaded in {load_dt:.1f}s    peak GPU: {gpu_mem_mb():.0f} MB")
    print(f"  device:    {model.device}")
    
    for row in dataset_image.itertuples():
        image_path = script_dir.parent.parent / 'data' / 'ui_dataset' / row.image_path
        section("LOADING IMAGE")
        image_tensor = load_image_tensor(image_path)
        print(f"  tensor shape: {tuple(image_tensor.shape)}  dtype: {image_tensor.dtype}")
        print(f"  range: [{image_tensor.min().item():.3f}, {image_tensor.max().item():.3f}]")

        # Run tests. Wrapped in try/except so partial results are saved even if
        # a later test crashes — useful for debugging on Spartan where you may
        # not get a second chance at a long-running allocation.
        all_results = {
            "model": cfg.name,
            "model_id": cfg.model_id,
            "image": str(args.image),
            "load_time_s": load_dt,
            "torch_version": torch.__version__,
            "smoke": None,
            "sanity": None,
            "error": None,
        }

        try:
            all_results["smoke"]  = run_smoke(model, image_tensor)
            all_results["sanity"] = run_sanity(model, image_tensor)
        except Exception as e:
            import traceback
            all_results["error"] = {"type": type(e).__name__, "msg": str(e),
                                    "traceback": traceback.format_exc()}
            print()
            print("!!! TEST FAILED !!!")
            traceback.print_exc()

        # Persist results: JSON for the structured data, plus a separate
        # transcribe-output file because full transcripts are long and clutter
        # the JSON when reading by eye.
        json_path = args.out_dir / f"{row.image_id}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {json_path}")

        if all_results["smoke"] is not None:
            txt_path = args.out_dir / f"{row.image_id}_transcripts.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== {row.image_id} — default prompt transcription ===\n\n")
                f.write(all_results["smoke"]["transcribe_default"]["output"])
                f.write(f"\n\n=== {row.image_id} — custom prompt transcription ===\n\n")
                f.write(all_results["smoke"]["transcribe_custom"]["output"])
            print(f"Full transcripts written to: {txt_path}")

        if all_results["error"] is not None:
            return 0
    return 1
            


if __name__ == "__main__":
    sys.exit(main())