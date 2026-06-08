"""
phase0_eval.py
==============
Phase 0 evaluation harness. For each (image, surrogate) pair, runs:

  1. answer_query() per entity → Binding Accuracy (Threat 1, primary)
     The model is asked the canonical question for each field.
     Score = how well the answer matches the GT value.

  2. transcribe() → Token Presence (Threat 1, secondary)
     Full transcription is searched for each entity value.
     High score on clean images confirms readability.
     Should STAY high after perturbation (proves human can still read).

  3. transcribe() → Content Fidelity (Threat 2)
     ROUGE-L on content blocks from the same transcription.

Outputs per model:
  <model>_results.csv          one row per (image_id, model)
  <model>_results_full.json    per-entity and per-block diagnostics
  <model>_transcripts.jsonl    raw transcription + per-entity answers

Run one model at a time:
    uv run -m vlm_suppress.tests.phase0_eval \\
        --model qwen2_5vl \\
        --manifest data/ui_dataset/manifest.csv \\
        --gt-dir  data/ui_dataset/ground_truth/pil \\
        --out-dir outputs/phase0_metrics

    # Smoke test (one image per domain)
    uv run ... --one-per-domain

    # DeepSeek (separate venv)
    .venv-deepseek/bin/python -m vlm_suppress.tests.phase0_eval --model deepseekvl2 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    name: str
    model_id: str
    max_new_tokens: int = 1024
    device: str | None = None


DEFAULT_REGISTRY: dict[str, ModelCfg] = {
    "qwen2_5vl":    ModelCfg("qwen2_5vl",    "Qwen/Qwen2.5-VL-7B-Instruct"),
    "internvl3_5":  ModelCfg("internvl3_5",  "OpenGVLab/InternVL3_5-8B"),
    "llama3_2vl":     ModelCfg("llama3_2vl",     "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "paligemma2":   ModelCfg("paligemma2",   "google/paligemma2-10b-mix-448"),
    "deepseekvl2": ModelCfg("deepseekvl2", "deepseek-ai/deepseek-vl2-small"),
}

# Aggressive verbatim prompt for transcription — maximises clean-image
# baseline so perturbation-induced drops are unambiguous.
TRANSCRIBE_PROMPT = (
    "Output the exact text content of this image. "
    "Do not summarize, paraphrase, or interpret. "
    "Reproduce every word exactly as it appears, preserving line breaks. "
    "Output only the text, nothing else."
)


def load_wrapper(model_key: str, cfg: ModelCfg):
    if model_key == "qwen2_5vl":
        from vlm_suppress.models.qwen2_5vl import Qwen2_5VL
        return Qwen2_5VL(cfg)
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
    raise ValueError(f"Unknown model key: {model_key!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_tensor(path: Path) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB"))


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def subset_one_per_domain(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for row in rows:
        dt = row.get("doc_type", "")
        if dt not in seen:
            seen.add(dt)
            result.append(row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "image_id", "doc_type", "model",
    "binding_accuracy",   # primary Threat 1: targeted query score
    "token_presence",     # secondary: entity search in transcription
    "content_fidelity",   # Threat 2: ROUGE-L on content blocks
    "n_entities",
    "n_entities_with_q",  # entities that have a question
    "n_content_blocks",
    "transcribe_time_s",
    "query_time_s",
    "error",
]


def make_csv_row(
    image_id: str,
    doc_type: str,
    model_name: str,
    ba: float | None,      # binding accuracy
    tp: float | None,      # token presence
    cf: float | None,      # content fidelity
    n_ent: int,
    n_ent_q: int,
    n_blocks: int,
    t_transcribe: float,
    t_query: float,
    error: str = "",
) -> dict:
    def _fmt(v):
        return f"{v:.4f}" if v is not None else ""
    return dict(
        image_id=image_id, doc_type=doc_type, model=model_name,
        binding_accuracy=_fmt(ba), token_presence=_fmt(tp),
        content_fidelity=_fmt(cf),
        n_entities=n_ent, n_entities_with_q=n_ent_q,
        n_content_blocks=n_blocks,
        transcribe_time_s=f"{t_transcribe:.2f}",
        query_time_s=f"{t_query:.2f}",
        error=error,
    )


def print_summary(rows: list[dict], model_name: str) -> None:
    from collections import defaultdict
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_domain[r["doc_type"]].append(r)

    def mean(vals):
        v = [float(x) for x in vals if x]
        return sum(v) / len(v) if v else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  n/a "

    print()
    print("=" * 82)
    print(f"  Phase 0 Results — {model_name}")
    print("=" * 82)
    print(f"  {'Domain':<18} {'N':>4}  "
          f"{'Binding Acc':>12}  {'Token Pres':>11}  {'Content Fid':>12}")
    print(f"  {'-'*18} {'-'*4}  {'-'*12}  {'-'*11}  {'-'*12}")

    for domain in sorted(by_domain):
        dr = by_domain[domain]
        ba = mean([r["binding_accuracy"] for r in dr])
        tp = mean([r["token_presence"]   for r in dr])
        cf = mean([r["content_fidelity"] for r in dr])
        err = sum(1 for r in dr if r["error"])
        n_str = f"{len(dr)}" + (f"({err}err)" if err else "")
        print(f"  {domain:<18} {n_str:>4}  "
              f"{fmt(ba):>12}  {fmt(tp):>11}  {fmt(cf):>12}")

    all_ba = mean([r["binding_accuracy"] for r in rows])
    all_tp = mean([r["token_presence"]   for r in rows])
    all_cf = mean([r["content_fidelity"] for r in rows])
    print(f"  {'-'*18} {'-'*4}  {'-'*12}  {'-'*11}  {'-'*12}")
    print(f"  {'OVERALL':<18} {len(rows):>4}  "
          f"{fmt(all_ba):>12}  {fmt(all_tp):>11}  {fmt(all_cf):>12}")
    print("=" * 82)

    print()
    print("  Binding Accuracy = targeted query answer scored vs GT value (primary Threat 1)")
    print("  Token Presence   = entity search in full transcription (secondary / readability)")
    print("  Content Fidelity = ROUGE-L on content blocks (Threat 2)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    cfg = DEFAULT_REGISTRY[args.model]
    if args.model_id:
        cfg.model_id = args.model_id
    if args.device:
        cfg.device = args.device
    cfg.max_new_tokens = args.max_new_tokens

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    rows = load_manifest(manifest_path)
    if args.one_per_domain:
        rows = subset_one_per_domain(rows)
        print(f"[subset] one-per-domain: {len(rows)} images")
    if args.domain:
        rows = [r for r in rows if r["doc_type"] == args.domain]
        print(f"[subset] domain={args.domain}: {len(rows)} images")
    if args.limit:
        rows = rows[:args.limit]
        print(f"[subset] limit={args.limit}: {len(rows)} images")

    print(f"[eval] model={cfg.name}  images={len(rows)}")

    # Load model
    print(f"\n[loading] {cfg.name} ({cfg.model_id})")
    t0 = time.perf_counter()
    model = load_wrapper(args.model, cfg)
    print(f"[loading] done in {time.perf_counter()-t0:.1f}s  device={model.device}")

    from vlm_suppress.metrics import evaluate, GroundTruth
    from vlm_suppress.metrics.comparators import COMPARATORS, EXTRACTORS
    from vlm_suppress.metrics.entity_recall import EntityRecallResult, PerEntityScore

    gt_dir = Path(args.gt_dir)

    csv_path        = out_dir / f"{cfg.name}_results.csv"
    json_path       = out_dir / f"{cfg.name}_results_full.json"
    transcript_path = out_dir / f"{cfg.name}_transcripts.jsonl"

    csv_rows:     list[dict] = []
    full_results: list[dict] = []
    n_ok = n_err = 0

    with open(transcript_path, "w", encoding="utf-8") as tf:

        for idx, row in enumerate(rows):
            image_id   = row["image_id"]
            doc_type   = row["doc_type"]
            image_path = Path(row["image_path"])
            gt_path    = gt_dir / f"{image_id}.json"

            print(f"\n[{idx+1:03d}/{len(rows):03d}] {image_id} ({doc_type})")

            # resolve image path
            if not image_path.exists():
                image_path = manifest_path.parent / row["image_path"]
            if not image_path.exists():
                print(f"  SKIP — image not found")
                n_err += 1
                csv_rows.append(make_csv_row(
                    image_id, doc_type, cfg.name,
                    None, None, None, 0, 0, 0, 0, 0,
                    error="image_not_found"))
                continue

            if not gt_path.exists():
                print(f"  SKIP — GT not found: {gt_path}")
                n_err += 1
                csv_rows.append(make_csv_row(
                    image_id, doc_type, cfg.name,
                    None, None, None, 0, 0, 0, 0, 0,
                    error="gt_not_found"))
                continue

            try:
                gt = GroundTruth.from_json(gt_path)
            except Exception as e:
                print(f"  SKIP — GT invalid: {e}")
                n_err += 1
                csv_rows.append(make_csv_row(
                    image_id, doc_type, cfg.name,
                    None, None, None, 0, 0, 0, 0, 0,
                    error=f"gt_invalid:{e}"))
                continue

            try:
                image_tensor = load_image_tensor(image_path)
            except Exception as e:
                print(f"  SKIP — image load failed: {e}")
                n_err += 1
                csv_rows.append(make_csv_row(
                    image_id, doc_type, cfg.name,
                    None, None, None,
                    len(gt.entities), 0, len(gt.content_blocks),
                    0, 0, error=f"image_load_error:{type(e).__name__}"))
                continue

            # ── Step 1: Transcription (Token Presence + Content Fidelity) ─────
            try:
                t0 = time.perf_counter()
                transcription = model.transcribe(image_tensor, prompt=TRANSCRIBE_PROMPT)
                t_transcribe  = time.perf_counter() - t0
                print(f"  transcribe: {t_transcribe:.1f}s  ({len(transcription)} chars)")
            except Exception as e:
                print(f"  ERROR — transcribe failed: {e}")
                n_err += 1
                csv_rows.append(make_csv_row(
                    image_id, doc_type, cfg.name,
                    None, None, None,
                    len(gt.entities), 0, len(gt.content_blocks),
                    0, 0, error=f"transcribe_error:{type(e).__name__}"))
                continue

            # Token presence: search transcription for each entity
            from vlm_suppress.metrics.entity_recall import entity_recall
            from vlm_suppress.metrics.content_fidelity import content_fidelity

            tp_result = entity_recall(transcription, gt)
            cf_result = content_fidelity(transcription, gt)

            tp_score = tp_result.score
            cf_score = cf_result.score if cf_result else None

            print(f"  token_presence={tp_score:.3f}  "
                  f"content_fidelity={f'{cf_score:.3f}' if cf_score is not None else 'n/a'}")

            # ── Step 2: Targeted queries (Binding Accuracy) ───────────────────
            entities_with_q = [e for e in gt.entities if e.question]
            ba_per_entity: list[dict] = []
            t_query_total = 0.0

            for ent in entities_with_q:
                try:
                    t0  = time.perf_counter()
                    ans = model.answer_query(image_tensor, ent.question)
                    t_query_total += time.perf_counter() - t0

                    matcher   = COMPARATORS.get(ent.type)
                    extractor = EXTRACTORS.get(ent.type)
                    score     = float(matcher(ent.value, ans)) if matcher else 0.0
                    predicted = extractor(ent.value, ans) if extractor else ans[:200]

                    ba_per_entity.append({
                        "label":     ent.label,
                        "type":      ent.type,
                        "question":  ent.question,
                        "answer":    ans,          # raw model answer
                        "target":    ent.value,
                        "predicted": predicted,    # comparator diagnostic
                        "score":     score,
                    })
                except Exception as e:
                    ba_per_entity.append({
                        "label": ent.label, "type": ent.type,
                        "question": ent.question, "answer": "",
                        "target": ent.value, "predicted": "", "score": 0.0,
                        "error": str(e),
                    })

            ba_score = (
                sum(e["score"] for e in ba_per_entity) / len(ba_per_entity)
                if ba_per_entity else None
            )
            print(f"  binding_accuracy={f'{ba_score:.3f}' if ba_score is not None else 'n/a'}"
                  f"  ({len(ba_per_entity)} queries, {t_query_total:.1f}s)")

            # ── Accumulate ─────────────────────────────────────────────────────
            n_ok += 1
            csv_rows.append(make_csv_row(
                image_id, doc_type, cfg.name,
                ba_score, tp_score, cf_score,
                len(gt.entities), len(entities_with_q),
                len(gt.content_blocks),
                t_transcribe, t_query_total,
            ))

            full_result = {
                "image_id":    image_id,
                "doc_type":    doc_type,
                "model":       cfg.name,
                "transcription": transcription,
                "binding_accuracy": {
                    "score":      ba_score,
                    "per_entity": ba_per_entity,
                },
                "token_presence": {
                    "score":      tp_score,
                    "per_entity": [
                        {
                            "label":     e.label,
                            "type":      e.type,
                            "target":    e.target,
                            "predicted": e.predicted,
                            "score":     e.score,
                        }
                        for e in tp_result.per_entity
                    ],
                },
                "content_fidelity": (
                    None if cf_result is None else {
                        "score":     cf_score,
                        "per_block": [
                            {"label": b.label, "rouge_l": b.rouge_l}
                            for b in cf_result.per_block
                        ],
                    }
                ),
            }
            full_results.append(full_result)

            tf.write(json.dumps({
                "image_id":     image_id,
                "doc_type":     doc_type,
                "transcription": transcription,
                "answers":      [
                    {"label": e["label"], "question": e["question"], "answer": e["answer"]}
                    for e in ba_per_entity
                ],
            }, ensure_ascii=False) + "\n")

    # Write outputs
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print_summary(csv_rows, cfg.name)
    print(f"Results:     {csv_path}")
    print(f"Full JSON:   {json_path}")
    print(f"Transcripts: {transcript_path}")
    print(f"  OK={n_ok}  ERR={n_err}  TOTAL={len(rows)}")

    return 0 if n_err == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",    required=True, choices=list(DEFAULT_REGISTRY.keys()))
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-dir",   required=True)
    p.add_argument("--out-dir",  required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--device",   default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--one-per-domain", action="store_true")
    p.add_argument("--domain", default=None,
                   choices=["banking","medical","news","copyright",
                            "legal","identity","communications"])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())