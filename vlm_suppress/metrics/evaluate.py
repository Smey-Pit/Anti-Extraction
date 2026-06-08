"""
Top-level evaluation entry point.

Combines Entity Recall (Threat 1) and Content Fidelity (Threat 2) into a
single result object for one (prediction, ground-truth) pair.

CLI usage:
    python -m vlm_suppress.metrics.evaluate \\
        --prediction-file path/to/prediction.txt \\
        --gt-file path/to/banking_0000_gt.json

The CLI is mainly for one-off testing of the metric against a stored
transcription output. The eval harness that runs across (image x model)
matrices lives elsewhere and imports `evaluate` programmatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from vlm_suppress.metrics.content_fidelity import (
    ContentFidelityResult,
    content_fidelity,
)
from vlm_suppress.metrics.entity_recall import EntityRecallResult, entity_recall
from vlm_suppress.metrics.ground_truth import GroundTruth


@dataclass
class EvaluationResult:
    entity_recall: EntityRecallResult
    content_fidelity: ContentFidelityResult | None

    def to_dict(self) -> dict:
        return {
            "entity_recall": {
                "score": self.entity_recall.score,
                "per_entity": [asdict(p) for p in self.entity_recall.per_entity],
                "by_type": self.entity_recall.by_type(),
            },
            "content_fidelity": (
                None
                if self.content_fidelity is None
                else {
                    "score": self.content_fidelity.score,
                    "per_block": [asdict(p) for p in self.content_fidelity.per_block],
                }
            ),
        }


def evaluate(prediction: str, gt: GroundTruth) -> EvaluationResult:
    """
    Run both metrics on one (prediction, ground-truth) pair.

    Parameters
    ----------
    prediction : raw transcription string from the VLM
    gt         : GroundTruth instance for the corresponding image

    Returns
    -------
    EvaluationResult with both scorers' outputs. .content_fidelity is None
    when the GT has no content_blocks (pure-field domains like identity).
    """
    return EvaluationResult(
        entity_recall=entity_recall(prediction, gt),
        content_fidelity=content_fidelity(prediction, gt),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one (prediction, ground-truth) pair."
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        required=True,
        help="Path to a text file containing the model's transcription output.",
    )
    parser.add_argument(
        "--gt-file",
        type=Path,
        required=True,
        help="Path to the ground-truth JSON file for the corresponding image.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Optional output JSON file. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    if not args.prediction_file.exists():
        print(f"ERROR: prediction file not found: {args.prediction_file}", file=sys.stderr)
        return 2

    prediction = args.prediction_file.read_text(encoding="utf-8")
    gt = GroundTruth.from_json(args.gt_file)
    result = evaluate(prediction, gt)

    payload = {
        "doc_type": gt.doc_type,
        "image_path": gt.image_path,
        "prediction_file": str(args.prediction_file),
        "result": result.to_dict(),
    }

    output = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out_file:
        args.out_file.write_text(output, encoding="utf-8")
        print(f"Results written to: {args.out_file}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(_main())