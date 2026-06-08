"""
Metric module for VLM-based extraction evaluation.

Two threats, two metrics:
  - Entity Recall  (Threat 1, factual leak): scored by entity_recall.py
  - Content Fidelity (Threat 2, structural/expressive leak): scored by
                     content_fidelity.py via ROUGE-L

See ground_truth.py for the GT schema and comparators.py for the type-aware
matchers used by entity recall.

Top-level entry point:
    from vlm_suppress.metrics import evaluate, GroundTruth
    gt = GroundTruth.from_json("banking_0000_gt.json")
    result = evaluate(prediction_string, gt)
"""

from vlm_suppress.metrics.content_fidelity import (
    ContentFidelityResult,
    content_fidelity,
)
from vlm_suppress.metrics.entity_recall import EntityRecallResult, entity_recall
from vlm_suppress.metrics.evaluate import EvaluationResult, evaluate
from vlm_suppress.metrics.ground_truth import ContentBlock, Entity, GroundTruth
from vlm_suppress.metrics.comparators import COMPARATORS, EXTRACTORS

__all__ = [
    "evaluate",
    "EvaluationResult",
    "entity_recall",
    "EntityRecallResult",
    "content_fidelity",
    "ContentFidelityResult",
    "GroundTruth",
    "Entity",
    "ContentBlock",
    "COMPARATORS",
    "EXTRACTORS",
]