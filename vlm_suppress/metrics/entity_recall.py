"""
Entity Recall (Threat 1): factual leak metric.

For each ground-truth entity, the type-appropriate comparator returns a
match score in [0, 1]. Entity Recall is the mean across entities.

References:
  - Pilán et al. (2022): "The Text Anonymization Benchmark." Used token-level
    recall over attribute values, which we generalize per-entity here.
  - Lison et al. (NeurIPS 2021): "Anonymisation Models for Text Data."
"""

from __future__ import annotations

from dataclasses import dataclass

from vlm_suppress.metrics.comparators import COMPARATORS, EXTRACTORS
from vlm_suppress.metrics.ground_truth import GroundTruth


@dataclass
class PerEntityScore:
    label: str
    type: str
    target: str
    predicted: str  # what the extractor found (or didn't find) in the prediction
    score: float    # [0, 1]


@dataclass
class EntityRecallResult:
    score: float  # mean across entities, in [0, 1]
    per_entity: list[PerEntityScore]

    def by_type(self) -> dict[str, float]:
        """Mean score grouped by entity type — useful for diagnostic tables."""
        from collections import defaultdict
        buckets: dict[str, list[float]] = defaultdict(list)
        for e in self.per_entity:
            buckets[e.type].append(e.score)
        return {t: sum(scores) / len(scores) for t, scores in buckets.items()}

    def n_recovered(self, threshold: float = 1.0) -> int:
        """Number of entities scored at or above `threshold`."""
        return sum(1 for e in self.per_entity if e.score >= threshold)


def entity_recall(prediction: str, gt: GroundTruth) -> EntityRecallResult:
    """
    Compute Entity Recall over a single (prediction, ground-truth) pair.

    Parameters
    ----------
    prediction : the model's transcription output, as a raw string
    gt         : ground-truth annotation for the image

    Returns
    -------
    EntityRecallResult with:
      - score      : mean per-entity score, in [0, 1]
      - per_entity : individual entity scores + what was found, for inspection

    If gt.entities is empty, score is 1.0 by convention.
    """
    per_entity: list[PerEntityScore] = []

    for ent in gt.entities:
        matcher   = COMPARATORS.get(ent.type)
        extractor = EXTRACTORS.get(ent.type)
        if matcher is None:
            raise ValueError(
                f"No comparator registered for entity type {ent.type!r} "
                f"(entity label={ent.label!r}). Add it to COMPARATORS in "
                f"vlm_suppress/metrics/comparators.py."
            )
        score     = float(matcher(ent.value, prediction))
        score     = max(0.0, min(1.0, score))
        predicted = extractor(ent.value, prediction) if extractor else ""
        per_entity.append(
            PerEntityScore(
                label=ent.label,
                type=ent.type,
                target=ent.value,
                predicted=predicted,
                score=score,
            )
        )

    overall = (
        sum(p.score for p in per_entity) / len(per_entity) if per_entity else 1.0
    )
    return EntityRecallResult(score=overall, per_entity=per_entity)