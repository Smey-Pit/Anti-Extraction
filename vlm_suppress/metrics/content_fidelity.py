"""
Content Fidelity (Threat 2): structural / expressive leak metric.

For each ground-truth content block, ROUGE-L is computed with the block as
the reference and the prediction as the hypothesis. Per-block scores are
averaged. Per-block diagnostics are exposed for inspection.

References:
  - Lin (2004): "ROUGE: A Package for Automatic Evaluation of Summaries."
  - Hans et al. (2024) "Be like a Goldfish, Don't Memorize!" — uses ROUGE-L
    for verbatim-memorization measurement in LLMs.

Returns None (not a score) when gt.content_blocks is empty, indicating
that this metric is not applicable to the document type. Callers must
handle the None case explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from rouge_score import rouge_scorer

from vlm_suppress.metrics.ground_truth import GroundTruth


@dataclass
class PerBlockScore:
    label: str
    rouge_l: float  # [0, 1] — F-measure of longest common subsequence


@dataclass
class ContentFidelityResult:
    score: float  # mean ROUGE-L across blocks, in [0, 1]
    per_block: list[PerBlockScore]


# Module-level scorer, instantiated once.
# use_stemmer=False to match the LLM-memorization literature convention
# (verbatim-leaning); stemming would over-credit paraphrase.
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def content_fidelity(prediction: str, gt: GroundTruth) -> ContentFidelityResult | None:
    """
    Compute Content Fidelity over a single (prediction, ground-truth) pair.

    We report ROUGE-L **recall** (not F-measure). Rationale: the threat is
    leakage of the protected content. If the prediction reproduces the
    full content block surrounded by extra context, that's a complete leak
    even though F-measure (which factors in precision) would be reduced by
    the extra output. Recall measures exactly what the threat model cares
    about: "what fraction of the protected content appears in the
    prediction." This matches the Goldfish-loss paper (Hans et al. 2024)
    and the near-verbatim recall metric in Ahmed et al. (2025).

    Parameters
    ----------
    prediction : the model's transcription output, as a raw string
    gt         : ground-truth annotation for the image

    Returns
    -------
    ContentFidelityResult with:
      - score     : mean per-block ROUGE-L recall, in [0, 1]
      - per_block : individual block ROUGE-L recall for diagnostic inspection
    None
      if gt.content_blocks is empty (this metric is not applicable to the
      document type).
    """
    if not gt.content_blocks:
        return None

    per_block: list[PerBlockScore] = []
    for block in gt.content_blocks:
        # rouge_score takes (target, prediction) — target is the reference
        scores = _SCORER.score(block.text, prediction)
        rouge_l = scores["rougeL"].recall  # recall, not fmeasure
        per_block.append(PerBlockScore(label=block.label, rouge_l=rouge_l))

    overall = sum(p.rouge_l for p in per_block) / len(per_block)
    return ContentFidelityResult(score=overall, per_block=per_block)