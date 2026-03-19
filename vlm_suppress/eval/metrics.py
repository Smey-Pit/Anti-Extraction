"""
Evaluation metrics.

All metrics accept raw strings — no model-specific preprocessing.
CER/WER via jiwer; exact match is simple normalised string equality.
GapGain requires pre-computed human accuracy (from user study CSV).
"""

from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer, wer


def normalise(s: str) -> str:
    """Lowercase, strip extra whitespace. Applied before all string metrics."""
    return " ".join(s.lower().split())


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate. Lower = better extraction."""
    h, r = normalise(hypothesis), normalise(reference)
    if not r:
        return 0.0 if not h else 1.0
    return float(cer(r, h))


def compute_wer(hypothesis: str, reference: str) -> float:
    h, r = normalise(hypothesis), normalise(reference)
    if not r:
        return 0.0 if not h else 1.0
    return float(wer(r, h))


def compute_exact_match(hypothesis: str, reference: str) -> bool:
    return normalise(hypothesis) == normalise(reference)


def compute_char_accuracy(hypothesis: str, reference: str) -> float:
    """1 - CER, clipped to [0, 1]."""
    return max(0.0, 1.0 - compute_cer(hypothesis, reference))


@dataclass
class ModelMetrics:
    """Metrics for one model on one image."""
    image_id: str
    model_name: str
    transcript_clean: str
    transcript_adv: str
    reference: str

    cer_clean: float
    cer_adv:   float
    wer_clean: float
    wer_adv:   float
    exact_clean: bool
    exact_adv:   bool

    cer_delta: float    # cer_adv - cer_clean: positive = more degraded = suppression success
    wer_delta: float

    @classmethod
    def compute(
        cls,
        image_id: str,
        model_name: str,
        transcript_clean: str,
        transcript_adv: str,
        reference: str,
    ) -> "ModelMetrics":
        cc = compute_cer(transcript_clean, reference)
        ca = compute_cer(transcript_adv,   reference)
        wc = compute_wer(transcript_clean, reference)
        wa = compute_wer(transcript_adv,   reference)
        return cls(
            image_id=image_id,
            model_name=model_name,
            transcript_clean=transcript_clean,
            transcript_adv=transcript_adv,
            reference=reference,
            cer_clean=cc, cer_adv=ca,
            wer_clean=wc, wer_adv=wa,
            exact_clean=compute_exact_match(transcript_clean, reference),
            exact_adv=compute_exact_match(transcript_adv, reference),
            cer_delta=ca - cc,
            wer_delta=wa - wc,
        )


def compute_gap_gain(
    h_clean: float,    # human char accuracy on clean images
    h_adv: float,      # human char accuracy on perturbed images
    m_clean: float,    # machine char accuracy on clean images
    m_adv: float,      # machine char accuracy on perturbed images
    h_adv_floor: float = 0.85,  # minimum acceptable h_adv; results below this are flagged
) -> dict:
    """
    GapGain = (H_adv - M_adv) - (H_clean - M_clean)

    Returns a dict with the value and a flag if h_adv is below the floor.
    """
    gap_gain = (h_adv - m_adv) - (h_clean - m_clean)
    return {
        "gap_gain": gap_gain,
        "h_clean": h_clean,
        "h_adv": h_adv,
        "m_clean": m_clean,
        "m_adv": m_adv,
        "h_adv_below_floor": h_adv < h_adv_floor,
    }


def compute_transfer_ratio(
    whitebox_cer_delta: float,
    heldout_cer_delta: float,
) -> float:
    """
    Transfer ratio = held-out delta / white-box delta.
    Values near 1.0 indicate strong transfer.
    """
    if abs(whitebox_cer_delta) < 1e-6:
        return 0.0
    return heldout_cer_delta / whitebox_cer_delta
