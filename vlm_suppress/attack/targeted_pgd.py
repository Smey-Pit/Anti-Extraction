"""
vlm_suppress/attack/targeted_pgd.py

Targeted PGD loop for the ghost-watermark substitution attack.

Loss design — full-transcript cross-entropy with span upweighting:
  Two teacher-forced forward passes per step (target and source transcripts).
  All T tokens contribute gradient; the substitution span is upweighted by
  span_weight (default 5.0) to keep focused pressure on the target word.
  This avoids the 1-token gradient sparsity that caused bf16 quantisation
  artifacts in the span-only formulation.
"""

from __future__ import annotations

import torch

from vlm_suppress.attack.importance import _align_tokens_to_words, _get_tokenizer
from vlm_suppress.eval.metrics import evaluate_targeted_substitution
from vlm_suppress.watermark.steplog import StepLogger


def make_word_mask(
    tensor_shape: tuple[int, int, int],   # (C, H, W)
    box: list[float],                      # [x0, y0, x1, y1] in tensor pixel coords
    padding: int = 4,
) -> torch.Tensor:
    """
    Binary mask: 1.0 inside the padded word bounding box, 0.0 elsewhere.

    Returns a (1, H, W) float tensor that broadcasts across channels.
    Concentrates the PGD budget on the target-word region, leaving the
    rest of the image unperturbed.
    """
    C, H, W = tensor_shape
    x0, y0, x1, y1 = box
    px0 = max(0, int(x0) - padding)
    py0 = max(0, int(y0) - padding)
    px1 = min(W, int(x1) + padding)
    py1 = min(H, int(y1) + padding)
    mask = torch.zeros(1, H, W)
    mask[:, py0:py1, px0:px1] = 1.0
    return mask   # caller moves to device


def run_targeted_pgd(
    wm_tensor: torch.Tensor,        # (3, H, W), [0,1], on device — ghost-watermarked
    source_transcript: str,
    target_transcript: str,
    source_word: str,
    target_word: str,
    surrogate,                       # SurrogateModel with targeted_ce_loss
    n_steps: int = 200,
    epsilon: float = 8 / 255,
    step_size: float = 1 / 255,
    eval_every: int = 25,
    span_weight: float = 5.0,        # upweight on substitution token span
    mask: torch.Tensor | None = None,
    logger: StepLogger | None = None,
    verbose: bool = True,
    delta_init: torch.Tensor | None = None,  # warm-start δ from a previous model's run
) -> tuple[torch.Tensor, str | None]:
    """
    Run targeted PGD and return (adv_tensor, final_outcome).

    adv_tensor  : (3, H, W) float32, detached, on same device as wm_tensor
    final_outcome : last evaluated outcome string, or None if never evaluated
    """
    device = wm_tensor.device
    if mask is not None:
        mask = mask.to(device)

    # ── Precompute substitution spans for upweighting (once, not per step) ────
    tokenizer = _get_tokenizer(surrogate)
    tgt_span: tuple[int, int] | None = None
    src_span: tuple[int, int] | None = None
    if tokenizer is not None:
        tgt_words = target_transcript.split()
        src_words = source_transcript.split()
        tgt_idx = next(
            (i for i, w in enumerate(tgt_words) if w.strip(".,:-") == target_word), None
        )
        src_idx = next(
            (i for i, w in enumerate(src_words) if w.strip(".,:-") == source_word), None
        )
        if tgt_idx is not None:
            tgt_span = _align_tokens_to_words(
                tokenizer, target_transcript, len(tgt_words)
            )[tgt_idx]
        if src_idx is not None:
            src_span = _align_tokens_to_words(
                tokenizer, source_transcript, len(src_words)
            )[src_idx]
    if verbose:
        print(f"  spans: {source_word}={src_span}  {target_word}={tgt_span}"
              f"  span_weight={span_weight}")

    δ_init = delta_init.detach().clamp(-epsilon, epsilon).to(device) if delta_init is not None \
        else torch.zeros_like(wm_tensor)
    δ = δ_init.requires_grad_(True)
    last_outcome: str | None = None

    for step in range(n_steps):
        adv = (wm_tensor + δ).clamp(0.0, 1.0)

        # ── Two forward passes → per-token log_probs ─────────────────────────
        lp_tgt = surrogate.targeted_ce_loss(adv, target_transcript)   # (T_tgt,)
        lp_src = surrogate.targeted_ce_loss(adv, source_transcript)   # (T_src,)

        T_tgt, T_src = lp_tgt.shape[0], lp_src.shape[0]

        # ── Span-upweighted full-transcript loss ──────────────────────────────
        w_tgt = torch.ones(T_tgt, device=device)
        w_src = torch.ones(T_src, device=device)
        if tgt_span is not None:
            s, e = tgt_span
            e = min(e, T_tgt)
            if s < T_tgt:
                w_tgt[s:e] = span_weight
        if src_span is not None:
            s, e = src_span
            e = min(e, T_src)
            if s < T_src:
                w_src[s:e] = span_weight

        L_attract = -(lp_tgt * w_tgt).sum()    # maximise p(target_transcript)
        L_repel   =  (lp_src * w_src).sum()    # minimise p(source_transcript)
        loss = L_attract + L_repel

        loss.backward()

        # ── Gradient sanity check at step 0 ───────────────────────────────────
        if step == 0 and verbose:
            g = δ.grad
            print(f"  grad sanity: max={g.abs().max():.6f}  mean={g.abs().mean():.6f}")

        with torch.no_grad():
            δ_new = δ - step_size * δ.grad.sign()         # gradient descent
            δ_new = δ_new.clamp(-epsilon, epsilon)         # L∞ projection
            if mask is not None:
                δ_new = δ_new * mask                       # zero outside word box

        δ = δ_new.detach().requires_grad_(True)

        # ── Eval checkpoint ───────────────────────────────────────────────────
        outcome: str | None = None
        transcript_out: str | None = None
        is_last = step == n_steps - 1
        if (step + 1) % eval_every == 0 or is_last:
            adv_eval = (wm_tensor + δ).clamp(0.0, 1.0).detach()
            result   = evaluate_targeted_substitution(
                adv_eval, source_word, target_word, surrogate
            )
            outcome        = result["outcome"]
            transcript_out = result["transcript"]   # full transcript — no truncation
            last_outcome   = outcome
            if verbose:
                short = transcript_out[:200].replace("\n", " ")
                print(
                    f"  step {step + 1:4d}/{n_steps}"
                    f"  loss={loss.item():.4f}"
                    f"  outcome={outcome}"
                )
                print(f"            {short!r}")

        if logger is not None:
            logger.write(step=step, loss=loss, outcome=outcome, transcript=transcript_out)

        if last_outcome == "exact_sub":
            if verbose:
                print(f"  Early stop: exact_sub at step {step + 1}.")
            break

    adv_tensor = (wm_tensor + δ).clamp(0.0, 1.0).detach()
    return adv_tensor, last_outcome
