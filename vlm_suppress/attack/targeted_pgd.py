"""
vlm_suppress/attack/targeted_pgd.py

Targeted PGD loop for the ghost-watermark substitution attack.

Minimises targeted_ce_loss over a perturbation δ constrained to an L∞ ε-ball,
starting from the ghost-watermarked image.  Calls evaluate_targeted_substitution
every `eval_every` steps for progress monitoring; stops early on "exact_sub".
"""

from __future__ import annotations

import torch

from vlm_suppress.eval.metrics import evaluate_targeted_substitution
from vlm_suppress.watermark.steplog import StepLogger


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
    logger: StepLogger | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, str | None]:
    """
    Run targeted PGD and return (adv_tensor, final_outcome).

    adv_tensor  : (3, H, W) float32, detached, on same device as wm_tensor
    final_outcome : last evaluated outcome string, or None if never evaluated
    """
    device = wm_tensor.device
    δ = torch.zeros_like(wm_tensor, requires_grad=True)

    last_outcome: str | None = None
    last_transcript: str | None = None

    for step in range(n_steps):
        adv = (wm_tensor + δ).clamp(0.0, 1.0)
        loss = surrogate.targeted_ce_loss(
            adv,
            source_transcript,
            target_transcript,
            source_word,
            target_word,
        )

        loss.backward()

        with torch.no_grad():
            δ_new = δ - step_size * δ.grad.sign()         # gradient descent
            δ_new = δ_new.clamp(-epsilon, epsilon)         # L∞ projection

        δ = δ_new.detach().requires_grad_(True)

        # ── Eval checkpoint ───────────────────────────────────────────────────
        outcome: str | None = None
        transcript: str | None = None
        is_last = step == n_steps - 1
        if (step + 1) % eval_every == 0 or is_last:
            adv_eval = (wm_tensor + δ).clamp(0.0, 1.0).detach()
            result   = evaluate_targeted_substitution(
                adv_eval, source_word, target_word, surrogate
            )
            outcome    = result["outcome"]
            transcript = result["transcript"][:120].replace("\n", " ")
            last_outcome    = outcome
            last_transcript = transcript
            if verbose:
                print(
                    f"  step {step + 1:4d}/{n_steps}"
                    f"  loss={loss.item():.4f}"
                    f"  outcome={outcome}"
                )

        if logger is not None:
            logger.write(step=step, loss=loss, outcome=outcome, transcript=transcript)

        if last_outcome == "exact_sub":
            if verbose:
                print(f"  Early stop: exact_sub at step {step + 1}.")
            break

    adv_tensor = (wm_tensor + δ).clamp(0.0, 1.0).detach()
    return adv_tensor, last_outcome
