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
    mask: torch.Tensor | None = None,  # (1, H, W) or (3, H, W); see make_word_mask()
    logger: StepLogger | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, str | None]:
    """
    Run targeted PGD and return (adv_tensor, final_outcome).

    adv_tensor  : (3, H, W) float32, detached, on same device as wm_tensor
    final_outcome : last evaluated outcome string, or None if never evaluated

    If `mask` is provided, δ is zeroed outside the mask region after every
    projection step — all perturbation budget stays inside the word box.
    """
    device = wm_tensor.device
    if mask is not None:
        mask = mask.to(device)

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
            if mask is not None:
                δ_new = δ_new * mask                       # zero outside word box

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
