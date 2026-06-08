"""
vlm_suppress/attack/pgd.py — Projected Gradient Descent for adversarial perturbation.

Model-agnostic PGD loop. Takes a callable loss_fn(x) → scalar tensor and
optimizes the perturbation δ under an L∞ constraint.

Sign-based update (FGSM-style steps) is used throughout — this is standard
for L∞ attacks and more stable than raw gradient magnitude steps.

Convention
----------
  - image_tensor : (3, H, W) float32 [0, 1]
  - perturbation : (3, H, W) float32, constrained to [-epsilon, epsilon]
  - all operations in float32 regardless of model dtype
"""

from __future__ import annotations

import torch


def pgd(
    image_tensor: torch.Tensor,       # (3, H, W) float32 [0,1]
    loss_fn: callable,                # loss_fn(x) -> scalar tensor with grad
    epsilon: float,                   # L∞ budget (e.g. 8/255)
    alpha: float,                     # step size (e.g. epsilon / 10)
    n_steps: int,                     # PGD iterations
    targeted: bool = False,           # False = maximize loss, True = minimize loss
    random_init: bool = True,         # random start within L∞ ball
    verbose: bool = False,
    eval_fn: callable | None = None,  # optional eval_fn(x_adv, step) for logging
    eval_every: int = 10,
) -> tuple[torch.Tensor, list[dict]]:
    """
    PGD attack under L∞ constraint.

    Parameters
    ----------
    image_tensor : clean image, (3, H, W) float32 [0, 1]
    loss_fn      : differentiable loss function, loss_fn(x_adv) -> scalar
    epsilon      : L∞ perturbation budget
    alpha        : step size per iteration
    n_steps      : number of gradient steps
    targeted     : if True, minimize loss (pull toward target)
                   if False, maximize loss (push away from current output)
    random_init  : start from random point in L∞ ball (more robust than zero init)
    verbose      : print loss every eval_every steps
    eval_fn      : optional callable(x_adv, step) -> dict for logging metrics
    eval_every   : evaluate every N steps

    Returns
    -------
    x_adv : (3, H, W) float32 [0,1] — perturbed image
    log   : list of dicts with step-by-step metrics
    """
    device = image_tensor.device
    x_orig = image_tensor.detach().clone().float()

    # Initialise perturbation
    if random_init:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x_orig)

    delta = torch.clamp(x_orig + delta, 0, 1) - x_orig
    delta.requires_grad_(True)

    log = []

    for step in range(n_steps):
        # Forward pass
        x_adv = torch.clamp(x_orig + delta, 0, 1)
        # x_adv needs grad for loss_fn — detach delta, reattach via x_adv
        x_adv = x_adv.detach().requires_grad_(True)

        loss = loss_fn(x_adv)

        # Gradient step
        loss.backward()
        grad = x_adv.grad.detach()

        with torch.no_grad():
            if targeted:
                # Minimize loss — step in negative gradient direction
                delta_update = delta.detach() - alpha * grad.sign()
            else:
                # Maximize loss — step in positive gradient direction
                delta_update = delta.detach() + alpha * grad.sign()

            # L∞ projection
            delta_update = torch.clamp(delta_update, -epsilon, epsilon)

            # Image bounds projection
            delta_update = torch.clamp(x_orig + delta_update, 0, 1) - x_orig

        delta = delta_update.requires_grad_(True)

        # Logging
        if verbose and (step % eval_every == 0 or step == n_steps - 1):
            loss_val = loss.item()
            print(f"  step {step:4d}/{n_steps}  loss={loss_val:.4f}", end="")

            if eval_fn is not None and (step % eval_every == 0 or step == n_steps - 1):
                x_eval = torch.clamp(x_orig + delta.detach(), 0, 1)
                metrics = eval_fn(x_eval, step)
                log.append({"step": step, "loss": loss_val, **metrics})
                metric_str = "  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                               if isinstance(v, float))
                print(metric_str, end="")
            else:
                log.append({"step": step, "loss": loss_val})

            print()

    x_adv = torch.clamp(x_orig + delta.detach(), 0, 1)
    return x_adv, log