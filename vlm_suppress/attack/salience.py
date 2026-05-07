# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/attack/salience.py
#
# Salience-conditioned perturbation budget map.
#
# Instead of assigning a flat epsilon to all text pixels, this module
# measures how sensitive each surrogate's CE loss is to local pixel changes
# (via the input-gradient L2 norm), then allocates a higher l-inf budget
# to pixels the ensemble finds most salient.
#
# Output contract
# ───────────────
#   Returns a (1, H, W) float32 tensor on `device` — same shape and dtype
#   as build_epsilon_map in masks.py, so it drops in directly as the eps_map
#   argument to project_onto_region_ball.
#
# Budget semantics
# ─────────────────
#   non-text pixels  →  epsilon_bg            (constant)
#   text pixels      →  epsilon_min + (epsilon_max − epsilon_min) · Ŝ
#                         where Ŝ ∈ [0, 1] is the 99th-pctile-normalised
#                         weighted ensemble salience, gated to text regions
#
# Caller contract
# ───────────────
#   • image_tensor must NOT have requires_grad — cloned internally.
#   • alpha_weights must sum to 1.0 (not enforced; caller's responsibility).
#   • epsilon_bg ≤ epsilon_min ≤ epsilon_max for the self-test assertions to
#     hold (not enforced here).
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch

from vlm_suppress.attack.masks import build_text_mask
from vlm_suppress.models.base import SurrogateModel


def build_salience_budget_map(
    image_tensor:  torch.Tensor,        # (1, 3, H, W) float32 on device, requires_grad=False
    transcript:    str,
    word_boxes:    list[list[int]],     # [[x0,y0,x1,y1], ...] already scaled to H, W
    surrogates:    list,                # list[SurrogateModel | LazySurrogate]
    alpha_weights: list[float],         # one per surrogate, must sum to 1.0
    epsilon_min:   float,               # budget floor  for text pixels
    epsilon_max:   float,               # budget ceiling for text pixels
    epsilon_bg:    float,               # budget for non-text pixels
    dilation:      int,                 # passed unchanged to build_text_mask
    device:        torch.device,
) -> torch.Tensor:                      # (1, H, W) float32
    """
    Build a salience-conditioned per-pixel epsilon budget map.

    Steps
    ─────
    1. For each surrogate k, compute input-gradient salience:
         a. Clone image_tensor, enable requires_grad.
         b. Forward: loss = f_k.ce_loss(clone[0], transcript)
         c. Backward to populate clone.grad.
         d. S_k = clone.grad.norm(dim=1, keepdim=True).squeeze(1)  # (1,H,W)
         e. Detach, move to CPU, clear per-iteration graph.

    2. Weighted aggregate on CPU:
         S = Σ_k  alpha_k · S_k          # (1, H, W)

    3. Normalise to [0, 1] via 99th-percentile clipping:
         p99    = quantile(S, 0.99)
         S_tilde = clamp(S, max=p99) / (p99 + 1e-8)

    4. Gate to text regions:
         text_mask = build_text_mask(H, W, word_boxes, dilation)
         S_tilde  *= text_mask            # zero outside bounding boxes

    5. Build budget map:
         E = epsilon_bg · ones(1, H, W)
         E[text_mask > 0] = epsilon_min + (epsilon_max − epsilon_min) · S_tilde[text_mask > 0]

    Returns E on `device`.
    """
    from vlm_suppress.models.lazy import LazySurrogate

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]

    # ── Step 1 & 2: per-surrogate salience, aggregated on CPU ─────────────────
    S = torch.zeros(1, H, W, dtype=torch.float32)

    for surrogate, alpha in zip(surrogates, alpha_weights):
        print(f"  [salience] {surrogate.name} ...")
        # Fresh leaf per surrogate — isolates the grad graph across iterations.
        img_clone = image_tensor.clone().detach().requires_grad_(True)

        if isinstance(surrogate, LazySurrogate):
            with surrogate as model:
                loss = model.ce_loss(img_clone[0], transcript)
                loss.backward()
        else:
            loss = surrogate.ce_loss(img_clone[0], transcript)
            loss.backward()

        if img_clone.grad is not None:
            # Norm over channel dim: (1,3,H,W) → (1,1,H,W) → (1,H,W)
            S_k = (
                img_clone.grad
                .norm(dim=1, keepdim=True)
                .squeeze(1)
                .detach()
                .cpu()
            )
        else:
            # Guard: surrogate broke the grad graph — contribute zeros.
            S_k = torch.zeros(1, H, W, dtype=torch.float32)

        S = S + alpha * S_k
        # img_clone goes out of scope; computation graph freed automatically.

    # ── Step 3: normalise to [0, 1] via 99th-percentile clipping ──────────────
    p99     = torch.quantile(S, 0.99)
    S_tilde = S.clamp(max=p99) / (p99 + 1e-8)

    # ── Step 4: gate to text regions ───────────────────────────────────────────
    text_mask = build_text_mask(
        H, W, word_boxes, dilation, device=torch.device("cpu")
    )                                       # (1, H, W) on CPU, values in {0, 1}
    S_tilde = S_tilde * text_mask

    # ── Step 5: build budget map ───────────────────────────────────────────────
    E = torch.full((1, H, W), epsilon_bg, dtype=torch.float32)

    text_pixels = text_mask > 0            # (1, H, W) bool
    E[text_pixels] = (
        epsilon_min
        + (epsilon_max - epsilon_min) * S_tilde[text_pixels]
    )

    return E.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def test_salience_budget() -> None:
    """
    Smoke test for build_salience_budget_map.

    Uses two dummy surrogates backed by nn.Linear layers.  Because the linear
    weights are randomly initialised (different seeds per surrogate) and vary
    across spatial positions, the two salience maps are distinct and
    spatially non-uniform — the budget map inside text regions will show a
    spread between epsilon_min and epsilon_max.
    """
    device = torch.device("cpu")
    H, W   = 64, 64

    # ── Dummy surrogate ────────────────────────────────────────────────────────
    class _DummySurrogate(SurrogateModel):
        """
        Linear map  (3·H·W,) → scalar.
        Gives a real autograd graph so img_clone.grad is non-trivially
        populated across all spatial positions.
        """
        def __init__(self, seed: int) -> None:
            torch.manual_seed(seed)
            self._device = device
            self.name    = f"dummy_{seed}"
            self._fc     = torch.nn.Linear(3 * H * W, 1)

        @property
        def device(self) -> torch.device:
            return self._device

        def ce_loss(
            self,
            image_tensor: torch.Tensor,   # (3, H, W), requires_grad
            transcript:   str,
        ) -> torch.Tensor:                # scalar
            return self._fc(image_tensor.reshape(1, -1)).squeeze()

        def align_loss(
            self,
            image_tensor: torch.Tensor,
            transcript:   str,
        ) -> torch.Tensor:
            return torch.tensor(0.0, device=self._device)

        def transcribe(self, image_tensor: torch.Tensor) -> str:
            return ""

    # ── Inputs ─────────────────────────────────────────────────────────────────
    surrogates    = [_DummySurrogate(seed=0), _DummySurrogate(seed=1)]
    alpha_weights = [0.6, 0.4]

    epsilon_min = 8.0  / 255.0   # > epsilon_bg → satisfies "all values >= epsilon_bg"
    epsilon_max = 16.0 / 255.0
    epsilon_bg  = 4.0  / 255.0

    # Three word boxes covering ~43 % of the 64×64 canvas (1 760 / 4 096 px).
    # Kept well inside the image boundary so dilation=0 leaves clean bg pixels.
    word_boxes: list[list[int]] = [
        [ 4,  4, 24, 24],   # 20 × 20 = 400 px
        [32,  4, 52, 24],   # 20 × 20 = 400 px
        [ 4, 36, 52, 56],   # 48 × 20 = 960 px
    ]

    torch.manual_seed(99)
    image_tensor = torch.rand(1, 3, H, W)

    budget_map = build_salience_budget_map(
        image_tensor  = image_tensor,
        transcript    = "dummy text",
        word_boxes    = word_boxes,
        surrogates    = surrogates,
        alpha_weights = alpha_weights,
        epsilon_min   = epsilon_min,
        epsilon_max   = epsilon_max,
        epsilon_bg    = epsilon_bg,
        dilation      = 0,             # no dilation → bg pixels exactly epsilon_bg
        device        = device,
    )

    # ── 1. Shape ───────────────────────────────────────────────────────────────
    assert budget_map.shape == (1, H, W), (
        f"Expected shape (1,{H},{W}), got {budget_map.shape}"
    )

    # ── 2. All values >= epsilon_bg ────────────────────────────────────────────
    assert budget_map.min().item() >= epsilon_bg - 1e-7, (
        f"Min {budget_map.min().item():.8f} < epsilon_bg {epsilon_bg:.8f}"
    )

    # ── 3. All values <= epsilon_max ───────────────────────────────────────────
    assert budget_map.max().item() <= epsilon_max + 1e-7, (
        f"Max {budget_map.max().item():.8f} > epsilon_max {epsilon_max:.8f}"
    )

    # ── 4. Background pixels exactly epsilon_bg ────────────────────────────────
    # With dilation=0, only pixels strictly inside word_boxes are text.
    # build_salience_budget_map uses the same text_mask internally, so those
    # pixels are left untouched at epsilon_bg.
    text_mask  = build_text_mask(H, W, word_boxes, dilation=0, device=device)
    bg_mask    = (text_mask == 0).squeeze(0)         # (H, W) bool
    bg_values  = budget_map[0][bg_mask]              # 1-D flat
    assert torch.allclose(bg_values, torch.full_like(bg_values, epsilon_bg)), (
        f"Background pixels must be exactly epsilon_bg={epsilon_bg:.8f}. "
        f"Got range [{bg_values.min():.10f}, {bg_values.max():.10f}]"
    )

    # ── 5. Text-region statistics ──────────────────────────────────────────────
    text_pixels = budget_map[0][text_mask.squeeze(0) > 0]   # 1-D flat
    print(
        f"Text region budget — "
        f"min: {text_pixels.min().item():.5f}  "
        f"mean: {text_pixels.mean().item():.5f}  "
        f"max: {text_pixels.max().item():.5f}"
    )

    # ── 6. Text pixels >= epsilon_min ──────────────────────────────────────────
    assert text_pixels.min().item() >= epsilon_min - 1e-7, (
        f"Text pixels must be >= epsilon_min={epsilon_min:.8f}, "
        f"got min {text_pixels.min().item():.8f}"
    )

    print("test_salience_budget: all assertions passed.")


if __name__ == "__main__":
    test_salience_budget()
