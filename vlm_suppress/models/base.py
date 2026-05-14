"""
Abstract base classes for surrogate and frontier models.

Any new model added to the project must implement these ABCs.
The attack code depends ONLY on these interfaces — never on a
specific model implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image


class SurrogateModel(ABC):
    """
    White-box Tier-2 surrogate. Must be differentiable w.r.t. image input.

    The attack operates on image tensors of shape (3, H, W) in [0, 1].
    All surrogate wrappers must handle their own preprocessing internally
    so the attack code stays model-agnostic.
    """

    name: str  # set in subclass __init__

    @abstractmethod
    def ce_loss(
        self,
        image_tensor: torch.Tensor,   # (3, H, W), float32, [0,1], on device, requires_grad
        transcript: str,
    ) -> torch.Tensor:
        """
        F_ce^k(delta) = L_ext(f(X+delta), T)

        Returns a scalar tensor with gradient w.r.t. image_tensor.
        Higher value = model fails more = more suppression.
        """
        ...

    @abstractmethod
    def align_loss(
        self,
        image_tensor: torch.Tensor,   # (3, H, W)
        transcript: str,
    ) -> torch.Tensor:
        """
        F_align^k(delta) = -sim(z_I^k(X+delta), z_T^k(T))

        Returns a scalar tensor (negative cosine similarity).
        Higher value = less image-text alignment = more suppression.
        """
        ...

    @abstractmethod
    def transcribe(
        self,
        image_tensor: torch.Tensor,   # (3, H, W), no grad needed
    ) -> str:
        """
        Run greedy/beam inference. Returns transcribed string.
        Used for evaluation only — no gradient required.
        """
        ...

    @torch.no_grad()
    def token_logprobs(
        self,
        image_tensor: torch.Tensor,
        transcript: str,
        return_top_k: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Per-token log probabilities for transcript given image.
        Subclasses that support importance mapping must override this.

        Parameters
        ----------
        return_top_k : if 0 (default), returns a 2-tuple.
                       if > 0, additionally returns top-K log probs and ids
                       at every transcript position.

        Returns
        -------
        Always:
          log_probs  : (T,) float32 on device
          token_ids  : (T,) int64  on device
        When return_top_k > 0:
          top_k_log_probs : (T, K) float32, sorted descending
          top_k_ids       : (T, K) int64
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement token_logprobs. "
            "Required for importance mapping (Stage 1). "
            "Add a token_logprobs method following the contract in base.py."
        )

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...


class FrontierModel(ABC):
    """
    Black-box Tier-3 frontier model. No gradient access.
    Evaluation only — called via API.
    """

    name: str

    @abstractmethod
    def transcribe(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Send image + prompt to the frontier API.
        Returns the raw transcription string.
        Handles retries and rate limiting internally.
        """
        ...