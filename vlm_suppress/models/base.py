"""
Abstract base class for surrogate models — Phase 0 scope.

Phase 0 needs ONLY inference: full-image transcription (for BoW coverage)
and free-form query answering (for field extraction). No gradients, no
differentiable preprocessing.

Later phases will extend this ABC with gradient-enabled methods
(ce_loss, align_loss, token_logprobs, targeted_ce_loss). The Phase 0
contract is intentionally small so that:

  1. New wrappers are easy to write and easy to verify.
  2. The attack code in later phases cannot accidentally rely on
     half-implemented gradient methods.
  3. Each surrogate's clean-image behaviour is established before any
     adversarial machinery is introduced.

Image input convention (kept consistent with later phases):
  - Shape: (3, H, W)
  - dtype: float32
  - Range: [0, 1]
  - Device: any (wrappers handle their own device placement)

Wrappers convert internally to whatever format their processor expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SurrogateModel(ABC):
    """
    Inference-only contract for a Tier-2 open-source VLM surrogate.

    All wrappers must accept image tensors of shape (3, H, W), float32 in
    [0, 1], and return raw decoded strings (no preamble stripping — the
    eval loop owns normalisation).
    """

    name: str  # set in subclass __init__

    @abstractmethod
    def transcribe(
        self,
        image_tensor: torch.Tensor,
        prompt: str | None = None,
    ) -> str:
        """
        Full-image transcription.

        Parameters
        ----------
        image_tensor : (3, H, W) float32 [0,1]
        prompt       : if None, the wrapper's default transcription prompt is
                       used. If a string is provided, it overrides the default
                       so the caller can sweep prompt variants for
                       noise-floor measurement.

        Returns
        -------
        Raw decoded string from the model. No stripping, no JSON parsing,
        no preamble removal. The eval loop is responsible for normalisation.
        """
        ...

    @abstractmethod
    def answer_query(
        self,
        image_tensor: torch.Tensor,
        question: str,
    ) -> str:
        """
        Free-form question answering over an image.

        Used for field-level extraction in Phase 0
        (e.g. question="What is the account holder?").

        Parameters
        ----------
        image_tensor : (3, H, W) float32 [0,1]
        question     : caller-controlled question string. The wrapper wraps
                       this in its model-appropriate chat template but does
                       not otherwise modify it.

        Returns
        -------
        Raw decoded string from the model.
        """
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the model lives on."""
        ...