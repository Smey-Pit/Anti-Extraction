"""
PaliGemma2-3B surrogate wrapper — Phase 0 (inference only).

Architecture:
  Vision encoder : SigLIP-So400m
  LM backbone    : Gemma-2B
  V-L connector  : Linear projection (no resampler, no tiling)

Reference checkpoint: google/paligemma2-3b-mix-448
Validated transformers: 5.4.0+

Notes
-----
PaliGemma uses a simple task-prefix prompt format. For OCR, the documented
prefix is "ocr\n"; for general visual question answering it is the question
itself with no special marker. The model expects the prompt tokenised
WITH the image so image-token positions match training.

Because PaliGemma's chat behaviour differs from the chat-template models,
we do NOT use apply_chat_template. We feed the user_text directly with the
"<image>" prefix that the processor expands into image tokens.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.models.base import SurrogateModel


class PaliGemma2(SurrogateModel):

    # PaliGemma's pre-training included an "ocr" task — using that prefix
    # routes the request to the OCR head behaviour the model was trained on.
    _DEFAULT_TRANSCRIBE_PROMPT = "ocr"

    def __init__(self, cfg) -> None:
        self.name = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_new_tokens = cfg.max_new_tokens

        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
        ).eval().to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Shared generation backend ──────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, image_tensor: torch.Tensor, user_text: str) -> str:
        pil = tensor_to_pil(image_tensor)

        # PaliGemma processor expects "<image>" prefix + task/question.
        # The processor expands "<image>" into the model's image-placeholder
        # tokens at the correct positions.
        prompt = f"<image>{user_text}\n"
        inputs = self.processor(
            text=prompt,
            images=pil,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self._device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        return self.processor.decode(
            out[0, prompt_len:], skip_special_tokens=True
        ).strip()

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(
        self,
        image_tensor: torch.Tensor,
        prompt: str | None = None,
    ) -> str:
        return self._generate(
            image_tensor,
            prompt if prompt is not None else self._DEFAULT_TRANSCRIBE_PROMPT,
        )

    def answer_query(
        self,
        image_tensor: torch.Tensor,
        question: str,
    ) -> str:
        # For free-form questions, PaliGemma works best with the question
        # used directly as the task prefix (no chat template).
        return self._generate(image_tensor, question)