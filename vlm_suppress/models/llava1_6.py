"""
LLaVA-1.6-Mistral-7B surrogate wrapper — Phase 0 (inference only).

Architecture:
  Vision encoder : CLIP ViT-L/14@336
  Connector      : 2-layer MLP (LlavaNextMultiModalProjector)
  LM backbone    : MistralForCausalLM
  Tiling         : AnyRes (sub-tiles + global thumbnail)

Reference checkpoint: llava-hf/llava-v1.6-mistral-7b-hf
Validated transformers: 5.4.0+
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.models.base import SurrogateModel


class LLaVA16(SurrogateModel):

    _DEFAULT_TRANSCRIBE_PROMPT = (
        "Transcribe all text in this image exactly as it appears. "
        "Do not add any explanation, formatting, or preamble. "
        "Output only the raw text content, nothing else."
    )

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
        # NB: do NOT pass device_map="auto" — on a single GPU with this model
        # size, accelerate sometimes shards the embedding layer to CPU while
        # leaving the rest on GPU, which breaks generate() with a device-mismatch
        # error inside the embedding lookup. Load on CPU first then .to(device).
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
        ).eval()
        self.model = self.model.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Shared generation backend ──────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, image_tensor: torch.Tensor, user_text: str) -> str:
        pil = tensor_to_pil(image_tensor)

        # LLaVA-1.6 uses the image token interleaved with user text.
        img_token_id = self.model.config.image_token_index
        img_token = self.processor.tokenizer.decode([img_token_id])

        # Build chat-formatted prompt. Strip leading BOS to avoid double-BOS
        # when the processor re-applies it during tokenisation.
        bos = self.processor.tokenizer.bos_token or "<s>"
        prompt_raw = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{img_token}\n{user_text}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_text = prompt_raw.removeprefix(bos)

        inputs = self.processor(
            text=prompt_text, images=pil, return_tensors="pt"
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
        return self.processor.tokenizer.decode(
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
        return self._generate(image_tensor, question)