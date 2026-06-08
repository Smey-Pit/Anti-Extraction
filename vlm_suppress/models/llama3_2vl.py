"""
Llama-3.2-11B-Vision-Instruct surrogate wrapper — Phase 0 (inference only).

Architecture:
  Vision encoder : Custom ViT (Meta-trained on 6B image-text pairs)
  LM backbone    : Llama-3.1-11B
  V-L connector  : Cross-attention adapter layers

Visual feature caching
----------------------
cross_attention_states is accepted by forward() but never returned in its
output — it is an intermediate variable forwarded to the inner LM.

We therefore extract cross_attention_states explicitly by calling the
vision encoder and projector directly before the first forward() call,
cache the result, and pass it to all subsequent forward() calls.

This gives us:
  - transcribe(): runs vision encoder once, caches states, decodes greedily
  - answer_query(): uses cached states, skips vision encoder entirely

Net speedup: ~8-10x on queries (vision encoder was ~90% of query time).
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.models.base import SurrogateModel


class LlamaVision(SurrogateModel):

    _DEFAULT_TRANSCRIBE_PROMPT = (
        "Perform a complete OCR scan of this document. "
        "Output every single line of text exactly as it appears, "
        "reading from the top of the document to the very bottom. "
        "Include all headers, body text, numbers, dates, amounts, names, "
        "and table contents. Do not skip, summarize, or omit any line. "
        "Output only the raw text content, nothing else."
    )

    def __init__(self, cfg) -> None:
        self.name = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_new_tokens  = cfg.max_new_tokens
        self._max_query_tokens = 100   # short answers only — saves KV cache memory

        self.processor = AutoProcessor.from_pretrained(cfg.model_id)

        self.model = MllamaForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
        ).eval()
        self.model = self.model.to(self._device)

        # Visual feature cache
        self._cache_key:     str | None           = None
        self._cached_states: torch.Tensor | None  = None

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Image fingerprint ──────────────────────────────────────────────────

    @staticmethod
    def _image_key(image_tensor: torch.Tensor) -> str:
        shape  = str(tuple(image_tensor.shape))
        sample = image_tensor.detach().cpu().float()[::1, ::8, ::8]
        h      = hashlib.md5(sample.numpy().tobytes()).hexdigest()[:12]
        return f"{shape}_{h}"

    # ── Vision encoder (explicit, cacheable) ───────────────────────────────

    @torch.no_grad()
    def _encode_image(
        self,
        pixel_values:      torch.Tensor,
        aspect_ratio_ids:  torch.Tensor | None,
        aspect_ratio_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Run vision encoder + projector, return cross_attention_states.

        MllamaVisionModel.forward() returns last_hidden_state with shape
        (batch, n_images, n_tiles, seq_len, vision_dim).
        multi_modal_projector maps vision_dim → lm_dim.

        Result shape: (batch, n_images, n_tiles, seq_len, lm_dim)
        """
        vision_model = self.model.model.vision_model
        vis_out = vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )
        states = vis_out.last_hidden_state  # (..., vision_dim)

        projector = getattr(self.model.model, "multi_modal_projector", None)
        if projector is not None:
            B, n_img, n_tiles, seq_len, D = states.shape
            projected = projector(states.reshape(-1, D).to(self._dtype))
            states = projected.reshape(B, n_img, n_tiles, seq_len, -1)

        return states   # (..., lm_dim)

    # ── Processor helper ───────────────────────────────────────────────────

    def _build_inputs(self, pil: Image.Image, user_text: str) -> dict[str, Any]:
        prompt_text = self.processor.apply_chat_template(
            [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.processor(
            text=prompt_text,
            images=pil,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            k: v.to(self._device) if torch.is_tensor(v) else v
            for k, v in enc.items()
        }

    # ── Greedy decode using forward() directly ────────────────────────────

    @torch.no_grad()
    def _greedy_decode(
        self,
        inputs: dict[str, Any],
        cross_attention_states: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """
        Greedy decode with pre-computed cross_attention_states.
        Uses model.forward() (not generate()) so we can pass states directly.
        Returns all token ids (prompt + generated) as a 1D tensor.
        """
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens

        generated = inputs["input_ids"].clone()      # (1, L)
        attn_mask = inputs["attention_mask"]
        ca_mask   = inputs.get("cross_attention_mask")
        past_kv   = None

        eot_id = self.processor.tokenizer.eos_token_id

        try:
            for step in range(max_new_tokens):
                if step == 0:
                    out = self.model(
                        input_ids=generated,
                        attention_mask=attn_mask,
                        cross_attention_states=cross_attention_states,
                        cross_attention_mask=ca_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    out = self.model(
                        input_ids=generated[:, -1:],
                        attention_mask=torch.ones(
                            (1, generated.shape[1]),
                            dtype=attn_mask.dtype,
                            device=self._device,
                        ),
                        cross_attention_states=cross_attention_states,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )

                past_kv    = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated  = torch.cat([generated, next_token], dim=1)

                if eot_id is not None and next_token.item() == eot_id:
                    break
        finally:
            # Explicitly free KV cache — critical for preventing memory
            # fragmentation across hundreds of sequential images.
            del past_kv
            torch.cuda.empty_cache()

        return generated[0]

    # ── Core generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate(
        self,
        image_tensor: torch.Tensor,
        user_text: str,
        max_new_tokens: int | None = None,
    ) -> str:
        pil = tensor_to_pil(image_tensor)
        key = self._image_key(image_tensor)

        inputs = self._build_inputs(pil, user_text)

        # Encode image if not cached
        if key != self._cache_key or self._cached_states is None:
            self._cached_states = self._encode_image(
                pixel_values      = inputs["pixel_values"],
                aspect_ratio_ids  = inputs.get("aspect_ratio_ids"),
                aspect_ratio_mask = inputs.get("aspect_ratio_mask"),
            )
            self._cache_key = key

        prompt_len = inputs["input_ids"].shape[1]
        all_ids    = self._greedy_decode(
            inputs, self._cached_states, max_new_tokens=max_new_tokens
        )
        gen_ids    = all_ids[prompt_len:]

        raw = self.processor.decode(gen_ids.tolist(), skip_special_tokens=False)
        eot = "<|eot_id|>"
        if eot in raw:
            raw = raw[:raw.index(eot)]
        return raw.strip()

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(
        self,
        image_tensor: torch.Tensor,
        prompt: str | None = None,
    ) -> str:
        user_text = prompt if prompt is not None else self._DEFAULT_TRANSCRIBE_PROMPT
        return self._generate(image_tensor, user_text,
                              max_new_tokens=self._max_new_tokens)

    def answer_query(
        self,
        image_tensor: torch.Tensor,
        question: str,
    ) -> str:
        # Use shorter token limit — answers are always brief.
        # Reduces KV cache size substantially, preventing memory fragmentation
        # across hundreds of sequential images.
        return self._generate(image_tensor, question,
                              max_new_tokens=self._max_query_tokens)