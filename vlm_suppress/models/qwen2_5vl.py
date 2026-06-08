"""
Qwen2.5-VL surrogate wrapper — Phase 0 (inference only).

Architecture:
  Vision encoder : Qwen-ViT (native resolution, dynamic patches)
  LM backbone    : Qwen2.5
  V-L connector  : MLP merger

Reference checkpoint: Qwen/Qwen2.5-VL-7B-Instruct
Validated transformers: 5.4.0+

Notes
-----
Qwen2.5-VL supports both Qwen2_5VLForConditionalGeneration (newer) and
Qwen2_5_VLForConditionalGeneration (older) naming. We try both and fall
back to the submodule import.
"""

from __future__ import annotations

import torch

from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.models.base import SurrogateModel


def _import_model_cls():
    try:
        from transformers import Qwen2_5VLForConditionalGeneration as _Cls
        return _Cls
    except ImportError:
        pass
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
        return _Cls
    except ImportError:
        pass
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as _Cls,
    )
    return _Cls


class Qwen2_5VL(SurrogateModel):

    _DEFAULT_TRANSCRIBE_PROMPT = (
        "Read the text in this image and output it exactly as written. "
        "Output the text only, no coordinates, no descriptions, no explanations."
    )

    def __init__(self, cfg) -> None:
        from transformers import AutoProcessor

        self.name = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_new_tokens = cfg.max_new_tokens

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )
        ModelCls = _import_model_cls()
        self.model = ModelCls.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device).eval()

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Shared generation backend ──────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, image_tensor: torch.Tensor, user_text: str) -> str:
        pil = tensor_to_pil(image_tensor)
        chat_text = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat_text], images=[pil], return_tensors="pt"
        )
        inputs = {
            k: (v.to(self._device).to(self._dtype) if k == "pixel_values"
                else v.to(self._device) if torch.is_tensor(v)
                else v)
            for k, v in inputs.items()
        }

        out = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(
            out[0][input_len:], skip_special_tokens=True
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