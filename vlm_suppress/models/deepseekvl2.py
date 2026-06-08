"""
DeepSeek-VL2 surrogate wrapper — Phase 0 (inference only).

Architecture:
  Vision encoder : SigLIP-SO400M-384 with dynamic tiling (up to 9 tiles + 1
                   global thumbnail per image)
  LM backbone    : DeepSeekMoE (small variant: 16B total / 2.8B activated)
  V-L connector  : Two-layer MLP projector

Reference checkpoint: deepseek-ai/deepseek-vl2-small
Validated install   : torch 2.0.1+cu117, transformers 4.38.2 (DeepSeek-VL2
                      pins these aggressively; must be installed in a
                      separate venv from the other Phase 0 surrogates).

Notes
-----
DeepSeek-VL2 is NOT a transformers-native model. The model class lives in
the `deepseek_vl2` package, which must be installed via:
    pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git

This wrapper assumes the package is importable. If you're invoking from a
shared codebase where some venvs don't have it, guard the import in the
test harness, not here.

Inference flow (differs from chat-template surrogates):
    1. Build a "conversation" list with explicit <|User|> / <|Assistant|>
       role tokens. The user content contains "<image>" placeholder + text.
       Images are attached as a separate `images=[pil]` field.
    2. processor(conversations, images, force_batchify=True, system_prompt="")
       returns a BatchedVLChatProcessorOutput with input_ids, attention_mask,
       images, images_seq_mask, images_spatial_crop.
    3. model.prepare_inputs_embeds(**prepare_inputs) runs the vision tower
       and splices visual embeddings into the LM input embedding sequence.
    4. model.language_model.generate(inputs_embeds=...) does decoding.
    5. tokenizer.decode(output_ids) returns the string.

The conversation list always ends with an empty Assistant turn — this is
how the chat template signals "model fills in this turn".
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.models.base import SurrogateModel


class DeepSeekVL2(SurrogateModel):

    _DEFAULT_TRANSCRIBE_PROMPT = (
        "<image>\n"
        "Transcribe all text in this image exactly as it appears. "
        "Output only the raw text content, with line breaks preserved. "
        "Do not add any explanation, formatting, or preamble."
    )

    def __init__(self, cfg) -> None:
        # Imported here (not at module level) so the wrapper file is still
        # importable in venvs that don't have deepseek_vl2 — the failure
        # surfaces only when the user actually tries to instantiate.
        from deepseek_vl2.models import DeepseekVLV2Processor

        self.name = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_new_tokens = cfg.max_new_tokens

        self.processor = DeepseekVLV2Processor.from_pretrained(cfg.model_id)
        self.tokenizer = self.processor.tokenizer

        # AutoModelForCausalLM + trust_remote_code is the documented load
        # path. The remote-code registration in deepseek_vl2 wires this up
        # to DeepseekVLV2ForCausalLM under the hood.
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id,
                                                        trust_remote_code=True,
                                                        torch_dtype=self._dtype,           # use self._dtype not hardcoded bf16
                                                        low_cpu_mem_usage=True,
                                                        device_map={"": self._device},
                                                        ).eval()

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Shared generation backend ──────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, image_tensor: torch.Tensor, user_text: str) -> str:
        """
        user_text is the full <|User|> content including the "<image>"
        placeholder. The transcribe/answer_query callers control whether
        the placeholder appears.
        """
        pil = tensor_to_pil(image_tensor)

        conversation = [
            {
                "role": "<|User|>",
                "content": user_text,
                "images": [pil],  # placeholder slot — the actual PIL goes via
                                  # the `images=[pil]` kwarg below
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = self.processor(
            conversations=conversation,
            images=[pil],
            force_batchify=True,
            system_prompt="",
        ).to(self._device)

        # Run vision tower + project + splice into LM embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Decoder-only generation from the spliced embedding sequence
        out = self.model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        return self.tokenizer.decode(
            out[0].cpu().tolist(), skip_special_tokens=True
        ).strip()

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(
        self,
        image_tensor: torch.Tensor,
        prompt: str | None = None,
    ) -> str:
        # When the caller overrides the prompt, we still need the "<image>"
        # placeholder to bind the image into the conversation. If the caller
        # already included it, respect their version; otherwise prepend.
        if prompt is None:
            user_text = self._DEFAULT_TRANSCRIBE_PROMPT
        elif "<image>" in prompt:
            user_text = prompt
        else:
            user_text = f"<image>\n{prompt}"
        return self._generate(image_tensor, user_text)

    def answer_query(
        self,
        image_tensor: torch.Tensor,
        question: str,
    ) -> str:
        # Field-extraction questions don't include "<image>" — wrap it in.
        user_text = (
            question if "<image>" in question
            else f"<image>\n{question}"
        )
        return self._generate(image_tensor, user_text)