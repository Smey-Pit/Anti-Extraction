from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vlm_suppress.models.base import SurrogateModel


class InternVL35(SurrogateModel):
    """
    InternVL3.5-8B surrogate wrapper.

    Validated by smoke test with:
      - clean OCR: strong
      - ce_loss: sane, non-catastrophic
      - gradient: reaches pixels
      - one-step perturbation: increases CE and CER

    Important environment note:
    This wrapper assumes the local Hugging Face cached InternVL3.5 remote-code
    files were already patched for compatibility with the current Transformers
    version:
      1) modeling_intern_vit.py: stochastic-depth schedule creation must avoid
         meta tensors during model init
      2) modeling_internvl_chat.py: tied-weights compatibility shim

    Conversation/template invariant:
    ce_loss() and transcribe() must condition on the same conversation template.
    For InternVL3.5-8B, the smoke test found the best CE template to be:

      <|im_start|>system
      {system}
      <|im_end|>
      <|im_start|>user
      <img>{IMG_CONTEXT * N}</img>
      {question}
      <|im_end|>
      <|im_start|>assistant
      \n

    with:
      - system=True
      - think=False

    This wrapper follows that exactly.
    """

    _SYSTEM = (
        "You are an OCR engine. "
        "Transcribe exactly all visible text in the image. "
        "Preserve line breaks. Output only the text."
    )
    _QUESTION = (
        "Transcribe exactly all visible text in the image. "
        "Preserve line breaks. Output only the text."
    )

    _IMG_START = "<img>"
    _IMG_END = "</img>"
    _IMG_TOKEN = "<IMG_CONTEXT>"

    _THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def __init__(self, cfg) -> None:
        self.name = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self._dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            use_fast=False,
        )

        # Important:
        # - use dtype= (not torch_dtype=)
        # - avoid device_map for the wrapper path
        # - low_cpu_mem_usage=False helps keep load behavior predictable
        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            dtype=self._dtype,
            low_cpu_mem_usage=False,
        ).eval().to(self._device)

        self._max_new_tokens = cfg.max_new_tokens

        self._n_img_tokens: int = self.model.num_image_token
        self._img_ctx_id: int = self.tokenizer.convert_tokens_to_ids(self._IMG_TOKEN)

        assert self._img_ctx_id != self.tokenizer.unk_token_id, (
            f"{self._IMG_TOKEN} is not in the tokenizer vocabulary."
        )

        # Required so model.chat() can also resolve image placeholders correctly
        self.model.img_context_token_id = self._img_ctx_id

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        (3, H, W) float32 [0,1] -> (1, 3, 448, 448) bfloat16 on model device.
        Gradient-transparent.
        """
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)

        x = image_tensor.to(device=self._device, dtype=self._dtype).unsqueeze(0)
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return x

    # ── Prompt construction ────────────────────────────────────────────────

    def _build_input_ids(self, question: str) -> torch.Tensor:
        """
        InternVL3.5-8B / Qwen3-style chat template found by smoke test:

          <|im_start|>system\n{system}<|im_end|>\n
          <|im_start|>user\n<img>{IMG_CONTEXT * N}</img>\n{question}<|im_end|>\n
          <|im_start|>assistant\n

        Returns (1, L) int64 on self._device.
        """
        img_placeholder = (
            self._IMG_START
            + self._IMG_TOKEN * self._n_img_tokens
            + self._IMG_END
        )

        conversation = (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{img_placeholder}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        ids = self.tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)

        return ids

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace <IMG_CONTEXT> token embedding rows with visual embeddings.

        Gradient flows:
          pixel_values -> extract_feature -> splice -> language_model
        """
        with torch.no_grad():
            tok_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        tok_embeds = tok_embeds.to(self._dtype)

        img_embeds = self.model.extract_feature(pixel_values)  # (1, N_img, D)

        img_mask = (input_ids[0] == self._img_ctx_id)
        n_found = int(img_mask.sum().item())
        assert n_found == self._n_img_tokens, (
            f"Expected {self._n_img_tokens} <IMG_CONTEXT> tokens, found {n_found}. "
            "Check InternVL3.5 conversation template."
        )

        inputs_embeds = tok_embeds.clone()
        inputs_embeds[0, img_mask] = img_embeds[0]
        return inputs_embeds

    # ── Differentiable losses ──────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | image, OCR chat prompt)

        Uses the same Qwen-style conversation prefix that the smoke test found
        to be the best CE template for InternVL3.5-8B.
        """
        pixel_values = self._preprocess(image_tensor)
        input_ids = self._build_input_ids(self._QUESTION)
        L = input_ids.size(1)

        inputs_embeds = self._build_inputs_embeds(input_ids, pixel_values)

        target_ids = self.tokenizer(
            transcript,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)
        T = target_ids.size(1)

        with torch.no_grad():
            tgt_embeds = self.model.language_model.get_input_embeddings()(target_ids)
        tgt_embeds = tgt_embeds.to(self._dtype)

        full_inputs_embeds = torch.cat([inputs_embeds, tgt_embeds], dim=1)

        labels = torch.cat(
            [
                torch.full((1, L), -100, device=self._device, dtype=torch.long),
                target_ids,
            ],
            dim=1,
        )

        out = self.model.language_model(
            inputs_embeds=full_inputs_embeds,
            labels=labels,
            return_dict=True,
        )
        return out.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cosine_sim(z_I, z_T)

        Same simple pooled embedding proxy used in InternVL2 wrapper.
        """
        pixel_values = self._preprocess(image_tensor)

        img_embeds = self.model.extract_feature(pixel_values)
        z_I = F.normalize(img_embeds.mean(dim=1).float(), dim=-1)

        with torch.no_grad():
            text_ids = self.tokenizer(
                transcript,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(self._device)

            z_T = self.model.language_model.get_input_embeddings()(text_ids)
            z_T = z_T.mean(dim=1).float()
            z_T = F.normalize(z_T, dim=-1)

        return -(z_I * z_T).sum(dim=-1).squeeze(0)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pixel_values = self._preprocess(image_tensor.detach())

        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=self._QUESTION,
            generation_config=dict(
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            ),
            history=None,
            return_history=True,
        )

        response = self._THINK_RE.sub("", response).strip()
        return response