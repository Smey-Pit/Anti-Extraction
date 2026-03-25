from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vlm_suppress.models.base import SurrogateModel


class InternVL2(SurrogateModel):
    """
    InternVL2-8B surrogate wrapper.

    Load strategy: always instantiate on CPU first, then .to(device).
    Never pass device_map to from_pretrained — InternVL2's custom __init__
    calls .item() on tensors during construction (stochastic depth schedule
    in modeling_intern_vit.py:312), which raises RuntimeError on meta tensors.
    HF's meta-device dispatch is triggered by ANY device_map= value in
    transformers >=4.38, including {"": 0}. CPU-first + .to() bypasses it.

    ce_loss:    teacher-forced CE via language_model(inputs_embeds=...).
                Gradients flow: pixel_values → vision_model → mlp1 → ce_loss.
    align_loss: -cosine_sim(z_I, z_T). Vision path is differentiable;
                text embedding is detached (no grad through tokeniser).
    """

    def __init__(self, cfg) -> None:
        self.name     = cfg.name
        self._dtype   = torch.bfloat16
        self._device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )

        # ── CPU-first load — do NOT pass device_map ────────────────────────
        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
            # device_map intentionally omitted
        ).eval().to(self._device)

        self._max_new_tokens = cfg.max_new_tokens

        self._ocr_prompt = (
            "You are an OCR engine. "
            "Transcribe exactly all visible text in this image. "
            "Preserve line breaks. Output only the text, nothing else."
        )

    # ── SurrogateModel ABC ─────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        (3, H, W) float32 [0,1]  →  (1, 3, 448, 448) bfloat16 on model device.
        InternVL2 uses ImageNet normalisation.
        Maintains gradient flow — no detach here.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=self._dtype)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=self._dtype)
        x = image_tensor.to(device=self._device, dtype=self._dtype)
        x = F.interpolate(x.unsqueeze(0), size=(448, 448),
                          mode="bilinear", align_corners=False)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return x  # (1, 3, 448, 448), grad flows through if image_tensor.requires_grad

    # ── Differentiable losses ──────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | image).

        Uses teacher-forced CE through the language model's inputs_embeds path.

        Gradient path:
          image_tensor → _preprocess → extract_feature
          → inputs_embeds → language_model → CE loss

        extract_feature() is InternVL2's own method that runs the full vision
        pipeline (ViT → pixel-unshuffle → mlp1) and returns (1, N_vis, 4096).
        Calling vision_model().last_hidden_state directly gives (1, N, 1024)
        which is the raw ViT dim — mlp1 then fails with a shape mismatch.
        """
        pixel_values = self._preprocess(image_tensor)  # (1, 3, 448, 448), bf16, grad

        # 1. Visual tokens — differentiable, correct 4096-dim output
        img_embeds = self.model.extract_feature(pixel_values)  # (1, N_vis, 4096)

        # 2. Text token embeddings — no grad needed through tokeniser
        target_ids = self.tokenizer(
            transcript,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)                             # (1, T)

        with torch.no_grad():
            tok_embeds = self.model.language_model \
                             .get_input_embeddings()(target_ids) # (1, T, D)

        # 3. Concatenate [img | text] and build labels
        #    Image positions are masked out (-100) so loss is only over text tokens.
        inputs_embeds = torch.cat(
            [img_embeds, tok_embeds.to(self._dtype)], dim=1
        )                                                        # (1, N_vis+T, D)

        labels = torch.cat([
            torch.full((1, img_embeds.size(1)), -100,
                       device=self._device, dtype=torch.long),   # ignore img tokens
            target_ids,                                          # supervise text tokens
        ], dim=1)                                                # (1, N_vis+T)

        out = self.model.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
        )
        return out.loss   # scalar; grad flows back to pixel_values ✓

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cosine_sim(z_I, z_T).
        Visual path is differentiable; text embedding is detached.
        """
        pixel_values = self._preprocess(image_tensor)

        # Visual embedding: use extract_feature for consistent 4096-dim repr
        img_embeds = self.model.extract_feature(pixel_values)  # (1, N_vis, 4096)
        z_I = img_embeds.mean(dim=1)                           # (1, 4096)
        z_I = F.normalize(z_I.float(), dim=-1)

        # Text embedding — detached, no grad
        with torch.no_grad():
            text_ids = self.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self._device)
            text_emb = self.model.language_model \
                           .get_input_embeddings()(text_ids)     # (1, T, D)
            z_T = text_emb.mean(dim=1).float()
            z_T = F.normalize(z_T, dim=-1)

        sim = (z_I * z_T).sum(dim=-1)
        return -sim.squeeze()   # higher = less aligned = more suppression

    # ── Inference (no grad) ────────────────────────────────────────────────

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pixel_values = self._preprocess(image_tensor.detach())  # same 448x448 as ce_loss

        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=(
                "Transcribe exactly all visible text in this image. "
                "Preserve line breaks. Output only the text, nothing else."
            ),
            generation_config=dict(
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            ),
            history=None,
            return_history=True,
        )
        return response.strip()