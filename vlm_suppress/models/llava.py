"""
vlm_suppress/models/llava16.py

LLaVA-1.6-Mistral-7B surrogate wrapper.

Architecture:
    Vision encoder       : CLIP ViT-L/14@336
    Connector            : LlavaNextMultiModalProjector (2-layer MLP)
    LM backbone          : MistralForCausalLM
    Tiling               : AnyRes — sub-tiles + 1 global thumbnail
    pixel_values shape   : (1, n_tiles, 3, 336, 336)

transformers 5.x behaviour:
    The processor pre-expands <image> into all patch token IDs in input_ids.
    The model's forward() handles image feature injection internally when
    pixel_values is passed alongside input_ids. Manual splicing is not needed
    and will fail (1890 image tokens found, not 1).

_preprocess — delta injection (same as llama3_2.py):
    pv_proc  = processor(pil)          — exact tiling, no grad
    x_norm   = clip_norm(resize(x))    — differentiable
    delta    = x_norm - sg(x_norm)     — zero value, live grad
    pv_diff  = pv_proc + expand(delta) — exact at init, grad-connected

ce_loss:
    Standard model forward with input_ids + pixel_values + labels.
    Tail-length label masking (-100 on prompt prefix, transcript on tail).
    cross_attention_mask extended to full sequence length if present.

Validated checkpoint : llava-hf/llava-v1.6-mistral-7b-hf
Validated transformers: 5.4.0
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from vlm_suppress.models.base import SurrogateModel

_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_TILE_SIZE = 336


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) float32 [0,1] → PIL RGB. Always detached."""
    arr = (image_tensor.detach().cpu().clamp(0, 1) * 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), mode="RGB")


class LLaVA16(SurrogateModel):

    _QUESTION = (
        "Transcribe all text in this image exactly as it appears. "
        "Do not add any explanation, formatting, or preamble. "
        "Output only the raw text content, nothing else."
    )

    def __init__(self, cfg) -> None:
        self.name            = cfg.name
        self._max_new_tokens = cfg.max_new_tokens

        _dev = getattr(cfg, "device", None)
        self._device = torch.device(_dev) if _dev else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(cfg.model_id)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            device_map=getattr(cfg, "device_map", "auto"),
        ).eval()

        self._img_token_id: int = self.model.config.image_token_index

        # Build and cache static prompt text — strip leading BOS to avoid double-BOS
        img_token  = self.processor.tokenizer.decode([self._img_token_id])
        bos        = self.processor.tokenizer.bos_token or "<s>"
        prompt_raw = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{img_token}\n{self._QUESTION}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        self._prompt_text = prompt_raw.removeprefix(bos)

        # CLIP normalisation constants
        self._clip_mean = torch.tensor(
            _CLIP_MEAN, device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)
        self._clip_std = torch.tensor(
            _CLIP_STD, device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)

        # _preprocess cache — keyed on image spatial size
        self._pv_proc_cache: torch.Tensor | None = None
        self._pv_proc_cache_size: tuple[int, int] | None = None
        self._proc_enc_cache: dict[str, Any] | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _clip_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._clip_mean) / self._clip_std

    def _processor_inputs(self, pil: Image.Image) -> dict[str, Any]:
        enc = self.processor(
            text=self._prompt_text, images=pil, return_tensors="pt"
        )
        return {k: v.to(self._device) if torch.is_tensor(v) else v
                for k, v in enc.items()}

    def _transcript_ids(self, transcript: str) -> torch.Tensor:
        return self.processor.tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)

    # ── Differentiable preprocessing ──────────────────────────────────────────

    def _preprocess(
        self, image_tensor: torch.Tensor, pil: Image.Image
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Return (pixel_values, prompt_enc) where pixel_values is grad-connected.

        Delta injection:
            pv_proc  = processor(pil)           — exact tiling, no grad, cached
            x_norm   = clip_norm(resize(x))     — differentiable
            delta    = x_norm - sg(x_norm)      — zero value, live grad
            pv_diff  = pv_proc + expand(delta)  — exact at init, grad-connected

        Cache invalidated per sample by setting _pv_proc_cache = None.
        """
        img_size = (image_tensor.shape[-2], image_tensor.shape[-1])

        if self._pv_proc_cache is None or self._pv_proc_cache_size != img_size:
            with torch.no_grad():
                enc = self._processor_inputs(pil)
            self._pv_proc_cache      = enc["pixel_values"].to(self._dtype)
            self._pv_proc_cache_size = img_size
            self._proc_enc_cache     = enc

        pv_proc    = self._pv_proc_cache    # (1, n_tiles, 3, 336, 336)
        prompt_enc = self._proc_enc_cache
        n_tiles    = pv_proc.shape[1]

        x = image_tensor.to(device=self._device, dtype=self._dtype).unsqueeze(0)
        x_r = F.interpolate(
            x, size=(_TILE_SIZE, _TILE_SIZE), mode="bilinear", align_corners=False
        )
        x_norm = self._clip_norm(x_r)       # (1, 3, 336, 336)

        delta = (x_norm - x_norm.detach()).unsqueeze(1).expand(
            -1, n_tiles, -1, -1, -1
        )  # (1, n_tiles, 3, 336, 336), zero value, grad-connected

        return pv_proc + delta, prompt_enc

    # ── SurrogateModel API ─────────────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | LLaVA chat prompt + image)

        Standard model forward: input_ids + pixel_values + labels.
        The processor has pre-expanded image tokens in input_ids — the model
        handles feature injection internally when pixel_values is passed.

        Tail-length label masking:
            full_ids = prompt_ids ++ transcript_ids
            labels   = [-100 ... -100 | transcript_ids]
        """
        pil                      = _tensor_to_pil(image_tensor)
        pixel_values, prompt_enc = self._preprocess(image_tensor, pil)

        transcript_ids = self._transcript_ids(transcript)   # (1, T)
        t_len          = transcript_ids.size(1)

        full_ids  = torch.cat([prompt_enc["input_ids"], transcript_ids], dim=1)
        full_attn = torch.cat([
            prompt_enc["attention_mask"],
            torch.ones((1, t_len), device=self._device, dtype=torch.long),
        ], dim=1)
        total_len = full_ids.size(1)

        labels = torch.full(
            (1, total_len), -100, device=self._device, dtype=torch.long
        )
        labels[0, -t_len:] = transcript_ids[0]

        out = self.model(
            input_ids=full_ids,
            attention_mask=full_attn,
            pixel_values=pixel_values,
            image_sizes=prompt_enc["image_sizes"],
            labels=labels,
            return_dict=True,
            use_cache=False,
        )

        if out.loss is None:
            raise RuntimeError("LLaVA16.ce_loss: model returned loss=None")
        return out.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cos(z_I, z_T)

        z_I: CLIP features of base tile (full image → 336×336),
             projected into LLM space via multi_modal_projector, mean-pooled.
        z_T: mean-pooled LM input embeddings of transcript tokens.
        """
        x    = image_tensor.to(device=self._device, dtype=self._dtype).unsqueeze(0)
        base = F.interpolate(
            x, size=(_TILE_SIZE, _TILE_SIZE), mode="bilinear", align_corners=False
        )
        pv_base = self._clip_norm(base)     # (1, 3, 336, 336)

        vision_out = self.model.model.vision_tower(pv_base, output_hidden_states=True)
        feat_layer = self.model.config.vision_feature_layer
        strategy   = self.model.config.vision_feature_select_strategy

        if strategy == "default":
            clip_feats = vision_out.hidden_states[feat_layer][:, 1:]   # drop CLS
        else:
            clip_feats = vision_out.hidden_states[feat_layer]
        # (1, n_vis_tokens, D_clip)

        proj_feats = self.model.model.multi_modal_projector(clip_feats)
        z_i = F.normalize(proj_feats.mean(dim=1).float(), dim=-1)      # (1, D_llm)

        with torch.no_grad():
            text_ids = self.processor.tokenizer(
                transcript, add_special_tokens=True, return_tensors="pt"
            ).input_ids.to(self._device)
            z_t = self.model.get_input_embeddings()(text_ids)
            z_t = F.normalize(z_t.mean(dim=1).float(), dim=-1)         # (1, D_llm)

        if z_i.shape[-1] != z_t.shape[-1]:
            return torch.zeros((), device=self._device, dtype=torch.float32)

        return -(z_i * z_t).sum(dim=-1).squeeze(0)

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pil    = _tensor_to_pil(image_tensor)
        inputs = self._processor_inputs(pil)

        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        raw = self.processor.tokenizer.decode(
            out[0, prompt_len:], skip_special_tokens=True
        ).strip()

        raw = re.sub(
            r'^(the (?:text (?:in the image )?(?:reads|is|says)|'
            r'image (?:shows|contains|displays|reads))[\s:\"\']*)',
            "", raw, flags=re.IGNORECASE,
        ).strip().strip('"').strip("'").strip()

        return raw