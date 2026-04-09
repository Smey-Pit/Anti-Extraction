"""
vlm_suppress/models/llama3_2.py

Llama 3.2 Vision-11B surrogate wrapper for the K=3 ensemble.

Architecture:
    Vision encoder : Custom ViT (trained on 6B image-text pairs)
    LM backbone    : Llama 3.1-11B
    V-L connector  : Cross-attention adapter layers

Why this model is in the K=3 ensemble:
    The cross-attention connector is architecturally distinct from both
    InternVL3.5 (MLP splice) and PaliGemma2 (linear projection). It provides
    a third gradient direction that neither InternViT nor SigLIP gradients
    cover, improving adversarial transferability.

SurrogateModel contract:
    ce_loss(image_tensor, transcript)    -> scalar tensor, grad-connected
    align_loss(image_tensor, transcript) -> scalar tensor, grad-connected
    transcribe(image_tensor)             -> str

Validated checkpoint : meta-llama/Llama-3.2-11B-Vision-Instruct
Validated transformers: 5.4.0
Validated image sizes : (192, 768)

Load strategy:
    CPU-first (no device_map, no low_cpu_mem_usage). Llama 3.2's
    cross-attention adapter __init__ performs operations that fail on
    meta tensors.

_preprocess — perturbation injection:
    MllamaImageProcessor applies proprietary tiling logic that cannot be
    replicated by crop+resize+normalise (cosine_sim < 0 when attempted).
    Instead, we use pv_proc + delta where delta = x_norm - sg(x_norm).
    delta is zero-valued at init but carries grad from image_tensor.
    All tiles receive the same spatial gradient signal (single resize),
    which is sufficient for PGD sign steps.

align_loss — projector dtype:
    multi_modal_projector is Linear(7680→4096) in bfloat16.
    last_hidden_state shape is (1, n_images, n_tiles, seq_len, 7680).
    Mean-pool over all non-batch, non-feature dims before projecting.
    Cast to model dtype before projector, then to float32 for cosine.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from vlm_suppress.models.base import SurrogateModel


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) float32 [0,1] → PIL RGB. Used for processor calls only."""
    arr = (image_tensor.detach().cpu().clamp(0, 1) * 255.0).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), mode="RGB")


class LlamaVision(SurrogateModel):

    _PROMPT = "Transcribe exactly all visible text. Output only the text, nothing else."

    def __init__(self, cfg) -> None:
        self.name            = cfg.name     
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self._dtype          = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_new_tokens = cfg.max_new_tokens
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)

        # CPU-first load — never use low_cpu_mem_usage or device_map.
        # Llama 3.2's cross-attention adapter __init__ crashes on meta tensors.
        self.model = MllamaForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
        ).eval()
        self.model = self.model.to(self._device)

        # Cache static prompt text and image token id — both are constants.
        self._prompt_text    = self._build_prompt_text()
        self._image_token_id = self._get_image_token_id()

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_image_token_id(self) -> int | None:
        tok = self.processor.tokenizer
        for candidate in ("<|image|>", "<image>"):
            try:
                tid = tok.convert_tokens_to_ids(candidate)
                if tid is not None and tid != tok.unk_token_id:
                    return tid
            except Exception:
                pass
        return None

    def _build_prompt_text(self) -> str:
        """Build the chat-formatted prompt string once at init."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._PROMPT},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _processor_inputs(self, pil: Image.Image) -> dict[str, Any]:
        enc = self.processor(
            text=self._prompt_text,
            images=pil,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            k: v.to(self._device) if torch.is_tensor(v) else v
            for k, v in enc.items()
        }

    def _vision_kwargs(self, enc: dict[str, Any]) -> dict[str, Any]:
        return {
            k: enc[k]
            for k in ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask")
            if k in enc
        }

    def _transcript_ids(self, transcript: str) -> torch.Tensor:
        """Tokenise transcript alone (no special tokens) → (1, T)."""
        return self.processor.tokenizer(
            transcript,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self._device)

    def _normalize_prediction(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^\s*assistant\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"^Transcribe exactly all visible text\. Output only the text, nothing else\.?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    # ── Differentiable image path ──────────────────────────────────────────

    def _preprocess(self, image_tensor: torch.Tensor, pil: Image.Image) -> torch.Tensor:
        """
        Return pixel_values that are numerically exact to the processor output
        at initialisation but carry gradient from image_tensor.

        MllamaImageProcessor applies aspect-ratio-aware tiling that cannot be
        replicated by crop+resize+normalise (produces cosine_sim < 0).

        Perturbation injection approach:
            pv_proc  = processor(pil)          — exact, no grad
            x_norm   = normalise(resize(x))    — differentiable
            delta    = x_norm - sg(x_norm)     — zero value, live grad
            return   pv_proc + expand(delta)   — exact at init, grad-connected

        All tiles share the same spatial delta (single global resize).
        This is sufficient for PGD sign updates.
        """
        ip = self.processor.image_processor

        # Step 1: exact processor output (no grad)
        with torch.no_grad():
            ref    = self.processor(
                text=self._prompt_text,
                images=pil,
                return_tensors="pt",
                add_special_tokens=False,
            )
        pv_proc = ref["pixel_values"].to(device=self._device, dtype=self._dtype)
        # shape: (1, 1, n_tiles, 3, H_tile, W_tile)

        H_tile  = pv_proc.shape[-2]
        W_tile  = pv_proc.shape[-1]
        n_tiles = pv_proc.shape[2]

        # Step 2: differentiable normalised representation of image_tensor
        x = image_tensor.to(device=self._device, dtype=self._dtype).unsqueeze(0)
        x = F.interpolate(x, size=(H_tile, W_tile), mode="bilinear", align_corners=False)

        def _to3(v):
            return list(v) if isinstance(v, (list, tuple)) else [v, v, v]

        mean_t = torch.tensor(_to3(ip.image_mean), device=self._device, dtype=self._dtype).view(1, 3, 1, 1)
        std_t  = torch.tensor(_to3(ip.image_std),  device=self._device, dtype=self._dtype).view(1, 3, 1, 1)
        x_norm = (x - mean_t) / std_t  # (1, 3, H_tile, W_tile)

        # Step 3: zero-valued delta with live gradient
        delta = (x_norm - x_norm.detach()).unsqueeze(1).unsqueeze(1).expand(
            -1, 1, n_tiles, -1, -1, -1
        )  # (1, 1, n_tiles, 3, H_tile, W_tile)

        return pv_proc + delta

    def _extend_cross_attention_mask(
        self, vision_kw: dict[str, Any], total_len: int
    ) -> dict[str, Any]:
        """
        cross_attention_mask is built from prompt tokens only.
        Extend it to total_len by zero-padding (transcript tokens don't
        attend to image).
        """
        if "cross_attention_mask" not in vision_kw:
            return vision_kw
        cam = vision_kw["cross_attention_mask"]
        if cam.size(1) >= total_len:
            return vision_kw
        pad = torch.zeros(
            cam.size(0), total_len - cam.size(1), *cam.shape[2:],
            device=self._device, dtype=cam.dtype,
        )
        return {**vision_kw, "cross_attention_mask": torch.cat([cam, pad], dim=1)}

    # ── SurrogateModel API ─────────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | image, chat prompt)

        Tail-length label masking:
            full_ids = prompt_ids ++ transcript_ids
            labels   = [-100 ... -100 | transcript_ids]
            image token positions also masked to -100
        """
        pil          = _tensor_to_pil(image_tensor)
        pixel_values = self._preprocess(image_tensor, pil)
        prompt_enc   = self._processor_inputs(pil)

        transcript_ids = self._transcript_ids(transcript)   # (1, T)
        t_len          = transcript_ids.size(1)

        full_ids  = torch.cat([prompt_enc["input_ids"], transcript_ids], dim=1)
        full_attn = torch.cat([
            prompt_enc["attention_mask"],
            torch.ones((1, t_len), device=self._device, dtype=prompt_enc["attention_mask"].dtype),
        ], dim=1)
        total_len = full_ids.size(1)

        labels = torch.full((1, total_len), -100, device=self._device, dtype=torch.long)
        labels[0, -t_len:] = transcript_ids[0]
        if self._image_token_id is not None:
            labels[full_ids == self._image_token_id] = -100

        vision_kw = self._extend_cross_attention_mask(
            self._vision_kwargs(prompt_enc), total_len
        )

        out = self.model(
            input_ids=full_ids,
            attention_mask=full_attn,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
            use_cache=False,
            **vision_kw,
        )
        if out.loss is None:
            raise RuntimeError("LlamaVision.ce_loss: model returned loss=None")
        return out.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cos(z_I, z_T)

        z_I: mean-pool last_hidden_state (1, n_images, n_tiles, seq_len, D_vision)
             over all non-batch/non-feature dims → (1, D_vision),
             then project through multi_modal_projector (bfloat16) → (1, D_lm),
             cast to float32, L2-normalise.

        z_T: mean-pool LM input embeddings of transcript → (1, D_lm),
             float32, L2-normalise.

        Returns 0 (no grad) if dimensions don't match after projection.
        """
        pil          = _tensor_to_pil(image_tensor)
        pixel_values = self._preprocess(image_tensor, pil)
        prompt_enc   = self._processor_inputs(pil)

        vis_kwargs = {
            "pixel_values": pixel_values,
            **{k: prompt_enc[k] for k in ("aspect_ratio_ids", "aspect_ratio_mask") if k in prompt_enc},
        }
        vis_out = self.model.model.vision_model(**vis_kwargs)

        # last_hidden_state: (1, n_images, n_tiles, seq_len, D_vision)
        lhs = vis_out.last_hidden_state
        z_i = lhs.reshape(lhs.size(0), -1, lhs.size(-1)).mean(dim=1)  # (1, D_vision)

        projector = getattr(self.model.model, "multi_modal_projector", None)
        if projector is not None:
            z_i = projector(z_i.to(self._dtype))  # (1, D_lm), bfloat16

        z_i = z_i.float()  # (1, D_lm), float32, grad-connected

        with torch.no_grad():
            text_ids = self.processor.tokenizer(
                transcript, add_special_tokens=True, return_tensors="pt"
            ).input_ids.to(self._device)
            z_t = self.model.get_input_embeddings()(text_ids).mean(dim=1).float()

        if z_i.shape[-1] != z_t.shape[-1]:
            return torch.zeros((), device=self._device, dtype=torch.float32)

        z_i = F.normalize(z_i, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        return -(z_i * z_t).sum(dim=-1).squeeze(0)

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pil = _tensor_to_pil(image_tensor)
        enc = self._processor_inputs(pil)

        out = self.model.generate(
            **enc,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        return self._normalize_prediction(
            self.processor.decode(gen_ids, skip_special_tokens=True)
        )