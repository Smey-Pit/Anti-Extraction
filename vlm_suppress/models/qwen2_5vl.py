from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from vlm_suppress.models.base import SurrogateModel

# Qwen2.5-VL uses the same patch geometry as Qwen2-VL:
#   temporal_patch_size = 2, patch_size = 14
#   pixel_values: (N_patches, 3 * 2 * 14 * 14) = (N_patches, 1176)
#   image_grid_thw: [[1, n_rows, n_cols]]
_PATCH_ROWS = 14
_PATCH_COLS = 28   # = temporal_patch_size * patch_size
_CHANNELS   = 3
_FLAT_DIM   = _CHANNELS * _PATCH_ROWS * _PATCH_COLS   # 1176

# Qwen2.5-VL normalisation constants (same as Qwen2-VL / CLIP)
_MEAN = [0.48145466, 0.4578275,  0.40821073]
_STD  = [0.26862954, 0.26130258, 0.27577711]


class Qwen2_5VL(SurrogateModel):
    """
    Qwen2.5-VL surrogate wrapper.

    Uses Qwen2_5VLForConditionalGeneration — the correct class for
    Qwen2.5-VL checkpoints (qwen2_5_vl architecture).  Do NOT use
    Qwen2VLForConditionalGeneration here; that class targets qwen2_vl
    and has a different vision-encoder MLP (fc1/fc2 vs SwiGLU).

    Differentiability design (same as Qwen2VL wrapper):
    ─────────────────────────────────────────────────────
    The processor pipeline (PIL → normalise → tile) breaks autograd.
    We therefore split the forward pass:

      1. STATIC (no grad, cached per image):
         Run processor once on a PIL copy to get input_ids,
         attention_mask, image_grid_thw.  These don't change with delta.

      2. DYNAMIC (differentiable, recomputed each step):
         Build pixel_values directly from the attack tensor using the
         grid geometry from image_grid_thw.  This stays in autograd.

    This guarantees gradients flow from the loss back to image_tensor.
    """

    def __init__(self, cfg, torch_dtype: torch.dtype | None = None) -> None:
        from transformers import AutoProcessor
        # Transformers uses two naming conventions across versions:
        #   Qwen2_5VLForConditionalGeneration  (no extra underscore, newer)
        #   Qwen2_5_VLForConditionalGeneration (double underscore, older)
        # Fall back to direct submodule import if neither is exported at top level.
        try:
            from transformers import Qwen2_5VLForConditionalGeneration as _ModelCls
        except ImportError:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
            except ImportError:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VLForConditionalGeneration as _ModelCls,
                )

        self.name    = cfg.name
        has_cuda     = torch.cuda.is_available()
        self._device = torch.device("cuda" if has_cuda else "cpu")
        self._dtype  = torch_dtype if torch_dtype else (
            torch.bfloat16 if has_cuda else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )
        self.model = _ModelCls.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device).eval()

        self._max_new_tokens = cfg.max_new_tokens
        self._prompt = (
            "Read the text in this image and output it exactly as written. "
            "Output the text only, no coordinates, no descriptions, no explanations."
        )

        # Locate the vision encoder — attribute path varies across versions.
        # Cache it once so align_loss doesn't repeat the search.
        self._visual = self._find_visual()

        # Normalisation tensors — built once, moved to device on first use
        self._mean: torch.Tensor | None = None
        self._std:  torch.Tensor | None = None

    def _find_visual(self) -> torch.nn.Module:
        """
        Return the vision-encoder submodule from self.model.

        Different transformers versions nest it differently:
          self.model.visual         — Qwen2-VL style (visual at top level)
          self.model.model.visual   — Qwen2.5-VL style (visual inside backbone)

        Falls back to scanning all named children for any module whose name
        contains 'visual' or 'vision'.
        """
        for attr_path in ("visual", "model.visual"):
            obj = self.model
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, torch.nn.Module):
                return obj

        # Last resort: first child whose name contains 'visual' or 'vision'
        for name, module in self.model.named_children():
            if "visual" in name or "vision" in name:
                return module

        raise AttributeError(
            f"Cannot locate vision encoder on {type(self.model).__name__}. "
            "Tried: .visual, .model.visual, named_children scan. "
            "Inspect model.named_children() and update _find_visual."
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def _get_norm(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mean is None:
            self._mean = torch.tensor(_MEAN, device=self._device, dtype=torch.float32).view(1, 3, 1, 1)
            self._std  = torch.tensor(_STD,  device=self._device, dtype=torch.float32).view(1, 3, 1, 1)
        return self._mean, self._std

    # ── Static inputs (no grad) ────────────────────────────────────────────────

    def _get_static_inputs(self, image_tensor: torch.Tensor) -> dict:
        """
        Run processor once on a PIL copy to get input_ids, attention_mask,
        image_grid_thw.  These are independent of the attack delta.
        Returns dict on self._device, pixel_values excluded.
        """
        arr  = (image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        pil  = Image.fromarray(arr)
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self._prompt},
            ]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[pil], return_tensors="pt")
        return {
            "input_ids":      inputs["input_ids"].to(self._device),
            "attention_mask": inputs["attention_mask"].to(self._device),
            "image_grid_thw": inputs["image_grid_thw"].to(self._device),
        }

    # ── Dynamic pixel_values (differentiable) ─────────────────────────────────

    def _build_pixel_values(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1], requires_grad
        grid_thw:     torch.Tensor,   # (1, 3) → [[T, n_rows, n_cols]]
    ) -> torch.Tensor:
        """
        Build pixel_values directly from image_tensor, staying in autograd graph.

        Grid geometry read from grid_thw (set by processor, stable per image).
        Output: (N_patches, 1176) in model dtype, gradients intact.
        """
        _, n_rows, n_cols = grid_thw[0].tolist()
        n_rows, n_cols    = int(n_rows), int(n_cols)
        target_h = n_rows * _PATCH_ROWS
        target_w = n_cols * _PATCH_COLS

        mean, std = self._get_norm()

        # 1. Move to device, keep float32 for grad stability
        x = image_tensor.to(device=self._device, dtype=torch.float32)

        # 2. Resize to exact patch-grid pixel dimensions
        x = F.interpolate(
            x.unsqueeze(0), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )  # (1, 3, target_h, target_w)

        # 3. Normalise — stays in graph
        x = (x - mean) / std

        # 4. Unfold into (n_rows, n_cols) patches of size (PATCH_ROWS, PATCH_COLS)
        x = x.unfold(2, _PATCH_ROWS, _PATCH_ROWS)  # (1, 3, n_rows, target_w, 14)
        x = x.unfold(3, _PATCH_COLS, _PATCH_COLS)  # (1, 3, n_rows, n_cols, 14, 28)

        # 5. Rearrange to (N_patches, C, PATCH_ROWS, PATCH_COLS)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (1, n_rows, n_cols, 3, 14, 28)
        x = x.reshape(n_rows * n_cols, _CHANNELS, _PATCH_ROWS, _PATCH_COLS)

        # 6. Flatten each patch to 1176
        x = x.reshape(n_rows * n_cols, _FLAT_DIM)  # (N_patches, 1176)

        # 7. Cast to model dtype — gradient still flows through float32 ops
        return x.to(dtype=self._dtype)

    # ── Label building ─────────────────────────────────────────────────────────

    def _build_labels(self, seq_len: int, transcript: str) -> torch.Tensor:
        """
        Build causal LM labels: -100 for all prompt tokens, target ids at tail.
        Shape: (1, seq_len) matching model logits.
        """
        target_ids = self.processor.tokenizer(
            transcript, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self._device)   # (1, T_target)

        t_len  = target_ids.shape[1]
        labels = torch.full((1, seq_len), -100, dtype=torch.long, device=self._device)
        if t_len <= seq_len:
            labels[:, -t_len:] = target_ids
        else:
            labels[:, :] = target_ids[:, :seq_len]
        return labels

    # ── SurrogateModel interface ───────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        static = self._get_static_inputs(image_tensor)
        pv     = self._build_pixel_values(image_tensor, static["image_grid_thw"])
        labels = self._build_labels(static["input_ids"].shape[1], transcript)

        out = self.model(
            input_ids=static["input_ids"],
            attention_mask=static["attention_mask"],
            pixel_values=pv,
            image_grid_thw=static["image_grid_thw"],
            labels=labels,
            return_dict=True,
        )
        return out.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        static = self._get_static_inputs(image_tensor)
        pv     = self._build_pixel_values(image_tensor, static["image_grid_thw"])

        # Visual embeddings — differentiable through pv
        vision_out = self._visual(
            pv.to(self._dtype),
            grid_thw=static["image_grid_thw"],
        )
        if hasattr(vision_out, "pooler_output"):
            z_I = vision_out.pooler_output
            if isinstance(z_I, (list, tuple)):
                z_I = torch.cat(z_I, dim=0)
        else:
            z_I = vision_out
        z_I = z_I.mean(dim=0, keepdim=True).float()
        z_I = F.normalize(z_I, dim=-1)

        with torch.no_grad():
            text_ids = self.processor.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True,
            ).input_ids.to(self._device)
            z_T = self.model.get_input_embeddings()(text_ids).mean(dim=1).float()
            z_T = F.normalize(z_T, dim=-1)

        return -(z_I * z_T).sum(dim=-1).squeeze()

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        static = self._get_static_inputs(image_tensor)
        pv     = self._build_pixel_values(image_tensor, static["image_grid_thw"])

        out = self.model.generate(
            input_ids=static["input_ids"],
            attention_mask=static["attention_mask"],
            pixel_values=pv.to(self._dtype),
            image_grid_thw=static["image_grid_thw"],
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        input_len = static["input_ids"].shape[1]
        return self.processor.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()

    @torch.no_grad()
    def token_logprobs(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
        transcript: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-token log probabilities for transcript given image.

        Returns
        -------
        log_probs : (T,) float32 on self._device
        token_ids : (T,) int64  on self._device
        """
        static = self._get_static_inputs(image_tensor)
        pv     = self._build_pixel_values(image_tensor, static["image_grid_thw"])

        target_ids = self.processor.tokenizer(
            transcript, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self._device)   # (1, T)
        T = target_ids.size(1)

        full_input_ids = torch.cat([static["input_ids"], target_ids], dim=1)
        full_attn = torch.cat([
            static["attention_mask"],
            torch.ones(1, T, device=self._device, dtype=static["attention_mask"].dtype),
        ], dim=1)

        out = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attn,
            pixel_values=pv,
            image_grid_thw=static["image_grid_thw"],
            return_dict=True,
        )

        transcript_logits = out.logits[0, -T - 1:-1, :].float()   # (T, vocab)
        log_probs = F.log_softmax(transcript_logits, dim=-1)
        tok_ids   = target_ids[0]                                  # (T,)
        token_lp  = log_probs.gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        return token_lp, tok_ids
