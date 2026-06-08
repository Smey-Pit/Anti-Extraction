"""
Qwen2.5-VL surrogate wrapper — Phase 0 (inference) + Phase 1 (gradient).

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

Phase 1 additions
-----------------
  build_pixel_values()         — differentiable preprocessing (verified mean
                                 diff < 0.004 vs processor output)
  _build_teacher_forced_seq()  — input_ids + labels for teacher-forced forward
  transcript_loss()            — CE loss for full transcription (∂L/∂image flows)
  query_loss()                 — CE loss for single entity query vs target value
  generate_answer()            — generation without grad (for answer-only masking)

Gradient flows: image_tensor → build_pixel_values → model forward → loss.backward()
The PIL conversion path (_generate) is inference-only and breaks the graph intentionally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

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

        # Cache preprocessor constants for gradient path
        self._img_proc       = self.processor.image_processor
        self._patch_size     = self._img_proc.patch_size          # 14
        self._merge_size     = self._img_proc.merge_size          # 2
        self._temporal_patch = self._img_proc.temporal_patch_size # 2
        self._rescale        = self._img_proc.rescale_factor      # 1/255
        self._norm_mean      = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self._device, dtype=torch.float32,
        ).view(3, 1, 1)
        self._norm_std       = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self._device, dtype=torch.float32,
        ).view(3, 1, 1)

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Phase 0: inference ────────────────────────────────────────────────

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

    # ── Phase 1: differentiable preprocessing ────────────────────────────

    def build_pixel_values(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1], requires_grad=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable preprocessing: image_tensor → (pixel_values, image_grid_thw).

        Replicates Qwen's _preprocess pipeline exactly in PyTorch so gradients
        flow back to image_tensor. Verified mean diff < 0.004 vs processor output.

        Returns
        -------
        pixel_values   : (n_patches, patch_dim) bfloat16 — input to model
        image_grid_thw : (1, 3) long — grid metadata
        """
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        _, H_orig, W_orig = image_tensor.shape
        factor = self._patch_size * self._merge_size

        H_r, W_r = smart_resize(
            H_orig, W_orig,
            factor=factor,
            min_pixels=self._img_proc.size.shortest_edge,
            max_pixels=self._img_proc.size.longest_edge,
        )

        # Float resize (grad-compatible) — verified mean diff < 0.004 vs uint8 path
        x = image_tensor.unsqueeze(0) * 255.0           # (1, 3, H, W)
        x = F.interpolate(x, size=(H_r, W_r), mode="bilinear", align_corners=False)
        x = x.squeeze(0)                                 # (3, H_r, W_r)
        x = x * self._rescale                            # [0, 1]
        x = (x - self._norm_mean) / self._norm_std       # normalize

        # Exact patch extraction from Qwen _preprocess source
        ph = pw = self._patch_size
        grid_h  = H_r // ph
        grid_w  = W_r // pw
        ms      = self._merge_size
        tp      = self._temporal_patch

        patches = x.unsqueeze(0).reshape(
            1, 3,
            grid_h // ms, ms, ph,
            grid_w // ms, ms, pw,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        pixel_values = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, tp, -1, -1)
            .reshape(1, grid_h * grid_w, 3 * tp * ph * pw)
            .squeeze(0)
        )   # (n_patches, patch_dim)

        image_grid_thw = torch.tensor(
            [[1, grid_h, grid_w]], dtype=torch.long, device=self._device
        )
        return pixel_values.to(self._dtype), image_grid_thw

    # ── Phase 1: sequence builder ─────────────────────────────────────────

    def _build_teacher_forced_seq(
        self,
        pil_image: Image.Image,
        prompt_text: str,
        target_text: str,
        gt_value: str | None = None,
    ) -> dict:
        """
        Build input_ids, attention_mask, labels for a teacher-forced forward pass.

        If gt_value is provided, labels are masked to -100 everywhere except
        the token span of gt_value within target_text (answer-only loss).
        Falls back to full-target loss if gt_value not found in target tokens.

        Returns dict with keys: input_ids, attention_mask, labels, prompt_len.
        Does NOT include pixel_values — those come from build_pixel_values().
        """
        chat_text = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = self.processor(
            text=[chat_text], images=[pil_image], return_tensors="pt"
        )
        prompt_ids = prompt_inputs["input_ids"].to(self._device)
        prompt_len = prompt_ids.shape[1]

        # Tokenize target
        target_ids_raw = self.processor.tokenizer(
            target_text, add_special_tokens=False
        ).input_ids
        eos_id = self.processor.tokenizer.eos_token_id
        target_ids_list = target_ids_raw + [eos_id]
        target_ids = torch.tensor(
            [target_ids_list], dtype=torch.long, device=self._device
        )

        full_input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(full_input_ids)

        # Default: full target loss
        labels = torch.full_like(full_input_ids, -100)
        labels[:, prompt_len:] = target_ids

        # Answer-only masking if gt_value provided
        if gt_value is not None:
            span = self._find_answer_span(target_ids_raw, gt_value)
            if span is not None:
                start, end = span
                answer_mask = torch.full_like(full_input_ids, -100)
                answer_mask[:, prompt_len + start: prompt_len + end] = \
                    target_ids[:, start:end]
                labels = answer_mask

        return {
            "input_ids":      full_input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "prompt_len":     prompt_len,
        }

    def _find_answer_span(
        self,
        target_ids: list[int],
        gt_value: str,
    ) -> tuple[int, int] | None:
        """
        Find token span of gt_value within target_ids using character mapping.
        Handles BPE context-dependence (e.g. ' Ella' vs 'Ella' in isolation).
        Returns (start, end) or None if not found. Takes last occurrence.
        """
        if not target_ids:
            return None
        decoded       = self.processor.tokenizer.decode(target_ids)
        token_strings = [self.processor.tokenizer.decode([i]) for i in target_ids]

        char_to_token: dict[int, int] = {}
        char_pos = 0
        for t_idx, t_str in enumerate(token_strings):
            for _ in t_str:
                char_to_token[char_pos] = t_idx
                char_pos += 1

        search_start = 0
        last_match   = None
        while True:
            idx = decoded.find(gt_value, search_start)
            if idx == -1:
                break
            end_char = idx + len(gt_value) - 1
            if idx in char_to_token and end_char in char_to_token:
                last_match = (char_to_token[idx], char_to_token[end_char] + 1)
            search_start = idx + 1
        return last_match

    # ── Phase 1: generation without grad (for answer-only masking) ────────

    @torch.no_grad()
    def generate_answer(
        self,
        image_tensor: torch.Tensor,
        prompt_text: str,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate the model's actual answer for a given prompt.
        Used to get a low-loss target for teacher-forcing in query_loss().
        """
        pil = tensor_to_pil(image_tensor)
        chat_text = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
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
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        return self.processor.decode(
            out[0][prompt_len:], skip_special_tokens=True
        ).strip()

    # ── Phase 1: differentiable losses ───────────────────────────────────

    def transcript_loss(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1], requires_grad=True
        full_text: str,
    ) -> torch.Tensor:
        """
        CE loss for full-image transcription.

        Gradient flows: image_tensor → build_pixel_values → model → loss.
        Maximize this loss to disrupt transcription (Stage 1 structural attack).
        Minimize to preserve transcription (readability preservation).

        Parameters
        ----------
        image_tensor : (3, H, W) float32 [0,1] with requires_grad=True
        full_text    : ground-truth full transcript (from labels_pil.jsonl)

        Returns
        -------
        Scalar CE loss tensor with gradient to image_tensor.
        """
        pil = tensor_to_pil(image_tensor)

        pixel_values, image_grid_thw = self.build_pixel_values(image_tensor)

        seq = self._build_teacher_forced_seq(
            pil, self._DEFAULT_TRANSCRIBE_PROMPT, full_text
        )

        outputs = self.model(
            input_ids=seq["input_ids"],
            attention_mask=seq["attention_mask"],
            pixel_values=pixel_values.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            labels=seq["labels"],
        )
        return outputs.loss

    def query_loss(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1], requires_grad=True
        question: str,
        target_value: str,
        use_generated: bool = False,
        gt_value: str | None = None,
    ) -> torch.Tensor:
        """
        CE loss for a single entity query against a target value.

        Use cases:
          - target_value = ground truth, minimize → preserve binding
          - target_value = decoy value,  minimize → inject decoy (Stage 2)
          - target_value = ground truth, maximize → disrupt binding (Stage 1 check)

        Parameters
        ----------
        image_tensor  : (3, H, W) float32 [0,1] with requires_grad=True
        question      : entity question string
        target_value  : value to score against (gt or decoy)
        use_generated : if True, generate model's answer first and use that
                        as target (ensures low loss, clean gradient signal)
        gt_value      : if provided alongside use_generated, apply answer-only
                        masking to the generated response

        Returns
        -------
        Scalar CE loss tensor with gradient to image_tensor.
        """
        pil = tensor_to_pil(image_tensor)

        if use_generated:
            tf_target = self.generate_answer(image_tensor, question)
            gv = gt_value
        else:
            tf_target = target_value
            gv = gt_value

        pixel_values, image_grid_thw = self.build_pixel_values(image_tensor)

        seq = self._build_teacher_forced_seq(
            pil, question, tf_target, gt_value=gv
        )

        outputs = self.model(
            input_ids=seq["input_ids"],
            attention_mask=seq["attention_mask"],
            pixel_values=pixel_values.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            labels=seq["labels"],
        )
        return outputs.loss