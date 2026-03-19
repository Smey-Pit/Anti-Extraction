
"""
LLaVA-1.6 wrapper — used as HELD-OUT Tier-2 transfer target (week 3+).
NOT included in the optimisation ensemble.
Implements the same SurrogateModel interface for fair eval.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from vlm_suppress.models.base import SurrogateModel


class LLaVA16(SurrogateModel):

    def __init__(self, cfg, torch_dtype=torch.bfloat16) -> None:
        self.name = cfg.name
        self._dtype = torch_dtype
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = LlavaNextProcessor.from_pretrained(cfg.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
        ).eval()

        self._max_new_tokens = cfg.max_new_tokens
        self._prompt = (
            "[INST] <image>\nTranscribe exactly all visible text. "
            "Output only the text, nothing else. [/INST]"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def _preprocess(self, image_tensor: torch.Tensor):
        """Returns processor inputs dict with pixel_values on device."""
        from PIL import Image
        import numpy as np
        arr = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr)
        inputs = self.processor(
            text=self._prompt, images=pil, return_tensors="pt"
        )
        # Re-extract pixel_values as a differentiable tensor from image_tensor
        # (processor output is non-differentiable; we reconstruct it)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=self._dtype)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=self._dtype)
        x = image_tensor.to(device=self._device, dtype=self._dtype)
        x = F.interpolate(x.unsqueeze(0), size=(336, 336), mode="bilinear", align_corners=False)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return inputs.to(self._device), x

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        inputs, pixel_values = self._preprocess(image_tensor)
        target_ids = self.processor.tokenizer(
            transcript, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)
        outputs = self.model(
            **{k: v for k, v in inputs.items() if k != "pixel_values"},
            pixel_values=pixel_values,
            labels=target_ids,
            return_dict=True,
        )
        return outputs.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        _, pixel_values = self._preprocess(image_tensor)
        vision_out = self.model.vision_tower(pixel_values, return_dict=True)
        z_I = vision_out.last_hidden_state.mean(dim=1)
        z_I = F.normalize(z_I.float(), dim=-1)
        with torch.no_grad():
            text_ids = self.processor.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self._device)
            text_emb = self.model.get_input_embeddings()(text_ids).mean(dim=1)
            z_T = F.normalize(text_emb.float(), dim=-1)
        return -(z_I * z_T).sum(dim=-1).squeeze()

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        from PIL import Image
        import numpy as np
        arr = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr)
        inputs = self.processor(
            text=self._prompt, images=pil, return_tensors="pt"
        ).to(self._device)
        out = self.model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self.processor.decode(out[0], skip_special_tokens=True)
