
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vlm_suppress.models.base import SurrogateModel


class InternVL2(SurrogateModel):
    """
    InternVL2 surrogate wrapper.

    ce_loss:    cross-entropy over generated token logits given ground-truth transcript.
    align_loss: negative cosine similarity between visual and text CLS embeddings.

    NOTE: InternVL2's visual encoder is differentiable w.r.t. pixel input.
    We detach the language decoder hidden states when computing align_loss
    to keep gradients localised to the vision encoder pathway.
    """

    def __init__(self, cfg, torch_dtype=torch.bfloat16) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.name = cfg.name
        self._dtype = torch_dtype
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=True,
        ).eval()

        self._max_new_tokens = cfg.max_new_tokens

        # Fixed extraction prompt
        self._prompt = (
            "<image>\nTranscribe exactly all visible text in this image. "
            "Output only the text, nothing else."
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize + normalise image tensor to InternVL2's expected input.
        image_tensor: (3, H, W) float32 [0,1]
        Returns: (1, 3, 448, 448) bfloat16 on model device.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=self._dtype)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=self._dtype)
        x = image_tensor.to(device=self._device, dtype=self._dtype)
        x = F.interpolate(x.unsqueeze(0), size=(448, 448), mode="bilinear", align_corners=False)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return x  # (1, 3, 448, 448)

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        pixel_values = self._preprocess(image_tensor)

        # Tokenise the target transcript
        target_ids = self.tokenizer(
            transcript,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)  # (1, T)

        # Teacher-forced forward pass to get logits over target tokens
        # This is the standard cross-entropy attack against autoregressive VLMs.
        # We pass pixel_values through the vision encoder (differentiable)
        # and compute CE over the decoder's logits for the target sequence.
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=target_ids,
            labels=target_ids,
            return_dict=True,
        )
        # outputs.loss is already -log p(T | X_delta) — this is F_ce^k
        return outputs.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        pixel_values = self._preprocess(image_tensor)

        # Visual embedding: mean-pool patch features from vision encoder
        vision_out = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        z_I = vision_out.last_hidden_state.mean(dim=1)  # (1, D)
        z_I = F.normalize(z_I, dim=-1)

        # Text embedding: encode the transcript with the language model's embeddings
        with torch.no_grad():
            text_ids = self.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self._device)
            text_emb = self.model.language_model.get_input_embeddings()(text_ids)
            z_T = text_emb.mean(dim=1)
            z_T = F.normalize(z_T, dim=-1)

        # F_align^k = -sim(z_I, z_T): higher = less aligned = more suppression
        sim = (z_I * z_T).sum(dim=-1)
        return -sim.squeeze()

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pixel_values = self._preprocess(image_tensor)
        prompt_ids = self.tokenizer(
            self._prompt, return_tensors="pt"
        ).input_ids.to(self._device)
        out = self.model.generate(
            pixel_values=pixel_values,
            input_ids=prompt_ids,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the prompt echo if present
        if self._prompt.strip() in decoded:
            decoded = decoded.replace(self._prompt.strip(), "").strip()
        return decoded