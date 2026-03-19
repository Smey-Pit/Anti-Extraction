from __future__ import annotations
import torch
import torch.nn.functional as F

from vlm_suppress.models.base import SurrogateModel


class QwenVL(SurrogateModel):
    """
    Qwen-VL-Chat surrogate wrapper.
    Same interface contract as InternVL2.
    """

    def __init__(self, cfg, torch_dtype=torch.bfloat16) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = cfg.name
        self._dtype = torch_dtype
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=True,
        ).eval()

        self._max_new_tokens = cfg.max_new_tokens

    @property
    def device(self) -> torch.device:
        return self._device

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=self._device, dtype=self._dtype)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=self._device, dtype=self._dtype)
        x = image_tensor.to(device=self._device, dtype=self._dtype)
        x = F.interpolate(x.unsqueeze(0), size=(448, 448), mode="bilinear", align_corners=False)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return x

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        pixel_values = self._preprocess(image_tensor)
        target_ids = self.tokenizer(
            transcript, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

        outputs = self.model(
            images=pixel_values,
            input_ids=target_ids,
            labels=target_ids,
            return_dict=True,
        )
        return outputs.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        pixel_values = self._preprocess(image_tensor)

        # Qwen-VL visual encoder output
        vision_out = self.model.transformer.visual(pixel_values)
        z_I = vision_out.mean(dim=1)
        z_I = F.normalize(z_I.float(), dim=-1)

        with torch.no_grad():
            text_ids = self.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self._device)
            text_emb = self.model.transformer.wte(text_ids).mean(dim=1)
            z_T = F.normalize(text_emb.float(), dim=-1)

        sim = (z_I * z_T).sum(dim=-1)
        return -sim.squeeze()

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pixel_values = self._preprocess(image_tensor)
        query = self.tokenizer.from_list_format([
            {"image": "placeholder"},
            {"text": "Transcribe exactly all visible text. Output only the text."},
        ])
        inputs = self.tokenizer(query, return_tensors="pt").to(self._device)
        # Inject pixel values
        inputs["images"] = pixel_values
        out = self.model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

