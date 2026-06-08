"""
InternVL3.5-8B surrogate wrapper — Phase 0 (inference only).

Architecture:
  Vision encoder : InternViT-300M
  Connector      : Pixel-shuffle + MLP
  LM backbone    : Qwen3 (8B variant)
  Tiling         : Dynamic high-resolution tiling (1..12 tiles + thumbnail)

Reference checkpoint: OpenGVLab/InternVL3_5-8B
Validated transformers: 5.4.0+

Notes
-----
InternVL exposes a `model.chat()` method that handles the Qwen-style chat
template, image-token expansion and dynamic tiling. For Phase 0 we use it
directly — it's what the original wrapper used for transcribe() and it
matches the model's training conditions.

The dynamic preprocessing function is replicated from the original wrapper
to avoid depending on its private code path.
"""

from __future__ import annotations

import re

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from vlm_suppress.models._utils import tensor_to_pil
from vlm_suppress.models.base import SurrogateModel


_TILE_SIZE = 448
_MAX_TILES = 12

# InternVL uses ImageNet normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class InternVL35(SurrogateModel):

    _DEFAULT_TRANSCRIBE_PROMPT = (
        "Transcribe exactly all visible text in the image. "
        "Preserve line breaks. Output only the text."
    )

    def __init__(self, cfg) -> None:
        self.name = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16
        self._max_new_tokens = cfg.max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            use_fast=False,
        )

        # InternVL's remote code (modeling_intern_vit.py) calls
        #   [x.item() for x in torch.linspace(...)]
        # inside InternVisionEncoder.__init__. With newer transformers
        # versions, model construction happens under a meta-device default
        # and that .item() call crashes. Force CPU as the default device
        # during construction so the linspace lands on real tensors, then
        # move to the target device after weights are loaded.
        with torch.device("cpu"):
            self.model = AutoModel.from_pretrained(
                cfg.model_id,
                trust_remote_code=True,
                dtype=self._dtype,
                low_cpu_mem_usage=False,
            ).eval()
        self.model = self.model.to(self._device)

        # Resolve the image-context token id so model.chat() can find it.
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Image preprocessing for transcribe (dynamic tiling) ────────────────

    def _dynamic_preprocess(self, pil: Image.Image) -> torch.Tensor:
        """
        InternVL's aspect-ratio-aware tiling.

        Chooses (ncols, nrows) within MAX_TILES so the grid aspect ratio
        matches the image, crops into TILE_SIZE patches, optionally appends
        a full-image thumbnail. Returns (N, 3, TILE_SIZE, TILE_SIZE).
        """
        img = pil.convert("RGB")
        W, H = img.size

        grids = {
            (i, j)
            for n in range(1, _MAX_TILES + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= _MAX_TILES
        }
        ncols, nrows = min(grids, key=lambda rc: abs(rc[0] / rc[1] - W / H))

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        resized = img.resize((ncols * _TILE_SIZE, nrows * _TILE_SIZE), Image.BICUBIC)
        patches = [
            tfm(resized.crop(
                (c * _TILE_SIZE, r * _TILE_SIZE,
                 (c + 1) * _TILE_SIZE, (r + 1) * _TILE_SIZE)
            ))
            for r in range(nrows) for c in range(ncols)
        ]
        if ncols * nrows > 1:
            patches.append(tfm(img.resize((_TILE_SIZE, _TILE_SIZE), Image.BICUBIC)))
        return torch.stack(patches).to(device=self._device, dtype=self._dtype)

    # ── Shared generation backend ──────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, image_tensor: torch.Tensor, user_text: str) -> str:
        pil = tensor_to_pil(image_tensor)
        pixel_values = self._dynamic_preprocess(pil)

        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=user_text,
            generation_config=dict(
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            ),
            history=None,
            return_history=True,
        )
        # Strip any <think>...</think> trace InternVL emits with its
        # reasoning-mode default templates.
        return _THINK_RE.sub("", response).strip()

    # ── Public API ────────────────────────────────────────────────────────

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