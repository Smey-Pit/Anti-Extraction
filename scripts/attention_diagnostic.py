"""
scripts/attention_diagnostic.py

Attention heatmap diagnostic for all five surrogate VLMs:
  qwen2_5vl, internvl3_5, llama3_2, llava1_6, paligemma2

For each target word, extracts and visualises how much the model attends to
image patches when generating that word's tokens (teacher-forced).

Self/cross-attention ratio (attn_on_word_patches / total_img_attn) is the key
metric: a low value means the model is NOT looking at the word's own pixels
when generating it.

Usage:
    uv run python scripts/attention_diagnostic.py \\
        --config configs/attack.yaml \\
        --words Thompson Ella 633-114 \\
        --category banking \\
        --surrogate qwen2_5vl \\
        --output-dir outputs/attention_diagnostics

Valid --surrogate values: qwen2_5vl, internvl3_5, llama3_2, llava1_6, paligemma2
"""

from __future__ import annotations

import abc
import argparse
import csv
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import dacite
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_suppress.config import (
    Domain, EnsembleWeighting, ExperimentConfig, ObjectiveConfig, ProxyStage,
)
from vlm_suppress.data.dataset import TextImageDataset
from vlm_suppress.attack.importance import _align_tokens_to_words

# ── Constants ──────────────────────────────────────────────────────────────────
LAST_N_LAYERS = 4

QWEN_PROMPT = (
    "Read the text in this image and output it exactly as written. "
    "Output the text only, no coordinates, no descriptions, no explanations."
)

LLAVA_QUESTION = (
    "Transcribe all text in this image exactly as it appears. "
    "Do not add any explanation, formatting, or preamble. "
    "Output only the raw text content, nothing else."
)

LLAMA_PROMPT = (
    "Perform a complete OCR scan of this document. "
    "Output every single line of text exactly as it appears, "
    "reading from the top of the document to the very bottom. "
    "Include all headers, body text, numbers, dates, amounts, names, and table contents. "
    "Do not skip, summarize, or omit any line. "
    "Output only the raw text content, nothing else."
)

INTERNVL_SYSTEM = (
    "You are an OCR engine. "
    "Transcribe exactly all visible text in the image. "
    "Preserve line breaks. Output only the text."
)
INTERNVL_QUESTION = (
    "Transcribe exactly all visible text in the image. "
    "Preserve line breaks. Output only the text."
)

PALIGEMMA_PROMPT = "<image>ocr\n"


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_cfg(config: Path) -> ExperimentConfig:
    with config.open() as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(
        data_class=ExperimentConfig,
        data=raw,
        config=dacite.Config(
            cast=[Path, Domain, ProxyStage, ObjectiveConfig, EnsembleWeighting],
            type_hooks={Optional[tuple[int, int]]: lambda v: tuple(v) if v else None},
        ),
    )


# ── Patch geometry helpers (model-agnostic) ────────────────────────────────────

def _patch_to_pixel(
    patch_idx: int,
    token_gh: int, token_gw: int,
    orig_h: int,   orig_w: int,
) -> tuple[int, int, int, int]:
    """Flat patch index → (x0, y0, x1, y1) in original image coordinates."""
    row = patch_idx // token_gw
    col = patch_idx % token_gw
    ph  = orig_h / token_gh
    pw  = orig_w / token_gw
    return int(col * pw), int(row * ph), int((col + 1) * pw), int((row + 1) * ph)


def _box_to_patch_indices(
    box: list[float],
    token_gh: int, token_gw: int,
    orig_h: int,   orig_w: int,
) -> list[int]:
    """Word pixel box [x0,y0,x1,y1] → list of overlapping flat patch indices."""
    x0, y0, x1, y1 = box
    ph, pw = orig_h / token_gh, orig_w / token_gw
    r0 = max(0, int(y0 / ph))
    r1 = min(token_gh - 1, int(y1 / ph))
    c0 = max(0, int(x0 / pw))
    c1 = min(token_gw - 1, int(x1 / pw))
    return [r * token_gw + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


# ── Word lookup helpers ────────────────────────────────────────────────────────

def _find_word_box(
    word: str, sample, labels_path: Path
) -> list[float] | None:
    """
    Look up pixel bounding box [x0, y0, x1, y1] for word.
    Priority: sample.word_boxes → labels_pil.json fallback.
    """
    if hasattr(sample, "word_strings") and hasattr(sample, "word_boxes"):
        for w, b in zip(sample.word_strings, sample.word_boxes):
            if w == word:
                return [float(v) for v in b]

    if labels_path.exists():
        with labels_path.open() as f:
            labels = json.load(f)
        label = next((l for l in labels if l["image_id"] == sample.image_id), None)
        if label is not None:
            for line in label["word_boxes"]:
                for entry in line:
                    if entry["word"] == word:
                        return [float(v) for v in entry["box"]]
    return None


def _find_word_index(target: str, words: list[str]) -> int | None:
    """Find index of target word in words list; allows trailing punctuation."""
    strip_re = re.compile(r'^[^\w\-]+|[^\w\-]+$')
    for i, w in enumerate(words):
        if w == target or strip_re.sub("", w) == target:
            return i
    return None


# ── Visualisation ──────────────────────────────────────────────────────────────

def _attn_to_grid(attn: torch.Tensor, token_gh: int, token_gw: int) -> np.ndarray:
    """Reshape (n_img_tokens,) → (token_gh, token_gw) normalised [0,1]."""
    g = attn.reshape(token_gh, token_gw).numpy().astype(np.float32)
    lo, hi = g.min(), g.max()
    return (g - lo) / (hi - lo + 1e-8)


def _save_word_figure(
    pil_img: Image.Image,
    word: str,
    word_box: list[float] | None,
    avg_attn: torch.Tensor,
    per_layer_attn: list[torch.Tensor],
    top5_indices: list[int],
    token_gh: int,
    token_gw: int,
    out_dir: Path,
) -> None:
    """Save 3-panel figure + per-layer heatmaps for one word."""
    orig_w, orig_h = pil_img.size
    safe_word = re.sub(r'[^A-Za-z0-9_\-]', '_', word)

    # Panel 1: annotated image
    ann = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(ann)
    if word_box:
        draw.rectangle(word_box, outline="cyan", width=3)
    for pidx in top5_indices[:5]:
        px0, py0, px1, py1 = _patch_to_pixel(pidx, token_gh, token_gw, orig_h, orig_w)
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)

    # Panel 2: avg attention grid
    grid_norm = _attn_to_grid(avg_attn, token_gh, token_gw)

    # Panel 3: overlay on image
    hm_rgb  = (cm.hot(grid_norm)[:, :, :3] * 255).astype(np.uint8)
    hm_pil  = Image.fromarray(hm_rgb).resize((orig_w, orig_h), Image.BILINEAR)
    overlay = Image.blend(pil_img.convert("RGB"), hm_pil, alpha=0.5)
    draw_ov = ImageDraw.Draw(overlay)
    if word_box:
        draw_ov.rectangle(word_box, outline="cyan", width=3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Attention diagnostic — '{word}'", fontsize=13)

    axes[0].imshow(ann)
    axes[0].set_title("Word box (cyan) · top-5 patches (red)")
    axes[0].axis("off")

    im = axes[1].imshow(grid_norm, cmap="hot", interpolation="nearest",
                        aspect="auto", vmin=0, vmax=1)
    axes[1].set_title(f"Avg attention grid ({token_gh}×{token_gw})")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Attention overlay (cyan = word box)")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"attn_{safe_word}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Per-layer heatmaps
    n_layers = len(per_layer_attn)
    fig2, axes2 = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes2 = [axes2]
    for li, (la, ax) in enumerate(zip(per_layer_attn, axes2)):
        g = _attn_to_grid(la, token_gh, token_gw)
        ax.imshow(g, cmap="hot", interpolation="nearest", aspect="auto",
                  vmin=0, vmax=1)
        ax.set_title(f"Layer -{n_layers - li}")
        ax.axis("off")
    fig2.suptitle(f"Per-layer attention — '{word}'", fontsize=12)
    plt.tight_layout()
    layer_path = out_dir / f"attn_{safe_word}_layers.png"
    fig2.savefig(layer_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {layer_path}")


# ── Abstract probe base ────────────────────────────────────────────────────────

class AttentionProbeBase(abc.ABC):
    """
    Base class for per-surrogate attention probes.

    After load():
      self._tokenizer  — the tokenizer (for _align_tokens_to_words)
      self._prompt_len — prompt token count (set by build_full_inputs)
    """

    _tokenizer = None
    _prompt_len: int = 0

    @abc.abstractmethod
    def load(self, s_cfg, device: str) -> None:
        """Load model and processor onto device."""
        ...

    @abc.abstractmethod
    def build_full_inputs(self, pil_img: Image.Image, transcript: str) -> dict:
        """
        Build all tensors for forward pass.
        Also sets self._prompt_len.
        Returns model-specific dict.
        """
        ...

    @abc.abstractmethod
    def run_forward(self, inputs: dict) -> tuple:
        """
        Single forward pass with output_attentions=True.
        Returns raw attentions tuple from model output.
        """
        ...

    @abc.abstractmethod
    def image_info(self, inputs: dict) -> tuple:
        """
        Returns (img_positions_or_None, token_gh, token_gw).
        img_positions: 1D tensor of seq positions for self-attn models;
                       None for cross-attention models (LLaMA 3.2).
        """
        ...

    @abc.abstractmethod
    def extract_word_attention(
        self,
        attentions: tuple,
        img_positions_or_none,
        seq_positions: range,
        token_gh: int,
        token_gw: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns (avg_attn, per_layer) each shape (token_gh*token_gw,).
        """
        ...

    @property
    def is_cross_attention(self) -> bool:
        return False

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def prompt_len(self) -> int:
        return self._prompt_len


# ── Shared self-attention extraction ──────────────────────────────────────────

def _extract_self_attention(
    attentions: tuple,
    img_positions: torch.Tensor,
    seq_positions: range,
    last_n: int = LAST_N_LAYERS,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Self-attention extraction for Qwen, InternVL, LLaVA, PaliGemma.

    For each of the last `last_n` layers:
      - average over heads
      - average over word span rows
      - extract image-patch columns
    Returns (avg_attn, per_layer) each (n_img_tokens,).
    """
    pos_list = list(seq_positions)
    per_layer: list[torch.Tensor] = []

    for layer_attn in attentions[-last_n:]:
        # (1, n_heads, seq_len, seq_len) → avg heads → (seq_len, seq_len)
        a = layer_attn[0].float().mean(dim=0)
        # avg over word span rows → (seq_len,)
        span = a[pos_list, :].mean(dim=0)
        # columns = image patch positions
        per_layer.append(span[img_positions].cpu())

    avg_attn = torch.stack(per_layer).mean(dim=0)
    return avg_attn, per_layer


# ── QwenProbe ──────────────────────────────────────────────────────────────────

class QwenProbe(AttentionProbeBase):
    """Qwen2.5-VL self-attention probe."""

    _IMAGE_TOKEN_ID = 151655   # <|image_pad|>
    _MERGE_FACTOR   = 2        # spatial 2×2 merge in VL connector

    def load(self, s_cfg, device: str) -> None:
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5VLForConditionalGeneration as _Cls
        except ImportError:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
            except ImportError:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VLForConditionalGeneration as _Cls,
                )

        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            s_cfg.model_id, trust_remote_code=True
        )
        self._tokenizer = self._processor.tokenizer

        try:
            self._model = _Cls.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
            ).eval().to(device)
        except Exception as e:
            warnings.warn(f"QwenProbe: eager attn load failed ({e}), retrying without it.")
            self._model = _Cls.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval().to(device)

    def build_full_inputs(self, pil_img: Image.Image, transcript: str) -> dict:
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": QWEN_PROMPT},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self._processor(text=[text], images=[pil_img], return_tensors="pt")

        prompt_ids = enc["input_ids"].to(self._device)
        pixel_vals = enc["pixel_values"].to(self._device, dtype=torch.bfloat16)
        grid_thw   = enc["image_grid_thw"].to(self._device)
        attn_mask  = enc["attention_mask"].to(self._device)

        transcript_ids = self._processor.tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)

        full_ids  = torch.cat([prompt_ids, transcript_ids], dim=1)
        full_mask = torch.cat([
            attn_mask,
            torch.ones(1, transcript_ids.shape[1], device=self._device, dtype=torch.long),
        ], dim=1)

        self._prompt_len = prompt_ids.shape[1]

        return dict(
            full_ids=full_ids,
            full_mask=full_mask,
            pixel_values=pixel_vals,
            image_grid_thw=grid_thw,
        )

    def run_forward(self, inputs: dict) -> tuple:
        with torch.no_grad():
            try:
                out = self._model(
                    input_ids=inputs["full_ids"],
                    attention_mask=inputs["full_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
            except Exception as e:
                if "attention" in str(e).lower() or "flash" in str(e).lower():
                    raise RuntimeError(
                        f"output_attentions=True failed: {e}\n"
                        "Hint: Flash Attention may be installed. Reload with "
                        "attn_implementation='eager'."
                    ) from e
                raise
        return out.attentions

    def image_info(self, inputs: dict) -> tuple:
        full_ids = inputs["full_ids"]
        grid_thw = inputs["image_grid_thw"]
        img_positions = (full_ids[0] == self._IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        _, gh_raw, gw_raw = [int(x) for x in grid_thw[0].tolist()]
        token_gh = gh_raw // self._MERGE_FACTOR
        token_gw = gw_raw // self._MERGE_FACTOR
        n_img = img_positions.numel()
        expected = token_gh * token_gw
        if expected != n_img:
            warnings.warn(
                f"QwenProbe: grid {token_gh}×{token_gw}={expected} vs "
                f"{n_img} image tokens.",
                RuntimeWarning,
            )
        return img_positions, token_gh, token_gw

    def extract_word_attention(
        self,
        attentions: tuple,
        img_positions_or_none,
        seq_positions: range,
        token_gh: int,
        token_gw: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return _extract_self_attention(attentions, img_positions_or_none, seq_positions)


# ── InternVLProbe ──────────────────────────────────────────────────────────────

class InternVLProbe(AttentionProbeBase):
    """InternVL3.5 self-attention probe (inputs_embeds path)."""

    _IMG_TOKEN = "<IMG_CONTEXT>"
    _IMG_START = "<img>"
    _IMG_END   = "</img>"

    def load(self, s_cfg, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(
            s_cfg.model_id, trust_remote_code=True, use_fast=False
        )

        try:
            self._model = AutoModel.from_pretrained(
                s_cfg.model_id,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                attn_implementation="eager",
            ).eval().to(device)
        except TypeError:
            self._model = AutoModel.from_pretrained(
                s_cfg.model_id,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).eval().to(device)

        self._n_img_tokens: int = self._model.num_image_token
        self._img_ctx_id: int = self._tokenizer.convert_tokens_to_ids(self._IMG_TOKEN)
        self._model.img_context_token_id = self._img_ctx_id

        # Compute grid from n_img_tokens (assumes square layout)
        sq = int(round(math.sqrt(self._n_img_tokens)))
        self._token_gh = sq
        self._token_gw = sq

    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """PIL → (1, 3, 448, 448) bfloat16 on device, ImageNet normalised."""
        import torchvision.transforms.functional as TF
        img = pil_img.convert("RGB").resize((448, 448), Image.BICUBIC)
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0).to(device=self._device, dtype=torch.bfloat16)

    def _build_input_ids(self) -> torch.Tensor:
        img_placeholder = (
            self._IMG_START
            + self._IMG_TOKEN * self._n_img_tokens
            + self._IMG_END
        )
        conversation = (
            f"<|im_start|>system\n{INTERNVL_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{img_placeholder}\n{INTERNVL_QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return self._tokenizer(
            conversation, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

    def build_full_inputs(self, pil_img: Image.Image, transcript: str) -> dict:
        pixel_values = self._preprocess(pil_img)
        input_ids    = self._build_input_ids()

        # Build base inputs_embeds (prompt only)
        with torch.no_grad():
            tok_embeds = self._model.language_model.get_input_embeddings()(input_ids)
        tok_embeds = tok_embeds.to(torch.bfloat16)

        img_embeds = self._model.extract_feature(pixel_values)   # (1, N_img, D)

        img_mask = (input_ids[0] == self._img_ctx_id)
        inputs_embeds = tok_embeds.clone()
        inputs_embeds[0, img_mask] = img_embeds[0]

        self._prompt_len = input_ids.shape[1]

        # Append transcript embeddings
        transcript_ids = self._tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)

        with torch.no_grad():
            tgt_embeds = self._model.language_model.get_input_embeddings()(
                transcript_ids
            ).to(torch.bfloat16)

        full_embeds = torch.cat([inputs_embeds, tgt_embeds], dim=1)

        # Build full_input_ids for img_positions lookup
        full_input_ids = torch.cat([input_ids, transcript_ids], dim=1)

        return dict(
            full_embeds=full_embeds,
            full_input_ids=full_input_ids,
        )

    def run_forward(self, inputs: dict) -> tuple:
        with torch.no_grad():
            try:
                out = self._model.language_model(
                    inputs_embeds=inputs["full_embeds"],
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
            except Exception as e:
                if "attention" in str(e).lower() or "flash" in str(e).lower():
                    raise RuntimeError(
                        f"output_attentions=True failed: {e}\n"
                        "Hint: Flash Attention may be installed. Reload with "
                        "attn_implementation='eager'."
                    ) from e
                raise
        return out.attentions

    def image_info(self, inputs: dict) -> tuple:
        full_ids = inputs["full_input_ids"]
        img_positions = (full_ids[0] == self._img_ctx_id).nonzero(as_tuple=True)[0]
        return img_positions, self._token_gh, self._token_gw

    def extract_word_attention(
        self,
        attentions: tuple,
        img_positions_or_none,
        seq_positions: range,
        token_gh: int,
        token_gw: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return _extract_self_attention(attentions, img_positions_or_none, seq_positions)


# ── LlamaProbe ────────────────────────────────────────────────────────────────

class LlamaProbe(AttentionProbeBase):
    """
    Llama 3.2 Vision cross-attention probe.

    Cross-attention layers have attn shape (1, n_heads, text_len, n_img_features),
    where n_img_features != text_len.  Image features are NOT in the self-attention
    sequence, so img_positions=None.
    """

    _PATCHES_PER_TILE = 1600   # 40×40

    @property
    def is_cross_attention(self) -> bool:
        return True

    def load(self, s_cfg, device: str) -> None:
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        self._device = device

        self._processor = AutoProcessor.from_pretrained(s_cfg.model_id)
        self._tokenizer = self._processor.tokenizer

        try:
            self._model = MllamaForConditionalGeneration.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval().to(device)
        except Exception as e:
            warnings.warn(f"LlamaProbe: eager attn load failed ({e}), retrying without it.")
            self._model = MllamaForConditionalGeneration.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
            ).eval().to(device)

        # Cache static prompt text
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": LLAMA_PROMPT},
        ]}]
        self._prompt_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Cached grid params (filled after first forward)
        self._token_gh: int | None = None
        self._token_gw: int | None = None

    def _extend_cross_attention_mask(
        self, vision_kw: dict, total_len: int
    ) -> dict:
        """
        cross_attention_mask is built from prompt tokens only.
        Extend it to total_len by zero-padding.
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

    def build_full_inputs(self, pil_img: Image.Image, transcript: str) -> dict:
        enc = self._processor(
            text=self._prompt_text,
            images=pil_img,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(self._device) if torch.is_tensor(v) else v
               for k, v in enc.items()}

        self._prompt_len = enc["input_ids"].shape[1]

        transcript_ids = self._processor.tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)
        t_len = transcript_ids.shape[1]

        full_ids  = torch.cat([enc["input_ids"], transcript_ids], dim=1)
        full_mask = torch.cat([
            enc["attention_mask"],
            torch.ones(1, t_len, device=self._device, dtype=torch.long),
        ], dim=1)
        total_len = full_ids.shape[1]

        vision_kw = {
            k: enc[k]
            for k in ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask")
            if k in enc
        }
        vision_kw = self._extend_cross_attention_mask(vision_kw, total_len)

        return dict(
            full_ids=full_ids,
            full_mask=full_mask,
            pixel_values=enc.get("pixel_values"),
            vision_kw=vision_kw,
        )

    def run_forward(self, inputs: dict) -> tuple:
        with torch.no_grad():
            try:
                out = self._model(
                    input_ids=inputs["full_ids"],
                    attention_mask=inputs["full_mask"],
                    pixel_values=inputs["pixel_values"],
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                    **inputs["vision_kw"],
                )
            except Exception as e:
                if "attention" in str(e).lower() or "flash" in str(e).lower():
                    raise RuntimeError(
                        f"output_attentions=True failed: {e}\n"
                        "Hint: Flash Attention may be installed. Reload with "
                        "attn_implementation='eager'."
                    ) from e
                raise
        # Mllama separates cross-attention weights into out.cross_attentions.
        # Merge with out.attentions so the non-square shape filter can find them
        # regardless of transformers version.
        cross     = tuple(getattr(out, "cross_attentions", None) or ())
        self_attn = tuple(out.attentions or ())
        return cross + self_attn

    def image_info(self, inputs: dict) -> tuple:
        # Grid derived from cross-attention weight shapes — call after run_forward
        # which has set self._token_gh / self._token_gw
        if self._token_gh is None:
            raise RuntimeError("LlamaProbe.image_info called before run_forward. "
                               "Call run_forward first, then image_info.")
        return None, self._token_gh, self._token_gw

    def _detect_grid_from_attentions(self, attentions: tuple) -> None:
        """Detect token_gh, token_gw from cross-attention weight shapes."""
        for layer_attn in reversed(attentions):
            if layer_attn is None:
                continue
            s = layer_attn.shape  # (1, n_heads, text_len, n_img_features) or self-attn
            if len(s) == 4 and s[-1] != s[-2]:
                n_img_features = s[-1]
                n_tiles = max(1, n_img_features // self._PATCHES_PER_TILE)
                self._token_gh = n_tiles * 40
                self._token_gw = 40
                return
        # Fallback: single tile
        self._token_gh = 40
        self._token_gw = 40

    def extract_word_attention(
        self,
        attentions: tuple,
        img_positions_or_none,
        seq_positions: range,
        token_gh: int,
        token_gw: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Cross-attention extraction.

        Finds layers where attn.shape[-1] != attn.shape[-2] (cross-attn).
        Uses last LAST_N_LAYERS cross-attention layers.
        For multi-tile: reshape n_img_features → (n_tiles, patches_per_tile),
        sum over tiles → (patches_per_tile,) = (1600,), then reshape to (40, 40).
        Final output flattened: (token_gh * token_gw,).
        """
        pos_list = list(seq_positions)
        cross_layers: list[torch.Tensor] = []

        for layer_attn in attentions:
            if layer_attn is None:
                continue
            s = layer_attn.shape
            if len(s) == 4 and s[-1] != s[-2]:
                cross_layers.append(layer_attn)

        if not cross_layers:
            raise RuntimeError(
                "LlamaProbe: no cross-attention layers found in attentions tuple. "
                "The model may not have returned cross-attention weights."
            )

        selected = cross_layers[-LAST_N_LAYERS:]
        n_img_features = selected[0].shape[-1]
        n_tiles = max(1, n_img_features // self._PATCHES_PER_TILE)
        patches_per_tile = self._PATCHES_PER_TILE  # 1600

        per_layer: list[torch.Tensor] = []
        for layer_attn in selected:
            # (1, n_heads, text_len, n_img_features) → avg heads → (text_len, n_img_features)
            a = layer_attn[0].float().mean(dim=0)
            # avg over word span rows → (n_img_features,)
            span = a[pos_list, :].mean(dim=0)
            # Multi-tile: sum over tiles → (patches_per_tile,)
            if n_tiles > 1:
                # trim to n_tiles * patches_per_tile if needed
                n_use = n_tiles * patches_per_tile
                span_trimmed = span[:n_use]
                span_reshaped = span_trimmed.reshape(n_tiles, patches_per_tile)
                span_pooled = span_reshaped.sum(dim=0)  # (patches_per_tile,)
            else:
                span_pooled = span[:patches_per_tile]

            per_layer.append(span_pooled.cpu())

        avg_attn = torch.stack(per_layer).mean(dim=0)  # (patches_per_tile,)
        return avg_attn, per_layer

    def run_forward_and_detect(self, inputs: dict) -> tuple:
        """Run forward pass and also detect grid geometry."""
        attentions = self.run_forward(inputs)
        self._detect_grid_from_attentions(attentions)
        return attentions


# ── LLaVAProbe ────────────────────────────────────────────────────────────────

class LLaVAProbe(AttentionProbeBase):
    """LLaVA-1.6 (LlavaNext) self-attention probe."""

    _BASE_PATCH_GRID = 24   # CLIP ViT-L/14@336: 336/14 = 24 patches per side

    def load(self, s_cfg, device: str) -> None:
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor

        self._device = device

        self._processor = AutoProcessor.from_pretrained(s_cfg.model_id)
        self._tokenizer = self._processor.tokenizer

        try:
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval().to(device)
        except Exception as e:
            warnings.warn(f"LLaVAProbe: eager attn load failed ({e}), retrying without it.")
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
            ).eval().to(device)

        self._image_token_index: int = self._model.config.image_token_index

        # Build and cache prompt text
        img_token  = self._processor.tokenizer.decode([self._image_token_index])
        bos        = self._processor.tokenizer.bos_token or "<s>"
        prompt_raw = self._processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{img_token}\n{LLAVA_QUESTION}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        self._prompt_text = prompt_raw.removeprefix(bos)

        # Grid params filled in build_full_inputs
        self._token_gh: int = self._BASE_PATCH_GRID
        self._token_gw: int = self._BASE_PATCH_GRID

    def build_full_inputs(self, pil_img: Image.Image, transcript: str) -> dict:
        enc = self._processor(
            text=self._prompt_text, images=pil_img, return_tensors="pt"
        )
        enc = {k: v.to(self._device) if torch.is_tensor(v) else v
               for k, v in enc.items()}

        self._prompt_len = enc["input_ids"].shape[1]

        transcript_ids = self._processor.tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)
        t_len = transcript_ids.shape[1]

        full_ids  = torch.cat([enc["input_ids"], transcript_ids], dim=1)
        full_mask = torch.cat([
            enc["attention_mask"],
            torch.ones(1, t_len, device=self._device, dtype=torch.long),
        ], dim=1)

        # Count image tokens from processor output (prompt input_ids only)
        n_img_tokens = int((enc["input_ids"][0] == self._image_token_index).sum().item())
        # Set grid: token_gh = 24, token_gw = n_img_tokens // 24
        # (all tiles laid out in a wide row for flat visualization)
        self._token_gh = self._BASE_PATCH_GRID
        self._token_gw = max(self._BASE_PATCH_GRID, n_img_tokens // self._BASE_PATCH_GRID)

        return dict(
            full_ids=full_ids,
            full_mask=full_mask,
            pixel_values=enc.get("pixel_values"),
            image_sizes=enc.get("image_sizes"),
        )

    def run_forward(self, inputs: dict) -> tuple:
        with torch.no_grad():
            try:
                out = self._model(
                    input_ids=inputs["full_ids"],
                    attention_mask=inputs["full_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_sizes=inputs.get("image_sizes"),
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
            except Exception as e:
                if "attention" in str(e).lower() or "flash" in str(e).lower():
                    raise RuntimeError(
                        f"output_attentions=True failed: {e}\n"
                        "Hint: Flash Attention may be installed. Reload with "
                        "attn_implementation='eager'."
                    ) from e
                raise
        return out.attentions

    def image_info(self, inputs: dict) -> tuple:
        full_ids = inputs["full_ids"]
        img_positions = (full_ids[0] == self._image_token_index).nonzero(as_tuple=True)[0]
        return img_positions, self._token_gh, self._token_gw

    def extract_word_attention(
        self,
        attentions: tuple,
        img_positions_or_none,
        seq_positions: range,
        token_gh: int,
        token_gw: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return _extract_self_attention(attentions, img_positions_or_none, seq_positions)


# ── PaliGemmaProbe ─────────────────────────────────────────────────────────────

class PaliGemmaProbe(AttentionProbeBase):
    """PaliGemma2 self-attention probe. SigLIP 448×448 → 32×32 = 1024 tokens."""

    _GRID_SIZE = 32   # 448 / 14 = 32 patches per side

    def load(self, s_cfg, device: str) -> None:
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self._device = device

        self._processor = AutoProcessor.from_pretrained(s_cfg.model_id)
        self._tokenizer = self._processor.tokenizer

        try:
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval().to(device)
        except Exception as e:
            warnings.warn(f"PaliGemmaProbe: eager attn load failed ({e}), retrying without it.")
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                s_cfg.model_id,
                torch_dtype=torch.bfloat16,
            ).eval().to(device)

        # Locate image token id
        self._image_token_id: int | None = None
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(self._processor, attr, None)
            if v is not None:
                self._image_token_id = int(v)
                break
        if self._image_token_id is None:
            v = getattr(self._model.config, "image_token_index", None)
            if v is not None:
                self._image_token_id = int(v)

    def build_full_inputs(self, pil_img: Image.Image, transcript: str) -> dict:
        enc = self._processor(
            text=PALIGEMMA_PROMPT,
            images=pil_img,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) if torch.is_tensor(v) else v
               for k, v in enc.items()}

        self._prompt_len = enc["input_ids"].shape[1]

        transcript_ids = self._processor.tokenizer(
            transcript, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)
        t_len = transcript_ids.shape[1]

        full_ids  = torch.cat([enc["input_ids"], transcript_ids], dim=1)
        full_mask = None
        if "attention_mask" in enc:
            full_mask = torch.cat([
                enc["attention_mask"],
                torch.ones(1, t_len, device=self._device, dtype=torch.long),
            ], dim=1)

        return dict(
            full_ids=full_ids,
            full_mask=full_mask,
            pixel_values=enc.get("pixel_values"),
        )

    def run_forward(self, inputs: dict) -> tuple:
        kwargs = dict(
            input_ids=inputs["full_ids"],
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        if inputs.get("full_mask") is not None:
            kwargs["attention_mask"] = inputs["full_mask"]

        with torch.no_grad():
            try:
                out = self._model(**kwargs)
            except Exception as e:
                if "attention" in str(e).lower() or "flash" in str(e).lower():
                    raise RuntimeError(
                        f"output_attentions=True failed: {e}\n"
                        "Hint: Flash Attention may be installed. Reload with "
                        "attn_implementation='eager'."
                    ) from e
                raise
        return out.attentions

    def image_info(self, inputs: dict) -> tuple:
        full_ids = inputs["full_ids"]
        token_gh = self._GRID_SIZE
        token_gw = self._GRID_SIZE

        if self._image_token_id is not None:
            img_positions = (full_ids[0] == self._image_token_id).nonzero(as_tuple=True)[0]
            if img_positions.numel() == 0:
                # Fallback: assume image tokens occupy first 1024 positions
                img_positions = torch.arange(
                    0, token_gh * token_gw, device=full_ids.device, dtype=torch.long
                )
        else:
            # Assume image tokens are at the start of the sequence
            n_img = token_gh * token_gw
            img_positions = torch.arange(0, n_img, device=full_ids.device, dtype=torch.long)

        return img_positions, token_gh, token_gw

    def extract_word_attention(
        self,
        attentions: tuple,
        img_positions_or_none,
        seq_positions: range,
        token_gh: int,
        token_gw: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return _extract_self_attention(attentions, img_positions_or_none, seq_positions)


# ── Probe registry ─────────────────────────────────────────────────────────────

PROBE_REGISTRY: dict[str, type[AttentionProbeBase]] = {
    "qwen2_5vl":   QwenProbe,
    "internvl3_5": InternVLProbe,
    "llama3_2":    LlamaProbe,
    "llava1_6":    LLaVAProbe,
    "paligemma2":  PaliGemmaProbe,
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention heatmap diagnostic for all five surrogate VLMs."
    )
    parser.add_argument("--config",      type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument("--words",       nargs="+", required=True,
                        help="Target words to probe (e.g. Thompson Ella 633-114)")
    parser.add_argument("--category",   type=str,  default="banking")
    parser.add_argument("--surrogate",  type=str,  default="qwen2_5vl",
                        choices=list(PROBE_REGISTRY))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_diagnostics"))
    parser.add_argument("--device",     type=str,  default=None)
    args = parser.parse_args()

    cfg    = _load_cfg(args.config)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = TextImageDataset(
        data_dir            = cfg.data.data_dir,
        data_dir_additional = cfg.data.data_dir_additional,
        image_size          = cfg.data.image_size,
        max_samples         = 1,
        category_filter     = args.category,
    )
    if len(dataset) == 0:
        sys.exit("ERROR: dataset is empty.")
    sample = dataset[0]
    print(f"Sample     : {sample.image_id}")

    pil_img        = sample.image
    orig_w, orig_h = pil_img.size
    transcript     = sample.transcript
    print(f"Transcript : {len(transcript)} chars  {len(transcript.split())} words")
    print(f"GT snippet : {transcript[:80].replace(chr(10), ' ')!r}")

    # Surrogate config
    s_cfg = next((s for s in cfg.surrogates if s.name == args.surrogate), None)
    if s_cfg is None:
        sys.exit(f"ERROR: surrogate '{args.surrogate}' not found in config.")

    # Instantiate probe
    probe_cls = PROBE_REGISTRY[args.surrogate]
    probe = probe_cls()

    is_cross = (args.surrogate == "llama3_2")
    attn_mode = "Cross-attention" if is_cross else "Self-attention"

    print(f"\nLoading {args.surrogate} ({attn_mode} mode, eager attn) …")
    probe.load(s_cfg, device)

    # Build inputs
    print("Building full_ids (prompt + teacher-forced transcript) …")
    inputs = probe.build_full_inputs(pil_img, transcript)

    prompt_len = probe.prompt_len

    if is_cross:
        # For LLaMA, sequence contains only text tokens
        n_total = inputs["full_ids"].shape[1]
        n_trans  = n_total - prompt_len
        print(f"Sequence   : {n_total} tokens  (prompt={prompt_len}, transcript={n_trans})")
        print("  Note     : cross-attn mode — no image tokens in text sequence")
    else:
        # For self-attn models, full_ids includes image tokens
        if "full_ids" in inputs:
            n_total = inputs["full_ids"].shape[1]
        elif "full_embeds" in inputs:
            n_total = inputs["full_embeds"].shape[1]
        else:
            n_total = prompt_len
        n_trans = n_total - prompt_len
        print(f"Sequence   : {n_total} tokens  (prompt={prompt_len}, transcript={n_trans})")

    # Forward pass
    print(f"Forward pass with output_attentions=True …")
    if isinstance(probe, LlamaProbe):
        attentions = probe.run_forward_and_detect(inputs)
    else:
        attentions = probe.run_forward(inputs)
    print(f"Attentions : {len(attentions)} layers  "
          f"shape={tuple(attentions[0].shape) if attentions[0] is not None else 'None'}")

    # Image info (after forward for LLaMA — grid detected from attn shapes)
    img_positions, token_gh, token_gw = probe.image_info(inputs)

    if is_cross:
        n_img = token_gh * token_gw
        print(f"Image grid : {token_gh}×{token_gw}={n_img} patches  "
              f"(cross-attn — features not in sequence)")
    else:
        n_img = img_positions.numel() if img_positions is not None else token_gh * token_gw
        print(f"Image toks : {n_img}  grid={token_gh}×{token_gw}")

    # Token-to-word alignment
    transcript_words = transcript.split()
    n_words          = len(transcript_words)
    spans            = _align_tokens_to_words(probe.tokenizer, transcript, n_words)

    # Labels / box fallback
    labels_path = Path(cfg.data.data_dir) / "labels_pil.json"

    # Output directory + CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict] = []

    # Per-word analysis
    for target_word in args.words:
        print(f"\n── Word: {target_word!r} {'─'*40}")

        word_idx = _find_word_index(target_word, transcript_words)
        if word_idx is None:
            print(f"  WARNING: '{target_word}' not found in transcript — skipping.")
            continue

        tok_start, tok_end = spans[word_idx]
        seq_start = prompt_len + tok_start
        seq_end   = prompt_len + tok_end
        seq_positions = range(seq_start, seq_end)

        print(f"  Token span:    [{tok_start}, {tok_end})")
        print(f"  Seq position:  {seq_start}")

        # Word bounding box
        word_box = _find_word_box(target_word, sample, labels_path)
        if word_box:
            x0, y0, x1, y1 = word_box
            print(f"  Word pixel box: ({int(x0)},{int(y0)})→({int(x1)},{int(y1)})")
            word_patch_idx = _box_to_patch_indices(
                word_box, token_gh, token_gw, orig_h, orig_w
            )
        else:
            print("  Word pixel box: not found in labels")
            word_patch_idx = []
        print(f"  Word patch indices: {word_patch_idx}")

        # Attention extraction
        avg_attn, per_layer_attn = probe.extract_word_attention(
            attentions, img_positions, seq_positions, token_gh, token_gw
        )

        # Attention on word's own patches
        n_attn_patches = avg_attn.numel()
        if word_patch_idx:
            # Clamp patch indices to valid range
            valid_idx = [i for i in word_patch_idx if i < n_attn_patches]
            if valid_idx:
                own_attn   = avg_attn[valid_idx]
                total_attn = avg_attn.sum().item()
                ratio      = own_attn.sum().item() / (total_attn + 1e-12)
                own_vals   = [f"{v:.5f}" for v in own_attn.tolist()]
                print(f"  Attn on word patches: {own_vals}")
                ratio_label = "Cross-attention ratio" if is_cross else "Self-attention ratio"
                print(f"  {ratio_label}: {ratio:.3f}  "
                      f"← attn_on_word_patches / total_attn")
            else:
                ratio = float("nan")
                print("  Word patch indices out of attn range.")
        else:
            ratio = float("nan")

        # Top-5 patches
        k = min(5, n_attn_patches)
        top_vals, top_idxs = torch.topk(avg_attn, k)
        top5_list = top_idxs.tolist()
        print("  Top 5 attended patches:")
        for rank, (pidx, pval) in enumerate(zip(top_idxs.tolist(), top_vals.tolist())):
            row_g, col_g = pidx // token_gw, pidx % token_gw
            px0, py0, px1, py1 = _patch_to_pixel(pidx, token_gh, token_gw, orig_h, orig_w)
            print(f"    rank {rank+1}: patch={pidx:4d}  grid=({row_g:2d},{col_g:2d})"
                  f"  pixel=({px0},{py0})→({px1},{py1})  attn={pval:.5f}")

        # Visualise
        _save_word_figure(
            pil_img, target_word, word_box,
            avg_attn, per_layer_attn,
            top5_list, token_gh, token_gw,
            args.output_dir,
        )

        # CSV row
        top1_px = (
            _patch_to_pixel(top5_list[0], token_gh, token_gw, orig_h, orig_w)
            if top5_list else ()
        )
        csv_rows.append(dict(
            word          = target_word,
            surrogate     = args.surrogate,
            attn_mode     = "cross" if is_cross else "self",
            token_span    = f"[{tok_start},{tok_end})",
            seq_pos       = seq_start,
            attn_ratio    = f"{ratio:.4f}",
            top1_patch    = top5_list[0] if top5_list else "",
            top1_pixel    = f"{top1_px}" if top1_px else "",
            top1_attn_val = f"{top_vals[0].item():.5f}" if len(top_vals) > 0 else "",
        ))

    # Summary CSV
    if csv_rows:
        csv_path = args.output_dir / "summary.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nSummary CSV : {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
