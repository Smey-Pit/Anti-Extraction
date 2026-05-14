"""
PaliGemma2-3B surrogate wrapper.

Architecture:
  Vision encoder : SigLIP-So400m  (Google's SigLIP family)
  LM backbone    : Gemma-2B
  V-L connector  : Linear projection (no resampler, no tiling)

Why this model is in the ensemble:
  Gemini's documented vision encoder is a frozen SigLIP-ViT tower.
  PaliGemma2 uses the same SigLIP-So400m family, making it the closest
  available open-source proxy for Gemini's visual feature space.

Key design decisions:
  ce_loss  — tail-length label masking (robust to SentencePiece boundary
              tokenization differences between prompt-alone vs full sequence)
  transcribe — prompt tokenized WITH a real PIL image so token positions
               match what the model was trained on
  align_loss — uses pooler_output (2304-dim) to match text embedding dim;
               dimension equality asserted at runtime

Validated checkpoint: google/paligemma2-3b-mix-448
"""

from __future__ import annotations

from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from vlm_suppress.models.base import SurrogateModel


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) float32 [0,1] → PIL RGB. Used for processor calls only."""
    arr = (image_tensor.detach().cpu().clamp(0, 1) * 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), mode="RGB")


class PaliGemma2(SurrogateModel):

    _PROMPT = "<image> Transcribe exactly all visible text. Output only the text, nothing else.\n"

    def __init__(self, cfg) -> None:
        self.name         = cfg.name
        _dev = getattr(cfg, "device", None)
        if _dev:
            self._device = torch.device(_dev)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        self._dtype      = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_new_tokens = cfg.max_new_tokens

        self.processor = AutoProcessor.from_pretrained(cfg.model_id)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,   # preferred kwarg; if your env warns, use dtype=
        ).eval().to(self._device)

        # Pull resize/norm config directly from processor so _preprocess stays
        # aligned with whatever the checkpoint expects, regardless of version.
        ip = self.processor.image_processor
        sz = ip.size
        # SizeDict is a dict subclass — must extract values, never cast directly
        if isinstance(sz, dict) or hasattr(sz, "__getitem__"):
            if "height" in sz and "width" in sz:
                self._h, self._w = int(sz["height"]), int(sz["width"])
            elif "shortest_edge" in sz:
                self._h = self._w = int(sz["shortest_edge"])
            else:
                # Last resort: take the first numeric value found
                vals = [v for v in sz.values() if isinstance(v, (int, float))]
                if not vals:
                    raise ValueError(f"Unrecognised processor size format: {sz}")
                self._h = self._w = int(vals[0])
        else:
            self._h = self._w = int(sz)

        def _to_list3(v):
            return list(v) if isinstance(v, (list, tuple)) else [v] * 3

        # Cached — never reallocated per forward pass
        self._mean = torch.tensor(
            _to_list3(ip.image_mean), device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            _to_list3(ip.image_std), device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Differentiable preprocessing.
        (3, H, W) float32 [0,1] → (1, 3, H_proc, W_proc) on device/dtype.
        Gradient-transparent — does not detach.
        """
        x = image_tensor.to(device=self._device, dtype=self._dtype).unsqueeze(0)
        x = F.interpolate(x, size=(self._h, self._w), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    # ── Token helpers ──────────────────────────────────────────────────────

    def _prompt_ids_with_image(self, pil_image: Image.Image) -> dict[str, torch.Tensor]:
        """
        Tokenize the prompt WITH a real image so the processor inserts image
        tokens at the correct positions — matching what the model was trained on.

        Used by transcribe() and as the reference length for ce_loss masking.
        """
        enc = self.processor(
            text=self._PROMPT,
            images=pil_image,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in enc.items() if k != "pixel_values"}

    def _transcript_ids(self, transcript: str) -> torch.Tensor:
        """
        Tokenize the transcript alone with no special tokens.
        Used to compute the tail length for label masking.
        Robust to SentencePiece boundary effects — we never rely on
        (full_length - prompt_length) == transcript_length.
        """
        return self.processor.tokenizer(
            transcript,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self._device)  # (1, T)

    # ── Differentiable losses ──────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | image, prompt)

        Label masking via tail-length:
          1. Tokenize transcript alone → T tokens
          2. Build full sequence: prompt_ids + transcript_ids (no special tokens)
          3. Mask everything except the last T positions
          4. Pass differentiable pixel_values separately

        This avoids the fragile (L_full - L_prompt) split that breaks under
        SentencePiece boundary tokenization differences.
        """
        pixel_values = self._preprocess(image_tensor)

        # Tail-length approach — get T from transcript tokenized alone
        transcript_ids = self._transcript_ids(transcript)  # (1, T)
        T = transcript_ids.size(1)

        # Build full input: prompt (with image tokens) + transcript
        pil = _tensor_to_pil(image_tensor)
        prompt_enc = self._prompt_ids_with_image(pil)

        # Append transcript tokens to prompt input_ids
        full_input_ids = torch.cat(
            [prompt_enc["input_ids"], transcript_ids], dim=1
        )  # (1, L_p + T)

        # attention_mask: extend to cover transcript tokens
        if "attention_mask" in prompt_enc:
            full_attn = torch.cat(
                [prompt_enc["attention_mask"],
                 torch.ones(1, T, device=self._device, dtype=torch.long)],
                dim=1,
            )
        else:
            full_attn = None

        L_total = full_input_ids.size(1)

        # Mask all but the last T tokens (the transcript)
        labels = torch.full(
            (1, L_total), -100, device=self._device, dtype=torch.long
        )
        labels[0, -T:] = transcript_ids[0]

        model_inputs = dict(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
        )
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn

        out = self.model(**model_inputs)
        return out.loss

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cosine_sim(z_I, z_T)

        Visual embedding: mean-pool over patch tokens from pooler_output.
        In this transformers build, get_image_features() returns:
          last_hidden_state : (1, 1024, 1152)  — raw SigLIP patch features
          pooler_output     : (1, 1024, 2304)  — projected into Gemma embedding space
          text embeddings   : (1, T,    2304)
        pooler_output is used because it is already in the same dim space as
        text embeddings, making cosine similarity meaningful without a projection.
        """
        pixel_values = self._preprocess(image_tensor)

        img_out = self.model.get_image_features(pixel_values=pixel_values)

        if not hasattr(img_out, "pooler_output") or img_out.pooler_output is None:
            raise AttributeError(
                "get_image_features() returned no pooler_output. "
                "Check transformers version — fall back to last_hidden_state "
                "with an added projection if needed."
            )

        # pooler_output: (1, N_patches, 2304) — mean-pool over patch dim
        z_I = F.normalize(img_out.pooler_output.mean(dim=1).float(), dim=-1)  # (1, 2304)

        with torch.no_grad():
            text_ids = self.processor.tokenizer(
                transcript,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids.to(self._device)
            txt_emb = self.model.get_input_embeddings()(text_ids)  # (1, T, 2304)
            z_T = F.normalize(txt_emb.mean(dim=1).float(), dim=-1)  # (1, 2304)

        assert z_I.shape[-1] == z_T.shape[-1], (
            f"align_loss dimension mismatch: z_I={z_I.shape}, z_T={z_T.shape}. "
            f"pooler_output dim={img_out.pooler_output.shape[-1]}, "
            f"text embed dim={z_T.shape[-1]}."
        )

        return -(z_I * z_T).sum(dim=-1).squeeze(0)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        """
        Greedy decoding. Prompt is tokenized WITH the actual image so
        image token positions match training — same path as ce_loss.
        """
        pixel_values = self._preprocess(image_tensor.detach())
        pil = _tensor_to_pil(image_tensor)
        prompt_enc = self._prompt_ids_with_image(pil)

        out_ids = self.model.generate(
            input_ids=prompt_enc["input_ids"],
            pixel_values=pixel_values,
            attention_mask=prompt_enc.get("attention_mask"),
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )

        # Decode only the generated tokens (strip the prompt prefix)
        prompt_len = prompt_enc["input_ids"].shape[1]
        gen_ids = out_ids[0, prompt_len:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def token_logprobs(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
        transcript: str,
        return_top_k: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Per-token log probabilities for transcript given image.

        Returns
        -------
        log_probs : (T,) float32 on self._device
        token_ids : (T,) int64  on self._device
        """
        pixel_values   = self._preprocess(image_tensor)
        transcript_ids = self._transcript_ids(transcript)   # (1, T)
        T              = transcript_ids.size(1)

        pil        = _tensor_to_pil(image_tensor)
        prompt_enc = self._prompt_ids_with_image(pil)

        full_input_ids = torch.cat(
            [prompt_enc["input_ids"], transcript_ids], dim=1
        )

        if "attention_mask" in prompt_enc:
            full_attn = torch.cat(
                [prompt_enc["attention_mask"],
                 torch.ones(1, T, device=self._device, dtype=torch.long)],
                dim=1,
            )
        else:
            full_attn = None

        model_inputs = dict(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            return_dict=True,
        )
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn

        out = self.model(**model_inputs)

        transcript_logits = out.logits[0, -T - 1:-1, :].float()   # (T, vocab)
        log_probs = F.log_softmax(transcript_logits, dim=-1)
        tok_ids   = transcript_ids[0]                              # (T,)
        token_lp  = log_probs.gather(1, tok_ids.unsqueeze(1)).squeeze(1)

        # ── Top-K extension (no-op when return_top_k == 0) ───────────────────
        if return_top_k > 0:
            K = min(return_top_k, log_probs.size(-1))
            top_k_lp, top_k_id = torch.topk(log_probs, k=K, dim=-1, sorted=True)
            return token_lp, tok_ids, top_k_lp, top_k_id

        return token_lp, tok_ids