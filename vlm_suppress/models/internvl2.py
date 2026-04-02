from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vlm_suppress.models.base import SurrogateModel


class InternVL2(SurrogateModel):
    """
    InternVL2-8B surrogate wrapper.

    Load strategy: CPU-first + .to(device). Never pass device_map —
    InternVL2's custom __init__ calls .item() during stochastic depth
    schedule construction, which crashes on meta tensors.

    ce_loss uses the same token-injection path as model.chat():
      1. Build prompt string with N image placeholder tokens (<IMG_CONTEXT>)
      2. Tokenize the full conversation (system prompt + image tokens + question)
      3. Replace <IMG_CONTEXT> positions in the embedding matrix with visual
         embeddings from extract_feature() — grad flows through this splice
      4. Run language_model() with the spliced inputs_embeds and CE labels
         over the transcript tokens only

    This matches what transcribe() sees exactly, so ce_loss and transcribe()
    are optimising and measuring the same conditional distribution.
    """

    # Conversation template constants (InternVL2-8B / InternLM2 chat format)
    _SYSTEM = (
        "You are an OCR engine. "
        "Transcribe exactly all visible text in this image. "
        "Preserve line breaks. Output only the text, nothing else."
    )
    _IMG_START  = "<img>"
    _IMG_END    = "</img>"
    _IMG_TOKEN  = "<IMG_CONTEXT>"

    def __init__(self, cfg) -> None:
        self.name    = cfg.name
        self._dtype  = torch.bfloat16
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).eval().to(self._device)

        self._max_new_tokens = cfg.max_new_tokens

        # Number of <IMG_CONTEXT> tokens per image — set by the model
        self._n_img_tokens: int = self.model.num_image_token  # typically 256

        # Token id for <IMG_CONTEXT>
        self._img_ctx_id: int = self.tokenizer.convert_tokens_to_ids(self._IMG_TOKEN)

        # Set on model so that model.chat() also works (used in transcribe)
        self.model.img_context_token_id = self._img_ctx_id

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        (3, H, W) float32 [0,1]  →  (1, 3, 448, 448) bfloat16 on model device.
        Gradient-transparent — does not detach.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=self._dtype)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=self._dtype)
        x = image_tensor.to(device=self._device, dtype=self._dtype)
        x = F.interpolate(x.unsqueeze(0), size=(448, 448),
                          mode="bilinear", align_corners=False)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return x  # (1, 3, 448, 448)

    # ── Prompt construction ────────────────────────────────────────────────

    def _build_input_ids(self, question: str) -> torch.Tensor:
        """
        Build the full conversation input_ids in InternVL2's chat format:

          <|im_start|>system\n{system}<|im_end|>\n
          <|im_start|>user\n<img>{IMG_CONTEXT * N}</img>\n{question}<|im_end|>\n
          <|im_start|>assistant\n

        Returns (1, L) int64 on self._device.
        The <IMG_CONTEXT> positions will be replaced with visual embeddings
        in _build_inputs_embeds().
        """
        img_placeholder = (
            self._IMG_START
            + self._IMG_TOKEN * self._n_img_tokens
            + self._IMG_END
        )
        conversation = (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{img_placeholder}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        ids = self.tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)  # (1, L)
        return ids

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,       # (1, L)
        pixel_values: torch.Tensor,    # (1, 3, 448, 448), grad-connected
    ) -> torch.Tensor:
        """
        Replace <IMG_CONTEXT> positions in the token embedding matrix with
        visual embeddings from extract_feature().

        This is the same splice that model.chat() performs internally.
        Gradient flows: pixel_values → extract_feature → splice → language_model.
        """
        # Token embeddings for the full sequence — (1, L, D)
        # Detach: we want grad to flow through vision path, not token path
        with torch.no_grad():
            tok_embeds = self.model.language_model \
                             .get_input_embeddings()(input_ids)  # (1, L, D)
        tok_embeds = tok_embeds.to(self._dtype)

        # Visual embeddings — (1, N_img, D), grad-connected via pixel_values
        img_embeds = self.model.extract_feature(pixel_values)    # (1, N_img, D)

        # Locate <IMG_CONTEXT> positions — shape (N_img,) of flat indices
        img_mask = (input_ids[0] == self._img_ctx_id)            # (L,)
        n_found  = img_mask.sum().item()
        assert n_found == self._n_img_tokens, (
            f"Expected {self._n_img_tokens} <IMG_CONTEXT> tokens, found {n_found}. "
            "Check _build_input_ids() template."
        )

        # In-place splice — replace IMG_CONTEXT rows with visual embeddings.
        # Use index_put_ with a cloned base so autograd tracks the operation.
        inputs_embeds = tok_embeds.clone()                       # (1, L, D)
        inputs_embeds[0, img_mask] = img_embeds[0]               # grad flows here

        return inputs_embeds  # (1, L, D)

    # ── Differentiable losses ──────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | chat-formatted image prompt).

        Uses the identical token context as model.chat() / transcribe(), so
        ce_loss and transcribe() optimise and measure the same distribution.

        Gradient path:
          image_tensor → _preprocess → extract_feature
          → _build_inputs_embeds (splice) → language_model CE → loss
        """
        pixel_values = self._preprocess(image_tensor)  # (1,3,448,448), grad

        # Build full conversation input_ids (IMG_CONTEXT placeholders intact)
        question = (
            "Transcribe exactly all visible text in this image. "
            "Preserve line breaks. Output only the text, nothing else."
        )
        input_ids = self._build_input_ids(question)    # (1, L)
        L = input_ids.size(1)

        # Splice visual embeddings into the conversation embedding matrix
        inputs_embeds = self._build_inputs_embeds(input_ids, pixel_values)  # (1, L, D)

        # Tokenize transcript for supervision (no special tokens)
        target_ids = self.tokenizer(
            transcript,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)                   # (1, T)
        T = target_ids.size(1)

        # Append transcript token embeddings to inputs_embeds
        with torch.no_grad():
            tgt_embeds = self.model.language_model \
                             .get_input_embeddings()(target_ids).to(self._dtype)  # (1,T,D)

        inputs_embeds = torch.cat([inputs_embeds, tgt_embeds], dim=1)  # (1, L+T, D)

        # Labels: ignore conversation prefix (-100), supervise transcript tokens
        labels = torch.cat([
            torch.full((1, L), -100, device=self._device, dtype=torch.long),
            target_ids,
        ], dim=1)                                      # (1, L+T)

        out = self.model.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
        )
        return out.loss  # scalar, grad flows to pixel_values ✓

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cosine_sim(z_I, z_T).
        Visual path differentiable; text embedding detached.
        """
        pixel_values = self._preprocess(image_tensor)

        img_embeds = self.model.extract_feature(pixel_values)  # (1, N_img, D)
        z_I = F.normalize(img_embeds.mean(dim=1).float(), dim=-1)

        with torch.no_grad():
            text_ids = self.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self._device)
            z_T = self.model.language_model \
                      .get_input_embeddings()(text_ids).mean(dim=1).float()
            z_T = F.normalize(z_T, dim=-1)

        return -(z_I * z_T).sum(dim=-1).squeeze()

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pixel_values = self._preprocess(image_tensor.detach())
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=(
                "Transcribe exactly all visible text in this image. "
                "Preserve line breaks. Output only the text, nothing else."
            ),
            generation_config=dict(
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            ),
            history=None,
            return_history=True,
        )
        return response.strip()