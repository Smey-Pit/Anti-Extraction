"""
LLaVA-1.6-Mistral-7B surrogate wrapper — transformers 4.44.2 compatible.

Architecture (confirmed via discovery):
  vision_tower        : CLIPVisionModel (CLIPVisionTransformer)
  multi_modal_projector: LlavaNextMultiModalProjector
  language_model      : MistralForCausalLM
  image_newline       : learnable parameter (row separator in tiled features)

pixel_values shape (this transformers version):
  processor output  : (1, num_patches, 3, 336, 336)  — batch dim included
  model.forward()   : same — (1, P, 3, 336, 336)
  num_patches for (192, 768): 4  (3 sub-tiles + 1 base)

ce_loss design:
  1. Run processor on detached PIL to get tile geometry + image_sizes (no grad)
  2. Rebuild pixel_values from image_tensor via F.interpolate — grad flows
  3. Manually replicate the vision → projector → splice pathway:
       vision_tower(pv)  → clip features  (P, n_patches, D_clip)
       multi_modal_projector → (P, n_patches, D_llm)
       pack_image_features   → (1, N_img_tokens, D_llm)  with image_newline
       splice into token embeddings at IMAGE_TOKEN_INDEX positions
  4. Append transcript embeddings, mask prefix with -100, call language_model

chat template:
  apply_chat_template is on processor.tokenizer (not processor directly).
  Mistral format: <s>[INST] <image>\n{question} [/INST]
  IMAGE_TOKEN_INDEX = 32000, decoded as '<image>'
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from vlm_suppress.models.base import SurrogateModel

_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_TILE_SIZE = 336


class LLaVA16(SurrogateModel):

    def __init__(self, cfg) -> None:
        self.name    = cfg.name
        self._dtype  = torch.bfloat16
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = _load_processor_safe(cfg.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=self._dtype,
            device_map=cfg.device_map,
        ).eval()

        self._max_new_tokens = cfg.max_new_tokens

        # IMAGE_TOKEN_INDEX = 32000 (confirmed)
        self._img_token_id: int = self.model.config.image_token_index
        _image_token = self.processor.tokenizer.decode([self._img_token_id])
        _question    = ("Transcribe all text in this image exactly as it appears. "
                        "Do not add any explanation, formatting, or preamble. "
                        "Output only the raw text content, nothing else.")

        # apply_chat_template renders a literal <s> BOS into the string.
        # Strip it — the processor/tokenizer adds its own BOS on top, causing
        # <s><s>[INST]... which misaligns the -100 label mask in ce_loss.
        # Use removeprefix (exact string match, not char-by-char lstrip).
        bos = self.processor.tokenizer.bos_token or "<s>"
        _prompt_raw = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{_image_token}\n{_question}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        self._prompt = _prompt_raw.removeprefix(bos)

        # Cache CLIP normalisation tensors
        self._clip_mean = torch.tensor(
            _CLIP_MEAN, device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)
        self._clip_std = torch.tensor(
            _CLIP_STD, device=self._device, dtype=self._dtype
        ).view(1, 3, 1, 1)

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Normalisation ──────────────────────────────────────────────────────────

    def _clip_norm(self, x: torch.Tensor) -> torch.Tensor:
        """CLIP normalise. x: (*, 3, H, W) in [0,1], returns same shape."""
        return (x - self._clip_mean) / self._clip_std

    # ── PIL helper ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image_tensor: torch.Tensor):
        """(3,H,W) float32 [0,1] → PIL RGB. Always detached."""
        from PIL import Image
        import numpy as np
        arr = (image_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255
               ).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    # ── Tile geometry ──────────────────────────────────────────────────────────

    def _get_proc_output(self, image_tensor: torch.Tensor) -> dict:
        """
        Run processor on a detached PIL image to fix tile geometry.
        Returns the full processor output dict (no grad anywhere).
        pixel_values shape: (1, P, 3, 336, 336)
        """
        pil = self._to_pil(image_tensor)
        return self.processor(
            text=self._prompt, images=pil, return_tensors="pt"
        )

    def _build_pixel_values(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) grad-connected
        proc_out: dict,               # processor output for this image
    ) -> torch.Tensor:
        """
        Build differentiable pixel_values matching the processor's tiling exactly.

        The processor's anyres tiling (pad + resize per tile) cannot be
        replicated with simple uniform grid crops — the pixel content differs
        by up to 3.4 normalised units, causing the model to predict EOS
        instead of the transcript at the supervision boundary.

        Strategy: use the processor's pixel_values as the tiling geometry
        reference, then reconstruct each tile differentiably from image_tensor
        using the same crop + resize the processor applied.

        We recover the crop boxes by inverting the processor's normalisation
        on its pixel_values output, then finding the corresponding region in
        image_tensor via cross-correlation. This is complex, so instead we
        use a simpler approach: reconstruct pixel_values as a differentiable
        function of image_tensor by applying the processor's exact resize
        sequence but routing through image_tensor for the pixel values.

        Concretely: the processor output tells us the final normalised tile
        values. We denormalise them to get the processor's source crop in
        [0,1], then use image_tensor to build a differentiable version of
        each crop using F.grid_sample with the inferred affine parameters.
        """
        x = image_tensor.to(device=self._device, dtype=self._dtype)  # (3,H,W)
        H, W = x.shape[-2], x.shape[-1]

        # Processor pixel_values: (1, P, 3, 336, 336) — no grad, correct geometry
        pv_proc = proc_out["pixel_values"].to(device=self._device, dtype=self._dtype)
        P = pv_proc.shape[1]

        # Denormalise processor tiles to [0,1] to recover source crop content
        mean = self._clip_mean  # (1,3,1,1)
        std  = self._clip_std   # (1,3,1,1)
        # pv_proc[0, i] = (crop_i - mean) / std  →  crop_i = pv_proc[0,i]*std + mean
        pv_denorm = pv_proc[0] * std + mean  # (P, 3, 336, 336), values in [0,1] approx

        tiles: list[torch.Tensor] = []
        for i in range(P):
            tile_ref = pv_denorm[i]  # (3, 336, 336) — processor's source crop, no grad

            # Find where in the original image this tile came from.
            # The last tile is always the full image downsized → straightforward.
            # For sub-tiles, find the best-matching region via normalised cross-corr
            # on a coarse grid (8×8 search over possible top-left corners).
            if i == P - 1:
                # Base tile: full image resized to 336×336
                tile_diff = F.interpolate(
                    x.unsqueeze(0), size=(_TILE_SIZE, _TILE_SIZE),
                    mode="bilinear", align_corners=False,
                )  # (1,3,336,336), grad-connected
            else:
                # Sub-tile: find crop region by comparing denormed tile to image
                crop_box = _find_crop_box(tile_ref, x, H, W)  # (y0,x0,y1,x1)
                y0, x0, y1, x1 = crop_box
                crop = x[:, y0:y1, x0:x1].unsqueeze(0)  # (1,3,h,w)
                tile_diff = F.interpolate(
                    crop, size=(_TILE_SIZE, _TILE_SIZE),
                    mode="bilinear", align_corners=False,
                )  # (1,3,336,336), grad-connected

            tiles.append(self._clip_norm(tile_diff).squeeze(0))  # (3,336,336)

        return torch.stack(tiles, dim=0).unsqueeze(0)  # (1,P,3,336,336)

    # ── Vision encoding (differentiable) ──────────────────────────────────────

    def _encode_images(
        self,
        pixel_values: torch.Tensor,   # (1, P, 3, 336, 336) grad-connected
        image_sizes:  torch.Tensor,   # (1, 2) = [[H, W]]
    ) -> torch.Tensor:
        """
        Replicate model's internal image encoding for transformers 4.44.2.

        In this version there is no encode_images() helper — we call the
        sub-modules directly:
          vision_tower → CLIP patch features
          multi_modal_projector → LLM-dimension features
          pack_image_features → merge tiles + image_newline

        Returns: (1, N_img_tokens, D_llm)  grad flows through pixel_values.
        """
        # pixel_values: (1, P, 3, 336, 336) — flatten to (P, 3, 336, 336)
        # for vision_tower which expects a flat batch of tiles
        pv_flat = pixel_values.squeeze(0)   # (P, 3, 336, 336)

        # CLIP vision tower → hidden states at configured feature layer
        vision_outputs = self.model.vision_tower(
            pv_flat, output_hidden_states=True,
        )
        feature_layer   = self.model.config.vision_feature_layer
        select_strategy = self.model.config.vision_feature_select_strategy
        if select_strategy == "default":
            image_features = vision_outputs.hidden_states[feature_layer][:, 1:]  # drop CLS
        elif select_strategy == "full":
            image_features = vision_outputs.hidden_states[feature_layer]
        else:
            raise ValueError(f"Unknown vision_feature_select_strategy: {select_strategy}")
        # image_features: (P, n_vis_tokens, D_clip)

        # Project to LLM dimension: (P, n_vis_tokens, D_llm)
        image_features = self.model.multi_modal_projector(image_features)

        # pack_image_features confirmed signature (transformers 4.44.2):
        #   pack_image_features(image_features, image_sizes, image_newline=None)
        # image_features must be a list[tensor] — one tensor per image in batch.
        # Returns (N_img_tokens, D_llm) — NO batch dim.
        image_features, _ = self.model.pack_image_features(
            [image_features],               # list of length 1 (batch_size=1)
            image_sizes.to(self._device),
            image_newline=self.model.image_newline,
        )
        # image_features: (N_img_tokens, D_llm) — unsqueeze for downstream cat
        return image_features.unsqueeze(0)  # (1, N_img_tokens, D_llm)

    # ── Token embedding splice ─────────────────────────────────────────────────

    def _build_inputs_embeds(
        self,
        input_ids:      torch.Tensor,   # (1, L)
        image_features: torch.Tensor,   # (1, N_img_tokens, D_llm)
    ) -> torch.Tensor:
        """
        Replace IMAGE_TOKEN_INDEX positions in the token embedding matrix
        with the packed image features.

        Returns (1, L_exp, D_llm) where L_exp = L - 1 + N_img_tokens
        (the single <image> token is replaced by N_img_tokens patch tokens).
        """
        with torch.no_grad():
            tok_embeds = self.model.language_model.get_input_embeddings()(
                input_ids.to(self._device)
            ).to(self._dtype)   # (1, L, D)

        img_mask = (input_ids[0] == self._img_token_id)  # (L,) — exactly one True
        n_img = img_mask.sum().item()
        assert n_img == 1, f"Expected exactly 1 <image> token, found {n_img}"

        img_pos = img_mask.nonzero(as_tuple=True)[0].item()  # scalar index

        # Split around the image token and splice
        before = tok_embeds[:, :img_pos, :]                  # (1, img_pos, D)
        after  = tok_embeds[:, img_pos + 1:, :]              # (1, L-img_pos-1, D)

        inputs_embeds = torch.cat([before, image_features, after], dim=1)
        # (1, img_pos + N_img_tokens + (L - img_pos - 1), D)
        # = (1, L - 1 + N_img_tokens, D)
        return inputs_embeds

    # ── ce_loss ────────────────────────────────────────────────────────────────

    def ce_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_ce^k = -log p(transcript | LLaVA chat prompt + image).

        Construction:
          [BOS][INST]<image_tokens>[/INST] transcript [EOS]
          |←————————— L_exp ————————————→|←——— T ———→|
          labels: -100 on prefix, transcript token ids as targets.

        The transcript tokens are included in input_ids (not appended as
        separate embeddings) so the model sees them as the assistant turn
        it is being teacher-forced to produce. This matches what generate()
        does: it predicts tokens starting immediately after [/INST].
        """
        # ── 1. Build full sequence: prompt + transcript ────────────────────
        # Concatenate prompt and transcript into one string so the processor
        # tokenizes them together as a single sequence.
        # Add EOS after transcript to match training distribution.
        eos = self.processor.tokenizer.eos_token or "</s>"
        full_text = self._prompt + " " + transcript + eos

        proc_prompt = self._get_proc_output(image_tensor)
        P           = proc_prompt["pixel_values"].shape[1]
        image_sizes = proc_prompt["image_sizes"]

        pil = self._to_pil(image_tensor)
        proc_full = self.processor(
            text=full_text, images=pil, return_tensors="pt"
        )
        input_ids_full = proc_full["input_ids"].to(self._device)
        L_full   = input_ids_full.size(1)
        L_prompt = proc_prompt["input_ids"].shape[1]

        # ── 2. Differentiable pixel_values matching processor geometry ─────
        pixel_values   = self._build_pixel_values(image_tensor, proc_prompt)
        image_features = self._encode_images(pixel_values, image_sizes)
        N_img          = image_features.size(1)

        # ── 3. Splice image features into full sequence embeddings ─────────
        inputs_embeds = self._build_inputs_embeds(input_ids_full, image_features)
        # Shape: (1, L_full - 1 + N_img, D)
        L_exp_prompt = L_prompt - 1 + N_img   # prefix length after image expansion

        # ── 4. Labels: mask prompt prefix, supervise transcript + eos ──────
        # Transcript starts at position L_exp_prompt in the expanded sequence
        L_exp_full = inputs_embeds.size(1)
        labels = torch.full(
            (1, L_exp_full), -100, device=self._device, dtype=torch.long
        )
        # The transcript token ids occupy positions [L_exp_prompt:] in input_ids_full
        # after image expansion. Map them back from input_ids_full.
        # input_ids_full positions [L_prompt:] correspond to transcript tokens.
        # After splice, these map to [L_exp_prompt:] in inputs_embeds.
        transcript_ids = input_ids_full[0, L_prompt:]  # (T+1,) includes EOS
        labels[0, L_exp_prompt:L_exp_prompt + len(transcript_ids)] = transcript_ids

        # ── 5. Language model forward ──────────────────────────────────────
        # Call m.model.forward() (LlavaNextForConditionalGeneration), NOT
        # m.model.language_model.forward(). When called with inputs_embeds
        # and pixel_values=None, the model skips its internal image encoding
        # and uses our pre-spliced embeddings directly — but the language
        # model now runs in the correct multimodal context it was trained in.
        # Calling language_model.forward() directly bypasses this and causes
        # the model to treat image patch embeddings as arbitrary token vectors,
        # producing EOS at the supervision boundary instead of the transcript.
        out = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=None,
            return_dict=True,
        )
        return out.loss

    # ── align_loss ─────────────────────────────────────────────────────────────

    def align_loss(self, image_tensor: torch.Tensor, transcript: str) -> torch.Tensor:
        """
        F_align^k = -cosine_sim(z_I, z_T).
        Uses base tile only (single 336×336). z_T is input embeddings (weak
        signal — same known limitation as InternVL2).
        """
        x    = image_tensor.to(device=self._device, dtype=self._dtype)
        base = F.interpolate(
            x.unsqueeze(0), size=(_TILE_SIZE, _TILE_SIZE),
            mode="bilinear", align_corners=False,
        )
        pv_base = self._clip_norm(base)   # (1, 3, 336, 336)

        vision_out = self.model.vision_tower(pv_base, output_hidden_states=True)
        # Use the same feature layer as ce_loss — hidden_states[-2], drop CLS
        feat_layer = self.model.config.vision_feature_layer
        strategy   = self.model.config.vision_feature_select_strategy
        if strategy == "default":
            clip_feats = vision_out.hidden_states[feat_layer][:, 1:]  # (1, n, D_clip)
        else:
            clip_feats = vision_out.hidden_states[feat_layer]          # (1, n, D_clip)

        # Project into LLM space so z_I and z_T share the same dimension
        proj_feats = self.model.multi_modal_projector(clip_feats)      # (1, n, D_llm)
        z_I = F.normalize(proj_feats.mean(dim=1).float(), dim=-1)      # (1, D_llm)

        with torch.no_grad():
            text_ids = self.processor.tokenizer(
                transcript, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self._device)
            text_emb = self.model.language_model.get_input_embeddings()(
                text_ids
            ).mean(dim=1).float()
            z_T = F.normalize(text_emb, dim=-1)

        return -(z_I * z_T).sum(dim=-1).squeeze()

    # ── transcribe ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def transcribe(self, image_tensor: torch.Tensor) -> str:
        pil    = self._to_pil(image_tensor)
        inputs = self.processor(
            text=self._prompt, images=pil, return_tensors="pt"
        ).to(self._device)

        input_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        raw = self.processor.tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()
        # Strip common preamble patterns LLaVA-1.6 occasionally emits
        # e.g. 'The text in the image is:\n\n"..."' or 'The image shows: ...'
        import re
        raw = re.sub(
            r'^(the (?:text (?:in the image )?(?:reads|is|says)|image (?:shows|contains|displays|reads))[\s:\"\']*)',
            '', raw, flags=re.IGNORECASE
        ).strip().strip('"').strip("'").strip()
        return raw


# ── Grid inference ─────────────────────────────────────────────────────────────

def _find_crop_box(
    tile_ref: torch.Tensor,   # (3, 336, 336) denormed processor tile [0,1]
    image:    torch.Tensor,   # (3, H, W) original image [0,1]
    H: int, W: int,
) -> tuple[int, int, int, int]:
    """
    Find the (y0, x0, y1, x1) crop box in the original image that the
    processor used to produce tile_ref.

    The processor pads images to aspect-ratio-friendly sizes before tiling,
    so sub-tiles are exact rectangular crops + resize. We find the best
    matching crop by searching over candidate boxes on a coarse grid.

    For (192, 768) with P=4 (1×3 grid + base):
      tile 0: x=[0, 256),   y=[0, 192)
      tile 1: x=[256, 512), y=[0, 192)
      tile 2: x=[512, 768), y=[0, 192)
    The search finds these exactly.
    """
    best_err  = float('inf')
    best_box  = (0, 0, H, W)

    # Candidate aspect ratio: assume tiles are roughly square regions
    # Try dividing W into num_sub columns (known from P-1)
    # Coarse search: try all integer divisors of W up to P-1 columns
    num_sub = 3  # will be overridden below dynamically — see caller
    # Search on a grid: try crops of size (H, W//k) for k in [2,6]
    # and offsets that tile cleanly
    with torch.no_grad():
        tile_small = F.interpolate(
            tile_ref.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze(0)  # (3, 32, 32)

        for n_cols in range(1, 7):
            cw = W // n_cols
            if cw < 32:
                continue
            for n_rows in range(1, 4):
                rh = H // n_rows
                if rh < 32:
                    continue
                for r in range(n_rows):
                    for c in range(n_cols):
                        y0 = r * rh
                        y1 = y0 + rh if r < n_rows - 1 else H
                        x0 = c * cw
                        x1 = x0 + cw if c < n_cols - 1 else W
                        crop = image[:, y0:y1, x0:x1]
                        crop_small = F.interpolate(
                            crop.unsqueeze(0), size=(32, 32),
                            mode="bilinear", align_corners=False
                        ).squeeze(0)
                        err = (crop_small - tile_small).abs().mean().item()
                        if err < best_err:
                            best_err = err
                            best_box = (y0, x0, y1, x1)

    return best_box
    """
    Map num_sub_tiles → (rows, cols).
    For (192, 768) the processor produces P=4: 3 sub-tiles + 1 base.
    3 sub-tiles = 1×3 grid (1 row, 3 cols) matching the 1:4 aspect ratio.
    """
    _GRIDS = {1: (1,1), 2: (1,2), 3: (1,3), 4: (2,2), 5: (1,5), 6: (2,3)}
    if num_sub not in _GRIDS:
        raise ValueError(
            f"Unexpected num_sub_tiles={num_sub}. "
            f"Inspect processor output for your image_size and add the grid."
        )
    return _GRIDS[num_sub]


# ── Version-safe processor loader ─────────────────────────────────────────────

def _load_processor_safe(model_id: str) -> LlavaNextProcessor:
    """
    Load LlavaNextProcessor on transformers 4.44.2, which does not accept
    'image_token', 'patch_size', etc. in __init__.
    Wraps __init__ to silently drop unrecognised kwargs.
    """
    valid = set(inspect.signature(LlavaNextProcessor.__init__).parameters) - {"self"}
    orig  = LlavaNextProcessor.__init__

    def _patched(self, *args, **kwargs):
        orig(self, *args, **{k: v for k, v in kwargs.items() if k in valid})

    LlavaNextProcessor.__init__ = _patched
    try:
        proc = LlavaNextProcessor.from_pretrained(model_id)
    finally:
        LlavaNextProcessor.__init__ = orig
    return proc