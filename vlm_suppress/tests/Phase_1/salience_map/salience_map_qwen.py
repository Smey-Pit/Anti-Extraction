"""
salience_map_qwen.py — Phase 1 Step 1: compute and visualize salience maps.

Computes three maps for a single image:
  S_transcript : ||∂CE_transcript/∂X||₂  — pixels that matter for full transcription
  S_query      : ||∂CE_query/∂X||₂       — pixels that matter for field-level binding
                 (averaged across all entities in the GT)
  S_bind       : S_query - S_transcript  — binding-specific signal

All maps are in image space (H, W) — gradients are unpatchified back to
the original spatial layout before visualization.

Visualization: 4-panel figure per entity + summary figure with averaged maps.

Usage (from Anti-Extraction-v2 root):
    uv run python salience_map_qwen.py \
        --image  data/ui_dataset/images/pil/banking_0000.png \
        --labels data/ui_dataset/labels_pil.jsonl \
        --gt     data/ui_dataset/ground_truth/pil/banking_0000.json \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --out    outputs/salience_maps/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True)
    p.add_argument("--labels",   required=True,  help="labels_pil.jsonl")
    p.add_argument("--gt",       required=True,  help="per-image GT JSON")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--out",      default="outputs/salience_maps/")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def get_full_text(labels_path: str, image_path: str) -> str:
    target = Path(image_path).name
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            if Path(row["image_path"]).name == target:
                return row["full_text"]
    raise ValueError(f"{target} not found in {labels_path}")


def load_gt(gt_path: str) -> dict:
    with open(gt_path) as f:
        return json.load(f)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_id: str, device: str):
    from transformers import AutoProcessor

    def _import_model_cls():
        try:
            from transformers import Qwen2_5VLForConditionalGeneration as C
            return C
        except ImportError:
            pass
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as C
            return C
        except ImportError:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration as C,
            )
            return C

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    ModelCls = _import_model_cls()
    model = ModelCls.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model, processor, dtype


# ── Differentiable preprocessing ──────────────────────────────────────────────

TRANSCRIBE_PROMPT = (
    "Read the text in this image and output it exactly as written. "
    "Output the text only, no coordinates, no descriptions, no explanations."
)


def build_pixel_values(
    image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1], requires_grad=True
    img_proc,
    device: str,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable preprocessing: image_tensor → pixel_values.
    Matches the processor pipeline exactly (verified to mean diff < 0.004).

    Returns (pixel_values, image_grid_thw).
    pixel_values: (n_patches, patch_dim) — grad flows back to image_tensor.
    """
    patch_size          = img_proc.patch_size
    merge_size          = img_proc.merge_size
    temporal_patch_size = img_proc.temporal_patch_size
    rescale_factor      = img_proc.rescale_factor
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073],
                        device=device, dtype=torch.float32).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=device, dtype=torch.float32).view(3,1,1)

    _, H_orig, W_orig = image_tensor.shape
    H_r, W_r = smart_resize(
        H_orig, W_orig,
        factor=patch_size * merge_size,
        min_pixels=img_proc.size.shortest_edge,
        max_pixels=img_proc.size.longest_edge,
    )

    # Float resize (grad-compatible) — tiny numerical diff vs uint8 resize is acceptable
    x = image_tensor.unsqueeze(0) * 255.0   # (1, 3, H, W)
    x = F.interpolate(
        x, size=(H_r, W_r), mode="bilinear", align_corners=False
    )
    x = x.squeeze(0)                        # (3, H_r, W_r)
    x = x * rescale_factor                  # rescale to [0,1]
    x = (x - mean) / std                    # normalize

    # Exact patch extraction from Qwen _preprocess source
    batch_size, channel = 1, 3
    grid_h = H_r // patch_size
    grid_w = W_r // patch_size

    patches = x.unsqueeze(0).reshape(
        batch_size, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    pixel_values = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            batch_size,
            grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        .squeeze(0)
    )   # (n_patches, patch_dim)

    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=device)
    return pixel_values, image_grid_thw


# ── Input sequence builder ────────────────────────────────────────────────────

def find_answer_span(
    target_ids: list[int],
    gt_value: str,
    tokenizer,
) -> tuple[int, int] | None:
    """
    Find the token span of gt_value within target_ids.

    Uses character-position mapping rather than token-id matching to handle
    BPE context-dependence (e.g. 'Ella' in isolation tokenizes differently
    from ' Ella' mid-sentence).

    Returns (start, end) as indices into target_ids, or None if not found.
    Takes the last occurrence to handle "The X is Y" sentence structure.
    """
    if not target_ids:
        return None

    # Decode full target sequence and each token individually
    decoded = tokenizer.decode(target_ids)
    token_strings = [tokenizer.decode([i]) for i in target_ids]

    # Build char → token index mapping
    char_to_token: dict[int, int] = {}
    char_pos = 0
    for t_idx, t_str in enumerate(token_strings):
        for _ in t_str:
            char_to_token[char_pos] = t_idx
            char_pos += 1

    # Find last occurrence of gt_value in decoded string
    search_start = 0
    last_match = None
    while True:
        idx = decoded.find(gt_value, search_start)
        if idx == -1:
            break
        end_char = idx + len(gt_value) - 1
        if idx in char_to_token and end_char in char_to_token:
            last_match = (char_to_token[idx], char_to_token[end_char] + 1)
        search_start = idx + 1

    return last_match


def build_teacher_forced_inputs(
    processor,
    device: str,
    dtype,
    pil_image: Image.Image,
    prompt_text: str,
    target_text: str,
    gt_value: str | None = None,   # if provided, loss computed only over gt_value span
) -> dict:
    """
    Build input_ids, attention_mask, labels for a teacher-forced forward pass.

    If gt_value is provided, labels are masked to -100 everywhere except the
    token span corresponding to gt_value within target_text. This gives
    answer-only loss — gradient reflects only the visually grounded tokens,
    not the boilerplate sentence prefix ("The account holder's name is...").

    Falls back to full-sentence loss if gt_value is not found in target tokens.
    """
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_inputs = processor(
        text=[chat_text], images=[pil_image], return_tensors="pt"
    )
    prompt_ids = prompt_inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Tokenize target (generated answer)
    target_ids_raw = processor.tokenizer(
        target_text, add_special_tokens=False
    ).input_ids
    eos_id = processor.tokenizer.eos_token_id
    target_ids_with_eos = target_ids_raw + [eos_id]

    target_ids = torch.tensor(
        [target_ids_with_eos], dtype=torch.long, device=device
    )

    # Full sequence
    full_input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.ones_like(full_input_ids)

    # Base labels: -100 for prompt, real ids for full target
    labels = torch.full_like(full_input_ids, -100)
    labels[:, prompt_len:] = target_ids

    # Answer-only masking: find gt_value span within target tokens
    span_info = "full sentence"
    if gt_value is not None:
        span = find_answer_span(target_ids_raw, gt_value, processor.tokenizer)
        if span is not None:
            start, end = span
            # Mask everything in target except the answer span
            # target starts at prompt_len in full_input_ids
            answer_mask = torch.full_like(labels, -100)
            answer_mask[:, prompt_len + start : prompt_len + end] = \
                target_ids[:, start:end]
            labels = answer_mask
            span_info = (f"answer-only [{start}:{end}] = "
                        f"{processor.tokenizer.decode(target_ids_raw[start:end])!r}")
        else:
            span_info = f"gt not found in generated — falling back to full sentence"

    print(f"  loss scope   : {span_info}")

    return {
        "input_ids":      full_input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "prompt_len":     prompt_len,
    }


# ── Salience computation ──────────────────────────────────────────────────────

def generate_answer(
    model,
    processor,
    img_proc,
    device: str,
    dtype,
    pil_image: Image.Image,
    prompt_text: str,
) -> str:
    """
    Generate the model's actual answer for a given prompt.
    Used to get a low-loss target for teacher-forcing.
    """
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_text], images=[pil_image], return_tensors="pt")
    inputs = {
        k: (v.to(device).to(dtype) if k == "pixel_values"
            else v.to(device) if torch.is_tensor(v)
            else v)
        for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def compute_salience(
    model,
    processor,
    img_proc,
    device: str,
    dtype,
    pil_image: Image.Image,
    prompt_text: str,
    target_text: str,
    image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
    use_generated: bool = True,
    gt_value: str | None = None,  # if set, answer-only loss over this span
) -> tuple[torch.Tensor, str, float]:
    """
    Compute salience map: ||∂CE/∂image_tensor||₂ in image space (H, W).

    Two-step approach (use_generated=True):
      1. Generate model's actual answer — guarantees low-loss target
      2. Teacher-force against generated answer — clean gradient signal

    Returns (salience, answer_used, loss_value).
    """
    # ── Step 1: generate model's answer ──────────────────────────────────
    if use_generated:
        tf_target = generate_answer(
            model, processor, img_proc, device, dtype, pil_image, prompt_text
        )
        print(f"  model answer : {tf_target!r}  (gt: {target_text!r})")
    else:
        tf_target = target_text

    # ── Step 2: teacher-force against generated answer ────────────────────
    x = image_tensor.detach().clone().to(device).float()
    x.requires_grad_(True)

    pixel_values, image_grid_thw = build_pixel_values(x, img_proc, device, dtype)
    pixel_values = pixel_values.to(dtype)

    seq = build_teacher_forced_inputs(
        processor, device, dtype, pil_image, prompt_text, tf_target,
        gt_value=gt_value,
    )

    outputs = model(
        input_ids=seq["input_ids"],
        attention_mask=seq["attention_mask"],
        pixel_values=pixel_values.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        labels=seq["labels"],
    )

    loss = outputs.loss
    loss_value = loss.item()
    print(f"  CE loss      : {loss_value:.4f}  "
          f"({'✓ low' if loss_value < 1.0 else '⚠ high'})")

    loss.backward()

    grad = x.grad.detach().cpu().float()   # (3, H, W)
    salience = grad.norm(dim=0)            # (H, W)

    print(f"  grad mean    : {grad.abs().mean():.2e}  "
          f"max: {grad.abs().max():.2e}  "
          f"nonzero: {(grad.abs() > 1e-10).float().mean()*100:.1f}%")

    return salience, tf_target, loss_value


# ── Visualization ─────────────────────────────────────────────────────────────

def normalize_map(m: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Percentile-clip then normalize to [0,1] — prevents extreme spikes crushing the map."""
    lo = np.percentile(m, 0)
    hi = np.percentile(m, percentile)
    if hi - lo < 1e-8:
        return np.zeros_like(m)
    return np.clip((m - lo) / (hi - lo), 0, 1)


def overlay_salience(
    pil_image: Image.Image,
    salience: np.ndarray,       # (H, W) normalized [0,1]
    alpha: float = 0.6,
    cmap: str = "inferno",
) -> np.ndarray:
    """Overlay salience heatmap on the original image."""
    img = np.array(pil_image.convert("RGB")).astype(float) / 255.0
    img_resized = np.array(
        pil_image.resize((salience.shape[1], salience.shape[0]), Image.BILINEAR)
        .convert("RGB")
    ).astype(float) / 255.0

    colormap = plt.get_cmap(cmap)
    heatmap = colormap(salience)[..., :3]   # (H, W, 3)

    blended = (1 - alpha) * img_resized + alpha * heatmap
    return np.clip(blended, 0, 1)


def save_entity_figure(
    pil_image: Image.Image,
    entity: dict,
    s_transcript: np.ndarray,
    s_query: np.ndarray,
    s_bind: np.ndarray,
    out_path: Path,
):
    """4-panel figure: original | S_transcript | S_query | S_bind."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"Entity: {entity['label']} = \"{entity['value']}\"\n"
        f"Q: {entity['question']}",
        fontsize=11, y=1.02,
    )

    # Panel 0: original
    axes[0].imshow(pil_image)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Panel 1: S_transcript
    axes[1].imshow(overlay_salience(pil_image, normalize_map(s_transcript)))
    axes[1].set_title("S_transcript\n(general readability)", fontsize=10)
    axes[1].axis("off")

    # Panel 2: S_query
    axes[2].imshow(overlay_salience(pil_image, normalize_map(s_query)))
    axes[2].set_title(f"S_query\n(field binding)", fontsize=10)
    axes[2].axis("off")

    # Panel 3: S_bind (positive regions only — where query > transcript)
    s_bind_pos = np.clip(s_bind, 0, None)
    axes[3].imshow(overlay_salience(pil_image, normalize_map(s_bind_pos), cmap="plasma"))
    axes[3].set_title("S_bind = S_query − S_transcript\n(binding-specific)", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_summary_figure(
    pil_image: Image.Image,
    s_transcript_avg: np.ndarray,
    s_query_avg: np.ndarray,
    s_bind_avg: np.ndarray,
    image_id: str,
    out_path: Path,
):
    """Summary 3-panel figure with maps averaged across all entities."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Summary (averaged across all entities) — {image_id}", fontsize=12)

    axes[0].imshow(pil_image)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay_salience(pil_image, normalize_map(s_transcript_avg)))
    axes[1].set_title("S_transcript (avg)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_salience(pil_image, normalize_map(s_query_avg)))
    axes[2].set_title("S_query (avg)", fontsize=10)
    axes[2].axis("off")

    s_bind_pos = np.clip(s_bind_avg, 0, None)
    axes[3].imshow(overlay_salience(pil_image, normalize_map(s_bind_pos), cmap="plasma"))
    axes[3].set_title("S_bind (avg, positive only)", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_id = Path(args.image).stem
    print(f"Image    : {args.image}  ({image_id})")
    print(f"Model    : {args.model_id}")
    print(f"Device   : {args.device}")
    print(f"Out dir  : {out_dir}")
    print()

    # ── Load data ────────────────────────────────────────────────────────
    full_text = get_full_text(args.labels, args.image)
    gt        = load_gt(args.gt)
    entities  = gt["entities"]
    pil_image = Image.open(args.image).convert("RGB")
    H, W      = pil_image.size[1], pil_image.size[0]

    print(f"Entities : {len(entities)}")
    print(f"Image size: {W}×{H}")
    print()

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading model…")
    model, processor, dtype = load_model_and_processor(args.model_id, args.device)
    img_proc = processor.image_processor
    print("Model loaded.")
    print()

    # Base image tensor — shared across all passes
    import numpy as np
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor_base = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)

    # ── Compute S_transcript (once — same for all entities) ───────────────
    print("Computing S_transcript…")
    s_transcript, _, transcript_loss = compute_salience(
        model, processor, img_proc, args.device, dtype,
        pil_image,
        prompt_text=TRANSCRIBE_PROMPT,
        target_text=full_text,
        image_tensor=image_tensor_base,
        use_generated=False,   # always use ground truth for transcript
    )
    s_transcript = s_transcript.numpy()
    print(f"  S_transcript: min={s_transcript.min():.4f} max={s_transcript.max():.4f} "
          f"mean={s_transcript.mean():.4f}")
    print()

    # Resize to original image dimensions for visualization
    def to_img_space(m: np.ndarray) -> np.ndarray:
        """Resize salience map to match original PIL image size (H, W)."""
        t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t.squeeze().numpy()

    s_transcript_img = to_img_space(s_transcript)

    # ── Compute S_query per entity ────────────────────────────────────────
    s_query_maps = []

    for i, entity in enumerate(entities):
        print(f"Computing S_query [{i+1}/{len(entities)}]: "
              f"{entity['label']} = \"{entity['value']}\"")

        s_query, model_answer, q_loss = compute_salience(
            model, processor, img_proc, args.device, dtype,
            pil_image,
            prompt_text=entity["question"],
            target_text=entity["value"],
            image_tensor=image_tensor_base,
            use_generated=True,
            gt_value=entity["value"],   # answer-only loss
        )
        s_query = s_query.numpy()

        # Flag if model answer doesn't match ground truth
        answer_match = model_answer.strip().lower() == entity["value"].strip().lower()
        match_str = "✓ match" if answer_match else f"✗ mismatch (gt: {entity['value']!r})"
        print(f"  answer check : {match_str}")
        s_query_img = to_img_space(s_query)
        s_query_maps.append(s_query_img)

        print(f"  S_query: min={s_query.min():.4f} max={s_query.max():.4f} "
              f"mean={s_query.mean():.4f}")

        # Per-entity S_bind
        s_bind_img = s_query_img - s_transcript_img

        # Pearson correlation between S_transcript and S_query
        st_flat = s_transcript_img.flatten()
        sq_flat = s_query_img.flatten()
        r = np.corrcoef(st_flat, sq_flat)[0, 1]
        print(f"  Pearson r(S_transcript, S_query) = {r:.4f}")

        # Save entity figure
        fname = out_dir / f"{image_id}_{entity['label']}.png"
        save_entity_figure(
            pil_image, entity,
            s_transcript_img, s_query_img, s_bind_img,
            fname,
        )
        print()

    # ── Summary figure ────────────────────────────────────────────────────
    s_query_avg  = np.stack(s_query_maps).mean(axis=0)
    s_bind_avg   = s_query_avg - s_transcript_img

    r_avg = np.corrcoef(s_transcript_img.flatten(), s_query_avg.flatten())[0, 1]
    print(f"Average Pearson r(S_transcript, S_query_avg) = {r_avg:.4f}")
    print()

    save_summary_figure(
        pil_image,
        s_transcript_img,
        s_query_avg,
        s_bind_avg,
        image_id,
        out_dir / f"{image_id}_summary.png",
    )

    # ── Save raw maps as numpy for later analysis ─────────────────────────
    np.save(out_dir / f"{image_id}_s_transcript.npy", s_transcript_img)
    np.save(out_dir / f"{image_id}_s_query_avg.npy",  s_query_avg)
    np.save(out_dir / f"{image_id}_s_bind_avg.npy",   s_bind_avg)
    print(f"Raw maps saved to {out_dir}")

    # ── Print interpretation guide ────────────────────────────────────────
    print()
    print("═" * 60)
    print("  INTERPRETATION")
    print("═" * 60)
    print(f"  Avg Pearson r  : {r_avg:.4f}")
    if r_avg < 0.5:
        verdict = "✓ S_transcript and S_query differ meaningfully — binding hypothesis supported."
    elif r_avg < 0.8:
        verdict = "⚠ Moderate correlation — some binding signal but maps overlap substantially."
    else:
        verdict = "✗ High correlation — model uses same pixels for binding as for reading. " \
                  "Whitespace attack may not work."
    print(f"  Verdict        : {verdict}")
    print("═" * 60)


if __name__ == "__main__":
    main()