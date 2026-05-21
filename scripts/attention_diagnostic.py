"""
scripts/attention_diagnostic.py

Attention heatmap diagnostic for surrogate models.

For each target word, extracts and visualises how much the model attends to
image patches when generating that word's tokens (teacher-forced).

The self-attention ratio (attn_on_word_patches / total_img_attn) is the key
metric: a low value means the model is NOT looking at the word's own pixels
when generating it — it is relying on context rather than local visual signal.

Usage:
    uv run python scripts/attention_diagnostic.py \\
        --config configs/attack.yaml \\
        --words Thompson Ella 633-114 \\
        --category banking \\
        --surrogate qwen2_5vl \\
        --output-dir outputs/attention_diagnostics

Supports only qwen2_5vl (needs eager attention + image_grid_thw).
"""

from __future__ import annotations

import argparse
import csv
import json
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
IMAGE_TOKEN_ID = 151655   # <|image_pad|> in Qwen2.5-VL vocabulary
MERGE_FACTOR   = 2        # spatial 2×2 merge in the VL connector
LAST_N_LAYERS  = 4

# Qwen2.5-VL pixel packing: each raw patch covers PATCH_H × PATCH_W pixels
# in the processed-image space.  PATCH_W = temporal_patch_size × patch_size.
PATCH_H = 14
PATCH_W = 28   # = 2 × 14

PROMPT = (
    "Read the text in this image and output it exactly as written. "
    "Output the text only, no coordinates, no descriptions, no explanations."
)


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


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_qwen(model_id: str, device: str):
    """Load Qwen2.5-VL with eager attention so output_attentions=True works."""
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

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = _Cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval().to(device)
    return processor, model


# ── Input construction ─────────────────────────────────────────────────────────

def _build_full_inputs(
    processor, pil_img: Image.Image, transcript: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Returns (full_ids, full_mask, pixel_values, image_grid_thw, prompt_len).

    full_ids = prompt_ids ++ transcript_ids (teacher-forced).
    """
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = processor(text=[text], images=[pil_img], return_tensors="pt")

    prompt_ids = enc["input_ids"].to(device)
    pixel_vals = enc["pixel_values"].to(device, dtype=torch.bfloat16)
    grid_thw   = enc["image_grid_thw"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    transcript_ids = processor.tokenizer(
        transcript, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    full_ids  = torch.cat([prompt_ids, transcript_ids], dim=1)
    full_mask = torch.cat([
        attn_mask,
        torch.ones(1, transcript_ids.shape[1], device=device, dtype=torch.long),
    ], dim=1)

    return full_ids, full_mask, pixel_vals, grid_thw, prompt_ids.shape[1]


# ── Forward pass ──────────────────────────────────────────────────────────────

def _forward_with_attentions(model, full_ids, full_mask, pixel_vals, grid_thw):
    """Single forward pass; returns tuple of per-layer attention tensors."""
    with torch.no_grad():
        out = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=pixel_vals,
            image_grid_thw=grid_thw,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
    return out.attentions   # (n_layers,), each (1, n_heads, seq_len, seq_len)


# ── Patch geometry ─────────────────────────────────────────────────────────────

def _patch_geometry(
    full_ids: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_factor: int = MERGE_FACTOR,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> tuple[torch.Tensor, int, int]:
    """
    Returns (img_positions, token_gh, token_gw).

    img_positions : (n_img_tokens,) int64 — seq positions of image tokens
    token_gh, token_gw : merged token grid height and width
    """
    img_positions = (full_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    n_img_tokens  = img_positions.numel()

    _, gh_raw, gw_raw = [int(x) for x in grid_thw[0].tolist()]
    token_gh = gh_raw // merge_factor
    token_gw = gw_raw // merge_factor

    expected = token_gh * token_gw
    if expected != n_img_tokens:
        warnings.warn(
            f"Token count mismatch: grid {token_gh}×{token_gw}={expected} "
            f"vs {n_img_tokens} image tokens in full_ids. "
            "Results may be misaligned — check merge_factor / image_token_id.",
            RuntimeWarning,
        )

    return img_positions, token_gh, token_gw


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


# ── Word lookup ────────────────────────────────────────────────────────────────

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


# ── Attention extraction ───────────────────────────────────────────────────────

def _extract_attention(
    attentions: tuple,
    img_positions: torch.Tensor,
    seq_positions: range,
    last_n: int = LAST_N_LAYERS,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Returns (avg_attn, per_layer) — each (n_img_tokens,).

    For each of the last `last_n` layers:
      - average attention over heads
      - average attention over the word's token span positions
      - extract image-patch columns
    """
    pos_list = list(seq_positions)
    per_layer: list[torch.Tensor] = []

    for layer_attn in attentions[-last_n:]:
        # (1, n_heads, seq_len, seq_len) → avg over heads → (seq_len, seq_len)
        a = layer_attn[0].float().mean(dim=0)
        # rows = word span; avg over span → (seq_len,)
        span = a[pos_list, :].mean(dim=0)
        # columns = image patch positions
        per_layer.append(span[img_positions].cpu())

    avg_attn = torch.stack(per_layer).mean(dim=0)
    return avg_attn, per_layer


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

    # ── Panel 1: annotated image ──────────────────────────────────────────────
    ann = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(ann)
    if word_box:
        draw.rectangle(word_box, outline="cyan", width=3)
    for pidx in top5_indices[:5]:
        px0, py0, px1, py1 = _patch_to_pixel(pidx, token_gh, token_gw, orig_h, orig_w)
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)

    # ── Panel 2: avg attention grid ───────────────────────────────────────────
    grid_norm = _attn_to_grid(avg_attn, token_gh, token_gw)

    # ── Panel 3: overlay on image ─────────────────────────────────────────────
    hm_rgb   = (cm.hot(grid_norm)[:, :, :3] * 255).astype(np.uint8)
    hm_pil   = Image.fromarray(hm_rgb).resize((orig_w, orig_h), Image.BILINEAR)
    overlay  = Image.blend(pil_img.convert("RGB"), hm_pil, alpha=0.5)
    draw_ov  = ImageDraw.Draw(overlay)
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

    # ── Per-layer heatmaps ────────────────────────────────────────────────────
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention heatmap diagnostic for Qwen2.5-VL surrogate."
    )
    parser.add_argument("--config",     type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument("--words",      nargs="+", required=True,
                        help="Target words to probe (e.g. Thompson Ella 633-114)")
    parser.add_argument("--category",  type=str,  default="banking")
    parser.add_argument("--surrogate", type=str,  default="qwen2_5vl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_diagnostics"))
    parser.add_argument("--device",    type=str,  default=None)
    args = parser.parse_args()

    if args.surrogate != "qwen2_5vl":
        print(f"WARNING: only qwen2_5vl is supported; got '{args.surrogate}'. Proceeding anyway.")

    cfg    = _load_cfg(args.config)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
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

    pil_img    = sample.image                  # PIL Image
    orig_w, orig_h = pil_img.size
    transcript = sample.transcript
    print(f"Transcript : {len(transcript)} chars  {len(transcript.split())} words")
    print(f"GT snippet : {transcript[:80].replace(chr(10),' ')!r}")

    # ── Model ─────────────────────────────────────────────────────────────────
    s_cfg = next((s for s in cfg.surrogates if s.name == args.surrogate), None)
    if s_cfg is None:
        sys.exit(f"ERROR: surrogate '{args.surrogate}' not found in config.")

    print(f"\nLoading {args.surrogate} (attn_implementation=eager) …")
    processor, model = _load_qwen(s_cfg.model_id, device)

    # ── Build inputs ──────────────────────────────────────────────────────────
    print("Building full_ids (prompt + teacher-forced transcript) …")
    full_ids, full_mask, pixel_vals, grid_thw, prompt_len = \
        _build_full_inputs(processor, pil_img, transcript, device)

    n_total = full_ids.shape[1]
    n_trans  = n_total - prompt_len
    print(f"Sequence   : {n_total} tokens  (prompt={prompt_len}, transcript={n_trans})")

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("Forward pass with output_attentions=True …")
    attentions = _forward_with_attentions(model, full_ids, full_mask, pixel_vals, grid_thw)
    print(f"Attentions : {len(attentions)} layers  "
          f"shape={tuple(attentions[0].shape)}")

    # ── Patch geometry ────────────────────────────────────────────────────────
    img_positions, token_gh, token_gw = _patch_geometry(full_ids, grid_thw)
    n_img = img_positions.numel()
    print(f"Image toks : {n_img}  grid={token_gh}×{token_gw}  "
          f"(raw {token_gh * MERGE_FACTOR}×{token_gw * MERGE_FACTOR}="
          f"{token_gh * token_gw * MERGE_FACTOR ** 2} raw patches)")

    # ── Token-to-word alignment ───────────────────────────────────────────────
    transcript_words = transcript.split()
    n_words          = len(transcript_words)
    spans            = _align_tokens_to_words(processor.tokenizer, transcript, n_words)

    # ── Labels / box fallback ─────────────────────────────────────────────────
    labels_path = Path(cfg.data.data_dir) / "labels_pil.json"

    # ── Output directory + CSV ────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict] = []

    # ── Per-word analysis ─────────────────────────────────────────────────────
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
            word_patch_idx = _box_to_patch_indices(word_box, token_gh, token_gw, orig_h, orig_w)
        else:
            print("  Word pixel box: not found in labels")
            word_patch_idx = []
        print(f"  Word patch indices: {word_patch_idx}")

        # Attention extraction
        avg_attn, per_layer_attn = _extract_attention(
            attentions, img_positions, seq_positions
        )

        # Attention on word's own patches
        if word_patch_idx:
            own_attn      = avg_attn[word_patch_idx]
            total_attn    = avg_attn.sum().item()
            self_ratio    = own_attn.sum().item() / (total_attn + 1e-12)
            own_vals      = [f"{v:.5f}" for v in own_attn.tolist()]
            print(f"  Attn on word patches: {own_vals}")
            print(f"  Self-attention ratio: {self_ratio:.3f}  "
                  f"← attn_on_word_patches / total_attn")
        else:
            self_ratio = float("nan")

        # Top-5 patches
        k = min(5, n_img)
        top_vals, top_idxs = torch.topk(avg_attn, k)
        top5_list = top_idxs.tolist()
        print("  Top 5 attended patches:")
        for rank, (pidx, pval) in enumerate(zip(top_idxs.tolist(), top_vals.tolist())):
            row, col = pidx // token_gw, pidx % token_gw
            px0, py0, px1, py1 = _patch_to_pixel(pidx, token_gh, token_gw, orig_h, orig_w)
            print(f"    rank {rank+1}: patch={pidx:4d}  grid=({row:2d},{col:2d})"
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
            token_span    = f"[{tok_start},{tok_end})",
            seq_pos       = seq_start,
            self_attn_ratio = f"{self_ratio:.4f}",
            top1_patch    = top5_list[0] if top5_list else "",
            top1_pixel    = f"{top1_px}" if top1_px else "",
            top1_attn_val = f"{top_vals[0].item():.5f}" if len(top_vals) > 0 else "",
        ))

    # ── Summary CSV ───────────────────────────────────────────────────────────
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
