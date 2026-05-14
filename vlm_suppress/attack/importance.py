# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/attack/importance.py
#
# Stage 1: Domain-agnostic token importance mapping.
#
# Produces a per-pixel importance map I(i,j) that identifies visually-grounded,
# semantically surprising tokens rather than structurally salient regions:
#
#   I(i,j) = normalize(S) × normalize(Surprise) × normalize(KL)
#
# Components:
#   S         — gradient salience (‖∇_x L_ce‖₂, existing signal)
#   Surprise  — -log p(word | blank_image, context)   [one forward pass]
#   KL        — log p(word | orig) - log p(word | masked_at_box)
#               [one pass per group of non-overlapping boxes]
#
# Caller contract:
#   • Surrogates must implement token_logprobs(image_tensor, transcript).
#     Models that do not are skipped with a warning; their component is set to 1.
#   • word_boxes and transcript.split() must have the same length (or close).
#   • This module does NOT modify the PGD attack loop or budget map construction.
#     It is Stage 1 diagnostic only.
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import warnings
from typing import Optional

import torch

from vlm_suppress.attack.masks import build_text_mask


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalize_01(t: torch.Tensor) -> torch.Tensor:
    """Scale tensor to [0, 1]. Returns all-zeros if range is zero."""
    lo, hi = t.min().item(), t.max().item()
    if hi - lo < 1e-9:
        return torch.zeros_like(t)
    return (t - lo) / (hi - lo)


def _get_tokenizer(model):
    """Return the transcript tokenizer from a surrogate model, or None."""
    if hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
        return model.processor.tokenizer
    if hasattr(model, "tokenizer"):
        return model.tokenizer
    return None


def _make_blank(image_tensor: torch.Tensor) -> torch.Tensor:
    """Mean-colour blank image, same shape as image_tensor."""
    return torch.full_like(image_tensor, image_tensor.mean().item())


def _boxes_overlap(a: list[int], b: list[int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _greedy_nonoverlap_groups(word_boxes: list[list[int]]) -> list[list[int]]:
    """
    Greedy independent-set partition of word indices so that no two boxes
    in a group overlap.  Used to batch masking forward passes for visual KL.
    """
    assigned = [False] * len(word_boxes)
    groups: list[list[int]] = []
    for i in range(len(word_boxes)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(word_boxes)):
            if assigned[j]:
                continue
            if all(not _boxes_overlap(word_boxes[k], word_boxes[j]) for k in group):
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


def _scores_to_pixel_map(
    word_scores: list[float],
    word_boxes:  list[list[int]],
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Paint each word box with its score.  Returns (H, W) float32 CPU tensor.
    Overlapping boxes take the max.
    """
    m = torch.zeros(H, W, dtype=torch.float32)
    for score, box in zip(word_scores, word_boxes):
        x0, y0, x1, y1 = (int(v) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = torch.maximum(
                m[y0:y1, x0:x1],
                torch.tensor(float(score)),
            )
    return m


# ── Token-to-word alignment ───────────────────────────────────────────────────

def _align_tokens_to_words(
    tokenizer,
    transcript: str,
    n_words: int,
) -> list[tuple[int, int]]:
    """
    Return (start_tok, end_tok) half-open spans for each of n_words in transcript.

    Uses character-level offset_mapping from the tokenizer if available.
    Falls back to proportional assignment on any failure.

    The words are obtained by splitting transcript on whitespace; the caller
    must ensure n_words matches the number of word_boxes.
    """
    words = transcript.split()
    if len(words) > n_words:
        words = words[:n_words]
    elif len(words) < n_words:
        warnings.warn(
            f"importance.py: transcript has {len(words)} whitespace tokens but "
            f"n_words={n_words} — proportional token assignment used.",
            RuntimeWarning, stacklevel=4,
        )

    try:
        enc = tokenizer(
            transcript,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets: list[tuple[int, int]] = enc.get("offset_mapping") or enc["offset_mapping"]
        T = len(offsets)

        # Build word character spans
        char_spans: list[tuple[int, int]] = []
        pos = 0
        for w in words:
            idx = transcript.find(w, pos)
            if idx == -1:
                raise ValueError(f"word {w!r} not found at pos {pos}")
            char_spans.append((idx, idx + len(w)))
            pos = idx + len(w)
        if len(char_spans) < n_words:
            raise ValueError("char span extraction incomplete")

        # Map char spans → token spans
        spans: list[tuple[int, int]] = []
        for w_start, w_end in char_spans:
            tok_start: Optional[int] = None
            tok_end: Optional[int] = None
            for i, (c_start, c_end) in enumerate(offsets):
                if c_end <= w_start:
                    continue
                if c_start >= w_end:
                    break
                if tok_start is None:
                    tok_start = i
                tok_end = i + 1
            if tok_start is None:
                tok_start = tok_end = 0
            spans.append((tok_start, tok_end or tok_start + 1))
        return spans

    except Exception:
        # Proportional fallback
        try:
            T = len(tokenizer(transcript, add_special_tokens=False).input_ids)
        except Exception:
            T = max(n_words, 1)
        spans = []
        for i in range(n_words):
            s = int(i * T / n_words)
            e = int((i + 1) * T / n_words)
            spans.append((s, max(s + 1, e)))
        return spans


# ── Per-word log-prob extraction ──────────────────────────────────────────────

def _word_logprobs(
    model,
    image_tensor: torch.Tensor,   # (3, H, W) on model.device
    transcript:   str,
    spans:        list[tuple[int, int]],
    n_words:      int,
) -> list[float]:
    """
    One forward pass → per-word sum of log probs.
    Returns list[float] of length n_words.
    """
    lp, _ = model.token_logprobs(image_tensor, transcript)   # (T,), on device
    lp = lp.cpu()
    T = lp.shape[0]

    scores = []
    for span_s, span_e in spans:
        if span_s is None or span_s >= T:
            scores.append(0.0)
        else:
            scores.append(float(lp[span_s:min(span_e, T)].sum()))
    return scores


# ── Core computation ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_token_surprise(
    model:        object,
    image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
    transcript:   str,
    word_boxes:   list[list[int]],
) -> torch.Tensor:                # (H, W) float32 CPU
    """
    Pixel-space surprise map: -log p(word | blank_image, context).

    Requires model.token_logprobs().  Returns zeros if not available.
    Cost: ONE forward pass (blank image).
    """
    if not hasattr(model, "token_logprobs"):
        warnings.warn(
            f"compute_token_surprise: {type(model).__name__} has no token_logprobs — "
            "returning zero map.",
            RuntimeWarning, stacklevel=2,
        )
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return torch.zeros(H, W)

    tokenizer = _get_tokenizer(model)
    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    n_words = len(word_boxes)

    spans = _align_tokens_to_words(tokenizer, transcript, n_words)

    blank = _make_blank(image_tensor).to(model.device)
    word_lp = _word_logprobs(model, blank, transcript, spans, n_words)

    # Surprise = -log p (positive; higher = model more surprised by this word)
    word_scores = [-lp for lp in word_lp]
    return _scores_to_pixel_map(word_scores, word_boxes, H, W)


@torch.no_grad()
def compute_visual_kl(
    model:        object,
    image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
    transcript:   str,
    word_boxes:   list[list[int]],
) -> torch.Tensor:                # (H, W) float32 CPU
    """
    Pixel-space visual-KL map: log p(word | orig) - log p(word | masked_at_box).

    Positive where the visual region meaningfully contributes to predicting
    the word.  Non-overlapping boxes are batched into single forward passes.

    Requires model.token_logprobs().  Returns zeros if not available.
    Cost: 1 (original) + N_groups forward passes (N_groups ≤ n_words).
    """
    if not hasattr(model, "token_logprobs"):
        warnings.warn(
            f"compute_visual_kl: {type(model).__name__} has no token_logprobs — "
            "returning zero map.",
            RuntimeWarning, stacklevel=2,
        )
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return torch.zeros(H, W)

    tokenizer = _get_tokenizer(model)
    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    n_words = len(word_boxes)
    mean_fill = float(image_tensor.mean())
    dev = model.device

    spans = _align_tokens_to_words(tokenizer, transcript, n_words)

    # Baseline: original image log probs
    orig_word_lp = _word_logprobs(model, image_tensor.to(dev), transcript, spans, n_words)

    # Masked passes — batch non-overlapping boxes
    groups = _greedy_nonoverlap_groups(word_boxes)
    masked_word_lp: list[Optional[float]] = [None] * n_words

    for g_idx, group in enumerate(groups):
        img_masked = image_tensor.clone()
        for idx in group:
            x0, y0, x1, y1 = (int(v) for v in word_boxes[idx])
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 > x0 and y1 > y0:
                img_masked[:, y0:y1, x0:x1] = mean_fill

        masked_lp_list = _word_logprobs(model, img_masked.to(dev), transcript, spans, n_words)
        for idx in group:
            masked_word_lp[idx] = masked_lp_list[idx]

    # KL ≈ log p_orig(w) - log p_masked(w); clamp to ≥ 0
    word_kl = [
        max(0.0, o - (m if m is not None else 0.0))
        for o, m in zip(orig_word_lp, masked_word_lp)
    ]
    return _scores_to_pixel_map(word_kl, word_boxes, H, W)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_importance_map(
    image_tensor:  torch.Tensor,       # (1, 3, H, W) or (3, H, W) float32 [0,1]
    transcript:    str,
    word_boxes:    list[list[int]],
    surrogates:    list,
    alpha_weights: list[float],
    epsilon_min:   float,
    epsilon_max:   float,
    epsilon_bg:    float,
    dilation:      int,
    device:        torch.device,
    use_surprise:  bool = True,
    use_visual_kl: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Build an importance-weighted epsilon budget map (diagnostic only).

    Pipeline:
      1. Gradient salience  — ‖∇_x L_ce‖₂  (existing, from build_salience_budget_map)
      2. Token surprise     — -log p(w | blank, context)
      3. Visual KL          — Δ log p(w | orig vs masked)
      4. I = normalize(S) × normalize(Surprise) × normalize(KL)
      5. Build eps map: text pixels ← epsilon_min + (epsilon_max-epsilon_min)·I
                        bg pixels   ← epsilon_bg

    Returns
    -------
    eps_map    : (1, H, W) float32 budget map on `device`
    components : dict with CPU (H, W) tensors: salience, surprise, kl, importance
    """
    from vlm_suppress.attack.salience import build_salience_budget_map
    from vlm_suppress.models.lazy import LazySurrogate

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)   # (3, H, W)

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    image_4d = image_tensor.unsqueeze(0)

    text_mask  = build_text_mask(H, W, word_boxes, dilation, device=torch.device("cpu"))
    text_flag  = text_mask.squeeze(0) > 0   # (H, W) bool

    # ── 1. Gradient salience ──────────────────────────────────────────────────
    print("  [importance] gradient salience ...")
    sal_4d = build_salience_budget_map(
        image_tensor  = image_4d,
        transcript    = transcript,
        word_boxes    = word_boxes,
        surrogates    = surrogates,
        alpha_weights = alpha_weights,
        epsilon_min   = epsilon_min,
        epsilon_max   = epsilon_max,
        epsilon_bg    = epsilon_bg,
        dilation      = dilation,
        device        = device,
    )

    # Extract raw salience in [0, 1] (undo linear epsilon scaling)
    sal_raw = sal_4d.squeeze(0).cpu()
    eps_range = epsilon_max - epsilon_min
    if eps_range > 1e-9:
        sal_raw = (sal_raw - epsilon_min) / eps_range
    sal_raw = sal_raw.clamp(0.0, 1.0) * text_flag.float()
    sal_norm = _normalize_01(sal_raw)

    # ── 2. Token surprise ─────────────────────────────────────────────────────
    surprise_map = torch.zeros(H, W, dtype=torch.float32)
    if use_surprise:
        for surrogate, alpha in zip(surrogates, alpha_weights):
            print(f"  [importance] surprise — {surrogate.name} ...")
            if isinstance(surrogate, LazySurrogate):
                with surrogate as model:
                    s_k = compute_token_surprise(model, image_tensor, transcript, word_boxes)
            else:
                s_k = compute_token_surprise(surrogate, image_tensor, transcript, word_boxes)
            surprise_map = surprise_map + alpha * s_k
        surprise_map = surprise_map * text_flag.float()

    # ── 3. Visual KL ──────────────────────────────────────────────────────────
    kl_map = torch.zeros(H, W, dtype=torch.float32)
    if use_visual_kl:
        n_groups = len(_greedy_nonoverlap_groups(word_boxes))
        for surrogate, alpha in zip(surrogates, alpha_weights):
            print(
                f"  [importance] visual KL — {surrogate.name} "
                f"({len(word_boxes)} words → {n_groups} batched passes) ..."
            )
            if isinstance(surrogate, LazySurrogate):
                with surrogate as model:
                    k_k = compute_visual_kl(model, image_tensor, transcript, word_boxes)
            else:
                k_k = compute_visual_kl(surrogate, image_tensor, transcript, word_boxes)
            kl_map = kl_map + alpha * k_k
        kl_map = kl_map * text_flag.float()

    # ── 4. Importance = product of normalized components ──────────────────────
    surp_n = _normalize_01(surprise_map) if use_surprise   else torch.ones(H, W)
    kl_n   = _normalize_01(kl_map)       if use_visual_kl  else torch.ones(H, W)
    imp_raw = sal_norm * surp_n * kl_n * text_flag.float()
    imp_n   = _normalize_01(imp_raw)

    # ── 5. Epsilon budget map ─────────────────────────────────────────────────
    E = torch.full((1, H, W), epsilon_bg, dtype=torch.float32)
    text_pixels = text_mask > 0
    E[text_pixels] = (
        epsilon_min + (epsilon_max - epsilon_min) * imp_n.unsqueeze(0)[text_pixels]
    )

    components = {
        "salience":   sal_norm,
        "surprise":   _normalize_01(surprise_map * text_flag.float()),
        "kl":         _normalize_01(kl_map        * text_flag.float()),
        "importance": imp_n,
    }
    return E.to(device), components
