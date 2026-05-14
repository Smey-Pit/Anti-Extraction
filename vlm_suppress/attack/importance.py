# ══════════════════════════════════════════════════════════════════════════════
# vlm_suppress/attack/importance.py
#
# Stage 1: Domain-agnostic token importance mapping.
#
# Produces a per-pixel importance map I(i,j) that identifies visually-grounded,
# semantically surprising tokens rather than structurally salient regions:
#
#   imp_raw = 0.2·S + 0.4·Surprise + 0.4·KL          (weighted sum)
#   imp_raw = max(imp_raw, 0.3·(Surprise+KL).clamp(0,1))   (floor)
#
# Components:
#   S         — gradient salience (‖∇_x L_ce‖₂, existing signal)
#   Surprise  — -log p(word | blank_image, field-reset context)
#               [one pass per field section; context resets at ALL_CAPS headers]
#   KL        — log p(word | orig) - log p(word | context-masked_image)
#               [one pass per non-context-overlapping group]
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


# ── Geometry helpers for contextual masking ───────────────────────────────────

def _box_center(box: list[int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _center_dist(a: list[int], b: list[int]) -> float:
    cx_a, cy_a = _box_center(a)
    cx_b, cy_b = _box_center(b)
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def _get_context_indices(
    word_idx: int,
    word_boxes: list[list[int]],
    context_radius_px: float,
) -> list[int]:
    """
    Return all word indices whose box centre is within context_radius_px of
    word_idx, including word_idx itself.  These are all masked together when
    computing the visual KL for word_idx.
    """
    box_i = word_boxes[word_idx]
    result = []
    for j, box_j in enumerate(word_boxes):
        if _center_dist(box_i, box_j) <= context_radius_px:
            result.append(j)
    return result


def _greedy_kl_groups(
    word_boxes: list[list[int]],
    context_radius_px: float,
) -> list[list[int]]:
    """
    Like _greedy_nonoverlap_groups but also forbids two words from the same
    group if they are within context_radius_px of each other.

    Rationale: if word k is within context_radius of word i, masking i also
    masks k (as context). Placing both in the same batch would create an
    inconsistent combined mask; measuring p(k | ...) from a pass that masked
    k-as-context-of-i AND k-as-principal would give the right number, but
    measuring p(i | ...) from a pass that also masked k-as-principal adds an
    uncontrolled extra mask to i's measurement.  Keeping them in separate
    groups eliminates this cross-contamination.
    """
    n = len(word_boxes)
    assigned = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            conflict = any(
                _boxes_overlap(word_boxes[k], word_boxes[j])
                or _center_dist(word_boxes[k], word_boxes[j]) <= context_radius_px
                for k in group
            )
            if not conflict:
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


# ── Section-boundary detection for surprise context reset ─────────────────────

def _same_row(box_i: list[int], box_j: list[int]) -> bool:
    """True if two boxes share approximately the same text row."""
    _, cy_i = _box_center(box_i)
    _, cy_j = _box_center(box_j)
    h_i = max(box_i[3] - box_i[1], 1)
    h_j = max(box_j[3] - box_j[1], 1)
    return abs(cy_i - cy_j) < 1.5 * max(h_i, h_j)


def _is_section_boundary(
    word_idx: int,
    words:     list[str],
    word_boxes: list[list[int]],
) -> bool:
    """
    True if this word is a section-header token that should reset the
    surprise context window.

    Heuristics (both must hold):
      1. The word (stripped of trailing punctuation) is ALL_CAPS and ≥2 chars.
      2. No word on the same bounding-box row is mixed/lower case — i.e., the
         word is not a mid-sentence caps abbreviation surrounded by prose.
    """
    raw = words[word_idx].strip()
    core = raw.rstrip(":：.-_#").strip()
    if len(core) < 2 or not core.isalpha() or not core.isupper():
        return False

    box_i = word_boxes[word_idx]
    for j, (w_j, box_j) in enumerate(zip(words, word_boxes)):
        if j == word_idx:
            continue
        if not _same_row(box_i, box_j):
            continue
        core_j = w_j.strip().rstrip(":：.-_#").strip()
        if core_j and not core_j.isupper():
            return False   # mixed-case neighbour → not a pure header row
    return True


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


def _word_top_k(
    model,
    image_tensor: torch.Tensor,
    transcript:   str,
    spans:        list[tuple[int, int]],
    n_words:      int,
    top_k:        int,
) -> list[tuple[float, torch.Tensor, torch.Tensor]]:
    """
    One forward pass → per-word (correct_lp, best_wrong_lp, best_wrong_ids).

    Calls model.token_logprobs(image_tensor, transcript, return_top_k=top_k).
    Expects a 4-tuple — caller must verify support before calling.

    For each word span (span_s, span_e):

      correct_lp : float
        sum of log p(correct_token_t) across token positions t in span

      best_wrong_lp : (span_len,) float32 cpu tensor
        for each position t in span, the log prob of the highest-ranked
        token in top_k whose id != correct token id at position t.
        If all top-k entries are the correct token, use the last entry.

      best_wrong_ids : (span_len,) int64 cpu tensor
        token ids corresponding to best_wrong_lp at each position.

    Returns list of length n_words.
    Out-of-range spans return (0.0, zeros(1), zeros(1)).
    """
    result = model.token_logprobs(
        image_tensor, transcript, return_top_k=top_k
    )
    token_lp, tok_ids, top_k_log_probs, top_k_id_tensor = result

    token_lp        = token_lp.cpu()
    tok_ids         = tok_ids.cpu()
    top_k_log_probs = top_k_log_probs.cpu()
    top_k_id_tensor = top_k_id_tensor.cpu()
    T = token_lp.shape[0]

    output = []
    for span_s, span_e in spans:
        if span_s is None or span_s >= T:
            output.append((0.0, torch.zeros(1), torch.zeros(1, dtype=torch.long)))
            continue

        span_e_clamped = min(span_e, T)

        correct_lp = float(token_lp[span_s:span_e_clamped].sum())

        best_wrong_lp  = []
        best_wrong_ids = []

        for pos in range(span_s, span_e_clamped):
            correct_id = int(tok_ids[pos].item())
            kk_lp  = top_k_log_probs[pos]   # (K,)
            kk_ids = top_k_id_tensor[pos]    # (K,)

            wrong_mask = (kk_ids != correct_id)
            if wrong_mask.any():
                first_wrong = int(wrong_mask.nonzero(as_tuple=True)[0][0])
                best_wrong_lp.append(float(kk_lp[first_wrong]))
                best_wrong_ids.append(int(kk_ids[first_wrong]))
            else:
                # All top-k are the correct token — model is extremely
                # confident. Use last entry as a floor.
                best_wrong_lp.append(float(kk_lp[-1]))
                best_wrong_ids.append(int(kk_ids[-1]))

        span_len = max(1, span_e_clamped - span_s)
        output.append((
            correct_lp / span_len,
            torch.tensor(best_wrong_lp,  dtype=torch.float32) / span_len,
            torch.tensor(best_wrong_ids, dtype=torch.int64),
        ))

    return output


# ── Core computation ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_token_surprise(
    model:        object,
    image_tensor: torch.Tensor,   # (3, H, W) float32 [0,1]
    transcript:   str,
    word_boxes:   list[list[int]],
) -> torch.Tensor:                # (H, W) float32 CPU
    """
    Pixel-space surprise map: -log p(word | blank_image, field-reset context).

    Context window resets at each ALL_CAPS section-header word that has no
    mixed-case neighbours on its bounding-box row.  This prevents value-word
    surprise from conditioning on the same token appearing in a prior field
    (e.g., "COPD" in a second clinical field no longer conditions on "COPD"
    seen in the diagnosis field).

    Requires model.token_logprobs().  Returns zeros if not available.
    Cost: ONE forward pass per detected field section (≥ 1, ≤ n_words).
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
    words   = transcript.split()[:n_words]
    blank   = _make_blank(image_tensor).to(model.device)

    # Identify section boundaries (indices into words[])
    # Word 0 always starts a section; additional boundaries reset context.
    section_starts = [0]
    for i in range(1, len(words)):   # len(words) ≤ n_words
        if _is_section_boundary(i, words, word_boxes[:len(words)]):
            section_starts.append(i)

    n_sections = len(section_starts)
    if n_sections > 1:
        print(f"    [surprise] {n_sections} field sections detected "
              f"(boundaries at words: {section_starts[1:]})")

    # Per-section forward pass on the blank image
    n_actual    = len(words)   # may be less than n_words if transcript is shorter
    word_scores = [0.0] * n_words
    for sec_idx, sec_start in enumerate(section_starts):
        sec_end = (
            section_starts[sec_idx + 1] if sec_idx + 1 < n_sections else n_actual
        )
        if sec_start >= sec_end:
            continue

        section_words      = words[sec_start:sec_end]
        section_transcript = " ".join(section_words)
        section_spans      = _align_tokens_to_words(
            tokenizer, section_transcript, len(section_words)
        )
        section_lp = _word_logprobs(
            model, blank, section_transcript, section_spans, len(section_words)
        )
        for i, lp in enumerate(section_lp):
            word_scores[sec_start + i] = -lp   # surprise = -log p

    return _scores_to_pixel_map(word_scores, word_boxes[:n_words], H, W)


@torch.no_grad()
def compute_visual_kl(
    model:             object,
    image_tensor:      torch.Tensor,   # (3, H, W) float32 [0,1]
    transcript:        str,
    word_boxes:        list[list[int]],
    context_radius_px: float = 50.0,
) -> torch.Tensor:                     # (H, W) float32 CPU
    """
    Pixel-space visual-KL map: log p(word | orig) - log p(word | context-masked).

    Context masking: when measuring word i, mask box_i AND all word boxes whose
    centres are within context_radius_px pixels of box_i's centre.  This
    approximates field-level masking — erasing the label next to a value along
    with the value itself avoids the model "reading" the label and guessing the
    value without needing the actual value pixels.

    Batching: words whose context masks would cross-contaminate each other are
    placed in separate groups (_greedy_kl_groups).  Within a group, all words
    and their contexts are masked simultaneously in one forward pass.

    Requires model.token_logprobs().  Returns zeros if not available.
    Cost: 1 (original) + N_groups forward passes (N_groups ≤ n_words,
    typically much smaller due to batching).
    """
    if not hasattr(model, "token_logprobs"):
        warnings.warn(
            f"compute_visual_kl: {type(model).__name__} has no token_logprobs — "
            "returning zero map.",
            RuntimeWarning, stacklevel=2,
        )
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return torch.zeros(H, W)

    tokenizer  = _get_tokenizer(model)
    H, W       = image_tensor.shape[-2], image_tensor.shape[-1]
    n_words    = len(word_boxes)
    mean_fill  = float(image_tensor.mean())
    dev        = model.device

    spans = _align_tokens_to_words(tokenizer, transcript, n_words)

    # Baseline: original image log probs (one pass)
    orig_word_lp = _word_logprobs(model, image_tensor.to(dev), transcript, spans, n_words)

    # Build batched groups that respect context-radius isolation
    groups = _greedy_kl_groups(word_boxes, context_radius_px)

    masked_word_lp: list[Optional[float]] = [None] * n_words
    for group in groups:
        img_masked = image_tensor.clone()
        # For each principal word in this group, mask it AND its context words
        masked_set: set[int] = set()
        for idx in group:
            for cidx in _get_context_indices(idx, word_boxes, context_radius_px):
                masked_set.add(cidx)

        for cidx in masked_set:
            x0, y0, x1, y1 = (int(v) for v in word_boxes[cidx])
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


@torch.no_grad()
def compute_confidence_drop(
    model:             object,
    image_tensor:      torch.Tensor,
    transcript:        str,
    word_boxes:        list[list[int]],
    context_radius_px: float = 50.0,
    top_k:             int   = 10,
) -> torch.Tensor:
    """
    Pixel-space confidence drop map.

    For each word w:
      ConfidenceDrop(w) = correct_lp(w | original)
                        - sum_over_positions(best_wrong_lp(w | masked))
      Clamped to >= 0.

    High where masking the pixel region causes the model to lose
    confidence in the correct token AND redistribute mass toward
    plausible wrong alternatives. Zero where the model remains
    confident despite masking.

    Uses identical context-radius masking and batching as
    compute_visual_kl. Same cost: 1 + N_groups forward passes.

    Requires model.token_logprobs() with return_top_k > 0 support.
    Verified by checking return tuple length before main computation.
    Returns zero map if not supported.
    """
    if not hasattr(model, "token_logprobs"):
        warnings.warn(
            f"compute_confidence_drop: {type(model).__name__} has no "
            "token_logprobs — returning zero map.",
            RuntimeWarning, stacklevel=2,
        )
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return torch.zeros(H, W)

    # Verify return_top_k support with minimal transcript
    try:
        _probe_transcript = transcript[:20] if len(transcript) > 20 \
                            else transcript
        _probe = model.token_logprobs(
            image_tensor, _probe_transcript, return_top_k=1
        )
        if len(_probe) != 4:
            warnings.warn(
                f"compute_confidence_drop: {type(model).__name__}."
                "token_logprobs did not return 4-tuple with return_top_k=1"
                " — returning zero map.",
                RuntimeWarning, stacklevel=2,
            )
            H, W = image_tensor.shape[-2], image_tensor.shape[-1]
            return torch.zeros(H, W)
    except Exception as exc:
        warnings.warn(
            f"compute_confidence_drop: probe failed ({exc})"
            " — returning zero map.",
            RuntimeWarning, stacklevel=2,
        )
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return torch.zeros(H, W)

    tokenizer = _get_tokenizer(model)
    H, W      = image_tensor.shape[-2], image_tensor.shape[-1]
    n_words   = len(word_boxes)
    mean_fill = float(image_tensor.mean())
    dev       = model.device

    spans = _align_tokens_to_words(tokenizer, transcript, n_words)

    # ── Baseline: original image ──────────────────────────────────────
    orig_data = _word_top_k(
        model, image_tensor.to(dev),
        transcript, spans, n_words, top_k,
    )
    orig_correct_lp = [entry[0] for entry in orig_data]

    # ── Masked passes — same batching as compute_visual_kl ────────────
    groups = _greedy_kl_groups(word_boxes, context_radius_px)
    masked_wrong_lp: list[float | None] = [None] * n_words

    for group in groups:
        img_masked = image_tensor.clone()
        masked_set: set[int] = set()
        for idx in group:
            for cidx in _get_context_indices(
                idx, word_boxes, context_radius_px
            ):
                masked_set.add(cidx)
        for cidx in masked_set:
            x0, y0, x1, y1 = (int(v) for v in word_boxes[cidx])
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 > x0 and y1 > y0:
                img_masked[:, y0:y1, x0:x1] = mean_fill

        masked_data = _word_top_k(
            model, img_masked.to(dev),
            transcript, spans, n_words, top_k,
        )
        for idx in group:
            # Sum best-wrong log probs across token positions in span
            masked_wrong_lp[idx] = float(masked_data[idx][1].sum())

    # ── Confidence drop per word ──────────────────────────────────────
    import math

    def _safe_exp(x: float) -> float:
        """exp clamped to avoid overflow on large positive values."""
        return math.exp(max(x, -500.0))

    word_cd = [
        max(0.0, _safe_exp(orig) - _safe_exp(mw if mw is not None else 0.0))
        for orig, mw in zip(orig_correct_lp, masked_wrong_lp)
    ]

    return _scores_to_pixel_map(word_cd, word_boxes[:n_words], H, W)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_importance_map(
    image_tensor:      torch.Tensor,       # (1, 3, H, W) or (3, H, W) float32 [0,1]
    transcript:        str,
    word_boxes:        list[list[int]],
    surrogates:        list,
    alpha_weights:     list[float],
    epsilon_min:       float,
    epsilon_max:       float,
    epsilon_bg:        float,
    dilation:          int,
    device:            torch.device,
    use_surprise:      bool  = True,
    use_visual_kl:     bool  = True,
    context_radius_px: float = 50.0,
) -> tuple[torch.Tensor, dict]:
    """
    Build an importance-weighted epsilon budget map (diagnostic only).

    Pipeline:
      1. Gradient salience  — ‖∇_x L_ce‖₂  (from build_salience_budget_map)
      2. Token surprise     — -log p(w | blank, field-reset context)
      3. Visual KL          — Δ log p(w | orig vs context-masked)
      4. Weighted sum + floor:
             imp_raw  = 0.2·S + 0.4·Surprise + 0.4·KL
             imp_raw  = max(imp_raw, 0.3·(Surprise+KL).clamp(0,1))
      5. Build eps map: text ← epsilon_min + (epsilon_max−epsilon_min)·I
                        bg   ← epsilon_bg

    Parameters
    ----------
    context_radius_px : radius (pixels) for field-level context masking in KL.

    Returns
    -------
    eps_map    : (1, H, W) float32 budget map on `device`
    components : dict with CPU (H, W) tensors: salience, surprise, kl, importance
    """
    from vlm_suppress.attack.salience import build_salience_budget_map
    from vlm_suppress.models.lazy import LazySurrogate

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)   # (3, H, W)

    H, W     = image_tensor.shape[-2], image_tensor.shape[-1]
    image_4d = image_tensor.unsqueeze(0)

    text_mask = build_text_mask(H, W, word_boxes, dilation, device=torch.device("cpu"))
    text_flag = text_mask.squeeze(0) > 0   # (H, W) bool

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

    # Undo linear epsilon scaling → raw salience in [0, 1]
    sal_raw   = sal_4d.squeeze(0).cpu()
    eps_range = epsilon_max - epsilon_min
    if eps_range > 1e-9:
        sal_raw = (sal_raw - epsilon_min) / eps_range
    sal_raw  = sal_raw.clamp(0.0, 1.0) * text_flag.float()
    sal_norm = _normalize_01(sal_raw)

    # ── 2. Token surprise (with field-reset context) ───────────────────────────
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

    # ── 3. Visual KL (with contextual span masking) ───────────────────────────
    kl_map = torch.zeros(H, W, dtype=torch.float32)
    if use_visual_kl:
        n_groups = len(_greedy_kl_groups(word_boxes, context_radius_px))
        for surrogate, alpha in zip(surrogates, alpha_weights):
            print(
                f"  [importance] visual KL — {surrogate.name} "
                f"({len(word_boxes)} words, r={context_radius_px:.0f}px "
                f"→ {n_groups} batched passes) ..."
            )
            if isinstance(surrogate, LazySurrogate):
                with surrogate as model:
                    k_k = compute_visual_kl(
                        model, image_tensor, transcript, word_boxes, context_radius_px
                    )
            else:
                k_k = compute_visual_kl(
                    surrogate, image_tensor, transcript, word_boxes, context_radius_px
                )
            kl_map = kl_map + alpha * k_k
        kl_map = kl_map * text_flag.float()

    # ── 4. Weighted sum + floor ───────────────────────────────────────────────
    surp_n = _normalize_01(surprise_map) if use_surprise  else torch.ones(H, W)
    kl_n   = _normalize_01(kl_map)       if use_visual_kl else torch.ones(H, W)

    imp_raw = (0.2 * sal_norm + 0.4 * surp_n + 0.4 * kl_n) * text_flag.float()
    floor   = 0.3 * (surp_n + kl_n).clamp(0.0, 1.0) * text_flag.float()
    imp_raw = torch.maximum(imp_raw, floor)
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
