"""
vlm_suppress/watermark/candidate.py

Surrogate candidate aggregator for targeted watermark substitution.

For each target word, queries each surrogate's top-K alternative tokens at
the first-token position of that word's span.  Scores are aggregated in
log-prob space (weighted by ensemble α_k) across all surrogates that support
the return_top_k API, then sorted to return the top n_candidates per word.

Only models that return a 4-tuple from token_logprobs(..., return_top_k=K)
contribute; models returning a 2-tuple are skipped with a warning.

Usage
-----
    from vlm_suppress.watermark.candidate import aggregate_surrogate_candidates

    results = aggregate_surrogate_candidates(
        image_tensor        = sample.image_tensor.cuda(),
        transcript          = sample.transcript,
        word_strings        = sample.word_strings,
        target_word_indices = [3, 7, 12],
        surrogates          = [qwen_model],
        top_k               = 20,
        n_candidates        = 3,
    )
    # results: {"Thompson": [("Henderson", -3.21), ("Johnson", -4.05), ...], ...}
"""

from __future__ import annotations

import warnings

import torch

from vlm_suppress.attack.importance import _align_tokens_to_words, _get_tokenizer


def aggregate_surrogate_candidates(
    image_tensor: torch.Tensor,
    transcript: str,
    word_strings: list[str],
    target_word_indices: list[int],
    surrogates: list,
    alpha_weights: list[float] | None = None,
    top_k: int = 20,
    n_candidates: int = 3,
    min_cand_len: int = 2,
) -> dict[str, list[tuple[str, float]]]:
    """
    Aggregate top-K surrogate token alternatives for each target word.

    Parameters
    ----------
    image_tensor        : (3, H, W) float32 in [0, 1], on surrogate device
    transcript          : full text string
    word_strings        : flat list of word strings aligned to word_boxes
    target_word_indices : indices into word_strings to query
    surrogates          : list of SurrogateModel instances (already loaded)
    alpha_weights       : ensemble weights; None → uniform 1/K
    top_k               : number of vocab alternatives to request per position
    n_candidates        : how many to return per word
    min_cand_len        : minimum decoded-string length to accept as a candidate

    Returns
    -------
    dict mapping word_string → [(candidate_str, agg_score), ...]
    sorted descending by agg_score (higher = more probable across ensemble).
    Score semantics: weighted sum of log-probs; a ranking signal, not a probability.
    """
    n = len(surrogates)
    if alpha_weights is None:
        alpha_weights = [1.0 / n] * n

    # accumulated scores: word_idx → {candidate_str → float}
    scores: dict[int, dict[str, float]] = {i: {} for i in target_word_indices}

    for surrogate, alpha in zip(surrogates, alpha_weights):
        tokenizer = _get_tokenizer(surrogate)
        if tokenizer is None:
            warnings.warn(
                f"candidate.py: {surrogate.name} has no accessible tokenizer — skipping.",
                RuntimeWarning, stacklevel=2,
            )
            continue

        try:
            result = surrogate.token_logprobs(image_tensor, transcript, return_top_k=top_k)
        except NotImplementedError:
            warnings.warn(
                f"candidate.py: {surrogate.name} does not implement token_logprobs — skipping.",
                RuntimeWarning, stacklevel=2,
            )
            continue

        if not isinstance(result, tuple) or len(result) != 4:
            warnings.warn(
                f"candidate.py: {surrogate.name}.token_logprobs returned a "
                f"{len(result) if isinstance(result, tuple) else type(result).__name__}-tuple "
                f"instead of 4 (return_top_k not supported) — skipping.",
                RuntimeWarning, stacklevel=2,
            )
            continue

        _, _, top_k_lp, top_k_id = result   # (T, K), (T, K)
        T = top_k_lp.shape[0]

        spans = _align_tokens_to_words(
            tokenizer, transcript, len(word_strings), word_strings
        )

        for word_idx in target_word_indices:
            if word_idx >= len(spans):
                continue
            start_tok, _ = spans[word_idx]
            if start_tok >= T:
                continue

            original_lower = word_strings[word_idx].lower()
            K_actual = top_k_id.shape[1]

            for rank in range(K_actual):
                tok_id  = int(top_k_id[start_tok, rank].item())
                lp      = float(top_k_lp[start_tok, rank].item())
                tok_str = tokenizer.decode([tok_id]).strip()

                if (
                    tok_str.isalpha()
                    and len(tok_str) >= min_cand_len
                    and tok_str.lower() != original_lower
                ):
                    scores[word_idx][tok_str] = scores[word_idx].get(tok_str, 0.0) + alpha * lp

    # Build output: word_string → sorted candidates
    output: dict[str, list[tuple[str, float]]] = {}
    for word_idx in target_word_indices:
        word = word_strings[word_idx]
        ranked = sorted(scores[word_idx].items(), key=lambda x: x[1], reverse=True)
        output[word] = ranked[:n_candidates]

    return output
