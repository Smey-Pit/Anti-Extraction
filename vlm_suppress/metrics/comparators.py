"""
Type-aware matchers for Entity Recall (Threat 1).

Each matcher: (target_value: str, prediction: str) -> float in [0, 1]
Each extractor: (target_value: str, prediction: str) -> str  (what was found)

  - name, short_phrase  : fractional, token-coverage of target tokens in prediction
  - digit_seq           : binary, all target digits appear as a contiguous run
  - amount              : binary, numeric value appears
  - date                : binary, canonical (Y, M, D) date appears
  - date_range          : binary, both endpoints match

The matchers do NOT require the prediction to be in any particular format —
they search the prediction string for the target's information content.
This makes Entity Recall robust to prompt-format variance across surrogates.
"""

from __future__ import annotations

import re
from datetime import date as _date

from dateutil import parser as date_parser

# ──────────────────────────────────────────────────────────────────────────
# Token-coverage matchers (fractional)
# ──────────────────────────────────────────────────────────────────────────

# Strip trailing punctuation from each token (but preserve internal apostrophes
# in names like O'Brien). Done after lowercasing.
_TOKEN_TRIM = ".,;:!?\"'()[]{}*_~`"


def _normalize_tokens(text: str) -> set[str]:
    """
    Lowercase, split on whitespace, strip surrounding punctuation,
    normalize English possessives ("thompson's" -> "thompson").
    Returns a set; order is not preserved.
    """
    tokens: set[str] = set()
    for raw in text.lower().split():
        t = raw.strip(_TOKEN_TRIM)
        if t.endswith("'s"):
            t = t[:-2]
        if t:
            tokens.add(t)
    return tokens


def match_name(target: str, prediction: str) -> float:
    """
    Token-coverage of target name tokens in prediction.
    "Ella Thompson" vs "...Bella Thompson..."  -> 0.5  (only "thompson" leaked)
    "Ella Thompson" vs "...Ella Thompson..."   -> 1.0
    """
    target_tokens = _normalize_tokens(target)
    if not target_tokens:
        return 0.0
    prediction_tokens = _normalize_tokens(prediction)
    matched = len(target_tokens & prediction_tokens)
    return matched / len(target_tokens)


def extract_name(target: str, prediction: str) -> str:
    """Return which target tokens were found and which were missing."""
    target_tokens = _normalize_tokens(target)
    prediction_tokens = _normalize_tokens(prediction)
    found   = sorted(target_tokens & prediction_tokens)
    missing = sorted(target_tokens - prediction_tokens)
    parts = []
    if found:
        parts.append(f"found={found}")
    if missing:
        parts.append(f"missing={missing}")
    return " | ".join(parts) if parts else "no_tokens"


def match_short_phrase(target: str, prediction: str) -> float:
    """
    Same as match_name for now — token-coverage, case-insensitive, order-free.
    "Everyday Savings" vs "...everyday spending and savings..." -> 1.0
    "Everyday Savings" vs "...everyday spending..."             -> 0.5
    """
    return match_name(target, prediction)


def extract_short_phrase(target: str, prediction: str) -> str:
    return extract_name(target, prediction)


# ──────────────────────────────────────────────────────────────────────────
# Digit sequence matcher (binary)
# ──────────────────────────────────────────────────────────────────────────


def _digits_only(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def match_digit_seq(target: str, prediction: str) -> float:
    """
    Strip non-digits from target → canonical digit string.
    Find every maximal digit-run in prediction, also concatenate adjacent
    digit-runs (model may have split groups). Look for target digit string
    as substring of any concatenation.

    "633-114 1409 5506 41" vs "...633114 1409 5506 41..."           -> 1.0
    "633-114 1409 5506 41" vs "...633-114-1409-5506-41..."           -> 1.0
    "633-114 1409 5506 41" vs "...633 114 1409 5506 41..."           -> 1.0
    """
    canonical = _digits_only(target)
    if not canonical:
        return 0.0
    digit_runs = re.findall(r"\d+", prediction)
    if not digit_runs:
        return 0.0
    n = len(digit_runs)
    for start in range(n):
        concat = ""
        for end in range(start, n):
            concat += digit_runs[end]
            if canonical in concat:
                return 1.0
            if len(concat) > len(canonical) * 4:
                break
    return 0.0


def extract_digit_seq(target: str, prediction: str) -> str:
    """Return canonical target digits and all digit-runs found in prediction."""
    canonical = _digits_only(target)
    runs = re.findall(r"\d+", prediction)
    concat_all = "".join(runs)
    found = canonical in concat_all
    # Show up to first 6 runs to keep output readable
    runs_preview = runs[:6]
    if len(runs) > 6:
        runs_preview.append(f"...+{len(runs)-6}more")
    return (
        f"target_digits={canonical!r} "
        f"pred_runs={runs_preview} "
        f"{'FOUND' if found else 'NOT_FOUND'}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Amount matcher (binary, numeric equality)
# ──────────────────────────────────────────────────────────────────────────

# Matches dollar amounts, signed amounts, "5 million" style multipliers,
# and bare numbers with optional comma grouping.
# Two alternatives in order of preference: (a) digit-grouped with commas
# ($4,821.33), (b) bare digits (4821.33). Anchored to require not breaking
# a longer number in half.
_AMOUNT_PATTERN = re.compile(
    r"""
    (?<![\d.])              # no digit or dot immediately before
    [-+]?                   # optional sign
    \$?\s?                  # optional currency symbol
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # comma-grouped: 4,821.33 or 1,000
        |
        \d+(?:\.\d+)?                  # bare number: 4821.33 or 5 or 5.5
    )
    (?![\d])                # no digit immediately after
    """,
    re.VERBOSE,
)

# Multipliers for word-suffixed amounts ("$5 million")
_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "mm": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

# Words that follow a number but don't change its value (currency words,
# unit-of-money words). Treat as "no multiplier, value is as-is."
_CURRENCY_WORDS = {
    "dollar", "dollars", "usd",
    "euro", "euros", "eur",
    "pound", "pounds", "gbp",
    "yen", "jpy",
}


def _parse_amount(s: str) -> float | None:
    """
    Parse a string like "$4,821.33", "-123.45", "+2,000.00", "5 million", "USD 50,000".
    Returns the numeric value, or None if it can't be parsed.
    """
    s = s.strip()
    # Remove common currency tokens
    s_clean = re.sub(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", "", s, flags=re.IGNORECASE).strip()
    # Try "N <multiplier>" form first
    m = re.match(
        r"^([-+]?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)?$",
        s_clean.strip(),
    )
    if not m:
        return None
    num_str, suffix = m.group(1), m.group(2)
    num_str = num_str.replace("$", "").replace(",", "").replace(" ", "")
    try:
        value = float(num_str)
    except ValueError:
        return None
    if suffix:
        mult = _MULTIPLIERS.get(suffix.lower())
        if mult is None:
            return None
        value *= mult
    return value


def _extract_amount_candidates(text: str) -> list[float]:
    """
    Find all amount-like substrings in `text` and parse each one.
    Returns the list of successfully parsed numeric values.
    """
    candidates: list[float] = []
    for m in _AMOUNT_PATTERN.finditer(text):
        candidate = m.group(0).strip()
        # Look ahead for a multiplier OR currency word ("5 million", "5 dollars")
        end_pos = m.end()
        suffix_match = re.match(r"\s+([a-zA-Z]+)", text[end_pos : end_pos + 20])
        if suffix_match:
            suffix_word = suffix_match.group(1).lower()
            if suffix_word in _MULTIPLIERS:
                candidate = f"{candidate} {suffix_word}"
            # else: currency words and unrelated words are ignored;
            # the bare numeric value stands.
        v = _parse_amount(candidate)
        if v is not None:
            candidates.append(v)
    return candidates


def match_amount(
    target: str, prediction: str, tolerance: float = 1e-6
) -> float:
    """
    Parse target to numeric value; extract all amounts from prediction;
    return 1.0 if any prediction amount equals target within tolerance,
    else 0.0.

    "$4,821.33" matches "4821.33", "$4,821.33", "USD 4,821.33", "4821.330"
    "$5 million" matches "5 million", "5000000", "$5,000,000"
    "$4,821.33" does NOT match "$4,821" (different value)
    """
    target_value = _parse_amount(target)
    if target_value is None:
        return 0.0
    for candidate in _extract_amount_candidates(prediction):
        if abs(candidate - target_value) <= tolerance:
            return 1.0
    return 0.0


def extract_amount(target: str, prediction: str) -> str:
    """Return parsed target value and all amounts found in prediction."""
    target_value = _parse_amount(target)
    candidates = _extract_amount_candidates(prediction)
    tolerance = 1e-6
    matched = [c for c in candidates if target_value is not None and abs(c - target_value) <= tolerance]
    candidates_preview = [round(c, 4) for c in candidates[:8]]
    if len(candidates) > 8:
        candidates_preview.append("...")
    return (
        f"target={target_value} "
        f"pred_amounts={candidates_preview} "
        f"{'FOUND' if matched else 'NOT_FOUND'}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Date matcher (binary, canonical equality)
# ──────────────────────────────────────────────────────────────────────────

# Heuristic regex to find date-like substrings in unstructured text.
# Catches: 2024-10-01, 10/01/2024, October 1, 2024, 1 Oct 2024, Oct 1 2024
_DATE_REGEX = re.compile(
    r"""
    \b(?:
        \d{4}-\d{1,2}-\d{1,2}                      # 2024-10-01
        | \d{1,2}/\d{1,2}/\d{2,4}                  # 10/01/2024
        | \d{1,2}-\d{1,2}-\d{2,4}                  # 10-01-2024
        | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{2,4})?
        | \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:\s+\d{2,4})?
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _parse_date(s: str, default_year: int | None = None) -> _date | None:
    """
    Parse a date string to a date object. Returns None if unparseable
    or ambiguous (no year and no default).
    """
    try:
        # Use a sentinel default so we can detect if the year was supplied
        sentinel_year = 1
        default = _date(sentinel_year, 1, 1)
        parsed = date_parser.parse(s, default=default, fuzzy=False)
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed.year == sentinel_year:
        # Year was not supplied in the string
        if default_year is None:
            return None
        return _date(default_year, parsed.month, parsed.day)
    return _date(parsed.year, parsed.month, parsed.day)


def match_date(target: str, prediction: str) -> float:
    """
    Parse target to (year, month, day); extract date candidates from
    prediction; return 1.0 if any candidate equals target, else 0.0.
    """
    target_date = _parse_date(target)
    if target_date is None:
        return 0.0
    for m in _DATE_REGEX.finditer(prediction):
        candidate_str = m.group(0)
        candidate_date = _parse_date(candidate_str, default_year=target_date.year)
        if candidate_date == target_date:
            return 1.0
    return 0.0


# ──────────────────────────────────────────────────────────────────────────
# Date-range matcher (binary, both endpoints required)
# ──────────────────────────────────────────────────────────────────────────

# Common range separators: "1 Oct - 31 Oct 2024", "1 Oct to 31 Oct", "1 Oct – 31 Oct"
_RANGE_SEPARATORS = re.compile(r"\s*(?:–|—|-|to|through|thru)\s*", re.IGNORECASE)


def _split_date_range(s: str) -> tuple[str, str] | None:
    """
    Split a date-range string into (start, end). Returns None if no
    recognized separator. The end-side string is what carries the year
    if only one year is given in the range (e.g. "1 Oct – 31 Oct 2024").
    """
    parts = _RANGE_SEPARATORS.split(s, maxsplit=1)
    if len(parts) != 2:
        return None
    start, end = parts[0].strip(), parts[1].strip()
    if not start or not end:
        return None
    return start, end


def extract_date(target: str, prediction: str) -> str:
    """Return parsed target date and all dates found in prediction."""
    target_date = _parse_date(target)
    found_dates = []
    if target_date:
        for m in _DATE_REGEX.finditer(prediction):
            d = _parse_date(m.group(0), default_year=target_date.year)
            if d:
                found_dates.append(str(d))
    matched = target_date and str(target_date) in found_dates
    return (
        f"target={target_date} "
        f"pred_dates={found_dates[:6]} "
        f"{'FOUND' if matched else 'NOT_FOUND'}"
    )


def match_date_range(target: str, prediction: str) -> float:
    """
    Parse target into (start_date, end_date). Both must be recoverable
    from the prediction. Returns 1.0 only if both endpoints are recovered.

    "1 Oct – 31 Oct 2024" -> start=Oct 1 2024, end=Oct 31 2024
    Both must appear in prediction (in any format) for credit.
    """
    parts = _split_date_range(target)
    if parts is None:
        return 0.0
    start_str, end_str = parts

    end_date = _parse_date(end_str)
    if end_date is None:
        return 0.0
    start_date = _parse_date(start_str, default_year=end_date.year)
    if start_date is None:
        return 0.0

    found_dates: set[_date] = set()
    for m in _DATE_REGEX.finditer(prediction):
        d = _parse_date(m.group(0), default_year=end_date.year)
        if d is not None:
            found_dates.add(d)

    if start_date in found_dates and end_date in found_dates:
        return 1.0
    return 0.0


def extract_date_range(target: str, prediction: str) -> str:
    """Return parsed start/end dates and which were found in prediction."""
    parts = _split_date_range(target)
    if parts is None:
        return "could_not_parse_range"
    start_str, end_str = parts
    end_date   = _parse_date(end_str)
    start_date = _parse_date(start_str, default_year=end_date.year if end_date else None)
    found_dates: set = set()
    if end_date:
        for m in _DATE_REGEX.finditer(prediction):
            d = _parse_date(m.group(0), default_year=end_date.year)
            if d:
                found_dates.add(str(d))
    start_found = str(start_date) in found_dates if start_date else False
    end_found   = str(end_date)   in found_dates if end_date   else False
    return (
        f"start={start_date}({'✓' if start_found else '✗'}) "
        f"end={end_date}({'✓' if end_found else '✗'}) "
        f"pred_dates={sorted(found_dates)[:4]}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────

COMPARATORS: dict[str, callable] = {
    "name":         match_name,
    "short_phrase": match_short_phrase,
    "digit_seq":    match_digit_seq,
    "amount":       match_amount,
    "date":         match_date,
    "date_range":   match_date_range,
}

EXTRACTORS: dict[str, callable] = {
    "name":         extract_name,
    "short_phrase": extract_short_phrase,
    "digit_seq":    extract_digit_seq,
    "amount":       extract_amount,
    "date":         extract_date,
    "date_range":   extract_date_range,
}