"""Transcript-dependent scoring signals for the v5 clip-detection pipeline.

Each public function takes plain text (or a first-sentence string) and returns
a float in [0.0, 1.0].  All functions are pure and dependency-free — no heavy
NLP libraries, only the standard library and regex.

Constants shared with ``clip_detector`` are imported from there rather than
redefined here so they stay in sync.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from processing.clip_detector import (
    GENERIC_KEYWORD_PHRASES,
    QUESTION_WORD_OPENERS,
    REACTION_MARKER_RE,  # noqa: F401 — re-exported for callers that import from here
    STRONG_EMOTION_WORDS,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# Pre-compiled regex used across multiple functions.
_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")
_NUMBER_RE = re.compile(r"\b\d+\b")
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+\b")
# Sentence-end boundary: period/bang/question followed by whitespace (or string end).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


def _load_settings() -> dict:
    """Load ``config/settings.yaml`` and return its parsed dict.

    Returns an empty dict if the file is missing or unparseable so callers
    always receive a safe default.
    """
    try:
        with open(SETTINGS_PATH) as fh:
            data = yaml.safe_load(fh)
    except OSError:
        return {}
    return data or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_sentence(text: str) -> str:
    """Return the first sentence of *text*.

    Splits on ``.``, ``?``, or ``!`` followed by whitespace.  If no such
    boundary is found the whole string is returned.  The result is
    whitespace-stripped.

    Args:
        text: Arbitrary plain text.

    Returns:
        First sentence as a stripped string.
    """
    parts = _SENTENCE_SPLIT_RE.split(text.strip(), maxsplit=1)
    sentence = parts[0].strip() if parts else text.strip()
    # Remove the trailing sentence-end punctuation that the lookbehind kept.
    return sentence.rstrip(".?!")


# ---------------------------------------------------------------------------
# Scoring functions — all return float in [0.0, 1.0]
# ---------------------------------------------------------------------------

# Words that indicate a continuation / context-dependent opener — a weak hook.
_BAD_STARTERS = {
    "and", "but", "so", "which", "because",
    "it", "that", "they", "this", "those",
}

# Patterns that signal a strong, concrete hook statement.
_HOOK_PATTERNS = re.compile(
    r"""
    \$[\d,]+                                # dollar amount
    | \d+%                                  # percentage
    | \b(nobody|no\s+one)\s+(talks|knows)\b # "nobody talks/knows"
    | \bthe\s+(truth|real\s+reason|problem)\b
    | \bi\s+(lost|made|quit|failed)\b
    | \bhere'?s\s+(why|what|how)\b
    | \b(most|everyone)\s+(people|thinks)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def hook_quality(first_sentence_text: str) -> float:
    """Score the hook strength of a clip's opening sentence.

    Starts at 1.0 and applies two negative adjustments:

    * ``-0.5`` if the first word is a continuation-style word (e.g. "and",
      "but", "because") — these require prior context and land weakly as
      standalone openers.
    * ``-0.2`` if the sentence exceeds 20 words — too verbose to land as a
      punchy hook.

    Then adds up to 0.4 bonus based on how many ``_HOOK_PATTERNS`` match
    (``min(hits * 0.2, 0.4)``).

    Args:
        first_sentence_text: The first sentence of the clip window (use
            ``first_sentence()`` to extract it).

    Returns:
        Float in [0.0, 1.0].
    """
    score = 1.0

    words = first_sentence_text.split()
    if not words:
        return 0.0

    # Strip trailing punctuation to normalise the first word.
    first_word = words[0].lower().rstrip(".,!?;:")
    if first_word in _BAD_STARTERS:
        score -= 0.5

    if len(words) > 20:
        score -= 0.2

    hits = len(_HOOK_PATTERNS.findall(first_sentence_text))
    score += min(hits * 0.2, 0.4)

    return max(0.0, min(1.0, score))


# Single-word openers that imply the clip depends on prior context.
_REQUIRES_CONTEXT = {
    "which", "this", "that", "it", "they", "them", "those",
    "he", "she", "because", "so", "therefore", "thus",
}

# Multi-word context-dependency phrases, lowercased for startswith checks.
_MULTIWORD_CONTEXT_PHRASES = (
    "as i said",
    "like i mentioned",
    "as we discussed",
)

# Single-word openers that produce a clean, self-contained start.
_CLEAN_OPENERS = {"i", "there", "the", "a", "an", "you", "we"}


def standalone_coherence(first_sentence_text: str) -> float:
    """Estimate whether the clip's opening sentence stands alone without context.

    Returns:
        * ``0.1`` — opener is a pronoun/conjunction that requires prior
          context (e.g. "they", "because", "as I said …").
        * ``0.8`` — opener is a clean, subject-launching word (e.g. "I",
          "The", "You").
        * ``0.6`` — neutral: anything else.

    Args:
        first_sentence_text: First sentence of the clip window.

    Returns:
        Float in {0.1, 0.6, 0.8}.
    """
    lowered = first_sentence_text.lower().strip()

    # Multi-word context phrases checked first (more specific).
    if any(lowered.startswith(phrase) for phrase in _MULTIWORD_CONTEXT_PHRASES):
        return 0.1

    words = lowered.split()
    if not words:
        return 0.6

    first_word = words[0].rstrip(".,!?;:")
    if first_word in _REQUIRES_CONTEXT:
        return 0.1
    if first_word in _CLEAN_OPENERS:
        return 0.8
    return 0.6


def sentiment(text: str) -> float:
    """Strong-emotion word density, normalised to [0.0, 1.0].

    Ports the logic from ``clip_detector._score_sentiment``: counts words
    that appear in ``STRONG_EMOTION_WORDS`` and normalises so that roughly
    one strong-emotion word per eight words of text saturates the score.

    Args:
        text: Raw transcript text for the window.

    Returns:
        Float in [0.0, 1.0].
    """
    words = re.findall(r"[a-z']+", text.lower())
    word_count = len(words)
    if word_count == 0:
        return 0.0
    hits = sum(1 for w in words if w in STRONG_EMOTION_WORDS)
    # 1 strong-emotion word per ~8 words of text reaches max score.
    return min(1.0, hits / max(1, word_count / 8))


def specificity(text: str) -> float:
    """Reward concrete, verifiable claims in *text*.

    Counts four categories of concrete-claim markers:

    * Dollar amounts (``$1,000``, ``$4.99``).
    * Percentages (``37%``, ``0.5%``).
    * Standalone integers (``\b\d+\b``).
    * Capitalised proper-noun-like tokens (``[A-Z][a-z]+``) that do *not*
      appear at the start of their sentence (a simple heuristic to skip
      sentence-initial capitalisation).

    Normalisation: ~3 concrete hits saturates to 1.0 (``min(1.0, hits / 3)``).

    Args:
        text: Raw transcript text for the window.

    Returns:
        Float in [0.0, 1.0].
    """
    hits = 0
    hits += len(_DOLLAR_RE.findall(text))
    hits += len(_PERCENT_RE.findall(text))
    hits += len(_NUMBER_RE.findall(text))

    # Capitalised tokens not at sentence start — proxy for proper nouns.
    # Build set of first tokens per sentence to exclude them.
    sentences = _SENTENCE_SPLIT_RE.split(text)
    sentence_start_tokens: set[str] = set()
    for sent in sentences:
        leading = sent.lstrip()
        if leading:
            first_tok = leading.split()[0]
            sentence_start_tokens.add(first_tok)

    for token in _CAP_TOKEN_RE.findall(text):
        if token not in sentence_start_tokens:
            hits += 1

    return min(1.0, hits / 3.0)


def curiosity(text: str) -> float:
    """Merge question and keyword signals into a single curiosity score.

    Combines three cue types:

    1. **Question marks** — literal ``?`` characters in the text.
    2. **Question-word openers** — sentences starting with words from
       ``QUESTION_WORD_OPENERS`` (e.g. "how", "why", "what").
    3. **Generic keyword phrases** — high-engagement hook phrases from
       ``GENERIC_KEYWORD_PHRASES`` found anywhere in the text.

    **Normalisation:** each category is independently capped at 1.0 via
    ``min(1.0, count / 3.0)`` (matching ``_score_question``) or
    ``min(1.0, count / 2.0)`` (matching ``_score_keyword``), then the
    two resulting values are averaged.  This preserves the calibration
    of the individual signals so their contribution is balanced.

    Args:
        text: Raw transcript text for the window.

    Returns:
        Float in [0.0, 1.0].
    """
    lowered = text.lower()

    # --- Question signal (ported from _score_question) ---
    mark_hits = lowered.count("?")
    sentences = re.split(r"(?<=[.?!])\s+|\n", text)
    opener_hits = 0
    for sentence in sentences:
        stripped = sentence.strip().lower()
        if stripped and stripped.startswith(QUESTION_WORD_OPENERS):
            opener_hits += 1
    question_score = min(1.0, (mark_hits + opener_hits) / 3.0)

    # --- Keyword signal (ported from _score_keyword) ---
    kw_hits = sum(1 for phrase in GENERIC_KEYWORD_PHRASES if phrase in lowered)
    keyword_score = min(1.0, kw_hits / 2.0)

    combined = (question_score + keyword_score) / 2.0
    return max(0.0, min(1.0, combined))
