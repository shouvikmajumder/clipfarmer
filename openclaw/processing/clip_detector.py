"""Clip detection stage: score transcript windows and select the best clips.

Slides a window across the transcript timeline and scores each window using
five dependency-free heuristic signals (see ``plan_v4_trimmed.md`` section 6):

| Signal                      | Weight |
|------------------------------|-------|
| Sentiment magnitude          | 0.30  |
| Speech pace                  | 0.20  |
| Question density             | 0.15  |
| Generic keyword hits         | 0.20  |
| Laughter / reaction markers  | 0.15  |

The top ``max_clips`` non-overlapping windows that clear ``min_clip_score``
(read from ``config/settings.yaml``, default ``0.45``) are returned, sorted by
score descending. Consumes the segment shape produced by
``processing.transcriber.transcribe``: ``[{"start": float, "end": float,
"text": str, ...}]``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# Path to settings.yaml relative to this file: openclaw/processing/clip_detector.py -> openclaw/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

DEFAULT_MIN_CLIP_SCORE = 0.45

# Sliding window bounds, in seconds, per plan section 6.
WINDOW_MIN_S = 20.0
WINDOW_MAX_S = 45.0
WINDOW_STEP_S = 5.0

# Signal weights — must sum to 1.0.
WEIGHT_SENTIMENT = 0.30
WEIGHT_PACE = 0.20
WEIGHT_QUESTION = 0.15
WEIGHT_KEYWORD = 0.20
WEIGHT_REACTION = 0.15

# Small built-in lexicon of strong-emotion words for sentiment magnitude.
# v1: a basic count-based lexicon, no external NLP dependency.
STRONG_EMOTION_WORDS = {
    "amazing", "incredible", "insane", "shocking", "unbelievable", "terrifying",
    "furious", "outraged", "devastated", "heartbroken", "thrilled", "ecstatic",
    "hate", "love", "disgusting", "horrible", "fantastic", "brutal", "savage",
    "crazy", "wild", "ridiculous", "exposed", "betrayed", "humiliated",
    "explosive", "scandal", "disaster", "nightmare", "miracle", "stunning",
}

# Question-word sentence openers, used alongside literal "?" counts.
QUESTION_WORD_OPENERS = (
    "who", "what", "why", "how", "when", "where", "which", "is it", "are you",
    "do you", "can you", "should",
)

# ~15-phrase list of generic high-engagement hook phrases.
GENERIC_KEYWORD_PHRASES = [
    "the truth is",
    "nobody talks about",
    "here's why",
    "what nobody tells you",
    "the real reason",
    "this changes everything",
    "you won't believe",
    "i can't believe",
    "the secret to",
    "nobody tells you",
    "the biggest mistake",
    "here's the thing",
    "what they don't want you to know",
    "the hard truth",
    "let me explain",
]

# Whisper-style bracketed reaction/laughter tags.
REACTION_MARKER_RE = re.compile(r"\[(laughter|applause|laughs|cheering|crosstalk)\]", re.IGNORECASE)


def _load_min_clip_score() -> float:
    """Read ``general.min_clip_score`` from ``config/settings.yaml``.

    Falls back to ``DEFAULT_MIN_CLIP_SCORE`` if the file or key is missing.
    """
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
    except OSError:
        return DEFAULT_MIN_CLIP_SCORE

    if not settings:
        return DEFAULT_MIN_CLIP_SCORE

    return settings.get("general", {}).get("min_clip_score", DEFAULT_MIN_CLIP_SCORE)


def _score_sentiment(text: str, word_count: int) -> float:
    """Strong-emotion word hits, normalised by word count, clamped to [0, 1]."""
    if word_count == 0:
        return 0.0
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    hits = sum(1 for w in words if w in STRONG_EMOTION_WORDS)
    # 1 strong-emotion word per ~8 words of text reaches max score.
    return min(1.0, hits / max(1, word_count / 8))


def _score_pace(word_count: int, duration_s: float) -> float:
    """Words per second, normalised against a typical engaging speaking pace.

    ~2.5-3.0 wps is energetic conversational speech; we treat 3.0 wps as the
    point that saturates the score, with a gentle penalty for very slow pace.
    """
    if duration_s <= 0:
        return 0.0
    wps = word_count / duration_s
    return max(0.0, min(1.0, wps / 3.0))


def _score_question(text: str) -> float:
    """Question-mark count and question-word openers, normalised per window."""
    lowered = text.lower()
    mark_hits = lowered.count("?")

    sentences = re.split(r"(?<=[.?!])\s+|\n", text)
    opener_hits = 0
    for sentence in sentences:
        stripped = sentence.strip().lower()
        if not stripped:
            continue
        if stripped.startswith(QUESTION_WORD_OPENERS):
            opener_hits += 1

    total_hits = mark_hits + opener_hits
    # 3+ question cues in a window is a strong signal.
    return min(1.0, total_hits / 3.0)


def _score_keyword(text: str) -> float:
    """Generic high-engagement hook phrase hits, normalised per window."""
    lowered = text.lower()
    hits = sum(1 for phrase in GENERIC_KEYWORD_PHRASES if phrase in lowered)
    # A single hook phrase is already a strong signal; 2+ saturates.
    return min(1.0, hits / 2.0)


def _score_reaction(text: str) -> float:
    """Laughter/applause/reaction tag hits, normalised per window."""
    hits = len(REACTION_MARKER_RE.findall(text))
    return min(1.0, hits / 2.0)


def score_window(segments: list[dict]) -> float:
    """Compute a composite engagement score for a window of transcript segments.

    Applies all five weighted signals (sentiment magnitude, speech pace,
    question density, generic keyword hits, laughter/reaction markers) to the
    concatenated text of *segments* and returns a value in [0.0, 1.0].

    Args:
        segments: Transcript segment dicts (``start``, ``end``, ``text``)
                   that fall within the window being scored.

    Returns:
        Float composite score in the range [0.0, 1.0].
    """
    if not segments:
        return 0.0

    text = " ".join(seg.get("text", "") for seg in segments)
    word_count = len(re.findall(r"\S+", text))
    start = segments[0].get("start", 0.0) or 0.0
    end = segments[-1].get("end", start) or start
    duration_s = max(0.0, end - start)

    sentiment = _score_sentiment(text, word_count)
    pace = _score_pace(word_count, duration_s)
    question = _score_question(text)
    keyword = _score_keyword(text)
    reaction = _score_reaction(text)

    composite = (
        sentiment * WEIGHT_SENTIMENT
        + pace * WEIGHT_PACE
        + question * WEIGHT_QUESTION
        + keyword * WEIGHT_KEYWORD
        + reaction * WEIGHT_REACTION
    )
    return max(0.0, min(1.0, composite))


def detect_clips(segments: list[dict], max_clips: int = 3) -> list[dict]:
    """Slide a window over *segments* and return the top non-overlapping clips.

    Builds candidate windows of ``WINDOW_MIN_S``-``WINDOW_MAX_S`` seconds
    (stepping by ``WINDOW_STEP_S``) anchored at each segment start, scores
    each window with ``score_window``, discards windows that score below
    ``general.min_clip_score`` (read from ``config/settings.yaml``, default
    0.45), then greedily selects up to *max_clips* of the highest-scoring
    windows with no time-range overlap.

    Args:
        segments: List of transcript segment dicts as returned by
                  ``processing.transcriber.transcribe``: each has at minimum
                  ``start`` (float), ``end`` (float), and ``text`` (str).
                  Native passthrough fields are ignored.
        max_clips: Maximum number of clips to return (default 3).

    Returns:
        List of dicts shaped ``{"start_s": float, "end_s": float, "score":
        float}``, sorted by score descending. May be empty if no window
        clears the minimum score threshold or *segments* is empty.
    """
    if not segments or max_clips <= 0:
        return []

    min_score = _load_min_clip_score()

    # Sort defensively by start time in case the caller passes unsorted segments.
    ordered = sorted(segments, key=lambda s: s.get("start", 0.0) or 0.0)
    timeline_end = max((s.get("end", 0.0) or 0.0) for s in ordered)

    candidates: list[dict[str, Any]] = []
    window_lengths = []
    length = WINDOW_MIN_S
    while length <= WINDOW_MAX_S:
        window_lengths.append(length)
        length += WINDOW_STEP_S
    if WINDOW_MAX_S not in window_lengths:
        window_lengths.append(WINDOW_MAX_S)

    for seg in ordered:
        window_start = seg.get("start", 0.0) or 0.0
        for window_len in window_lengths:
            window_end = window_start + window_len
            if window_end > timeline_end:
                continue
            window_segments = [
                s
                for s in ordered
                if (s.get("start", 0.0) or 0.0) < window_end
                and (s.get("end", 0.0) or 0.0) > window_start
            ]
            if not window_segments:
                continue
            score = score_window(window_segments)
            if score >= min_score:
                candidates.append(
                    {"start_s": window_start, "end_s": window_end, "score": score}
                )

    # Highest score first; stable tie-break by earlier start time.
    candidates.sort(key=lambda c: (-c["score"], c["start_s"]))

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(selected) >= max_clips:
            break
        overlaps = any(
            candidate["start_s"] < chosen["end_s"] and candidate["end_s"] > chosen["start_s"]
            for chosen in selected
        )
        if not overlaps:
            selected.append(candidate)

    return selected
