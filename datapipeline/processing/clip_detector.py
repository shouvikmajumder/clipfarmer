"""Clip detection orchestrator: fuse content + audio + social signals, refine
boundaries, and deduplicate to the final clip list.

v5 — rewrites the old single-signal scorer into a three-layer fusion engine:

1. **Content signals** (``processing.signals``): hook_quality, standalone_coherence,
   sentiment, specificity, curiosity, plus audio proxies when available.
2. **Social signal** (``processing.comments``): timestamp cluster density, weighted
   by adaptive confidence derived from comment volume.
3. **Sponsor penalty** (``processing.comments``): multiplier that suppresses
   windows dominated by sponsor-related comment activity.

Public API
----------
- ``detect(transcript, comments, audio_signals, max_clips)``  — primary entry point.
- ``score_window(segments, audio_signals)``                    — back-compat content-only scorer.
- ``detect_clips(transcript, max_clips)``                      — thin back-compat wrapper.

Module-level names kept for ``signals.py`` and existing tests
-------------------------------------------------------------
``STRONG_EMOTION_WORDS``, ``QUESTION_WORD_OPENERS``, ``GENERIC_KEYWORD_PHRASES``,
``REACTION_MARKER_RE``, ``WINDOW_MIN_S``, ``WINDOW_MAX_S``, ``WINDOW_STEP_S``,
``DEFAULT_MIN_CLIP_SCORE``, ``SETTINGS_PATH``, ``_load_min_clip_score``.

Circular-import note
--------------------
``signals.py`` imports several constants FROM this module at its top level.
Therefore this module MUST NOT import ``processing.signals`` or
``processing.comments`` at module level — those imports are done lazily inside
``detect()`` and ``score_window()`` at call time.  ``AudioSignals`` is received
as a parameter and never imported here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    # Only used for type annotations, never executed at runtime.
    from processing.audio_analyzer import AudioSignals  # noqa: F401

# ---------------------------------------------------------------------------
# Settings path
# ---------------------------------------------------------------------------

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# ---------------------------------------------------------------------------
# Module-level constants (kept for signals.py and test imports)
# ---------------------------------------------------------------------------

DEFAULT_MIN_CLIP_SCORE = 0.45

# Sliding window bounds in seconds — kept as module-level defaults; the
# detection config loader may override these at call time.
WINDOW_MIN_S = 20.0
WINDOW_MAX_S = 45.0
WINDOW_STEP_S = 5.0

# Small built-in lexicon of strong-emotion words.
STRONG_EMOTION_WORDS = {
    "amazing", "incredible", "insane", "shocking", "unbelievable", "terrifying",
    "furious", "outraged", "devastated", "heartbroken", "thrilled", "ecstatic",
    "hate", "love", "disgusting", "horrible", "fantastic", "brutal", "savage",
    "crazy", "wild", "ridiculous", "exposed", "betrayed", "humiliated",
    "explosive", "scandal", "disaster", "nightmare", "miracle", "stunning",
}

# Question-word sentence openers.
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
REACTION_MARKER_RE = re.compile(
    r"\[(laughter|applause|laughs|cheering|crosstalk)\]", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Settings loaders
# ---------------------------------------------------------------------------

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


def _load_detection_config() -> dict:
    """Return the ``detection:`` block from ``config/settings.yaml`` with safe
    fallbacks for every key.

    Returns a flat dict containing:
        window_min_s, window_max_s, window_step_s,
        top_k, final_clips,
        social_confidence_cap, comment_cluster_window_s,
        boundary_search_s, laughter_extend_s,
        min_clip_score, max_clip_length_s,
        weights (sub-dict with 'default' and 'no_transcript' profiles),
        sponsor_keywords (list[str]).
    """
    defaults: dict[str, Any] = {
        "window_min_s": 20.0,
        "window_max_s": 45.0,
        "window_step_s": 5.0,
        "top_k": 6,
        "final_clips": 3,
        "social_confidence_cap": 0.65,
        "comment_cluster_window_s": 15.0,
        "boundary_search_s": 10.0,
        "laughter_extend_s": 1.5,
        "min_clip_score": 0.45,
        "max_clip_length_s": 60.0,
        "weights": {
            "default": {
                "hook_quality": 0.18,
                "coherence": 0.12,
                "sentiment": 0.20,
                "specificity": 0.15,
                "curiosity": 0.12,
                "audio_energy": 0.08,
                "laughter": 0.08,
                "dramatic_pause": 0.07,
            },
            "no_transcript": {
                "audio_energy": 0.45,
                "laughter": 0.35,
                "dramatic_pause": 0.20,
            },
        },
        "sponsor_keywords": [
            "use code", "sponsored", "check out",
            "ad starts", "promo code", "discount code",
        ],
    }

    try:
        with open(SETTINGS_PATH) as f:
            raw = yaml.safe_load(f)
    except OSError:
        return defaults

    if not raw:
        return defaults

    det = raw.get("detection", {}) or {}
    gen = raw.get("general", {}) or {}

    cfg: dict[str, Any] = {
        "window_min_s": float(det.get("window_min_s", defaults["window_min_s"])),
        "window_max_s": float(det.get("window_max_s", defaults["window_max_s"])),
        "window_step_s": float(det.get("window_step_s", defaults["window_step_s"])),
        "top_k": int(det.get("top_k", defaults["top_k"])),
        "final_clips": int(det.get("final_clips", defaults["final_clips"])),
        "social_confidence_cap": float(det.get("social_confidence_cap", defaults["social_confidence_cap"])),
        "comment_cluster_window_s": float(det.get("comment_cluster_window_s", defaults["comment_cluster_window_s"])),
        "boundary_search_s": float(det.get("boundary_search_s", defaults["boundary_search_s"])),
        "laughter_extend_s": float(det.get("laughter_extend_s", defaults["laughter_extend_s"])),
        "min_clip_score": float(gen.get("min_clip_score", defaults["min_clip_score"])),
        "max_clip_length_s": float(gen.get("max_clip_length_s", defaults["max_clip_length_s"])),
        "sponsor_keywords": det.get("sponsor_keywords", defaults["sponsor_keywords"]),
        "weights": defaults["weights"].copy(),
    }

    # Merge weights if present in config.
    if "weights" in det and isinstance(det["weights"], dict):
        for profile in ("default", "no_transcript"):
            if profile in det["weights"] and isinstance(det["weights"][profile], dict):
                cfg["weights"][profile] = dict(det["weights"][profile])

    return cfg


# ---------------------------------------------------------------------------
# Window generation helpers
# ---------------------------------------------------------------------------

def _build_window_lengths(window_min_s: float, window_max_s: float, window_step_s: float) -> list[float]:
    """Build the list of window lengths from min to max, stepping by step.

    Always includes window_max_s even when the step doesn't land exactly on it.
    """
    lengths: list[float] = []
    length = window_min_s
    while length <= window_max_s + 1e-9:
        lengths.append(length)
        length += window_step_s
    if not lengths or abs(lengths[-1] - window_max_s) > 1e-9:
        lengths.append(window_max_s)
    return lengths


def _segments_in_window(ordered: list[dict], start: float, end: float) -> list[dict]:
    """Return segments that overlap the half-open interval [start, end)."""
    return [
        s for s in ordered
        if (s.get("start") or 0.0) < end and (s.get("end") or 0.0) > start
    ]


# ---------------------------------------------------------------------------
# Content scoring helpers (no imports from signals — those happen lazily)
# ---------------------------------------------------------------------------

def _content_score_transcript(
    window_start: float,
    window_end: float,
    ordered: list[dict],
    audio_signals: Any,
    weights: dict,
) -> float:
    """Compute a content score for a transcript window using v5 signals.

    Lazily imports ``processing.signals`` to avoid the circular import.
    """
    # Lazy import — signals.py imports constants FROM this module at the top level.
    from processing import signals as sig_mod  # noqa: PLC0415

    segs = _segments_in_window(ordered, window_start, window_end)
    window_text = " ".join(s.get("text", "") for s in segs)

    fs = sig_mod.first_sentence(window_text)

    hook = sig_mod.hook_quality(fs)
    coherence = sig_mod.standalone_coherence(fs)
    sent_score = sig_mod.sentiment(window_text)
    spec = sig_mod.specificity(window_text)
    cur = sig_mod.curiosity(window_text)

    if audio_signals is not None:
        audio_energy = audio_signals.energy(window_start, window_end)
        audio_laugh = audio_signals.laughter(window_start, window_end)
        dramatic = audio_signals.dramatic_pause(window_start, window_end)
    else:
        audio_energy = 0.0
        audio_laugh = 0.0
        dramatic = 0.0

    # Text-based laughter marker count.
    text_laugh = min(1.0, len(REACTION_MARKER_RE.findall(window_text)) / 2.0)
    laughter = max(text_laugh, audio_laugh)

    w = weights
    score = (
        w.get("hook_quality", 0.18) * hook
        + w.get("coherence", 0.12) * coherence
        + w.get("sentiment", 0.20) * sent_score
        + w.get("specificity", 0.15) * spec
        + w.get("curiosity", 0.12) * cur
        + w.get("audio_energy", 0.08) * audio_energy
        + w.get("laughter", 0.08) * laughter
        + w.get("dramatic_pause", 0.07) * dramatic
    )
    return max(0.0, min(1.0, score))


def _content_score_no_transcript(
    window_start: float,
    window_end: float,
    audio_signals: Any,
    weights: dict,
) -> float:
    """Compute a content score for the no-transcript path using audio only."""
    audio_energy = audio_signals.energy(window_start, window_end)
    laughter = audio_signals.laughter(window_start, window_end)
    dramatic = audio_signals.dramatic_pause(window_start, window_end)

    w = weights
    score = (
        w.get("audio_energy", 0.45) * audio_energy
        + w.get("laughter", 0.35) * laughter
        + w.get("dramatic_pause", 0.20) * dramatic
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Boundary refinement helpers
# ---------------------------------------------------------------------------

def _refine_boundaries_transcript(
    transcript: list[dict],
    start: float,
    end: float,
    search_s: float,
    laughter_extend_s: float,
) -> tuple[float, float]:
    """Snap start/end to natural sentence boundaries within the transcript.

    Start snapping: prefer a segment whose *previous* segment ends with
    ``.``, ``?``, or ``!`` (i.e. the segment starts a new sentence).  Fall
    back to the segment start closest to *start*.

    End snapping: prefer a segment whose own text ends with ``.``, ``?``,
    or ``!``.  Fall back to the segment end closest to *end*.

    Laughter extension: if a segment beginning within 3 s after the refined
    end matches ``REACTION_MARKER_RE``, extend the end to that segment's end
    plus *laughter_extend_s*.
    """
    if not transcript:
        return start, end

    ordered = sorted(transcript, key=lambda s: s.get("start", 0.0) or 0.0)

    # --- Refine start ---
    candidates_start = [
        s for s in ordered
        if abs((s.get("start") or 0.0) - start) <= search_s
    ]

    new_start = start
    if candidates_start:
        # Prefer a segment that is preceded by a sentence-ending segment.
        sentence_start_segs = []
        for i, seg in enumerate(ordered):
            seg_start = seg.get("start") or 0.0
            if abs(seg_start - start) > search_s:
                continue
            if i == 0:
                # First segment of the whole transcript — always a clean start.
                sentence_start_segs.append(seg)
                continue
            prev_text = (ordered[i - 1].get("text") or "").rstrip()
            if prev_text and prev_text[-1] in ".?!":
                sentence_start_segs.append(seg)

        if sentence_start_segs:
            best = min(sentence_start_segs, key=lambda s: abs((s.get("start") or 0.0) - start))
            new_start = float(best.get("start") or start)
        else:
            best = min(candidates_start, key=lambda s: abs((s.get("start") or 0.0) - start))
            new_start = float(best.get("start") or start)

    # --- Refine end ---
    candidates_end = [
        s for s in ordered
        if abs((s.get("end") or 0.0) - end) <= search_s
    ]

    new_end = end
    if candidates_end:
        sentence_end_segs = [
            s for s in candidates_end
            if (s.get("text") or "").rstrip() and (s.get("text") or "").rstrip()[-1] in ".?!"
        ]
        if sentence_end_segs:
            best = min(sentence_end_segs, key=lambda s: abs((s.get("end") or 0.0) - end))
            new_end = float(best.get("end") or end)
        else:
            best = min(candidates_end, key=lambda s: abs((s.get("end") or 0.0) - end))
            new_end = float(best.get("end") or end)

    # --- Laughter extension ---
    for seg in ordered:
        seg_start = seg.get("start") or 0.0
        seg_end = seg.get("end") or 0.0
        if 0.0 <= seg_start - new_end <= 3.0:
            if REACTION_MARKER_RE.search(seg.get("text") or ""):
                new_end = float(seg_end) + laughter_extend_s
                break  # only extend once

    return new_start, new_end


def _refine_boundaries_audio(
    audio_signals: Any,
    start: float,
    end: float,
    search_s: float,
) -> tuple[float, float]:
    """Snap start/end to nearest silence-gap midpoints within ±search_s."""
    new_start = start
    new_end = end

    # Search window around start edge.
    start_midpoints = audio_signals.silence_boundaries(
        max(0.0, start - search_s), start + search_s
    )
    if start_midpoints:
        best = min(start_midpoints, key=lambda m: abs(m - start))
        new_start = best

    # Search window around end edge.
    end_midpoints = audio_signals.silence_boundaries(
        max(0.0, end - search_s), end + search_s
    )
    if end_midpoints:
        best = min(end_midpoints, key=lambda m: abs(m - end))
        new_end = best

    return new_start, new_end


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def detect(
    transcript: list[dict] | None,
    comments: list[dict] | None = None,
    audio_signals: Any = None,
    max_clips: int | None = None,
) -> list[dict]:
    """Detect the best clips from a video by fusing content, audio, and social signals.

    Args:
        transcript:    List of ``{start, end, text}`` segment dicts (sorted by
                       start time), or ``None`` / ``[]`` for the no-transcript
                       fallback path.
        comments:      Comment dicts from ``comments.fetch_comments``; may be
                       ``[]`` or ``None`` — treated as zero comments.
        audio_signals: ``AudioSignals`` instance, or ``None`` when librosa /
                       audio analysis is unavailable.
        max_clips:     Final clip count; defaults to ``detection.final_clips``
                       (3) when ``None``.

    Returns:
        List of ``{"start_s": float, "end_s": float, "score": float}`` dicts
        sorted by score descending.  May be empty.
    """
    # --- 1. Load config ---
    cfg = _load_detection_config()

    window_min_s: float = cfg["window_min_s"]
    window_max_s: float = cfg["window_max_s"]
    window_step_s: float = cfg["window_step_s"]
    top_k: int = cfg["top_k"]
    final_clips: int = cfg["final_clips"]
    social_confidence_cap: float = cfg["social_confidence_cap"]
    comment_cluster_window_s: float = cfg["comment_cluster_window_s"]
    boundary_search_s: float = cfg["boundary_search_s"]
    laughter_extend_s: float = cfg["laughter_extend_s"]
    min_clip_score: float = cfg["min_clip_score"]
    max_clip_length_s: float = cfg["max_clip_length_s"]
    weights_default: dict = cfg["weights"]["default"]
    weights_no_transcript: dict = cfg["weights"]["no_transcript"]

    effective_max_clips = max_clips if max_clips is not None else final_clips
    if effective_max_clips <= 0:
        return []

    has_transcript = bool(transcript)

    # --- 2. Social signal setup (lazy import) ---
    from processing import comments as comments_mod  # noqa: PLC0415

    safe_comments: list[dict] = comments or []

    clusters = comments_mod.cluster_timestamps(safe_comments, window_s=comment_cluster_window_s)
    ts_count = comments_mod.timestamped_comment_count(safe_comments)
    max_count = max((c["count"] for c in clusters), default=0)

    # Adaptive confidence: 0 timestamps => pure content (social_conf = 0.0).
    if ts_count == 0:
        social_conf = 0.0
    else:
        social_conf = min(social_confidence_cap, 0.2 + (ts_count / 50) * 0.45)

    # --- 3. Generate candidate windows ---
    window_lengths = _build_window_lengths(window_min_s, window_max_s, window_step_s)

    if has_transcript:
        ordered = sorted(transcript, key=lambda s: s.get("start", 0.0) or 0.0)
        timeline_end = max((s.get("end") or 0.0 for s in ordered), default=0.0)
        anchors = [s.get("start") or 0.0 for s in ordered]
    else:
        # No-transcript path requires audio_signals.
        if audio_signals is None:
            return []
        duration = audio_signals.duration_s
        timeline_end = duration
        ordered = []
        anchors = []
        t = 0.0
        while t < duration:
            anchors.append(t)
            t += window_step_s

    candidates: list[dict[str, Any]] = []

    for anchor in anchors:
        for wlen in window_lengths:
            w_start = anchor
            w_end = anchor + wlen
            if w_end > timeline_end + 1e-9:
                continue

            if has_transcript:
                content = _content_score_transcript(
                    w_start, w_end, ordered, audio_signals, weights_default
                )
            else:
                content = _content_score_no_transcript(
                    w_start, w_end, audio_signals, weights_no_transcript
                )

            social = comments_mod.social_score(clusters, w_start, w_end, max_count=max_count)
            fused = social * social_conf + content * (1.0 - social_conf)

            sponsor_mult = comments_mod.sponsor_penalty(
                safe_comments, clusters, w_start, w_end
            )
            fused *= sponsor_mult
            fused = max(0.0, min(1.0, fused))

            if fused >= min_clip_score:
                candidates.append({"start_s": w_start, "end_s": w_end, "score": fused})

    # --- 5. Select up to top_k DISTINCT (non-overlapping) candidate moments.
    # A raw ``candidates[:top_k]`` slice would pick many near-duplicate windows
    # clustered around the single hottest moment (adjacent anchors / lengths all
    # score high); the final dedup would then collapse them to one clip and drop
    # the 2nd/3rd best distinct moments. Greedily picking non-overlapping windows
    # here guarantees top_k distinct moments enter boundary refinement.
    candidates.sort(key=lambda c: (-c["score"], c["start_s"]))
    top_candidates: list[dict[str, Any]] = []
    for cand in candidates:
        if len(top_candidates) >= top_k:
            break
        overlaps = any(
            cand["start_s"] < chosen["end_s"] and cand["end_s"] > chosen["start_s"]
            for chosen in top_candidates
        )
        if not overlaps:
            top_candidates.append(cand)

    # --- 6. Boundary refinement and re-scoring ---
    refined: list[dict[str, Any]] = []

    for cand in top_candidates:
        raw_start = cand["start_s"]
        raw_end = cand["end_s"]

        if has_transcript:
            new_start, new_end = _refine_boundaries_transcript(
                ordered, raw_start, raw_end, boundary_search_s, laughter_extend_s
            )
        else:
            new_start, new_end = _refine_boundaries_audio(
                audio_signals, raw_start, raw_end, boundary_search_s
            )

        # Clamp.
        new_start = max(0.0, new_start)
        new_end = min(timeline_end, new_end)
        if new_end <= new_start:
            new_end = raw_end
            new_start = raw_start
            new_start = max(0.0, new_start)
            new_end = min(timeline_end, new_end)
        if new_end - new_start > max_clip_length_s:
            new_end = new_start + max_clip_length_s

        # Re-score the refined window.
        if has_transcript:
            content_refined = _content_score_transcript(
                new_start, new_end, ordered, audio_signals, weights_default
            )
        else:
            content_refined = _content_score_no_transcript(
                new_start, new_end, audio_signals, weights_no_transcript
            )

        social_refined = comments_mod.social_score(clusters, new_start, new_end, max_count=max_count)
        fused_refined = social_refined * social_conf + content_refined * (1.0 - social_conf)
        sponsor_mult_refined = comments_mod.sponsor_penalty(
            safe_comments, clusters, new_start, new_end
        )
        fused_refined *= sponsor_mult_refined
        fused_refined = max(0.0, min(1.0, fused_refined))

        refined.append({"start_s": new_start, "end_s": new_end, "score": fused_refined})

    # --- 7. Dedup: greedy non-overlapping selection ---
    refined.sort(key=lambda c: (-c["score"], c["start_s"]))
    selected: list[dict[str, Any]] = []

    for cand in refined:
        if len(selected) >= effective_max_clips:
            break
        overlaps = any(
            cand["start_s"] < chosen["end_s"] and cand["end_s"] > chosen["start_s"]
            for chosen in selected
        )
        if not overlaps:
            selected.append(cand)

    # --- 8. Return sorted by score desc ---
    selected.sort(key=lambda c: -c["score"])
    return selected


# ---------------------------------------------------------------------------
# Back-compat wrappers
# ---------------------------------------------------------------------------

def score_window(segments: list[dict], audio_signals: Any = None) -> float:
    """Compute a content-only score for a window of transcript segments.

    Uses the v5 content signals with the ``default`` weight profile.  No
    social signal, no sponsor penalty.

    Args:
        segments:      Transcript segment dicts (``start``, ``end``, ``text``)
                       covering the window to score.
        audio_signals: Optional ``AudioSignals`` instance.

    Returns:
        Float composite score in [0.0, 1.0].
    """
    if not segments:
        return 0.0

    cfg = _load_detection_config()
    weights = cfg["weights"]["default"]

    ordered = sorted(segments, key=lambda s: s.get("start", 0.0) or 0.0)
    w_start = float(ordered[0].get("start") or 0.0)
    w_end = float(ordered[-1].get("end") or w_start)

    return _content_score_transcript(w_start, w_end, ordered, audio_signals, weights)


def detect_clips(transcript: list[dict], max_clips: int = 3) -> list[dict]:
    """Thin back-compat wrapper around ``detect``.

    Equivalent to ``detect(transcript, comments=[], audio_signals=None,
    max_clips=max_clips)``.

    Args:
        transcript: List of segment dicts ``{start, end, text}``.
        max_clips:  Maximum clips to return (default 3).

    Returns:
        List of ``{"start_s": float, "end_s": float, "score": float}`` dicts
        sorted by score descending.
    """
    return detect(transcript, comments=[], audio_signals=None, max_clips=max_clips)
