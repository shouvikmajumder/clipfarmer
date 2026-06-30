"""Fetch a YouTube transcript via the YouTube Transcript API (no API key required).

Uses ``youtube-transcript-api`` to retrieve auto-generated or manually
uploaded captions for a video directly from YouTube's public endpoints.
No API key is needed — only publicly available caption tracks are accessible.

When this module returns ``None``, the pipeline's audio/comment fallback
path in ``processing.clip_detector`` takes over. Callers must not raise on
a ``None`` return; it is an expected, non-fatal outcome.

Dependencies:
    youtube-transcript-api >= 0.6.2  (lazy-imported inside ``fetch_transcript``)
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Path to settings.yaml relative to this file:
# datapipeline/processing/transcript_fetcher.py -> datapipeline/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_settings() -> dict:
    """Load ``config/settings.yaml`` and return the parsed dict.

    Returns an empty dict on any read/parse error so callers can use
    ``.get()`` safely without further error handling.
    """
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
        return settings or {}
    except OSError:
        return {}


def _get_raw_transcript(api_cls, youtube_id: str) -> list[dict]:
    """Return raw transcript entries, supporting both library API generations.

    ``youtube-transcript-api`` changed its surface in 1.0: the ``get_transcript``
    classmethod was removed in favour of an instance method ``fetch`` that
    returns a ``FetchedTranscript`` (convertible to the legacy list-of-dicts
    shape via ``to_raw_data()``). Because ``requirements.txt`` allows either
    line, try the 1.x instance API first and fall back to the 0.6.x classmethod.

    Each returned entry has at least ``text``, ``start`` and ``duration`` keys.
    """
    # 1.x: instance .fetch(...).to_raw_data()
    fetch = getattr(api_cls, "fetch", None)
    if callable(fetch):
        try:
            fetched = api_cls().fetch(youtube_id)
        except TypeError:
            # Some builds expose fetch as a classmethod taking the id directly.
            fetched = fetch(youtube_id)
        to_raw = getattr(fetched, "to_raw_data", None)
        return to_raw() if callable(to_raw) else list(fetched)

    # 0.6.x: classmethod get_transcript(...)
    return api_cls.get_transcript(youtube_id)


def fetch_transcript(youtube_id: str) -> list[dict] | None:
    """Fetch the transcript for *youtube_id* without using Whisper.

    Retrieves the first available caption track (auto-generated or manual)
    from YouTube's public captions API via ``youtube-transcript-api``.

    Args:
        youtube_id: The bare YouTube video ID (e.g. ``"dQw4w9WgXcQ"``),
                    not a full URL.

    Returns:
        Sorted list of segment dicts shaped::

            {
                "start": float,  # segment start time in seconds
                "end":   float,  # segment end time in seconds (start + duration)
                "text":  str,    # stripped caption text for the segment
            }

        Returns ``None`` (does **not** raise) if the transcript cannot be
        retrieved for any reason: captions disabled, no caption track found,
        video unavailable, network error, or empty response. A ``None``
        return triggers the audio/comment fallback path in the detector.
    """
    # Lazy import: the library is only required when this function runs, so the
    # module still imports cleanly even if youtube-transcript-api is absent.
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import]
    except ImportError:
        logger.warning(
            "youtube-transcript-api is not installed; cannot fetch transcript "
            "for video %r. Returning None to trigger audio/comment fallback.",
            youtube_id,
        )
        return None

    # Import the known exception classes defensively: older and newer versions
    # of the library have exposed them at different paths.  Fall back to plain
    # Exception for any version that doesn't expose them at the expected path.
    try:
        from youtube_transcript_api import (  # type: ignore[import]
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
        _known_errors: tuple[type[Exception], ...] = (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
    except ImportError:
        # Library version doesn't expose these names at the top level; we will
        # catch generic Exception below and inspect the message instead.
        _known_errors = ()

    try:
        raw: list[dict] = _get_raw_transcript(YouTubeTranscriptApi, youtube_id)
    except _known_errors as exc:
        logger.warning(
            "No transcript available for video %r (%s: %s). "
            "Returning None to trigger audio/comment fallback.",
            youtube_id,
            type(exc).__name__,
            exc,
        )
        return None
    except Exception as exc:  # network errors, unexpected API changes, etc.
        logger.warning(
            "Unexpected error fetching transcript for video %r (%s: %s). "
            "Returning None to trigger audio/comment fallback.",
            youtube_id,
            type(exc).__name__,
            exc,
        )
        return None

    if not raw:
        logger.warning(
            "YouTubeTranscriptApi returned an empty list for video %r. "
            "Returning None to trigger audio/comment fallback.",
            youtube_id,
        )
        return None

    segments: list[dict] = []
    for entry in raw:
        start: float = float(entry.get("start", 0.0) or 0.0)
        duration: float = float(entry.get("duration", 0.0) or 0.0)
        text: str = str(entry.get("text", "") or "").strip()
        segments.append(
            {
                "start": start,
                "end": start + duration,
                "text": text,
            }
        )

    segments.sort(key=lambda s: s["start"])

    if not segments:
        logger.warning(
            "All transcript entries for video %r were empty after conversion. "
            "Returning None to trigger audio/comment fallback.",
            youtube_id,
        )
        return None

    return segments
