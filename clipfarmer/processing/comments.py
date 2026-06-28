"""Fetch and analyse YouTube comments for the clip-detection social signal.

Retrieves top-level comments from the YouTube Data API v3 ``commentThreads``
endpoint, extracts timestamp mentions embedded in comment text, and exposes
clustering and scoring helpers consumed by the clip detector's social-fusion
layer.

This module is **best-effort**: every network call wraps itself in try/except
and returns a graceful empty result rather than raising.  The detector treats
zero comments as a pure content score, so failures here degrade gracefully
without blocking a job.

``social_confidence_cap`` and final fusion arithmetic live in
``processing.clip_detector``, not here — this module only computes per-window
social and sponsor-penalty values.

Dependencies:
    google-api-python-client >= 2.140.0  (lazy-imported inside ``fetch_comments``)
    YOUTUBE_API_KEY env var               (or pass ``api_key`` directly)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

# Path to settings.yaml relative to this file:
# clipfarmer/processing/comments.py -> clipfarmer/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# Default values mirroring detection: block in settings.yaml
_DEFAULT_CLUSTER_WINDOW_S: float = 15.0
_DEFAULT_SPONSOR_KEYWORDS: list[str] = [
    "use code",
    "sponsored",
    "check out",
    "ad starts",
    "promo code",
    "discount code",
]

# Regex for mm:ss and h:mm:ss timestamp patterns.
# Captures: group(1) = optional hours, group(2) = minutes, group(3) = seconds.
_TIMESTAMP_RE = re.compile(
    r"\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b"
)


def _load_detection_settings() -> dict:
    """Load the ``detection:`` block from ``config/settings.yaml``.

    Returns an empty dict on any read/parse error so callers can call
    ``.get()`` safely without additional error handling.
    """
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
        return (settings or {}).get("detection", {})
    except OSError:
        return {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_timestamps(text: str) -> list[int]:
    """Extract all timestamp mentions from a comment string.

    Recognises ``mm:ss`` and ``h:mm:ss`` formats.  Only timestamps where
    minutes **and** seconds are each strictly less than 60 are kept (values
    produced by the regex already enforce this via the ``[0-5]?\\d``
    character-class, but we validate explicitly for safety).

    Args:
        text: Raw comment text, possibly containing multiple timestamps.

    Returns:
        List of integer seconds, in the order they appear in *text*.  Empty
        list if no valid timestamps are found.

    Examples::

        >>> parse_timestamps("great moment at 1:23 and also 2:05")
        [83, 125]
        >>> parse_timestamps("see 1:02:45 for the highlight")
        [3765]
    """
    results: list[int] = []
    for match in _TIMESTAMP_RE.finditer(text):
        hours_str, minutes_str, seconds_str = match.group(1), match.group(2), match.group(3)

        hours = int(hours_str) if hours_str is not None else 0
        minutes = int(minutes_str)
        seconds = int(seconds_str)

        # Belt-and-suspenders: regex already restricts minutes/seconds to
        # [0-5]?\d, which allows 0-59, but we re-validate explicitly.
        if minutes >= 60 or seconds >= 60:
            continue

        results.append(hours * 3600 + minutes * 60 + seconds)
    return results


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_comments(
    youtube_id: str,
    api_key: Optional[str] = None,
    max_comments: int = 500,
) -> list[dict]:
    """Fetch top-level comments for *youtube_id* from the YouTube Data API v3.

    Uses the ``commentThreads.list`` endpoint ordered by relevance.  Pages
    through results until *max_comments* entries have been collected or no
    ``nextPageToken`` is returned.

    Args:
        youtube_id: Bare YouTube video ID (e.g. ``"dQw4w9WgXcQ"``).
        api_key:    YouTube Data API v3 server key.  If ``None``, the key is
                    read from the ``YOUTUBE_API_KEY`` environment variable.
                    If no key is available at all, logs a warning and returns
                    ``[]`` — the detector falls back to pure content scoring.
        max_comments: Upper bound on comment count to retrieve (default 500).

    Returns:
        List of comment dicts shaped::

            {
                "text":        str,       # raw comment text
                "like_count":  int,       # number of likes on the comment
                "timestamps":  list[int], # seconds extracted by parse_timestamps
            }

        All comments are included, whether or not they contain timestamps.
        On any API error (quota exceeded, comments disabled, network failure)
        returns whatever comments were collected up to that point — or ``[]``
        on immediate failure.  Never raises.
    """
    # Resolve API key
    resolved_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not resolved_key:
        logger.warning(
            "No YOUTUBE_API_KEY available; skipping comment fetch for video %r. "
            "The detector will use pure content scoring.",
            youtube_id,
        )
        return []

    # Lazy import: google-api-python-client is only required when this function
    # runs, keeping the module importable even if the library is absent.
    try:
        from googleapiclient.discovery import build  # type: ignore[import]
    except ImportError:
        logger.warning(
            "google-api-python-client is not installed; cannot fetch comments "
            "for video %r. Returning [] to use pure content scoring.",
            youtube_id,
        )
        return []

    comments: list[dict] = []
    page_token: Optional[str] = None

    try:
        service = build("youtube", "v3", developerKey=resolved_key)

        while len(comments) < max_comments:
            remaining = max_comments - len(comments)
            # API max per page is 100.
            page_size = min(100, remaining)

            kwargs: dict = {
                "part": "snippet",
                "videoId": youtube_id,
                "maxResults": page_size,
                "order": "relevance",
            }
            if page_token:
                kwargs["pageToken"] = page_token

            response = service.commentThreads().list(**kwargs).execute()

            items = response.get("items") or []
            for item in items:
                top_comment = (
                    item.get("snippet", {})
                    .get("topLevelComment", {})
                    .get("snippet", {})
                )
                text: str = str(top_comment.get("textDisplay") or "")
                like_count: int = int(top_comment.get("likeCount") or 0)
                timestamps: list[int] = parse_timestamps(text)
                comments.append(
                    {
                        "text": text,
                        "like_count": like_count,
                        "timestamps": timestamps,
                    }
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    except Exception as exc:  # quota exceeded, comments disabled, network, etc.
        logger.warning(
            "Error fetching comments for video %r after collecting %d comment(s) "
            "(%s: %s). Returning partial results.",
            youtube_id,
            len(comments),
            type(exc).__name__,
            exc,
        )

    return comments


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_timestamps(
    comments: list[dict],
    window_s: float = _DEFAULT_CLUSTER_WINDOW_S,
) -> list[dict]:
    """Cluster timestamp mentions from *comments* into time-contiguous groups.

    Flattens all timestamps from all comments into a single sorted sequence,
    then performs a greedy sliding-window merge: a new cluster is started
    whenever the next timestamp exceeds ``window_s`` seconds beyond the
    first timestamp of the current cluster.

    This simple O(n log n) approach is sufficient for typical comment volumes.
    DBSCAN would offer better density sensitivity but is deferred until videos
    consistently produce >50 timestamps per hour; at that density the greedy
    merge produces nearly identical clusters anyway.

    Args:
        comments: List of comment dicts as returned by ``fetch_comments``.
        window_s: Maximum span (in seconds) of a single cluster.  Read from
                  ``detection.comment_cluster_window_s`` in settings by the
                  callers that want the configured default; this parameter
                  accepts an explicit override.

    Returns:
        List of cluster dicts sorted by ``count`` descending::

            {
                "center_s": float,  # mean timestamp across the cluster
                "count":    int,    # number of timestamps in the cluster
                "start_s":  float,  # earliest timestamp in the cluster
                "end_s":    float,  # latest timestamp in the cluster
            }

        Returns ``[]`` if no timestamps were found across all comments.
    """
    # Flatten all timestamps from every comment.
    all_ts: list[int] = []
    for comment in comments:
        all_ts.extend(comment.get("timestamps") or [])

    if not all_ts:
        return []

    all_ts.sort()

    clusters: list[dict] = []
    cluster_start: float = float(all_ts[0])
    current: list[float] = [float(all_ts[0])]

    for ts in all_ts[1:]:
        ts_f = float(ts)
        if ts_f - cluster_start <= window_s:
            current.append(ts_f)
        else:
            # Finalise current cluster and start a new one.
            clusters.append(
                {
                    "center_s": sum(current) / len(current),
                    "count": len(current),
                    "start_s": current[0],
                    "end_s": current[-1],
                }
            )
            cluster_start = ts_f
            current = [ts_f]

    # Finalise the last open cluster.
    clusters.append(
        {
            "center_s": sum(current) / len(current),
            "count": len(current),
            "start_s": current[0],
            "end_s": current[-1],
        }
    )

    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def social_score(
    clusters: list[dict],
    window_start_s: float,
    window_end_s: float,
    max_count: Optional[int] = None,
) -> float:
    """Return a social engagement score in [0, 1] for a candidate window.

    Finds the strongest timestamp cluster whose ``center_s`` falls within
    ``[window_start_s, window_end_s]`` and normalises its ``count`` by the
    busiest cluster across all clusters.

    Args:
        clusters:       List of cluster dicts as returned by
                        ``cluster_timestamps``.
        window_start_s: Inclusive start of the candidate window in seconds.
        window_end_s:   Inclusive end of the candidate window in seconds.
        max_count:      Normalisation denominator.  If ``None``, derived as
                        the maximum ``count`` across *clusters*.

    Returns:
        Float in [0.0, 1.0].  Returns ``0.0`` if no cluster center falls
        within the window or if *clusters* is empty.
    """
    if not clusters:
        return 0.0

    if max_count is None:
        max_count = max(c["count"] for c in clusters)

    if max_count <= 0:
        return 0.0

    best_count = 0
    for cluster in clusters:
        if window_start_s <= cluster["center_s"] <= window_end_s:
            if cluster["count"] > best_count:
                best_count = cluster["count"]

    return min(1.0, best_count / max_count)


def sponsor_penalty(
    comments: list[dict],
    clusters: list[dict],
    window_start_s: float,
    window_end_s: float,
    sponsor_keywords: Optional[List[str]] = None,
    proximity_s: float = 5.0,
) -> float:
    """Return a sponsor-penalty multiplier in [0, 1] for a candidate window.

    Guards against emoji-dense sponsor-mocking comments scoring high: if
    more than 30% of timestamped comments near the window contain a sponsor
    keyword, the window receives a reduced multiplier (0.3).

    ``1.0`` means no penalty; lower values penalise the window's final score.

    Args:
        comments:        Comment dicts from ``fetch_comments``.
        clusters:        Cluster dicts from ``cluster_timestamps`` (currently
                         unused but kept for future proximity-cluster logic).
        window_start_s:  Start of the candidate window in seconds.
        window_end_s:    End of the candidate window in seconds.
        sponsor_keywords: Keywords that indicate a sponsor segment
                         (case-insensitive substring match).  If ``None``,
                         loaded from ``detection.sponsor_keywords`` in
                         ``config/settings.yaml``.
        proximity_s:     How far outside the window boundary (in seconds) a
                         timestamp must be to be excluded.  Defaults to 5 s.

    Returns:
        ``1.0`` if fewer than 30% of nearby timestamped comments mention
        a sponsor keyword, or if there are no timestamped comments near the
        window.  Returns ``0.3`` otherwise.
    """
    if sponsor_keywords is None:
        detection = _load_detection_settings()
        sponsor_keywords = detection.get("sponsor_keywords", _DEFAULT_SPONSOR_KEYWORDS)

    # Normalise keywords once.
    lower_keywords = [kw.lower() for kw in (sponsor_keywords or [])]
    if not lower_keywords:
        return 1.0

    # Determine the search band around the window.
    search_start = window_start_s - proximity_s
    search_end = window_end_s + proximity_s

    # Collect comments that have at least one timestamp within the search band.
    nearby_comments: list[dict] = []
    for comment in comments:
        timestamps = comment.get("timestamps") or []
        if any(search_start <= ts <= search_end for ts in timestamps):
            nearby_comments.append(comment)

    if not nearby_comments:
        return 1.0

    sponsor_count = 0
    for comment in nearby_comments:
        text_lower = comment.get("text", "").lower()
        if any(kw in text_lower for kw in lower_keywords):
            sponsor_count += 1

    ratio = sponsor_count / len(nearby_comments)
    if ratio > 0.30:
        return 0.3

    return 1.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def timestamped_comment_count(comments: list[dict]) -> int:
    """Return the total number of timestamp mentions across all *comments*.

    Used by the detector's adaptive fusion logic to determine how much
    weight to assign the social signal: a video with many timestamped
    comments warrants higher social trust than one with few or none.

    Args:
        comments: Comment dicts as returned by ``fetch_comments``.

    Returns:
        Integer total of all timestamps found across all comment texts.
    """
    return sum(len(comment.get("timestamps") or []) for comment in comments)
