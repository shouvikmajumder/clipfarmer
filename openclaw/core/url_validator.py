"""YouTube URL validation and video preflight checks.

Runs two layers of validation:
1. Regex/pattern check — fast, no network call.
2. yt-dlp ``extract_info`` preflight — confirms the video is accessible and
   meets the duration and type constraints defined in ``config/settings.yaml``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# Path to settings.yaml relative to this file: openclaw/core/url_validator.py -> openclaw/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# Matches youtube.com/watch?v=, youtube.com/shorts/, youtu.be/, m.youtube.com, etc.
YOUTUBE_URL_RE = re.compile(
    r"^https?://"
    r"(www\.|m\.)?"
    r"(youtube\.com/(watch\?v=|shorts/|embed/)|youtu\.be/)",
    re.IGNORECASE,
)


class InvalidURLError(ValueError):
    """Raised when a URL is not a recognised YouTube video URL."""


class VideoUnavailableError(ValueError):
    """Raised when yt-dlp cannot access the video (private, deleted, geo-blocked)."""


class LiveStreamNotSupportedError(ValueError):
    """Raised when the target video is a live stream (not yet ended)."""


class VideoTooLongError(ValueError):
    """Raised when the video duration exceeds the configured hard limit."""


def _load_settings() -> dict[str, Any]:
    """Load and return the ``general`` section of ``config/settings.yaml``."""
    with open(SETTINGS_PATH) as f:
        settings = yaml.safe_load(f)
    return settings.get("general", {}) if settings else {}


def _is_youtube_url(url: str) -> bool:
    """Return True if *url* matches a recognised YouTube domain/path pattern."""
    return bool(YOUTUBE_URL_RE.match(url.strip()))


def validate_url(url: str) -> dict:
    """Validate a YouTube URL and return basic video metadata.

    Performs the following checks in order, raising a domain-specific
    exception (all subclasses of ``ValueError``) on the first failure:

    1. The URL must match a recognised YouTube domain
       (``youtube.com/watch?v=``, ``youtu.be/``, ``youtube.com/shorts/``).
       Raises ``InvalidURLError`` otherwise.
    2. ``yt-dlp`` ``extract_info`` (download=False) must succeed — a failure
       here typically means the video is private, deleted, or geo-blocked.
       Raises ``VideoUnavailableError`` otherwise.
    3. The video must not be a live stream (``is_live`` must be falsy).
       Raises ``LiveStreamNotSupportedError`` otherwise.
    4. The video duration must not exceed the hard limit from settings
       (default 21600 s / 6 hours). Raises ``VideoTooLongError`` otherwise.

    Args:
        url: The candidate YouTube URL string.

    Returns:
        A dict containing at minimum::

            {
                "youtube_id": str,
                "video_title": str,
                "video_duration_s": int,
            }

    Raises:
        InvalidURLError: The URL is not a recognised YouTube URL.
        VideoUnavailableError: yt-dlp could not extract video info (private,
            deleted, geo-blocked, or any other access failure).
        LiveStreamNotSupportedError: The video is an active live stream.
        VideoTooLongError: The video's duration exceeds the configured hard
            limit (``general.max_video_duration_hard_limit_s``).
    """
    if not url or not _is_youtube_url(url):
        raise InvalidURLError(f"Not a recognised YouTube URL: {url!r}")

    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt_dlp raises yt_dlp.utils.DownloadError, etc.
        raise VideoUnavailableError(
            f"Could not access video at {url!r}: {exc}"
        ) from exc

    if info is None:
        raise VideoUnavailableError(f"Could not access video at {url!r}: no info returned")

    if info.get("is_live"):
        raise LiveStreamNotSupportedError(
            f"Live streams are not supported: {url!r}"
        )

    settings = _load_settings()
    hard_limit_s = settings.get("max_video_duration_hard_limit_s", 21600)

    duration_s = info.get("duration") or 0
    if duration_s > hard_limit_s:
        raise VideoTooLongError(
            f"Video duration {duration_s}s exceeds hard limit of {hard_limit_s}s: {url!r}"
        )

    return {
        "youtube_id": info.get("id"),
        "video_title": info.get("title"),
        "video_duration_s": duration_s,
    }
