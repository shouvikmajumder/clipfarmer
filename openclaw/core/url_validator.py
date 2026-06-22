"""YouTube URL validation and video preflight checks.

Runs two layers of validation:
1. Regex/pattern check — fast, no network call.
2. yt-dlp ``extract_info`` preflight — confirms the video is accessible and
   meets the duration and type constraints defined in ``config/settings.yaml``.
"""

from __future__ import annotations


def validate_url(url: str) -> dict:
    """Validate a YouTube URL and return basic video metadata.

    Performs the following checks in order, raising ``ValueError`` on the
    first failure:

    1. The URL must match a recognised YouTube domain
       (``youtube.com/watch?v=``, ``youtu.be/``, ``youtube.com/shorts/``).
    2. ``yt-dlp`` ``extract_info`` (download=False) must succeed — a failure
       here typically means the video is private, deleted, or geo-blocked.
    3. The video must not be a live stream (``is_live`` must be falsy).
    4. The video duration must not exceed the hard limit from settings
       (default 21600 s / 6 hours).

    Args:
        url: The candidate YouTube URL string.

    Returns:
        A dict containing at minimum::

            {
                "id": str,           # YouTube video ID
                "title": str,
                "duration": int,     # seconds
                "channel": str,
                "thumbnail": str,    # best-quality thumbnail URL
            }

    Raises:
        ValueError: With a human-readable message describing exactly which
                    check failed (non-YouTube URL, private video, live stream,
                    or duration exceeded).
    """
    raise NotImplementedError
