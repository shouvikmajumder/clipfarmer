"""TikTok video upload via TikTok Content Posting API.

Authenticates with client credentials stored in ``TIKTOK_CLIENT_KEY``,
``TIKTOK_CLIENT_SECRET``, and ``TIKTOK_ACCESS_TOKEN`` env vars.
Uses the ``/v2/post/publish/video/init`` + ``/v2/post/publish/video/upload``
two-phase upload flow.
"""

from __future__ import annotations


def post_to_tiktok(clip: dict) -> str:
    """Upload *clip* to TikTok and return the public video URL.

    Steps:
    1. Read credentials from ``TIKTOK_CLIENT_KEY``, ``TIKTOK_CLIENT_SECRET``,
       ``TIKTOK_ACCESS_TOKEN`` env vars.
    2. Call ``POST /v2/post/publish/video/init`` to obtain an upload URL and
       ``publish_id``.
    3. Upload the MP4 binary (from ``clip["file_path"]``) to the provided
       upload URL via a PUT request.
    4. Poll ``POST /v2/post/publish/status/fetch`` until the video status
       transitions to ``"PUBLISH_COMPLETE"``.
    5. Construct and return the public TikTok share URL.

    Args:
        clip: Clip dict containing at minimum ``clip_id``, ``file_path``,
              and ``title`` fields.

    Returns:
        Public TikTok video URL
        (e.g. ``"https://www.tiktok.com/@<user>/video/<video_id>"``).

    Raises:
        RuntimeError: If the TikTok API returns an error or polling times out.
        EnvironmentError: If any required env var is absent.
    """
    raise NotImplementedError
