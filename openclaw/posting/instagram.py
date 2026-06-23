"""Instagram Reels upload via Instagram Graph API.

Authenticates using ``INSTAGRAM_ACCESS_TOKEN`` and ``INSTAGRAM_ACCOUNT_ID``
env vars. Uses the two-step container-create + publish flow required by the
Graph API for video content.
"""

from __future__ import annotations


def post_to_instagram(clip: dict) -> str:
    """Upload *clip* as an Instagram Reel and return the public media URL.

    Steps:
    1. Read credentials from ``INSTAGRAM_ACCESS_TOKEN`` and
       ``INSTAGRAM_ACCOUNT_ID`` env vars.
    2. The video must be hosted at a publicly accessible URL; if
       ``clip["cdn_url"]`` is not set, upload to a temporary CDN first
       (implementation detail left for Phase B).
    3. Call ``POST /<account_id>/media`` with ``media_type=REELS``,
       ``video_url``, ``caption``, and ``share_to_feed=true`` to create a
       media container. Receive ``container_id``.
    4. Poll ``GET /<container_id>?fields=status_code`` until
       ``status_code == "FINISHED"``.
    5. Call ``POST /<account_id>/media_publish`` with the ``container_id``
       to publish the Reel.
    6. Retrieve and return the permalink from the published media object.

    Args:
        clip: Clip dict containing at minimum ``clip_id``, ``file_path``,
              ``cdn_url`` (optional), and ``title`` fields.

    Returns:
        Public Instagram permalink URL
        (e.g. ``"https://www.instagram.com/reel/<shortcode>/"``).

    Raises:
        RuntimeError: If any Graph API call fails or container polling times out.
        EnvironmentError: If any required env var is absent.
    """
    raise NotImplementedError
