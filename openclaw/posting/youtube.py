"""YouTube Shorts upload via YouTube Data API v3.

Authenticates using OAuth 2.0 credentials stored in environment variables
(``YOUTUBE_CLIENT_ID``, ``YOUTUBE_CLIENT_SECRET``, ``YOUTUBE_REFRESH_TOKEN``).
Uses resumable upload to handle files up to 50 MB reliably.
"""

from __future__ import annotations


def post_to_youtube(clip: dict) -> str:
    """Upload *clip* as a YouTube Short and return the public video URL.

    Steps:
    1. Build OAuth2 credentials from ``YOUTUBE_CLIENT_ID``,
       ``YOUTUBE_CLIENT_SECRET``, and ``YOUTUBE_REFRESH_TOKEN`` env vars.
    2. Construct video metadata: title (from clip), description, tags,
       ``#Shorts`` hashtag, category 22 (People & Blogs), and
       ``privacyStatus="public"``.
    3. Initiate a resumable upload using ``googleapiclient.http.MediaFileUpload``
       with the final MP4 at ``clip["file_path"]``.
    4. Poll upload status until complete, then return the watch URL.

    Args:
        clip: Clip dict containing at minimum ``clip_id``, ``file_path``,
              and ``title`` fields.

    Returns:
        Public YouTube watch URL for the uploaded Short
        (e.g. ``"https://www.youtube.com/shorts/<video_id>"``).

    Raises:
        RuntimeError: If authentication fails or the upload API returns an
                      unrecoverable error after the configured retry count.
        EnvironmentError: If any required env var is absent.
    """
    raise NotImplementedError
