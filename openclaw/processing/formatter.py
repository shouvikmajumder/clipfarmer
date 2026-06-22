"""Final formatting stage: re-encode for platform delivery constraints.

Enforces the two hard limits that apply to all short-form platforms:
- Maximum 60 seconds duration (trims from the end if needed).
- Maximum 50 MB file size (two-pass bitrate targeting if needed).

Output is a production-ready H.264/AAC MP4 at 1080x1920 with sensible
audio normalisation and fast-start moov atom for streaming.
"""

from __future__ import annotations


def format_clip(job: dict, clip: dict, video_path: str) -> str:
    """Final encode *video_path* to a platform-ready MP4 and return the path.

    Steps:
    1. Probe the clip duration; if > 60 s, trim to exactly 60 s from the
       start (preserving the highest-scoring portion per clip metadata).
    2. Perform a two-pass H.264 encode targeting a bitrate that keeps the
       output under 50 MB.
    3. Normalise audio to -14 LUFS (standard for mobile/social platforms).
    4. Set the ``moov`` atom to the start of the file (``-movflags faststart``).
    5. Write output to ``data/jobs/<job_id>/clips/<clip_id>/final.mp4``.

    Args:
        job: Job metadata dict.
        clip: Clip dict with ``clip_id``.
        video_path: Absolute path to the captioned clip file.

    Returns:
        Absolute path to the final encoded clip file.

    Raises:
        RuntimeError: If ffmpeg processing fails or the output exceeds 50 MB
                      after bitrate targeting (e.g. source is too high quality
                      to compress sufficiently).
    """
    raise NotImplementedError
