"""Video editing stage: crop source footage to 9:16 vertical format.

Uses ffmpeg (via ffmpeg-python) and OpenCV for face/motion detection to
determine the best horizontal crop window. Falls back to a center crop if
smart crop analysis fails or is disabled.
"""

from __future__ import annotations


def edit_clip(job: dict, clip: dict, video_path: str) -> str:
    """Crop and trim *video_path* to produce a 9:16 vertical clip.

    Steps:
    1. Seek to ``clip["start_s"]`` and cut to ``clip["end_s"]``.
    2. Attempt smart crop: use OpenCV face detection on sampled frames to
       find the horizontal region of interest. If a face is detected, centre
       the crop on it.
    3. Fall back to a centre crop if no face is found or smart crop is
       disabled in options.
    4. Output is written to ``data/jobs/<job_id>/clips/<clip_id>/edit.mp4``
       at 1080 x 1920 (or the nearest valid resolution for the source).

    Args:
        job: Job metadata dict (provides ``job["id"]`` and options).
        clip: Clip dict with ``clip_id``, ``start_s``, and ``end_s``.
        video_path: Absolute path to the raw source video.

    Returns:
        Absolute path to the cropped clip file.

    Raises:
        RuntimeError: If ffmpeg or OpenCV processing fails.
    """
    raise NotImplementedError
