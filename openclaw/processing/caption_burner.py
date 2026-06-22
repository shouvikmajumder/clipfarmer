"""Caption burning stage: overlay styled subtitles onto a clip.

Filters the master transcript *segments* to those overlapping the clip's
time window, converts them to an ASS/SRT subtitle file, then uses ffmpeg's
``subtitles`` filter to burn them into the video. Captions are styled for
short-form mobile viewing (large font, centred, bottom third of frame).
"""

from __future__ import annotations


def burn_captions(
    job: dict,
    clip: dict,
    video_path: str,
    segments: list[dict],
) -> str:
    """Burn word-level captions into *video_path* and return the output path.

    Steps:
    1. Filter *segments* to those whose ``start``/``end`` overlap the clip
       window (``clip["start_s"]`` to ``clip["end_s"]``).
    2. Re-base timestamps relative to the clip start.
    3. Render segments to a temporary ``.ass`` subtitle file with the
       OpenClaw caption style (bold, drop-shadow, centred).
    4. Use ffmpeg ``-vf subtitles=<ass_file>`` to burn them in.
    5. Write output to ``data/jobs/<job_id>/clips/<clip_id>/captioned.mp4``.

    Args:
        job: Job metadata dict.
        clip: Clip dict with ``clip_id``, ``start_s``, ``end_s``.
        video_path: Absolute path to the edited (9:16 cropped) clip.
        segments: Full list of transcript segment dicts for the job.

    Returns:
        Absolute path to the captioned clip file.

    Raises:
        RuntimeError: If subtitle generation or ffmpeg processing fails.
    """
    raise NotImplementedError
