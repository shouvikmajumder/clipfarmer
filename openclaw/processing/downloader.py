"""Video download stage.

Uses yt-dlp to fetch the best-quality MP4 (or remux to MP4) for the source
YouTube video associated with a job. The raw file is written to
``data/jobs/<job_id>/raw/``.
"""

from __future__ import annotations


def download(job: dict) -> str:
    """Download the YouTube video for *job* and return the local file path.

    Uses yt-dlp with format selection ``bestvideo[ext=mp4]+bestaudio[ext=m4a]/
    best[ext=mp4]/best`` to prefer an MP4 container. The output file is
    placed at ``data/jobs/<job_id>/raw/<video_id>.mp4``.

    Args:
        job: Job metadata dict as returned by ``state.get_job``. Must contain
             at minimum ``job["id"]`` and ``job["youtube_url"]``.

    Returns:
        Absolute path to the downloaded video file.

    Raises:
        RuntimeError: If yt-dlp exits with a non-zero status or the expected
                      output file is not present after the download completes.
    """
    raise NotImplementedError
