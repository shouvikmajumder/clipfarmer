"""Video download stage.

Uses yt-dlp to fetch the best-quality MP4 (or remux to MP4) for the source
YouTube video associated with a job. The raw file is written to
``data/jobs/<job_id>/raw/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Root of all job data. Resolved relative to this file so it works regardless
# of the caller's current working directory: datapipeline/processing/downloader.py -> datapipeline/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOBS_DIR = DATA_DIR / "jobs"

# Cap resolution at 1080p — final output is short vertical clips, no need for
# the bandwidth/disk cost of 4K source video.
MAX_HEIGHT = 1080


def _job_raw_dir(job_id: str) -> Path:
    """Return (and ensure exists) the ``raw/`` directory for *job_id*."""
    raw_dir = JOBS_DIR / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _progress_hook(d: dict) -> None:
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "?%").strip()
        speed = d.get("_speed_str", "?/s").strip()
        eta = d.get("_eta_str", "?s").strip()
        print(f"\r[download] {pct} at {speed} ETA {eta}    ", end="", flush=True)
    elif d["status"] == "finished":
        print(f"\r[download] Done — merging to MP4…" + " " * 20, flush=True)


def download(job: dict) -> str:
    """Download the YouTube video for *job* and return the local file path.

    Uses yt-dlp with format selection capped at 1080p, preferring an MP4
    container: ``bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/
    best[height<=1080][ext=mp4]/best[height<=1080]``. The output file is
    placed at ``data/jobs/<job_id>/raw/<video_id>.<ext>``.

    Args:
        job: Job metadata dict as returned by ``state.get_job``. Must contain
             at minimum ``job["id"]`` and ``job["youtube_url"]``.

    Returns:
        Absolute path to the downloaded video file.

    Raises:
        RuntimeError: If yt-dlp exits with a non-zero status or the expected
                      output file is not present after the download completes.
    """
    import yt_dlp

    job_id = job["id"]
    youtube_url = job["youtube_url"]
    raw_dir = _job_raw_dir(job_id)

    out_tmpl = str(raw_dir / "%(id)s.%(ext)s")
    ydl_opts: dict[str, Any] = {
        "format": (
            f"bestvideo[height<={MAX_HEIGHT}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={MAX_HEIGHT}][ext=mp4]/"
            f"best[height<={MAX_HEIGHT}]/best"
        ),
        "outtmpl": out_tmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "progress_hooks": [_progress_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
    except Exception as exc:  # yt_dlp raises yt_dlp.utils.DownloadError, etc.
        raise RuntimeError(f"yt-dlp download failed for {youtube_url!r}: {exc}") from exc

    # merge_output_format means the muxed file ends in .mp4 even if
    # prepare_filename (computed pre-merge) reports a different extension.
    final_path = Path(filename)
    if ydl_opts.get("merge_output_format"):
        final_path = final_path.with_suffix(".mp4")

    if not final_path.exists():
        # Fall back to whatever prepare_filename predicted, in case the merge
        # step didn't run (e.g. a pre-muxed format was selected directly).
        fallback_path = Path(filename)
        if fallback_path.exists():
            final_path = fallback_path
        else:
            raise RuntimeError(
                f"yt-dlp reported success but output file is missing: {final_path}"
            )

    return str(final_path.resolve())
