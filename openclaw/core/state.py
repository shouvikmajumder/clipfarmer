"""Job state management backed by JSON files on disk.

Each job lives under ``data/jobs/<job_id>/`` and consists of:
- ``job.json``   — top-level job metadata and current state
- ``clips.json`` — list of detected clip dicts
- ``posts.json`` — list of social-media post records

All functions are intentionally side-effect-free with respect to external
services; they only read from and write to the local filesystem.

Storage is plain JSON files, not a database. Every write is performed
atomically: the new content is written to a sibling ``.tmp`` file and then
moved into place with ``os.replace()``, which is atomic on POSIX and Windows.
This guarantees that a crash mid-write can never leave ``job.json`` (or
``clips.json`` / ``posts.json``) truncated or corrupted — readers either see
the old complete file or the new complete file, never a partial one.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Root of all job data. Resolved relative to this file so it works regardless
# of the caller's current working directory: openclaw/core/state.py -> openclaw/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOBS_DIR = DATA_DIR / "jobs"


# ---------------------------------------------------------------------------
# Low-level path / IO helpers
# ---------------------------------------------------------------------------


def _job_dir(job_id: str) -> Path:
    """Return the working directory for *job_id* (does not guarantee it exists)."""
    return JOBS_DIR / job_id


def _job_json_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _clips_json_path(job_id: str) -> Path:
    return _job_dir(job_id) / "clips.json"


def _posts_json_path(job_id: str) -> Path:
    return _job_dir(job_id) / "posts.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* atomically.

    Writes to ``<path>.tmp`` in the same directory first, flushes and fsyncs
    it to disk, then calls ``os.replace()`` to atomically move it into place.
    Because the temp file lives in the same directory as the target, the
    replace is guaranteed atomic on the same filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _now() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Job operations
# ---------------------------------------------------------------------------


def enqueue_job(youtube_url: str, options: dict | None = None) -> str:
    """Create a new job directory, write an initial job.json, and return the job ID.

    The generated job ID is a UUID4 string. ``job.json`` is written with
    ``state="queued"``, the supplied URL, and any caller-provided options.

    Args:
        youtube_url: The fully-qualified YouTube video URL to process.
        options: Optional dict of per-job overrides (e.g. max_clips, platforms).

    Returns:
        The new job's UUID string.
    """
    job_id = str(uuid.uuid4())
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "raw").mkdir(parents=True, exist_ok=True)
    (job_dir / "clips").mkdir(parents=True, exist_ok=True)

    job = {
        "id": job_id,
        "youtube_url": youtube_url,
        "youtube_id": None,
        "video_title": None,
        "video_duration_s": None,
        "submitted_at": _now(),
        "state": "queued",
        "current_stage": None,
        "last_stage_completed": None,
        "error_message": None,
        "retry_count": 0,
        "options": options or {},
    }
    _atomic_write_json(_job_json_path(job_id), job)
    return job_id


def get_next_queued_job() -> dict | None:
    """Scan all job directories for the oldest job with ``state=="queued"``.

    Jobs are ordered by their ``submitted_at`` ISO timestamp. Returns ``None``
    when no queued jobs exist.

    Returns:
        The parsed contents of the queued job's ``job.json``, or ``None``.
    """
    if not JOBS_DIR.exists():
        return None

    queued_jobs: list[dict] = []
    for job_json_path in JOBS_DIR.glob("*/job.json"):
        try:
            job = _read_json(job_json_path)
        except (json.JSONDecodeError, OSError):
            continue
        if job.get("state") == "queued":
            queued_jobs.append(job)

    if not queued_jobs:
        return None

    queued_jobs.sort(key=lambda j: j.get("submitted_at") or "")
    return queued_jobs[0]


def get_job(job_id: str) -> dict:
    """Read and return a job's ``job.json`` as a dict.

    Args:
        job_id: The UUID of the job to retrieve.

    Returns:
        Parsed job metadata dict.

    Raises:
        FileNotFoundError: If the job directory or ``job.json`` does not exist.
    """
    path = _job_json_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    return _read_json(path)


def _update_job(job_id: str, **fields: Any) -> dict:
    """Read job.json, apply *fields*, write it back atomically, return the new dict."""
    job = get_job(job_id)
    job.update(fields)
    _atomic_write_json(_job_json_path(job_id), job)
    return job


def update_job_stage(job_id: str, stage: str) -> None:
    """Set the ``current_stage`` field in ``job.json`` to *stage*.

    Called at the beginning of each processing stage so that the job record
    reflects what is actively happening. Also sets ``state`` to *stage* to
    keep the top-level state machine in sync with the active stage.

    Args:
        job_id: The UUID of the job to update.
        stage: The name of the stage now starting (e.g. ``"downloading"``).
    """
    _update_job(job_id, current_stage=stage, state=stage)


def update_job_metadata(job_id: str, **fields: Any) -> None:
    """Merge arbitrary metadata fields into ``job.json``.

    Used by ``core.job_runner`` after pre-flight URL validation to persist
    ``youtube_id``, ``video_title``, and ``video_duration_s`` onto the job
    record. Follows the same atomic read-modify-write pattern as the other
    job mutators in this module.

    Args:
        job_id: The UUID of the job to update.
        **fields: Arbitrary key/value pairs to merge into the job dict.
    """
    _update_job(job_id, **fields)


def mark_stage_complete(job_id: str, stage: str) -> None:
    """Record *stage* as the most recently completed stage in ``job.json``.

    Used by ``JobRunner.process`` to track resume position so that a crashed
    job can restart from where it left off.

    Args:
        job_id: The UUID of the job.
        stage: The name of the stage that just finished.
    """
    _update_job(job_id, last_stage_completed=stage)


def mark_job_complete(job_id: str) -> None:
    """Set ``state="complete"`` and record a ``completed_at`` timestamp.

    Args:
        job_id: The UUID of the job to mark complete.
    """
    _update_job(job_id, state="complete", current_stage=None, completed_at=_now())


def mark_job_failed(job_id: str, error: str) -> None:
    """Set ``state="failed"``, record ``error_message``, and timestamp.

    Args:
        job_id: The UUID of the job to mark as failed.
        error: Human-readable description of the failure reason.
    """
    _update_job(job_id, state="failed", error_message=error, failed_at=_now())


def mark_job_cancelled(job_id: str, reason: str) -> None:
    """Set ``state="cancelled"``, record the reason, and timestamp.

    Used for jobs rejected during pre-flight (bad URL, private/live video,
    duration over the hard limit) — no retries are attempted.

    Args:
        job_id: The UUID of the job to cancel.
        reason: Human-readable explanation of why the job was cancelled.
    """
    _update_job(job_id, state="cancelled", error_message=reason, cancelled_at=_now())


# ---------------------------------------------------------------------------
# Clip operations
# ---------------------------------------------------------------------------


def get_clips(job_id: str) -> list[dict]:
    """Read and return the list of clips from ``data/jobs/<job_id>/clips.json``.

    Args:
        job_id: The UUID of the job.

    Returns:
        List of clip dicts. Returns an empty list if ``clips.json`` does not
        yet exist for the job.
    """
    path = _clips_json_path(job_id)
    if not path.exists():
        return []
    return _read_json(path)


# Alias matching the brief's requested name.
get_clips_for_job = get_clips


def save_clips(job_id: str, clips: list[dict]) -> None:
    """Write the list of detected clips to ``data/jobs/<job_id>/clips.json``.

    Overwrites any existing ``clips.json`` for the job.

    Args:
        job_id: The UUID of the job.
        clips: List of clip dicts, each containing at minimum
               ``{id, job_id, start_s, end_s, score}``.
    """
    _atomic_write_json(_clips_json_path(job_id), clips)


def add_clip(job_id: str, clip: dict) -> str:
    """Append a single clip record to ``data/jobs/<job_id>/clips.json``.

    Generates a UUID for the clip if it does not already define ``id``, and
    stamps ``job_id`` and ``created_at`` onto the record (storing ``job_id``
    on each clip lets ``create_post_record`` resolve a clip's parent job
    without a separate index).

    Args:
        job_id: The UUID of the parent job.
        clip: Clip dict (e.g. ``{file_path, score, start_s, end_s,
              transcript_snippet, status}``).

    Returns:
        The clip's UUID string.
    """
    clip = dict(clip)
    clip_id = clip.get("id") or str(uuid.uuid4())
    clip["id"] = clip_id
    clip["job_id"] = job_id
    clip.setdefault("status", "ready")
    clip.setdefault("created_at", _now())

    clips = get_clips(job_id)
    clips.append(clip)
    save_clips(job_id, clips)
    return clip_id


def _find_job_id_for_clip(clip_id: str) -> str:
    """Search every job's clips.json for *clip_id* and return its job_id.

    Args:
        clip_id: The UUID of the clip to locate.

    Returns:
        The owning job's UUID string.

    Raises:
        FileNotFoundError: If no clip with that ID exists in any job.
    """
    if not JOBS_DIR.exists():
        raise FileNotFoundError(f"No such clip: {clip_id}")

    for clips_json_path in JOBS_DIR.glob("*/clips.json"):
        try:
            clips = _read_json(clips_json_path)
        except (json.JSONDecodeError, OSError):
            continue
        for clip in clips:
            if clip.get("id") == clip_id:
                return clip.get("job_id") or clips_json_path.parent.name

    raise FileNotFoundError(f"No such clip: {clip_id}")


# ---------------------------------------------------------------------------
# Post operations
# ---------------------------------------------------------------------------


def get_posts(job_id: str) -> list[dict]:
    """Read and return the list of post records from ``posts.json``.

    Args:
        job_id: The UUID of the job.

    Returns:
        List of post record dicts. Returns an empty list if ``posts.json``
        does not yet exist.
    """
    path = _posts_json_path(job_id)
    if not path.exists():
        return []
    return _read_json(path)


def _save_posts(job_id: str, posts: list[dict]) -> None:
    _atomic_write_json(_posts_json_path(job_id), posts)


def create_post_record(clip_id: str, platform: str) -> str:
    """Append a new post record to the owning job's ``posts.json``.

    Resolves the parent job by looking up ``clip_id`` across all jobs' clip
    records (each clip stores its own ``job_id``, written by ``add_clip``),
    so ``posts.json`` always lives alongside the job that owns the clip.

    Args:
        clip_id: The ID of the clip being posted.
        platform: Target platform name (e.g. ``"youtube"``, ``"tiktok"``).

    Returns:
        The new post's UUID string.

    Raises:
        FileNotFoundError: If no clip with that ID exists in any job.
    """
    job_id = _find_job_id_for_clip(clip_id)
    post_id = str(uuid.uuid4())
    post = {
        "id": post_id,
        "clip_id": clip_id,
        "platform": platform,
        "status": "queued",
        "posted_at": None,
        "post_url": None,
        "error": None,
        "retry_count": 0,
    }
    posts = get_posts(job_id)
    posts.append(post)
    _save_posts(job_id, posts)
    return post_id


def _update_post(job_id: str, post_id: str, **fields: Any) -> None:
    posts = get_posts(job_id)
    for post in posts:
        if post.get("id") == post_id:
            post.update(fields)
            break
    else:
        raise FileNotFoundError(f"No such post: {post_id} (job {job_id})")
    _save_posts(job_id, posts)


def mark_post_success(post_id: str, post_url: str) -> None:
    """Update a post record to ``status="success"``.

    Also records the returned public ``post_url`` and a ``posted_at``
    timestamp. The owning job is resolved via the post's ``clip_id``.

    Args:
        post_id: The UUID of the post record to update.
        post_url: The public URL of the uploaded post.

    Raises:
        FileNotFoundError: If no post with that ID exists in any job.
    """
    job_id = _find_job_id_for_post(post_id)
    _update_post(job_id, post_id, status="success", post_url=post_url, posted_at=_now())


def mark_post_failed(post_id: str, error: str) -> None:
    """Update a post record to ``status="failed"``.

    Also records the ``error`` message. The owning job is resolved via the
    post's ``clip_id``.

    Args:
        post_id: The UUID of the post record to update.
        error: Human-readable description of the failure.

    Raises:
        FileNotFoundError: If no post with that ID exists in any job.
    """
    job_id = _find_job_id_for_post(post_id)
    _update_post(job_id, post_id, status="failed", error=error)


def _find_job_id_for_post(post_id: str) -> str:
    """Search every job's posts.json for *post_id* and return its job_id."""
    if not JOBS_DIR.exists():
        raise FileNotFoundError(f"No such post: {post_id}")

    for posts_json_path in JOBS_DIR.glob("*/posts.json"):
        try:
            posts = _read_json(posts_json_path)
        except (json.JSONDecodeError, OSError):
            continue
        for post in posts:
            if post.get("id") == post_id:
                return posts_json_path.parent.name

    raise FileNotFoundError(f"No such post: {post_id}")
