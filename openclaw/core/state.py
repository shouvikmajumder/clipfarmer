"""Job state management backed by JSON files on disk.

Each job lives under ``data/jobs/<job_id>/`` and consists of:
- ``job.json``   — top-level job metadata and current state
- ``clips.json`` — list of detected clip dicts
- ``posts.json`` — list of social-media post records

All functions are intentionally side-effect-free with respect to external
services; they only read from and write to the local filesystem.
"""

from __future__ import annotations


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
    raise NotImplementedError


def get_next_queued_job() -> dict | None:
    """Scan all job directories for the oldest job with ``state=="queued"``.

    Jobs are ordered by their ``created_at`` ISO timestamp. Returns ``None``
    when no queued jobs exist.

    Returns:
        The parsed contents of the queued job's ``job.json``, or ``None``.
    """
    raise NotImplementedError


def get_job(job_id: str) -> dict:
    """Read and return a job's ``job.json`` as a dict.

    Args:
        job_id: The UUID of the job to retrieve.

    Returns:
        Parsed job metadata dict.

    Raises:
        FileNotFoundError: If the job directory or ``job.json`` does not exist.
    """
    raise NotImplementedError


def update_job_stage(job_id: str, stage: str) -> None:
    """Set the ``current_stage`` field in ``job.json`` to *stage*.

    Called at the beginning of each processing stage so that the job record
    reflects what is actively happening.

    Args:
        job_id: The UUID of the job to update.
        stage: The name of the stage now starting (e.g. ``"downloading"``).
    """
    raise NotImplementedError


def mark_stage_complete(job_id: str, stage: str) -> None:
    """Record *stage* as the most recently completed stage in ``job.json``.

    Used by ``JobRunner.process`` to track resume position so that a crashed
    job can restart from where it left off.

    Args:
        job_id: The UUID of the job.
        stage: The name of the stage that just finished.
    """
    raise NotImplementedError


def mark_job_complete(job_id: str) -> None:
    """Set ``state="complete"`` and record a ``completed_at`` timestamp.

    Args:
        job_id: The UUID of the job to mark complete.
    """
    raise NotImplementedError


def mark_job_failed(job_id: str, error: str) -> None:
    """Set ``state="failed"``, record ``error_message``, and timestamp.

    Args:
        job_id: The UUID of the job to mark as failed.
        error: Human-readable description of the failure reason.
    """
    raise NotImplementedError


def save_clips(job_id: str, clips: list[dict]) -> None:
    """Write the list of detected clips to ``data/jobs/<job_id>/clips.json``.

    Overwrites any existing ``clips.json`` for the job.

    Args:
        job_id: The UUID of the job.
        clips: List of clip dicts, each containing at minimum
               ``{clip_id, start_s, end_s, score}``.
    """
    raise NotImplementedError


def get_clips(job_id: str) -> list[dict]:
    """Read and return the list of clips from ``data/jobs/<job_id>/clips.json``.

    Args:
        job_id: The UUID of the job.

    Returns:
        List of clip dicts.

    Raises:
        FileNotFoundError: If ``clips.json`` does not exist for the job.
    """
    raise NotImplementedError


def create_post_record(clip_id: str, job_id: str, platform: str) -> str:
    """Append a new post record to ``data/jobs/<job_id>/posts.json``.

    Creates ``posts.json`` if it does not yet exist. The record is written
    with ``status="pending"`` and a UUID post ID.

    Args:
        clip_id: The ID of the clip being posted.
        job_id: The UUID of the parent job.
        platform: Target platform name (e.g. ``"youtube"``, ``"tiktok"``).

    Returns:
        The new post's UUID string.
    """
    raise NotImplementedError


def mark_post_success(job_id: str, post_id: str, url: str) -> None:
    """Update a post record in ``posts.json`` to ``status="success"``.

    Also records the returned public ``url`` and a ``posted_at`` timestamp.

    Args:
        job_id: The UUID of the parent job.
        post_id: The UUID of the post record to update.
        url: The public URL of the uploaded post.
    """
    raise NotImplementedError


def mark_post_failed(job_id: str, post_id: str, error: str) -> None:
    """Update a post record in ``posts.json`` to ``status="failed"``.

    Also records the ``error`` message and a ``failed_at`` timestamp.

    Args:
        job_id: The UUID of the parent job.
        post_id: The UUID of the post record to update.
        error: Human-readable description of the failure.
    """
    raise NotImplementedError


def get_posts(job_id: str) -> list[dict]:
    """Read and return the list of post records from ``posts.json``.

    Args:
        job_id: The UUID of the job.

    Returns:
        List of post record dicts. Returns an empty list if ``posts.json``
        does not yet exist.
    """
    raise NotImplementedError
