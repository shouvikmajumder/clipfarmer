"""Tests for core.job_runner — JobRunner orchestration and resume logic.

``processing.downloader.download`` and ``core.url_validator.validate_url``
are mocked, so these tests never touch yt-dlp or the network beyond what
pytest's ``tmp_path`` fixture provides.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core import state
from core.job_runner import JobRunner
from core.url_validator import VideoTooLongError


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect core.state file IO into a throwaway temp directory."""
    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    monkeypatch.setattr(state, "DATA_DIR", data_dir)
    monkeypatch.setattr(state, "JOBS_DIR", jobs_dir)
    return jobs_dir


@pytest.fixture
def runner() -> JobRunner:
    return JobRunner()


def _make_job(youtube_url: str = "https://www.youtube.com/watch?v=abc123") -> dict:
    job_id = state.enqueue_job(youtube_url)
    return state.get_job(job_id)


def _patch_pipeline(
    *,
    validate_url_return=None,
    validate_url_side_effect=None,
    download_return="raw/video.mp4",
    download_side_effect=None,
):
    """Build a dict of patch targets -> behaviours for the download-only pipeline."""
    if validate_url_return is None and validate_url_side_effect is None:
        validate_url_return = {
            "youtube_id": "abc123",
            "video_title": "Test Video",
            "video_duration_s": 600,
        }

    patches = {}

    validate_url_mock = MagicMock()
    if validate_url_side_effect is not None:
        validate_url_mock.side_effect = validate_url_side_effect
    else:
        validate_url_mock.return_value = validate_url_return
    patches["core.job_runner.validate_url"] = validate_url_mock

    download_mock = MagicMock()
    if download_side_effect is not None:
        download_mock.side_effect = download_side_effect
    else:
        download_mock.return_value = download_return
    patches["processing.downloader.download"] = download_mock

    return patches


def _apply_patches(patches: dict):
    """Return a list of started mock.patch context managers (caller stops them)."""
    ctxs = [patch(target, new) for target, new in patches.items()]
    for ctx in ctxs:
        ctx.start()
    return ctxs


def _stop_patches(ctxs):
    for ctx in ctxs:
        ctx.stop()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_download_only_pipeline_reaches_complete(runner: JobRunner):
    job = _make_job()
    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"
    assert final_job["youtube_id"] == "abc123"
    assert final_job["video_title"] == "Test Video"
    assert final_job["last_stage_completed"] == "downloading"

    patches["processing.downloader.download"].assert_called_once()


def test_download_failure_fails_job(runner: JobRunner):
    job = _make_job()
    patches = _patch_pipeline(download_side_effect=RuntimeError("yt-dlp crashed"))
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "failed"
    assert "yt-dlp crashed" in final_job["error_message"]


# ---------------------------------------------------------------------------
# Pre-flight rejection
# ---------------------------------------------------------------------------


def test_preflight_rejection_cancels_job(runner: JobRunner):
    job = _make_job()
    patches = _patch_pipeline(
        validate_url_side_effect=VideoTooLongError("video too long: 99999s")
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "cancelled"
    assert "too long" in final_job["error_message"]

    patches["processing.downloader.download"].assert_not_called()


# ---------------------------------------------------------------------------
# Crash-resume
# ---------------------------------------------------------------------------


def test_get_resume_index_with_no_last_stage(runner: JobRunner):
    job = _make_job()
    assert runner.get_resume_index(job) == 0


def test_get_resume_index_after_download_complete(runner: JobRunner):
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=resume")
    state.update_job_metadata(
        job_id, youtube_id="resume", video_title="Resume Video", video_duration_s=120
    )
    state.mark_stage_complete(job_id, "downloading")
    job = state.get_job(job_id)

    assert runner.get_resume_index(job) == len(JobRunner.STAGE_ORDER)


def test_crash_resume_after_successful_download_marks_complete_without_redownload(
    runner: JobRunner,
):
    """If the download already completed before a crash, resuming must mark
    the job complete directly without calling download() again -- the file
    is already on disk and is the deliverable for the next workflow.
    """
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=crash")
    state.update_job_metadata(
        job_id, youtube_id="crash", video_title="Crash Video", video_duration_s=120
    )
    state.mark_stage_complete(job_id, "downloading")
    job = state.get_job(job_id)

    assert runner.get_resume_index(job) == len(JobRunner.STAGE_ORDER)

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    patches["processing.downloader.download"].assert_not_called()

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"


def test_run_resumes_inflight_jobs_on_startup(runner: JobRunner, monkeypatch):
    """run() should pick up any non-terminal, non-queued job before polling."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=startup")
    state.update_job_metadata(
        job_id, youtube_id="startup", video_title="Startup Video", video_duration_s=60
    )
    state.update_job_stage(job_id, "downloading")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)

    # After resuming the in-flight job, get_next_queued_job should return None
    # so run() sleeps -- raise KeyboardInterrupt from the sleep to stop the loop.
    monkeypatch.setattr(
        "core.job_runner.time.sleep", MagicMock(side_effect=KeyboardInterrupt())
    )

    try:
        runner.run()
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"
    patches["processing.downloader.download"].assert_called_once()


# ---------------------------------------------------------------------------
# FIFO queue ordering
# ---------------------------------------------------------------------------


def test_get_next_queued_job_is_fifo_by_submission_order():
    job_id_1 = state.enqueue_job("https://www.youtube.com/watch?v=first")
    job_id_2 = state.enqueue_job("https://www.youtube.com/watch?v=second")

    # Force a deterministic ordering regardless of clock resolution.
    state._update_job(job_id_1, submitted_at="2026-01-01T00:00:00")
    state._update_job(job_id_2, submitted_at="2026-01-01T00:00:01")

    next_job = state.get_next_queued_job()
    assert next_job["id"] == job_id_1


# ---------------------------------------------------------------------------
# Disk-space pre-check before downloading
# ---------------------------------------------------------------------------


def test_low_disk_space_fails_job_before_downloading(runner: JobRunner, monkeypatch):
    """If shutil.disk_usage reports free space below general.min_free_disk_gb,
    the job must be marked failed with a clear message, and download() must
    never be called -- the worker process itself must not crash.
    """
    job = _make_job()

    class _FakeUsage:
        total = 100 * 1024 ** 3
        used = 99 * 1024 ** 3
        free = 1 * 1024 ** 3  # 1GB free, below the 5GB default threshold

    monkeypatch.setattr(
        "core.job_runner.shutil.disk_usage", MagicMock(return_value=_FakeUsage())
    )

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "failed"
    assert "disk space" in final_job["error_message"].lower()

    patches["processing.downloader.download"].assert_not_called()


def test_sufficient_disk_space_proceeds_to_download(runner: JobRunner, monkeypatch):
    job = _make_job()

    class _FakeUsage:
        total = 100 * 1024 ** 3
        used = 10 * 1024 ** 3
        free = 90 * 1024 ** 3  # plenty of room

    monkeypatch.setattr(
        "core.job_runner.shutil.disk_usage", MagicMock(return_value=_FakeUsage())
    )

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"
    patches["processing.downloader.download"].assert_called_once()


def test_disk_usage_check_failure_does_not_crash_worker(runner: JobRunner, monkeypatch):
    """If shutil.disk_usage itself raises (e.g. odd filesystem), the worker
    process must not crash -- the job should proceed as if space were
    sufficient rather than propagating the exception.
    """
    job = _make_job()

    monkeypatch.setattr(
        "core.job_runner.shutil.disk_usage",
        MagicMock(side_effect=OSError("disk_usage unsupported")),
    )

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"
    patches["processing.downloader.download"].assert_called_once()
