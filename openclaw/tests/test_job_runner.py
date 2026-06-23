"""Tests for core.job_runner — JobRunner orchestration and resume logic.

Every processing-module function and every core.state function is mocked,
so these tests never touch ffmpeg, yt-dlp, whisper, or the real filesystem
beyond what pytest's ``tmp_path`` fixture provides.
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


SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "Hello world, this is amazing."},
    {"start": 5.0, "end": 10.0, "text": "Here's the thing nobody tells you."},
    {"start": 10.0, "end": 15.0, "text": "What happens next will shock you?"},
]

CLIP_WINDOWS = [
    {"start_s": 0.0, "end_s": 10.0, "score": 0.9},
    {"start_s": 10.0, "end_s": 20.0, "score": 0.7},
]


def _patch_pipeline(
    *,
    validate_url_return=None,
    validate_url_side_effect=None,
    download_return="raw/video.mp4",
    transcribe_return=None,
    detect_clips_return=None,
    crop_to_vertical_side_effect=None,
    burn_captions_side_effect=None,
    format_clip_side_effect=None,
):
    """Build a dict of patch targets -> behaviours for the full pipeline."""
    if validate_url_return is None and validate_url_side_effect is None:
        validate_url_return = {
            "youtube_id": "abc123",
            "video_title": "Test Video",
            "video_duration_s": 600,
        }
    if transcribe_return is None:
        transcribe_return = SEGMENTS
    if detect_clips_return is None:
        detect_clips_return = CLIP_WINDOWS

    patches = {}

    validate_url_mock = MagicMock()
    if validate_url_side_effect is not None:
        validate_url_mock.side_effect = validate_url_side_effect
    else:
        validate_url_mock.return_value = validate_url_return
    patches["core.job_runner.validate_url"] = validate_url_mock

    download_mock = MagicMock(return_value=download_return)
    patches["processing.downloader.download"] = download_mock

    transcribe_mock = MagicMock(return_value=transcribe_return)
    patches["processing.transcriber.transcribe"] = transcribe_mock

    detect_clips_mock = MagicMock(return_value=detect_clips_return)
    patches["processing.clip_detector.detect_clips"] = detect_clips_mock

    crop_mock = MagicMock(side_effect=crop_to_vertical_side_effect)
    if crop_to_vertical_side_effect is None:
        crop_mock.side_effect = lambda input_path, start_s, end_s, output_path: output_path
    patches["processing.editor.crop_to_vertical"] = crop_mock

    burn_mock = MagicMock(side_effect=burn_captions_side_effect)
    if burn_captions_side_effect is None:
        burn_mock.side_effect = lambda input_path, segments, output_path: output_path
    patches["processing.caption_burner.burn_captions"] = burn_mock

    format_mock = MagicMock(side_effect=format_clip_side_effect)
    if format_clip_side_effect is None:
        format_mock.side_effect = lambda input_path, output_path, **kw: output_path
    patches["processing.formatter.format_clip"] = format_mock

    patches["posting.youtube.post_to_youtube"] = MagicMock(side_effect=NotImplementedError())
    patches["posting.tiktok.post_to_tiktok"] = MagicMock(side_effect=NotImplementedError())
    patches["posting.instagram.post_to_instagram"] = MagicMock(side_effect=NotImplementedError())

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


def test_full_pipeline_reaches_complete(runner: JobRunner):
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

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 2
    for clip in clips:
        assert clip["file_path"]
        assert clip["status"] == "ready"

    posts = state.get_posts(job["id"])
    # 2 clips x 3 platforms (default settings.yaml platforms list)
    assert len(posts) == 6


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

    # No clips or posts should have been created.
    assert state.get_clips_for_job(job["id"]) == []
    assert state.get_posts(job["id"]) == []


# ---------------------------------------------------------------------------
# Zero clips detected
# ---------------------------------------------------------------------------


def test_zero_clips_detected_fails_job(runner: JobRunner):
    job = _make_job()
    patches = _patch_pipeline(detect_clips_return=[])
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "failed"
    assert "no clips" in final_job["error_message"].lower()
    assert state.get_clips_for_job(job["id"]) == []


# ---------------------------------------------------------------------------
# One clip fails edit, others succeed
# ---------------------------------------------------------------------------


def test_one_clip_failing_edit_does_not_fail_job(runner: JobRunner):
    job = _make_job()

    def crop_side_effect(input_path, start_s, end_s, output_path):
        if start_s == 0.0:
            raise RuntimeError("ffmpeg crashed on clip 0")
        return output_path

    patches = _patch_pipeline(crop_to_vertical_side_effect=crop_side_effect)
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 1
    assert clips[0]["start_s"] == 10.0


def test_all_clips_failing_edit_fails_job(runner: JobRunner):
    job = _make_job()
    patches = _patch_pipeline(
        crop_to_vertical_side_effect=RuntimeError("ffmpeg always crashes")
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "failed"
    assert "editing" in final_job["error_message"]
    assert state.get_clips_for_job(job["id"]) == []


# ---------------------------------------------------------------------------
# Posting failure does not fail the job
# ---------------------------------------------------------------------------


def test_posting_failure_does_not_fail_job(runner: JobRunner):
    job = _make_job()
    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    # posting/* are stubs raising NotImplementedError -- job must still complete.
    assert final_job["state"] == "complete"

    posts = state.get_posts(job["id"])
    assert len(posts) > 0
    assert all(p["status"] == "failed" for p in posts)


# ---------------------------------------------------------------------------
# Crash-resume
# ---------------------------------------------------------------------------


def test_get_resume_index_with_no_last_stage(runner: JobRunner):
    job = _make_job()
    assert runner.get_resume_index(job) == 0


def test_get_resume_index_after_partial_completion(runner: JobRunner):
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=resume")
    state.update_job_metadata(
        job_id, youtube_id="resume", video_title="Resume Video", video_duration_s=120
    )
    state.mark_stage_complete(job_id, "downloading")
    state.mark_stage_complete(job_id, "transcribing")
    job = state.get_job(job_id)

    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("detecting")


def test_crash_resume_skips_already_completed_stages(runner: JobRunner):
    """A job that crashed after transcribing should resume at detecting.

    The pipeline loop itself does not re-invoke download/transcribe stage
    handlers for stages already marked complete (it starts iterating
    STAGE_ORDER at the resume index), but since their in-memory artifacts
    (video_path/segments) don't survive a crash, the runner recomputes them
    once up front before resuming -- this is a deliberate trade-off
    (idempotent re-download/re-transcribe) in exchange for correctness.
    The resumed run must still reach `detecting` exactly once via the main
    loop and proceed through to completion.
    """
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=crash")
    state.update_job_metadata(
        job_id, youtube_id="crash", video_title="Crash Video", video_duration_s=120
    )
    state.update_job_stage(job_id, "transcribing")
    state.mark_stage_complete(job_id, "downloading")
    state.mark_stage_complete(job_id, "transcribing")
    job = state.get_job(job_id)

    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("detecting")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    # Recomputed once each to recover lost in-memory state, not once per stage.
    assert patches["processing.downloader.download"].call_count == 1
    assert patches["processing.transcriber.transcribe"].call_count == 1
    patches["processing.clip_detector.detect_clips"].assert_called_once()

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"


def test_run_resumes_inflight_jobs_on_startup(runner: JobRunner, monkeypatch):
    """run() should pick up any non-terminal, non-queued job before polling."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=startup")
    state.update_job_metadata(
        job_id, youtube_id="startup", video_title="Startup Video", video_duration_s=60
    )
    state.update_job_stage(job_id, "formatting")
    state.mark_stage_complete(job_id, "downloading")
    state.mark_stage_complete(job_id, "transcribing")
    state.mark_stage_complete(job_id, "detecting")
    state.mark_stage_complete(job_id, "editing")
    state.mark_stage_complete(job_id, "captioning")

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
    # Resume started at formatting; earlier stages' in-memory artifacts are
    # gone after a crash, so they are recomputed once each before formatting
    # runs -- but the main stage loop itself only executes formatting/posting.
    patches["processing.downloader.download"].assert_called_once()
    patches["processing.transcriber.transcribe"].assert_called_once()
    patches["processing.clip_detector.detect_clips"].assert_called_once()
    patches["processing.editor.crop_to_vertical"].assert_called()
    patches["processing.caption_burner.burn_captions"].assert_called()
    patches["processing.formatter.format_clip"].assert_called()


# ---------------------------------------------------------------------------
# Resume from a fully-completed "formatting" stage (fix #2)
# ---------------------------------------------------------------------------


def test_resume_from_formatting_complete_skips_reformatting_goes_straight_to_posting(
    runner: JobRunner,
):
    """Resuming with last_stage_completed="formatting" must NOT re-invoke
    crop_to_vertical/burn_captions/format_clip -- formatting already fully
    completed and persisted clips for this job. The pipeline should jump
    straight to posting, reading clips from disk via get_clips_for_job.
    """
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=formatted")
    state.update_job_metadata(
        job_id, youtube_id="formatted", video_title="Formatted Video", video_duration_s=60
    )
    state.mark_stage_complete(job_id, "downloading")
    state.mark_stage_complete(job_id, "transcribing")
    state.mark_stage_complete(job_id, "detecting")
    state.mark_stage_complete(job_id, "editing")
    state.mark_stage_complete(job_id, "captioning")
    state.mark_stage_complete(job_id, "formatting")

    # Simulate clips already fully persisted from the completed formatting run.
    state.add_clip(
        job_id,
        {
            "file_path": f"data/jobs/{job_id}/clips/clip_0.mp4",
            "score": 0.9,
            "start_s": 0.0,
            "end_s": 10.0,
            "transcript_snippet": "Hello world",
        },
    )

    job = state.get_job(job_id)
    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("posting")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    patches["processing.editor.crop_to_vertical"].assert_not_called()
    patches["processing.caption_burner.burn_captions"].assert_not_called()
    patches["processing.formatter.format_clip"].assert_not_called()
    # downloading/transcribing/detecting are also skipped since start_idx is
    # past formatting -- the early-return branch never touches them either.
    patches["processing.downloader.download"].assert_not_called()
    patches["processing.transcriber.transcribe"].assert_not_called()
    patches["processing.clip_detector.detect_clips"].assert_not_called()

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job_id)
    assert len(clips) == 1

    posts = state.get_posts(job_id)
    # 1 clip x 3 platforms (default settings.yaml platforms list)
    assert len(posts) == 3


# ---------------------------------------------------------------------------
# Duplicate clip prevention on resume mid-formatting (fix #1)
# ---------------------------------------------------------------------------


def test_resume_mid_formatting_does_not_duplicate_persisted_clips(runner: JobRunner):
    """Simulate a crash mid-formatting: one clip was already persisted via
    state.add_clip() before the crash, but mark_stage_complete(job_id,
    "formatting") never got written (so last_stage_completed is still
    "captioning"). Resuming must re-run _format_clips without re-adding the
    already-persisted clip, and posting must not double-post it.
    """
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=midformat")
    state.update_job_metadata(
        job_id, youtube_id="midformat", video_title="Mid Format Video", video_duration_s=60
    )
    state.mark_stage_complete(job_id, "downloading")
    state.mark_stage_complete(job_id, "transcribing")
    state.mark_stage_complete(job_id, "detecting")
    state.mark_stage_complete(job_id, "editing")
    state.mark_stage_complete(job_id, "captioning")
    # Note: formatting NOT marked complete -- simulates a crash partway
    # through _format_clips after add_clip() for the first clip window.

    # CLIP_WINDOWS[0] is {"start_s": 0.0, "end_s": 10.0, "score": 0.9}.
    state.add_clip(
        job_id,
        {
            "file_path": f"data/jobs/{job_id}/clips/clip_0.mp4",
            "score": 0.9,
            "start_s": 0.0,
            "end_s": 10.0,
            "transcript_snippet": "Hello world, this is amazing.",
        },
    )

    job = state.get_job(job_id)
    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("formatting")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job_id)
    # Exactly 2 clips total (one for each CLIP_WINDOWS entry) -- no duplicate
    # for the window that was already persisted before the simulated crash.
    assert len(clips) == 2
    windows = [(c["start_s"], c["end_s"]) for c in clips]
    assert len(windows) == len(set(windows)), "duplicate clip window detected"

    # format_clip should only be called for the NOT-yet-persisted window
    # (start_s=10.0); the already-persisted clip (start_s=0.0) is skipped.
    format_clip_mock = patches["processing.formatter.format_clip"]
    assert format_clip_mock.call_count == 1

    # Posting: 2 clips x 3 platforms = 6 posts, never more (no double-posting
    # of the clip that survived the crash).
    posts = state.get_posts(job_id)
    assert len(posts) == 6


# ---------------------------------------------------------------------------
# Cleanup of raw downloads / intermediate files on completion (fix #3)
# ---------------------------------------------------------------------------


def test_completion_cleans_up_raw_and_intermediate_files_keeps_final_clips(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """After mark_job_complete, raw/ downloads and edited_*/captioned_*
    intermediate files must be deleted, while the final clip_*.mp4 files
    referenced by persisted clip records survive.
    """
    job = _make_job()
    job_id = job["id"]
    job_dir = isolated_jobs_dir / job_id
    raw_dir = job_dir / "raw"
    clips_dir = job_dir / "clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    raw_video = raw_dir / "abc123.mp4"
    raw_video.write_bytes(b"raw video bytes")

    def make_intermediate_and_final(idx: int):
        (clips_dir / f"edited_{idx}.mp4").write_bytes(b"edited")
        (clips_dir / f"captioned_{idx}.mp4").write_bytes(b"captioned")
        final_path = clips_dir / f"clip_{idx}.mp4"
        final_path.write_bytes(b"final")
        return str(final_path)

    def crop_side_effect(input_path, start_s, end_s, output_path):
        Path(output_path).write_bytes(b"edited")
        return output_path

    def burn_side_effect(input_path, segments, output_path):
        Path(output_path).write_bytes(b"captioned")
        return output_path

    def format_side_effect(input_path, output_path, **kw):
        Path(output_path).write_bytes(b"final")
        return output_path

    patches = _patch_pipeline(
        crop_to_vertical_side_effect=crop_side_effect,
        burn_captions_side_effect=burn_side_effect,
        format_clip_side_effect=format_side_effect,
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job_id)
    assert len(clips) == 2

    # Raw downloads must be gone.
    assert not raw_video.exists()
    assert list(raw_dir.iterdir()) == []

    # Intermediate files must be gone.
    assert list(clips_dir.glob("edited_*.mp4")) == []
    assert list(clips_dir.glob("captioned_*.mp4")) == []

    # Final clip files referenced by persisted clip records must survive.
    for clip in clips:
        assert Path(clip["file_path"]).exists()


# ---------------------------------------------------------------------------
# Disk-space pre-check before downloading (fix #5)
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
