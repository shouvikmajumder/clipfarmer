"""Tests for core.job_runner — JobRunner orchestration and resume logic.

``processing.downloader.download``, ``core.url_validator.validate_url``, and
the detection-pipeline modules are mocked, so these tests never touch yt-dlp,
librosa, or the network beyond what pytest's ``tmp_path`` fixture provides.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core import state
from core.job_runner import JobRunner
from core.url_validator import VideoTooLongError

# Sentinel value meaning "use the default" in optional keyword arguments.
_UNSET = object()


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
    detect_return=_UNSET,
    detect_side_effect=None,
    fetch_transcript_return=None,
    fetch_comments_return=None,
    audio_signals_return=None,
    edit_clip_side_effect=None,
    transcribe_words_return=None,
    caption_clip_side_effect=None,
):
    """Build a dict of patch targets -> behaviours for the three-stage pipeline.

    Detection mocks default to returning an empty clip list (so the job still
    reaches COMPLETE) unless *detect_return* / *detect_side_effect* are given.
    The individual component mocks (transcript, comments, audio) can also be
    overridden; they default to None / [] / None respectively.

    The editing stage's ``processing.editor.edit_clip`` is mocked too: by
    default it simply returns the ``output_path`` it was handed (so a clip's
    ``file_path`` is recorded) without invoking FFmpeg. Pass
    *edit_clip_side_effect* to override (e.g. to raise).
    """
    if validate_url_return is None and validate_url_side_effect is None:
        validate_url_return = {
            "youtube_id": "abc123",
            "video_title": "Test Video",
            "video_duration_s": 600,
        }
    if detect_return is _UNSET:
        detect_return = []
    if fetch_comments_return is None:
        fetch_comments_return = []

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

    # Detection components — lazily imported by _detect at call time.
    detect_mock = MagicMock()
    if detect_side_effect is not None:
        detect_mock.side_effect = detect_side_effect
    else:
        detect_mock.return_value = detect_return
    patches["processing.clip_detector.detect"] = detect_mock

    transcript_mock = MagicMock(return_value=fetch_transcript_return)
    patches["processing.transcript_fetcher.fetch_transcript"] = transcript_mock

    comments_mock = MagicMock(return_value=fetch_comments_return)
    patches["processing.comments.fetch_comments"] = comments_mock

    audio_mock = MagicMock(return_value=audio_signals_return)
    patches["processing.audio_analyzer.AudioSignals.from_file"] = audio_mock

    # Editing stage — lazily imported by _edit at call time. Default: echo the
    # output_path back so file_path gets recorded, without running FFmpeg.
    def _default_edit(input_path, start_s, end_s, output_path, **kwargs):
        return output_path

    edit_mock = MagicMock()
    edit_mock.side_effect = edit_clip_side_effect or _default_edit
    patches["processing.editor.edit_clip"] = edit_mock

    # Captioning stage — lazily imported by _caption at call time. Default:
    # transcribe_words returns no words, caption_clip echoes its output_path,
    # so the stage completes without running Whisper or FFmpeg.
    patches["processing.transcriber.transcribe_words"] = MagicMock(
        return_value=transcribe_words_return if transcribe_words_return is not None else []
    )

    def _default_caption(input_path, words, srt_path, output_path, offset_s=0.0):
        return output_path

    caption_mock = MagicMock()
    caption_mock.side_effect = caption_clip_side_effect or _default_caption
    patches["processing.caption_burner.caption_clip"] = caption_mock

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
    """A queued job must pass through all four stages and end COMPLETE."""
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
    # All four stages must have completed; the last one recorded is "captioning".
    assert final_job["last_stage_completed"] == "captioning"

    patches["processing.downloader.download"].assert_called_once()
    patches["processing.clip_detector.detect"].assert_called_once()


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
    """After downloading, resume index must point at detecting (index 1), not past the end."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=resume")
    state.update_job_metadata(
        job_id, youtube_id="resume", video_title="Resume Video", video_duration_s=120
    )
    state.mark_stage_complete(job_id, "downloading")
    job = state.get_job(job_id)

    # STAGE_ORDER == ["downloading", "detecting"]
    # After "downloading" completes, resume index == 1 == STAGE_ORDER.index("detecting").
    expected_index = JobRunner.STAGE_ORDER.index("detecting")
    assert runner.get_resume_index(job) == expected_index
    assert expected_index == 1  # Belt-and-suspenders: index must be 1, not 2.


def test_crash_resume_after_successful_download_runs_detecting_without_redownload(
    runner: JobRunner,
    isolated_jobs_dir: Path,
):
    """If the download already completed before a crash, resuming must skip
    download, run the detecting stage, and then mark the job complete.
    """
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=crash")
    state.update_job_metadata(
        job_id, youtube_id="crash", video_title="Crash Video", video_duration_s=120
    )
    state.mark_stage_complete(job_id, "downloading")
    job = state.get_job(job_id)

    # After "downloading" completes, resume_index == 1, which is detecting.
    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("detecting")

    # Create the video file so _locate_video can find it.
    raw_dir = isolated_jobs_dir / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    video_file = raw_dir / "crash.mp4"
    video_file.write_bytes(b"fake video content")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    # Download must NOT be called again — the file is already on disk.
    patches["processing.downloader.download"].assert_not_called()

    # Detection MUST have been called — that's the resumed stage.
    patches["processing.clip_detector.detect"].assert_called_once()

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


# ---------------------------------------------------------------------------
# Detecting stage — end-to-end and unit coverage
# ---------------------------------------------------------------------------


def _make_fake_video(isolated_jobs_dir: Path, job_id: str, youtube_id: str = "abc123") -> Path:
    """Create a fake raw video file and return its path."""
    raw_dir = isolated_jobs_dir / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    video_file = raw_dir / f"{youtube_id}.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


def test_process_end_to_end_with_two_clips(runner: JobRunner, isolated_jobs_dir: Path):
    """A queued job with two detected clips must end COMPLETE and persist both."""
    two_clips = [
        {"start_s": 10.0, "end_s": 40.0, "score": 0.85},
        {"start_s": 60.0, "end_s": 90.0, "score": 0.72},
    ]

    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    # _download mock must return the real path so _detect has it.
    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=two_clips,
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 2
    clip_tuples = [(c["start_s"], c["end_s"], c["score"]) for c in clips]
    assert (10.0, 40.0, 0.85) in clip_tuples
    assert (60.0, 90.0, 0.72) in clip_tuples


def test_transcript_snippet_included_when_transcript_available(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """When fetch_transcript returns overlapping segments, persisted records must
    include a non-empty transcript_snippet."""
    transcript = [
        {"start": 5.0, "end": 15.0, "text": "Hello world, this is a test."},
        {"start": 15.0, "end": 25.0, "text": "More interesting content here."},
    ]
    clip = {"start_s": 8.0, "end_s": 20.0, "score": 0.80}

    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[clip],
        fetch_transcript_return=transcript,
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 1
    assert "transcript_snippet" in clips[0]
    assert clips[0]["transcript_snippet"]  # non-empty


def test_transcript_snippet_absent_when_transcript_is_none(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """When fetch_transcript returns None, persisted clip records must NOT contain
    a transcript_snippet key."""
    clip = {"start_s": 10.0, "end_s": 35.0, "score": 0.75}

    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[clip],
        fetch_transcript_return=None,
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 1
    assert "transcript_snippet" not in clips[0]


def test_idempotent_clip_persistence_on_resume(runner: JobRunner, isolated_jobs_dir: Path):
    """Calling _detect twice on the same job must not double the clip count,
    because save_clips([]) clears existing records before re-adding."""
    two_clips = [
        {"start_s": 0.0, "end_s": 30.0, "score": 0.90},
        {"start_s": 50.0, "end_s": 80.0, "score": 0.65},
    ]

    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    # Call _detect directly twice to simulate a crash-resume re-run.
    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=two_clips,
    )
    ctxs = _apply_patches(patches)
    try:
        runner._detect(str(video_file), job)
        runner._detect(str(video_file), job)
    finally:
        _stop_patches(ctxs)

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 2, (
        f"Expected 2 clips (idempotent); got {len(clips)}. "
        "save_clips([]) must clear before re-adding."
    )


def test_zero_clips_job_is_still_complete(runner: JobRunner, isolated_jobs_dir: Path):
    """detect() returning an empty list must not fail the job — zero clips is a
    valid outcome (a warning is logged but the job reaches COMPLETE)."""
    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[],
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job["id"])
    assert clips == []


def test_audio_failure_is_nonfatal(runner: JobRunner, isolated_jobs_dir: Path):
    """If AudioSignals.from_file raises, _detect must still call detect() with
    audio_signals=None and the job must complete."""
    clip = {"start_s": 5.0, "end_s": 35.0, "score": 0.70}

    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    # Capture the actual call args to detect() so we can inspect audio_signals.
    detect_calls: list = []

    def _fake_detect(transcript, *, comments=None, audio_signals=None, max_clips=None):
        detect_calls.append({"audio_signals": audio_signals})
        return [clip]

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_side_effect=_fake_detect,
    )
    # AudioSignals.from_file raises an ImportError (mimicking librosa absent).
    patches["processing.audio_analyzer.AudioSignals.from_file"] = MagicMock(
        side_effect=ImportError("No module named 'librosa'")
    )

    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    assert detect_calls, "detect() was never called"
    assert detect_calls[0]["audio_signals"] is None, (
        "detect() must receive audio_signals=None when AudioSignals.from_file fails"
    )

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"


# ---------------------------------------------------------------------------
# _locate_video
# ---------------------------------------------------------------------------


def test_locate_video_finds_deterministic_path(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """_locate_video must return raw/<youtube_id>.mp4 when that file exists."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=loctest")
    state.update_job_metadata(
        job_id, youtube_id="loctest", video_title="Loc Test", video_duration_s=60
    )
    job = state.get_job(job_id)

    video_file = _make_fake_video(isolated_jobs_dir, job_id, youtube_id="loctest")

    result = runner._locate_video(job)
    assert result == str(video_file)


def test_locate_video_returns_none_when_raw_dir_empty(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """_locate_video must return None when the raw directory exists but is empty."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=empty")
    state.update_job_metadata(
        job_id, youtube_id="empty", video_title="Empty", video_duration_s=60
    )
    job = state.get_job(job_id)

    # Ensure the raw dir exists but has no files.
    raw_dir = isolated_jobs_dir / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    result = runner._locate_video(job)
    assert result is None


def test_crash_resume_into_detecting_with_missing_video_fails_job(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """Resuming into the detecting stage with no video file present must mark the
    job failed with a descriptive error message."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=missing")
    state.update_job_metadata(
        job_id, youtube_id="missing", video_title="Missing", video_duration_s=60
    )
    state.mark_stage_complete(job_id, "downloading")
    job = state.get_job(job_id)

    # raw dir exists but no video file — _locate_video returns None.
    raw_dir = isolated_jobs_dir / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job_id)
    assert final_job["state"] == "failed"
    assert "downloaded video not found" in final_job["error_message"]

    # Download must not have been called — we resumed past it.
    patches["processing.downloader.download"].assert_not_called()


# ---------------------------------------------------------------------------
# State transitions through detecting
# ---------------------------------------------------------------------------


def test_state_passes_through_all_processing_stages(runner: JobRunner, isolated_jobs_dir: Path):
    """process() must move through 'detecting', 'editing' AND 'captioning', and the
    final job must have last_stage_completed == 'captioning'."""
    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    stage_updates: list[str] = []
    stage_completions: list[str] = []

    original_update_stage = state.update_job_stage
    original_mark_complete = state.mark_stage_complete

    def _capture_update(job_id: str, stage: str) -> None:
        stage_updates.append(stage)
        original_update_stage(job_id, stage)

    def _capture_complete(job_id: str, stage: str) -> None:
        stage_completions.append(stage)
        original_mark_complete(job_id, stage)

    patches = _patch_pipeline(download_return=str(video_file))

    ctxs = _apply_patches(patches)
    try:
        with (
            patch("core.job_runner.state.update_job_stage", side_effect=_capture_update),
            patch("core.job_runner.state.mark_stage_complete", side_effect=_capture_complete),
        ):
            runner.process(job)
    finally:
        _stop_patches(ctxs)

    for stage in ("downloading", "detecting", "editing", "captioning"):
        assert stage in stage_updates
        assert stage in stage_completions

    final_job = state.get_job(job["id"])
    assert final_job["last_stage_completed"] == "captioning"


# ---------------------------------------------------------------------------
# Editing stage — end-to-end and unit coverage
# ---------------------------------------------------------------------------


def test_editing_persists_file_path(runner: JobRunner, isolated_jobs_dir: Path):
    """_edit must render each clip and set file_path to clip_<idx>_edited.mp4.

    Tested against _edit in isolation (the later captioning stage would rewrite
    file_path to the captioned output)."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=editfp")
    state.update_job_metadata(
        job_id, youtube_id="editfp", video_title="EditFP", video_duration_s=120
    )
    state.add_clip(job_id, {"start_s": 10.0, "end_s": 40.0, "score": 0.85})
    state.add_clip(job_id, {"start_s": 60.0, "end_s": 90.0, "score": 0.72})
    job = state.get_job(job_id)
    video_file = _make_fake_video(isolated_jobs_dir, job_id, youtube_id="editfp")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner._edit(str(video_file), job)
    finally:
        _stop_patches(ctxs)

    clips = state.get_clips_for_job(job_id)
    assert len(clips) == 2
    for idx, clip in enumerate(clips):
        assert clip["file_path"].endswith(f"clip_{idx}_edited.mp4")

    # edit_clip must have run once per clip.
    assert patches["processing.editor.edit_clip"].call_count == 2


def test_editing_forwards_profile_and_crop_bias_from_options(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """content_profile / crop_bias set in the job options must be forwarded to edit_clip."""
    job_id = state.enqueue_job(
        "https://www.youtube.com/watch?v=prof",
        options={"content_profile": "interview", "crop_bias": "right"},
    )
    job = state.get_job(job_id)
    video_file = _make_fake_video(isolated_jobs_dir, job_id)

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[{"start_s": 1.0, "end_s": 20.0, "score": 0.9}],
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    edit_mock = patches["processing.editor.edit_clip"]
    edit_mock.assert_called_once()
    _, kwargs = edit_mock.call_args
    assert kwargs["profile"] == "interview"
    assert kwargs["crop_bias"] == "right"


def test_editing_uses_default_profile_from_settings(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """When the job options omit content_profile, the settings default ('default')
    must be used."""
    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[{"start_s": 1.0, "end_s": 20.0, "score": 0.9}],
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    _, kwargs = patches["processing.editor.edit_clip"].call_args
    assert kwargs["profile"] == "default"


def test_all_clips_fail_editing_marks_job_failed(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """If edit_clip raises for every clip, the job must be marked FAILED."""
    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[{"start_s": 1.0, "end_s": 20.0, "score": 0.9}],
        edit_clip_side_effect=RuntimeError("ffmpeg boom"),
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "failed"
    assert "all clips failed during editing" in final_job["error_message"]


def test_editing_partial_failure_is_isolated(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """If one clip fails editing but another succeeds, the job completes and only
    the succeeding clip carries a file_path."""
    two_clips = [
        {"start_s": 1.0, "end_s": 20.0, "score": 0.9},
        {"start_s": 30.0, "end_s": 50.0, "score": 0.8},
    ]
    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    calls = {"n": 0}

    def _edit(input_path, start_s, end_s, output_path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first clip boom")
        return output_path

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=two_clips,
        edit_clip_side_effect=_edit,
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 2
    with_path = [c for c in clips if c.get("file_path")]
    assert len(with_path) == 1


def test_resume_into_editing_runs_only_editing(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """Resuming a job whose detecting stage already completed must run ONLY
    editing — download and detect must not be called."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=redit")
    state.update_job_metadata(
        job_id, youtube_id="redit", video_title="Redit", video_duration_s=120
    )
    # Pre-seed the clips that detecting would have produced.
    state.add_clip(job_id, {"start_s": 1.0, "end_s": 20.0, "score": 0.9})
    state.add_clip(job_id, {"start_s": 30.0, "end_s": 50.0, "score": 0.8})
    state.mark_stage_complete(job_id, "detecting")
    job = state.get_job(job_id)

    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("editing")

    _make_fake_video(isolated_jobs_dir, job_id, youtube_id="redit")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    patches["processing.downloader.download"].assert_not_called()
    patches["processing.clip_detector.detect"].assert_not_called()
    assert patches["processing.editor.edit_clip"].call_count == 2

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"
    # Resuming at editing also runs the subsequent captioning stage.
    assert final_job["last_stage_completed"] == "captioning"


def test_crash_resume_into_editing_with_missing_video_fails_job(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """Resuming into editing with no video file present must mark the job failed."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=emiss")
    state.update_job_metadata(
        job_id, youtube_id="emiss", video_title="Emiss", video_duration_s=60
    )
    state.add_clip(job_id, {"start_s": 1.0, "end_s": 20.0, "score": 0.9})
    state.mark_stage_complete(job_id, "detecting")
    job = state.get_job(job_id)

    # raw dir exists but is empty — _locate_video returns None.
    raw_dir = isolated_jobs_dir / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job_id)
    assert final_job["state"] == "failed"
    assert "downloaded video not found for editing" in final_job["error_message"]

    patches["processing.downloader.download"].assert_not_called()
    patches["processing.clip_detector.detect"].assert_not_called()


def test_get_resume_index_after_detecting_editing_captioning(runner: JobRunner):
    """Resume index tracks each completed stage: detecting → editing (2),
    editing → captioning (3), captioning → past the end (== len(STAGE_ORDER))."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=ridx")
    state.update_job_metadata(
        job_id, youtube_id="ridx", video_title="Ridx", video_duration_s=60
    )

    state.mark_stage_complete(job_id, "detecting")
    job = state.get_job(job_id)
    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("editing")

    state.mark_stage_complete(job_id, "editing")
    job = state.get_job(job_id)
    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("captioning")

    state.mark_stage_complete(job_id, "captioning")
    job = state.get_job(job_id)
    assert runner.get_resume_index(job) == len(JobRunner.STAGE_ORDER)


# ---------------------------------------------------------------------------
# Captioning stage
# ---------------------------------------------------------------------------


def _make_job_with_profile(profile: str, youtube_url: str = "https://www.youtube.com/watch?v=prof") -> dict:
    job_id = state.enqueue_job(youtube_url, options={"content_profile": profile})
    return state.get_job(job_id)


@pytest.mark.parametrize("profile", ["gaming", "irl"])
def test_profile_skips_captioning(runner: JobRunner, isolated_jobs_dir: Path, profile: str):
    """The gaming and irl profiles must skip captioning entirely: caption_clip is
    never called and each clip keeps its edited file_path; the job still COMPLETEs."""
    job = _make_job_with_profile(profile)
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[{"start_s": 1.0, "end_s": 20.0, "score": 0.9}],
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"
    assert final_job["last_stage_completed"] == "captioning"

    # Captioning was skipped for this profile.
    patches["processing.caption_burner.caption_clip"].assert_not_called()

    clips = state.get_clips_for_job(job["id"])
    assert clips[0]["file_path"].endswith("clip_0_edited.mp4")


def test_captioning_persists_captioned_file_path(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """For a captioned profile, each clip's file_path is rewritten to its
    captioned output and caption_clip runs once per clip."""
    two_clips = [
        {"start_s": 10.0, "end_s": 40.0, "score": 0.85},
        {"start_s": 60.0, "end_s": 90.0, "score": 0.72},
    ]
    job = _make_job()  # default profile → captioned
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(download_return=str(video_file), detect_return=two_clips)
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "complete"

    clips = state.get_clips_for_job(job["id"])
    assert len(clips) == 2
    for idx, clip in enumerate(clips):
        assert clip["file_path"].endswith(f"clip_{idx}_captioned.mp4")

    assert patches["processing.caption_burner.caption_clip"].call_count == 2
    # Whisper word extraction must have run once per clip.
    assert patches["processing.transcriber.transcribe_words"].call_count == 2


def test_all_clips_fail_captioning_marks_job_failed(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """If caption_clip raises for every clip, the job must be marked FAILED."""
    job = _make_job()
    video_file = _make_fake_video(isolated_jobs_dir, job["id"])

    patches = _patch_pipeline(
        download_return=str(video_file),
        detect_return=[{"start_s": 1.0, "end_s": 20.0, "score": 0.9}],
        caption_clip_side_effect=RuntimeError("ffmpeg caption boom"),
    )
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    final_job = state.get_job(job["id"])
    assert final_job["state"] == "failed"
    assert "all clips failed during captioning" in final_job["error_message"]


def test_resume_into_captioning_runs_only_captioning(
    runner: JobRunner, isolated_jobs_dir: Path
):
    """Resuming a job whose editing stage already completed must run ONLY
    captioning — download, detect, and edit must not be called."""
    job_id = state.enqueue_job("https://www.youtube.com/watch?v=rcap")
    state.update_job_metadata(
        job_id, youtube_id="rcap", video_title="Rcap", video_duration_s=120
    )
    # Pre-seed clips as editing would have left them (file_path = edited mp4).
    state.add_clip(
        job_id, {"start_s": 1.0, "end_s": 20.0, "score": 0.9, "file_path": "/x/clip_0_edited.mp4"}
    )
    state.add_clip(
        job_id, {"start_s": 30.0, "end_s": 50.0, "score": 0.8, "file_path": "/x/clip_1_edited.mp4"}
    )
    state.mark_stage_complete(job_id, "editing")
    job = state.get_job(job_id)

    assert runner.get_resume_index(job) == JobRunner.STAGE_ORDER.index("captioning")

    patches = _patch_pipeline()
    ctxs = _apply_patches(patches)
    try:
        runner.process(job)
    finally:
        _stop_patches(ctxs)

    patches["processing.downloader.download"].assert_not_called()
    patches["processing.clip_detector.detect"].assert_not_called()
    patches["processing.editor.edit_clip"].assert_not_called()
    assert patches["processing.caption_burner.caption_clip"].call_count == 2

    final_job = state.get_job(job_id)
    assert final_job["state"] == "complete"
    assert final_job["last_stage_completed"] == "captioning"
