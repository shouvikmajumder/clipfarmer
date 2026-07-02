"""Tests for core.state — job lifecycle and persistence helpers.

All tests monkeypatch ``core.state.DATA_DIR`` / ``core.state.JOBS_DIR`` to a
pytest ``tmp_path`` so nothing is ever written to the real ``data/`` dir.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core import state


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all state.py file IO into a throwaway temp directory."""
    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    monkeypatch.setattr(state, "DATA_DIR", data_dir)
    monkeypatch.setattr(state, "JOBS_DIR", jobs_dir)
    return jobs_dir


# ---------------------------------------------------------------------------
# enqueue_job
# ---------------------------------------------------------------------------


def test_enqueue_job_creates_correct_file_structure(isolated_jobs_dir: Path) -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=abc123")

    job_dir = isolated_jobs_dir / job_id
    assert job_dir.is_dir()
    assert (job_dir / "job.json").is_file()
    assert (job_dir / "raw").is_dir()
    assert (job_dir / "clips").is_dir()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["id"] == job_id
    assert job["youtube_url"] == "https://youtube.com/watch?v=abc123"
    assert job["state"] == "queued"
    assert job["current_stage"] is None
    assert job["last_stage_completed"] is None
    assert job["retry_count"] == 0
    assert job["options"] == {}
    assert "submitted_at" in job and job["submitted_at"]


def test_enqueue_job_stores_options() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=xyz", options={"max_clips": 5})
    job = state.get_job(job_id)
    assert job["options"] == {"max_clips": 5}


def test_enqueue_job_ids_are_unique() -> None:
    id1 = state.enqueue_job("https://youtube.com/watch?v=a")
    id2 = state.enqueue_job("https://youtube.com/watch?v=b")
    assert id1 != id2


# ---------------------------------------------------------------------------
# atomic write behaviour
# ---------------------------------------------------------------------------


def test_atomic_write_does_not_leave_tmp_file(isolated_jobs_dir: Path) -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=abc")
    job_json = isolated_jobs_dir / job_id / "job.json"
    tmp_path = job_json.with_suffix(".json.tmp")

    state.update_job_stage(job_id, "downloading")

    assert job_json.exists()
    assert not tmp_path.exists()


def test_atomic_write_leaves_old_file_intact_if_interrupted(tmp_path: Path) -> None:
    """Simulate a crash mid-write: a leftover .tmp file must never be picked
    up in place of the real file, and the original file must remain valid.
    """
    target = tmp_path / "job.json"
    original = {"state": "queued"}
    target.write_text(json.dumps(original))

    # Simulate a crash: a partial .tmp file is left on disk (e.g. process
    # killed after open() but before os.replace()).
    tmp_file = target.with_suffix(".json.tmp")
    tmp_file.write_text('{"state": "down')  # truncated / invalid JSON

    # The "real" file must still be readable and correct, regardless of the
    # stray .tmp file sitting next to it.
    on_disk = json.loads(target.read_text())
    assert on_disk == original

    # A subsequent atomic write must succeed and clean up its own tmp file.
    state._atomic_write_json(target, {"state": "downloading"})
    assert json.loads(target.read_text()) == {"state": "downloading"}


# ---------------------------------------------------------------------------
# get_next_queued_job
# ---------------------------------------------------------------------------


def test_get_next_queued_job_returns_none_when_empty(isolated_jobs_dir: Path) -> None:
    assert state.get_next_queued_job() is None


def test_get_next_queued_job_skips_non_queued(isolated_jobs_dir: Path) -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=a")
    state.update_job_stage(job_id, "downloading")  # moves state off "queued"

    assert state.get_next_queued_job() is None


def test_get_next_queued_job_returns_oldest_by_submitted_at(isolated_jobs_dir: Path) -> None:
    id1 = state.enqueue_job("https://youtube.com/watch?v=first")
    time.sleep(0.01)
    id2 = state.enqueue_job("https://youtube.com/watch?v=second")
    time.sleep(0.01)
    id3 = state.enqueue_job("https://youtube.com/watch?v=third")

    next_job = state.get_next_queued_job()
    assert next_job is not None
    assert next_job["id"] == id1

    # After the oldest is no longer queued, the next-oldest should surface.
    state.update_job_stage(id1, "downloading")
    next_job = state.get_next_queued_job()
    assert next_job["id"] == id2

    state.update_job_stage(id2, "downloading")
    next_job = state.get_next_queued_job()
    assert next_job["id"] == id3


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


def test_get_job_raises_for_missing_job(isolated_jobs_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        state.get_job("does-not-exist")


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------


def test_update_job_stage_sets_current_stage_and_state() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=stage")
    state.update_job_stage(job_id, "downloading")

    job = state.get_job(job_id)
    assert job["current_stage"] == "downloading"
    assert job["state"] == "downloading"


def test_mark_stage_complete_sets_last_stage_completed() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=stage2")
    state.update_job_stage(job_id, "downloading")
    state.mark_stage_complete(job_id, "downloading")

    job = state.get_job(job_id)
    assert job["last_stage_completed"] == "downloading"


def test_mark_job_complete() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=done")
    state.mark_job_complete(job_id)

    job = state.get_job(job_id)
    assert job["state"] == "complete"
    assert job["current_stage"] is None
    assert job["completed_at"]


def test_mark_job_failed() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=fail")
    state.mark_job_failed(job_id, "yt-dlp timed out")

    job = state.get_job(job_id)
    assert job["state"] == "failed"
    assert job["error_message"] == "yt-dlp timed out"
    assert job["failed_at"]


def test_mark_job_cancelled() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=cancel")
    state.mark_job_cancelled(job_id, "video is private")

    job = state.get_job(job_id)
    assert job["state"] == "cancelled"
    assert job["error_message"] == "video is private"
    assert job["cancelled_at"]


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------


def test_add_clip_appends_and_returns_id() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=clips")
    clip_id = state.add_clip(
        job_id,
        {
            "file_path": f"data/jobs/{job_id}/clips/clip_0.mp4",
            "score": 0.78,
            "start_s": 142,
            "end_s": 198,
            "transcript_snippet": "Here's the thing nobody tells you...",
        },
    )

    clips = state.get_clips_for_job(job_id)
    assert len(clips) == 1
    assert clips[0]["id"] == clip_id
    assert clips[0]["job_id"] == job_id
    assert clips[0]["score"] == 0.78
    assert clips[0]["status"] == "ready"
    assert clips[0]["created_at"]


def test_add_clip_multiple_preserves_order() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=multiclip")
    id1 = state.add_clip(job_id, {"start_s": 0, "end_s": 10, "score": 0.5})
    id2 = state.add_clip(job_id, {"start_s": 10, "end_s": 20, "score": 0.6})

    clips = state.get_clips_for_job(job_id)
    assert [c["id"] for c in clips] == [id1, id2]


def test_get_clips_for_job_empty_when_no_clips() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=noclips")
    assert state.get_clips_for_job(job_id) == []


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------


def test_create_post_record_resolves_job_from_clip() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=posts")
    clip_id = state.add_clip(job_id, {"start_s": 0, "end_s": 10, "score": 0.9})

    post_id = state.create_post_record(clip_id, "youtube")

    posts = state.get_posts(job_id)
    assert len(posts) == 1
    assert posts[0]["id"] == post_id
    assert posts[0]["clip_id"] == clip_id
    assert posts[0]["platform"] == "youtube"
    assert posts[0]["status"] == "queued"


def test_create_post_record_raises_for_unknown_clip(isolated_jobs_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        state.create_post_record("nonexistent-clip-id", "tiktok")


def test_mark_post_success() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=postsuccess")
    clip_id = state.add_clip(job_id, {"start_s": 0, "end_s": 10, "score": 0.9})
    post_id = state.create_post_record(clip_id, "tiktok")

    state.mark_post_success(post_id, "https://tiktok.com/@user/video/123")

    posts = state.get_posts(job_id)
    post = next(p for p in posts if p["id"] == post_id)
    assert post["status"] == "success"
    assert post["post_url"] == "https://tiktok.com/@user/video/123"
    assert post["posted_at"]


def test_mark_post_failed() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=postfail")
    clip_id = state.add_clip(job_id, {"start_s": 0, "end_s": 10, "score": 0.9})
    post_id = state.create_post_record(clip_id, "instagram")

    state.mark_post_failed(post_id, "rate limited")

    posts = state.get_posts(job_id)
    post = next(p for p in posts if p["id"] == post_id)
    assert post["status"] == "failed"
    assert post["error"] == "rate limited"


def test_multiple_posts_for_same_clip_tracked_independently() -> None:
    job_id = state.enqueue_job("https://youtube.com/watch?v=multiplatform")
    clip_id = state.add_clip(job_id, {"start_s": 0, "end_s": 10, "score": 0.9})

    yt_post = state.create_post_record(clip_id, "youtube")
    tt_post = state.create_post_record(clip_id, "tiktok")

    state.mark_post_success(yt_post, "https://youtube.com/shorts/abc")
    state.mark_post_failed(tt_post, "auth expired")

    posts = {p["id"]: p for p in state.get_posts(job_id)}
    assert posts[yt_post]["status"] == "success"
    assert posts[tt_post]["status"] == "failed"
