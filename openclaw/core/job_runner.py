"""Job orchestration: polls for queued jobs and drives them through all stages.

``JobRunner`` owns the main processing loop. It is designed to run as a
long-lived process, continuously polling ``state.get_next_queued_job()`` and
routing each job through the ordered pipeline of stages. It supports
resumption: if a job's ``last_stage_completed`` is set, execution picks up
from the next uncompleted stage rather than starting over.

Pipeline stage order (mirrors ``core.job_states.JobState``):
    downloading -> detecting -> editing -> captioning ->
    formatting -> posting -> complete

The detecting stage fetches the YouTube transcript, comments, and audio
signals in parallel, then runs clip detection over all three sources.
Whisper transcription runs per-clip inside the captioning stage (against
the already-edited, already-t=0-relative clip file) using the smaller
``worker.clip_whisper_model`` model.

Per-clip failure isolation: a single clip failing at the editing or
captioning stage is logged and skipped; the rest of the pipeline continues
with the remaining clips. If *every* clip fails at a stage, the job itself
is marked failed. Posting is always best-effort: a posting failure never
fails the job once at least one clip was produced.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from core import state
from core.url_validator import validate_url

logger = logging.getLogger(__name__)

# Path to settings.yaml relative to this file: openclaw/core/job_runner.py -> openclaw/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# Terminal job states — these are excluded from crash-recovery resumption.
_TERMINAL_STATES = {"complete", "failed", "cancelled"}

DEFAULT_JOB_POLL_INTERVAL_S = 5
DEFAULT_MAX_CLIPS = 3
DEFAULT_PLATFORMS = ["youtube", "tiktok", "instagram"]
DEFAULT_MIN_FREE_DISK_GB = 5


def _load_settings() -> dict[str, Any]:
    """Load and return the full parsed contents of ``config/settings.yaml``."""
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
    except OSError:
        return {}
    return settings or {}


class JobRunner:
    """Orchestrates end-to-end processing of queued jobs.

    Responsibilities:
    - Poll for the next queued job in a tight loop (with configurable sleep).
    - Delegate each processing stage to the appropriate ``processing.*`` module.
    - Update job state in ``core.state`` before and after each stage.
    - Handle exceptions per-stage: mark the job failed and continue polling.
    - Resume in-flight jobs from their last completed stage after a crash.
    """

    # Canonical linear order of pipeline stages. Used both to drive
    # ``process()`` and to compute crash-recovery resume points.
    STAGE_ORDER: list[str] = [
        "downloading",
        "detecting",
        "editing",
        "captioning",
        "formatting",
        "posting",
    ]

    def __init__(self) -> None:
        """Initialise the runner, loading worker settings from settings.yaml."""
        settings = _load_settings()
        worker_settings = settings.get("worker", {}) or {}
        general_settings = settings.get("general", {}) or {}
        posting_settings = settings.get("posting", {}) or {}

        self.poll_interval_s: float = worker_settings.get(
            "job_poll_interval_s", DEFAULT_JOB_POLL_INTERVAL_S
        )
        self.default_max_clips: int = general_settings.get(
            "max_clips_per_job", DEFAULT_MAX_CLIPS
        )
        self.platforms: list[str] = posting_settings.get("platforms", DEFAULT_PLATFORMS)
        self.min_free_disk_gb: float = general_settings.get(
            "min_free_disk_gb", DEFAULT_MIN_FREE_DISK_GB
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main polling loop.

        On startup, resumes any in-flight jobs left behind by a crash (any
        job whose state is not terminal). Then continuously polls
        ``state.get_next_queued_job()``; when no jobs are queued, sleeps for
        ``worker.job_poll_interval_s`` before checking again. Runs until
        interrupted (``KeyboardInterrupt`` is caught and triggers a graceful
        shutdown log).
        """
        try:
            for job in self._find_resumable_jobs():
                logger.info(
                    "Resuming in-flight job %s from stage after %r",
                    job["id"],
                    job.get("last_stage_completed"),
                )
                self.process(job)

            while True:
                job = state.get_next_queued_job()
                if job is None:
                    time.sleep(self.poll_interval_s)
                    continue
                self.process(job)
        except KeyboardInterrupt:
            logger.info("JobRunner interrupted, shutting down gracefully.")

    def _find_resumable_jobs(self) -> list[dict]:
        """Return all jobs whose state is not terminal, oldest submitted first.

        These are jobs that were mid-pipeline when the worker process last
        stopped (e.g. crash, kill -9) and need to resume rather than restart.
        """
        if not state.JOBS_DIR.exists():
            return []

        resumable = []
        for job_dir in state.JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                job = state.get_job(job_dir.name)
            except (FileNotFoundError, OSError):
                continue
            if job.get("state") not in _TERMINAL_STATES and job.get("state") != "queued":
                resumable.append(job)

        resumable.sort(key=lambda j: j.get("submitted_at") or "")
        return resumable

    # ------------------------------------------------------------------
    # Per-job processing
    # ------------------------------------------------------------------

    def get_resume_index(self, job: dict) -> int:
        """Return the index into ``STAGE_ORDER`` from which processing should start.

        If ``job["last_stage_completed"]`` is set, returns the index of the
        *next* stage after the completed one. If it is absent, ``None``, or
        unrecognised, returns 0 (start from the beginning).

        Args:
            job: Job metadata dict.

        Returns:
            Integer index into ``STAGE_ORDER``.
        """
        last_completed = job.get("last_stage_completed")
        if not last_completed:
            return 0
        try:
            return self.STAGE_ORDER.index(last_completed) + 1
        except ValueError:
            return 0

    def process(self, job: dict) -> None:
        """Run *job* through the pipeline, resuming from its last completed stage.

        Pre-flight URL validation runs unconditionally on every call (it is
        cheap and idempotent); on rejection the job is cancelled and this
        method returns immediately without touching the stage pipeline.

        Any unhandled exception raised by a *required* stage (downloading,
        detecting, formatting) marks the job failed and returns — it does
        not propagate, so the worker loop keeps running.
        Editing/captioning isolate failures per-clip instead (see module
        docstring). Posting is always best-effort and never fails the job.

        Args:
            job: Job metadata dict as returned by ``state.get_job``.
        """
        job_id = job["id"]

        # ---- Pre-flight (only meaningful the first time through; cheap to
        # re-run on resume since validate_url is idempotent and read-only) ----
        if job.get("state") == "queued" or not job.get("youtube_id"):
            try:
                metadata = validate_url(job["youtube_url"])
            except ValueError as exc:
                logger.warning("Job %s rejected at pre-flight: %s", job_id, exc)
                state.mark_job_cancelled(job_id, reason=str(exc))
                return
            state.update_job_metadata(job_id, **metadata)
            job = state.get_job(job_id)

        try:
            self._run_pipeline(job)
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("Job %s failed during processing", job_id)
            state.mark_job_failed(job_id, str(exc))

    def _run_pipeline(self, job: dict) -> None:
        """Drive *job* through STAGE_ORDER starting at its resume index.

        Distinct from ``process`` so that an early return (e.g. zero clips
        detected, or all clips failing edit) can short-circuit cleanly
        without being mistaken for an unhandled exception by the caller.
        """
        job_id = job["id"]
        start_idx = self.get_resume_index(job)
        formatting_idx = self.STAGE_ORDER.index("formatting")

        video_path: str | None = None
        clip_windows: list[dict] = []
        edited_clips: list[dict] = []
        captioned_clips: list[dict] = []

        # Crash-resume case: if we're resuming from "formatting" already
        # complete (i.e. start_idx is past formatting, headed straight to
        # posting), the persisted clips in clips.json are the full, final
        # output of the formatting stage -- crop_to_vertical/burn_captions/
        # format_clip must NOT be re-invoked, since doing so would re-process
        # clips that are already fully formatted and posted/postable. Skip
        # straight to posting, which reads clips from disk via
        # state.get_clips_for_job.
        if start_idx > formatting_idx:
            if start_idx > self.STAGE_ORDER.index("posting"):
                self._cleanup_job_files(job_id)
                state.mark_job_complete(job_id)
                return
            state.update_job_stage(job_id, "posting")
            self._post_all(job)
            state.mark_stage_complete(job_id, "posting")
            self._cleanup_job_files(job_id)
            state.mark_job_complete(job_id)
            return

        # Crash-resume case: if we're skipping past downloading (already
        # marked complete on disk from a prior run), video_path is gone from
        # memory — later stages (detecting, editing, captioning) still need
        # it.  Re-running download() is safe: yt-dlp skips the download if
        # the file already exists on disk.
        if start_idx > self.STAGE_ORDER.index("downloading"):
            if not self._has_enough_disk_space(job_id):
                return
            video_path = self._download(job)
        if start_idx > self.STAGE_ORDER.index("detecting"):
            clip_windows = self._detect(video_path, job)
            if not clip_windows:
                state.mark_job_failed(job_id, "no clips met the score threshold")
                return
        if start_idx > self.STAGE_ORDER.index("editing"):
            edited_clips = self._edit(job, video_path, clip_windows)
            if not edited_clips:
                state.mark_job_failed(job_id, "all clips failed during editing")
                return
        if start_idx > self.STAGE_ORDER.index("captioning"):
            captioned_clips = self._add_captions(job, edited_clips)
            if not captioned_clips:
                state.mark_job_failed(job_id, "all clips failed during captioning")
                return

        for stage_name in self.STAGE_ORDER[start_idx:]:
            state.update_job_stage(job_id, stage_name)

            if stage_name == "downloading":
                if not self._has_enough_disk_space(job_id):
                    return
                video_path = self._download(job)
            elif stage_name == "detecting":
                clip_windows = self._detect(video_path, job)
                if not clip_windows:
                    state.mark_job_failed(
                        job_id, "no clips met the score threshold"
                    )
                    return
            elif stage_name == "editing":
                edited_clips = self._edit(job, video_path, clip_windows)
                if not edited_clips:
                    state.mark_job_failed(
                        job_id, "all clips failed during editing"
                    )
                    return
            elif stage_name == "captioning":
                captioned_clips = self._add_captions(job, edited_clips)
                if not captioned_clips:
                    state.mark_job_failed(
                        job_id, "all clips failed during captioning"
                    )
                    return
            elif stage_name == "formatting":
                final_clips = self._format_clips(job, captioned_clips)
                if not final_clips:
                    state.mark_job_failed(
                        job_id, "all clips failed during formatting"
                    )
                    return
            elif stage_name == "posting":
                self._post_all(job)

            state.mark_stage_complete(job_id, stage_name)

        self._cleanup_job_files(job_id)
        state.mark_job_complete(job_id)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _has_enough_disk_space(self, job_id: str) -> bool:
        """Check free disk space against ``general.min_free_disk_gb`` before
        starting the downloading stage.

        Uses ``shutil.disk_usage`` against the job data directory's
        filesystem. If free space is below the configured threshold, marks
        the job failed with a clear message and returns ``False`` (the
        caller must not proceed to download). Never raises: any unexpected
        error from ``disk_usage`` itself is logged and treated as "enough
        space" so a misconfigured/odd filesystem never crashes the worker
        process outright.

        Args:
            job_id: The UUID of the job about to start downloading.

        Returns:
            ``True`` if there is enough free disk space to proceed,
            ``False`` if the job was marked failed due to insufficient space.
        """
        check_path = state.JOBS_DIR
        check_path.mkdir(parents=True, exist_ok=True)
        try:
            usage = shutil.disk_usage(check_path)
        except OSError as exc:
            logger.warning(
                "Job %s: disk_usage check failed (%s), proceeding without it",
                job_id,
                exc,
            )
            return True

        free_gb = usage.free / (1024 ** 3)
        if free_gb < self.min_free_disk_gb:
            message = (
                f"Insufficient free disk space: {free_gb:.2f}GB available, "
                f"{self.min_free_disk_gb}GB required (general.min_free_disk_gb)"
            )
            logger.error("Job %s: %s", job_id, message)
            state.mark_job_failed(job_id, message)
            return False

        return True

    def _download(self, job: dict) -> str:
        """Download the source video for *job* and return its local path."""
        from processing.downloader import download

        return download(job)

    def _detect(self, video_path: str, job: dict) -> list[dict]:
        """Fetch transcript, comments, and audio signals in parallel; run clip detection.

        All three data sources are fetched concurrently via a
        ``ThreadPoolExecutor``.  Any individual source failing is treated as
        unavailable (``None`` / ``[]``) rather than aborting the whole job —
        the detector degrades gracefully when any signal is absent.

        Args:
            video_path: Local path to the downloaded source video (needed for
                        audio analysis).
            job:        Job metadata dict; used for ``youtube_id`` and
                        ``options.max_clips``.

        Returns:
            List of ``{"start_s", "end_s", "score"}`` clip-window dicts,
            sorted by score descending.  May be empty.
        """
        import concurrent.futures

        from processing.clip_detector import detect
        from processing.transcript_fetcher import fetch_transcript
        from processing.comments import fetch_comments
        from processing.audio_analyzer import AudioSignals

        youtube_id = job.get("youtube_id")

        transcript: list[dict] | None = None
        comments: list[dict] = []
        audio: AudioSignals | None = None

        def _fetch_audio() -> AudioSignals | None:
            try:
                return AudioSignals.from_file(video_path)
            except (RuntimeError, ModuleNotFoundError, Exception) as exc:
                logger.warning(
                    "Job %s: audio analysis failed, continuing without audio signals: %s",
                    job["id"],
                    exc,
                )
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_transcript = executor.submit(fetch_transcript, youtube_id)
            future_comments = executor.submit(fetch_comments, youtube_id)
            future_audio = executor.submit(_fetch_audio)

            transcript = future_transcript.result()
            comments = future_comments.result()
            audio = future_audio.result()

        logger.info(
            "Job %s: detect stage — transcript=%s, comments=%d, audio=%s",
            job["id"],
            "found" if transcript else "not found",
            len(comments),
            "ok" if audio is not None else "unavailable",
        )

        max_clips = (job.get("options") or {}).get("max_clips", self.default_max_clips)
        return detect(transcript, comments=comments, audio_signals=audio, max_clips=max_clips)

    def _edit(
        self, job: dict, video_path: str, clip_windows: list[dict]
    ) -> list[dict]:
        """Crop each detected window to 9:16 vertical format.

        Per plan section 11: a single clip's edit failure is logged and the
        clip is skipped; the rest continue. Returns the list of clip dicts
        (each carrying its original window metadata plus ``edited_path``)
        that succeeded.
        """
        from processing.editor import crop_to_vertical

        job_id = job["id"]
        clips_dir = state.JOBS_DIR / job_id / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        edited: list[dict] = []
        for idx, window in enumerate(clip_windows):
            output_path = str(clips_dir / f"edited_{idx}.mp4")
            try:
                crop_to_vertical(
                    video_path,
                    window["start_s"],
                    window["end_s"],
                    output_path,
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
                logger.error(
                    "Job %s: clip %d failed during editing, skipping: %s",
                    job_id,
                    idx,
                    exc,
                )
                continue
            edited.append({**window, "index": idx, "edited_path": output_path})

        return edited

    def _add_captions(self, job: dict, edited_clips: list[dict]) -> list[dict]:
        """Transcribe each edited clip with Whisper and burn captions into it.

        Each edited clip file is already t=0-relative (it starts at second 0),
        so the segments returned by ``transcribe`` can be passed directly to
        ``burn_captions`` without any slice/rebase step.  Uses the smaller
        ``worker.clip_whisper_model`` (default ``"base"``) rather than the
        full-video model, keeping per-clip transcription fast.

        Same per-clip failure isolation as ``_edit``: a single clip failing
        transcription or caption burning is logged and skipped; the remaining
        clips continue.

        Args:
            job:         Job metadata dict.
            edited_clips: List of clip dicts carrying ``edited_path`` and
                          ``index`` (as returned by ``_edit``).

        Returns:
            List of clip dicts augmented with ``captioned_path`` and
            ``transcript_snippet``.
        """
        from processing.transcriber import transcribe
        from processing.caption_burner import burn_captions

        job_id = job["id"]
        clips_dir = state.JOBS_DIR / job_id / "clips"

        # Read the clip-level Whisper model once for all clips in this job.
        settings = _load_settings()
        clip_model: str = (settings.get("worker") or {}).get(
            "clip_whisper_model", "base"
        )

        captioned: list[dict] = []
        for clip in edited_clips:
            idx = clip["index"]
            output_path = str(clips_dir / f"captioned_{idx}.mp4")
            try:
                # The edited clip already starts at t=0, so Whisper segments
                # are clip-relative by construction — no rebasing needed.
                clip_segments = transcribe(clip["edited_path"], model_name=clip_model)
                burn_captions(clip["edited_path"], clip_segments, output_path)
            except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
                logger.error(
                    "Job %s: clip %d failed during captioning, skipping: %s",
                    job_id,
                    idx,
                    exc,
                )
                continue
            captioned.append(
                {
                    **clip,
                    "captioned_path": output_path,
                    "transcript_snippet": " ".join(
                        s.get("text", "") for s in clip_segments
                    ).strip(),
                }
            )

        return captioned

    def _format_clips(self, job: dict, captioned_clips: list[dict]) -> list[dict]:
        """Final-encode each captioned clip and persist successful ones via state.add_clip.

        Same per-clip failure isolation as the prior stages. For every clip
        that survives all three processing stages, calls ``state.add_clip``
        with the final file path, score, time range, and transcript snippet.

        Returns:
            List of persisted clip dicts (the dicts returned by
            ``state.get_clips_for_job``-shaped records, augmented with the
            in-memory ``id`` just assigned).
        """
        from processing.formatter import format_clip

        job_id = job["id"]
        clips_dir = state.JOBS_DIR / job_id / "clips"

        # Crash-resume safety: if a prior run already persisted a clip for
        # this (start_s, end_s) window (or file_path) before crashing
        # mid-formatting -- i.e. state.add_clip() succeeded but
        # mark_stage_complete(job_id, "formatting") never got written --
        # re-running this stage from scratch must not re-add it. Re-adding
        # would create a duplicate clip record and lead to the clip being
        # posted twice during the posting stage.
        existing_clips = state.get_clips_for_job(job_id)
        existing_windows = {
            (existing.get("start_s"), existing.get("end_s"))
            for existing in existing_clips
        }
        existing_paths = {existing.get("file_path") for existing in existing_clips}

        final_clips: list[dict] = []
        for clip in captioned_clips:
            idx = clip["index"]
            output_path = str(clips_dir / f"clip_{idx}.mp4")

            window_key = (clip.get("start_s"), clip.get("end_s"))
            if window_key in existing_windows or output_path in existing_paths:
                logger.info(
                    "Job %s: clip %d already persisted (resume), skipping re-add",
                    job_id,
                    idx,
                )
                matching = next(
                    (
                        existing
                        for existing in existing_clips
                        if (existing.get("start_s"), existing.get("end_s")) == window_key
                        or existing.get("file_path") == output_path
                    ),
                    None,
                )
                if matching is not None:
                    final_clips.append(matching)
                continue

            try:
                format_clip(clip["captioned_path"], output_path)
            except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
                logger.error(
                    "Job %s: clip %d failed during formatting, skipping: %s",
                    job_id,
                    idx,
                    exc,
                )
                continue

            clip_record = {
                "file_path": output_path,
                "score": clip.get("score"),
                "start_s": clip.get("start_s"),
                "end_s": clip.get("end_s"),
                "transcript_snippet": clip.get("transcript_snippet", ""),
            }
            clip_id = state.add_clip(job_id, clip_record)
            final_clips.append({**clip_record, "id": clip_id})

        return final_clips

    def _post_all(self, job: dict) -> None:
        """Best-effort auto-post every persisted clip to every configured platform.

        For each clip/platform pair, creates a post record up front, then
        attempts the platform-specific upload. Any failure (including the
        ``NotImplementedError`` raised by the current posting/* stubs) is
        caught and recorded via ``state.mark_post_failed`` — a posting
        failure must never fail the job.
        """
        from posting.instagram import post_to_instagram
        from posting.tiktok import post_to_tiktok
        from posting.youtube import post_to_youtube

        platform_fns = {
            "youtube": post_to_youtube,
            "tiktok": post_to_tiktok,
            "instagram": post_to_instagram,
        }

        job_id = job["id"]
        clips = state.get_clips_for_job(job_id)

        for clip in clips:
            clip_id = clip["id"]
            for platform in self.platforms:
                post_id = state.create_post_record(clip_id, platform)
                post_fn = platform_fns.get(platform)
                if post_fn is None:
                    state.mark_post_failed(post_id, f"Unknown platform: {platform!r}")
                    continue
                try:
                    post_url = post_fn(clip)
                except Exception as exc:  # noqa: BLE001 - posting is best-effort
                    logger.warning(
                        "Job %s: posting clip %s to %s failed: %s",
                        job_id,
                        clip_id,
                        platform,
                        exc,
                    )
                    state.mark_post_failed(post_id, str(exc))
                    continue
                state.mark_post_success(post_id, post_url)

    def _cleanup_job_files(self, job_id: str) -> None:
        """Delete raw downloads and intermediate per-clip files once a job
        is done, keeping only the final ``clip_*.mp4`` outputs referenced in
        ``clips.json``.

        Removes everything under ``data/jobs/<job_id>/raw/`` (the original
        downloaded source video, no longer needed post-pipeline) and any
        ``edited_*.mp4`` / ``captioned_*.mp4`` intermediate files left behind
        in ``data/jobs/<job_id>/clips/`` by the editing/captioning stages.
        Final ``clip_*.mp4`` files (the ones persisted in clips.json via
        ``state.add_clip``) are left untouched.

        Best-effort: any OSError while deleting a given file/dir is logged
        and skipped rather than propagated, so cleanup failures never fail
        an otherwise-successful job.
        """
        job_dir = state.JOBS_DIR / job_id
        raw_dir = job_dir / "raw"
        clips_dir = job_dir / "clips"

        if raw_dir.exists():
            for raw_file in raw_dir.iterdir():
                try:
                    if raw_file.is_file():
                        raw_file.unlink()
                    elif raw_file.is_dir():
                        shutil.rmtree(raw_file)
                except OSError as exc:
                    logger.warning(
                        "Job %s: failed to remove raw file %s during cleanup: %s",
                        job_id,
                        raw_file,
                        exc,
                    )

        if clips_dir.exists():
            for pattern in ("edited_*.mp4", "captioned_*.mp4"):
                for intermediate_file in clips_dir.glob(pattern):
                    try:
                        intermediate_file.unlink()
                    except OSError as exc:
                        logger.warning(
                            "Job %s: failed to remove intermediate file %s during cleanup: %s",
                            job_id,
                            intermediate_file,
                            exc,
                        )
