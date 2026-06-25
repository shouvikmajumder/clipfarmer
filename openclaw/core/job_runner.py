"""Job orchestration: polls for queued jobs, downloads them, detects clips,
and edits each detected clip to vertical format.

``JobRunner`` owns the main processing loop. It is designed to run as a
long-lived process, continuously polling ``state.get_next_queued_job()`` and
driving each job through the full four-stage pipeline:

    queued -> downloading -> detecting -> editing -> captioning -> complete

Stage summary:

- **downloading**: validates the URL, checks disk space, and downloads the
  source video to ``data/jobs/<job_id>/raw/<youtube_id>.mp4`` via
  ``processing.downloader``.

- **detecting**: fetches the transcript (``processing.transcript_fetcher``),
  comments (``processing.comments``), and audio signals
  (``processing.audio_analyzer``) concurrently, then calls
  ``processing.clip_detector.detect`` to score and select clip windows.
  Results are persisted to ``data/jobs/<job_id>/clips.json`` via
  ``core.state``.

- **editing**: crops each detected clip window to 9:16 vertical format using
  ``processing.editor.edit_clip``. Clips are written to
  ``data/jobs/<job_id>/clips/clip_<idx>_edited.mp4`` and the ``file_path``
  field on each clip record is updated in ``clips.json``. The editing step is
  idempotent: already-rendered output files are skipped on crash-resume.

- **captioning**: burns word-level captions onto each edited clip using
  ``processing.transcriber.transcribe_words`` (per-clip Whisper) and
  ``processing.caption_burner``. The ``gaming`` and ``irl`` content profiles
  skip captioning entirely. Captioned clips are written to
  ``data/jobs/<job_id>/clips/clip_<idx>_captioned.mp4`` and ``file_path`` is
  updated. Idempotent: already-captioned outputs are skipped on crash-resume.

Resumption is supported at all stage boundaries: if a crash left
``last_stage_completed`` at any earlier stage, the runner skips the completed
stages and resumes from the next one.

Jobs are processed one at a time, FIFO, oldest ``submitted_at`` first.
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
    """Orchestrates download and clip-detection processing of queued jobs.

    Responsibilities:
    - Poll for the next queued job in a tight loop (with configurable sleep).
    - Pre-flight validate the job's URL before downloading.
    - Check available disk space before starting a download.
    - Delegate the download to ``processing.downloader.download``.
    - Fetch transcript, comments, and audio signals concurrently, then run
      ``processing.clip_detector.detect`` and persist resulting clip windows.
    - Update job state in ``core.state`` before and after each stage.
    - Handle exceptions: mark the job failed and keep polling.
    - Resume in-flight jobs after a crash: if a stage already completed before
      the crash, skip it and continue from the next stage.
    """

    # Canonical four-stage pipeline order. Used to compute crash-recovery
    # resume points.
    STAGE_ORDER: list[str] = ["downloading", "detecting", "editing", "captioning"]

    def __init__(self) -> None:
        """Initialise the runner, loading worker settings from settings.yaml.

        Drives each queued job through the full pipeline:
        downloading -> detecting -> editing -> complete.
        """
        settings = _load_settings()
        worker_settings = settings.get("worker", {}) or {}
        general_settings = settings.get("general", {}) or {}

        self.poll_interval_s: float = worker_settings.get(
            "job_poll_interval_s", DEFAULT_JOB_POLL_INTERVAL_S
        )
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
        unrecognised, returns 0 (start from the beginning). With
        ``STAGE_ORDER == ["downloading"]`` this returns either 0 (not yet
        downloaded) or 1 (download already completed before a crash).

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
        """Run *job* through pre-flight validation, downloading, clip detection,
        and clip editing.

        Pre-flight URL validation runs unconditionally the first time
        through (it is cheap and idempotent); on rejection the job is
        cancelled and this method returns immediately.

        Crash-resume: if all stages already completed before a prior crash
        (``get_resume_index`` is past the last stage), the job is marked
        complete directly without repeating any work.

        Any unhandled exception inside the stage loop marks the job failed
        and returns -- it does not propagate, so the worker loop keeps
        running.

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

        # ---- Crash-resume: all stages already completed before a crash ----
        if self.get_resume_index(job) >= len(self.STAGE_ORDER):
            state.mark_job_complete(job_id)
            return

        try:
            resume_index = self.get_resume_index(job)
            video_path: str | None = None

            # --- downloading ---
            if resume_index <= self.STAGE_ORDER.index("downloading"):
                if not self._has_enough_disk_space(job_id):
                    return
                state.update_job_stage(job_id, "downloading")
                video_path = self._download(job)
                state.mark_stage_complete(job_id, "downloading")

            # --- detecting ---
            if resume_index <= self.STAGE_ORDER.index("detecting"):
                if video_path is None:  # crash-resume into detecting
                    video_path = self._locate_video(job)
                    if video_path is None:
                        state.mark_job_failed(
                            job_id, "downloaded video not found for detection"
                        )
                        return
                state.update_job_stage(job_id, "detecting")
                self._detect(video_path, job)
                state.mark_stage_complete(job_id, "detecting")

            # --- editing ---
            if resume_index <= self.STAGE_ORDER.index("editing"):
                if video_path is None:  # crash-resume into editing
                    video_path = self._locate_video(job)
                    if video_path is None:
                        state.mark_job_failed(
                            job_id, "downloaded video not found for editing"
                        )
                        return
                state.update_job_stage(job_id, "editing")
                self._edit(video_path, job)
                state.mark_stage_complete(job_id, "editing")

            # --- captioning ---
            # Operates on the edited clip files (clip["file_path"]); needs no
            # source video, so no _locate_video step here.
            if resume_index <= self.STAGE_ORDER.index("captioning"):
                state.update_job_stage(job_id, "captioning")
                self._caption(job)
                state.mark_stage_complete(job_id, "captioning")

            state.mark_job_complete(job_id)

        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("Job %s failed", job_id)
            state.mark_job_failed(job_id, str(exc))

    # ------------------------------------------------------------------
    # Stage implementation
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

    def _locate_video(self, job: dict) -> str | None:
        """Locate the already-downloaded video file for *job*.

        Tries the deterministic path ``raw/<youtube_id>.mp4`` first, then
        falls back to the first sorted match of ``raw/*.*``. Returns ``None``
        if neither yields an existing file. Never raises.

        Args:
            job: Job metadata dict (must contain ``"id"`` and optionally
                 ``"youtube_id"``).

        Returns:
            Absolute path string to the video file, or ``None`` if not found.
        """
        raw_dir = state.JOBS_DIR / job["id"] / "raw"
        youtube_id = job.get("youtube_id")

        if youtube_id:
            candidate = raw_dir / f"{youtube_id}.mp4"
            if candidate.exists():
                return str(candidate)

        try:
            matches = sorted(raw_dir.glob("*.*"))
        except OSError:
            return None

        for path in matches:
            if path.is_file():
                return str(path)

        return None

    def _detect(self, video_path: str, job: dict) -> list[dict]:
        """Run clip detection for *job* using *video_path* as the source file.

        Fetches transcript, comments, and audio signals concurrently, then
        calls ``processing.clip_detector.detect`` to score and select clip
        windows. Results are persisted idempotently to ``clips.json`` via
        ``core.state``.

        A run that produces zero clips is treated as a valid completion (a
        warning is logged but the job is not failed).

        Args:
            video_path: Absolute path to the downloaded video file.
            job: Job metadata dict.

        Returns:
            List of clip dicts returned by ``detect``, possibly empty.
        """
        import concurrent.futures

        from processing.clip_detector import detect
        from processing.transcript_fetcher import fetch_transcript
        from processing.comments import fetch_comments
        from processing.audio_analyzer import AudioSignals

        youtube_id = job.get("youtube_id")

        def _audio() -> "AudioSignals | None":
            try:
                return AudioSignals.from_file(video_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Job %s: audio analysis failed, continuing without audio: %s",
                    job["id"],
                    exc,
                )
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_transcript = executor.submit(fetch_transcript, youtube_id)
            f_comments = executor.submit(fetch_comments, youtube_id)
            f_audio = executor.submit(_audio)

            transcript = f_transcript.result()
            comments = f_comments.result()
            audio = f_audio.result()

        logger.info(
            "Job %s: detect — transcript=%s, comments=%d, audio=%s",
            job["id"],
            "found" if transcript else "none",
            len(comments),
            "ok" if audio is not None else "unavailable",
        )

        max_clips = (job.get("options") or {}).get("max_clips")

        clips = detect(transcript, comments=comments, audio_signals=audio, max_clips=max_clips)

        # Persist idempotently: clear first so a crash-resumed re-run never
        # duplicates records, then append each clip individually.
        state.save_clips(job["id"], [])

        for c in clips:
            record: dict = {
                "start_s": c["start_s"],
                "end_s": c["end_s"],
                "score": c["score"],
            }

            if transcript:
                start_s = c["start_s"]
                end_s = c["end_s"]
                snippet_parts = [
                    seg.get("text", "").strip()
                    for seg in transcript
                    if (seg.get("start") or 0.0) < end_s
                    and (seg.get("end") or 0.0) > start_s
                ]
                snippet = " ".join(p for p in snippet_parts if p)
                record["transcript_snippet"] = snippet[:200]

            state.add_clip(job["id"], record)

        if not clips:
            logger.warning(
                "Job %s: detection produced no clips above threshold", job["id"]
            )

        return clips

    def _edit(self, video_path: str, job: dict) -> list[dict]:
        """Crop each detected clip to 9:16 vertical format using the configured
        editing profile.

        Reads clip windows from ``clips.json``, renders each one via
        ``processing.editor.edit_clip``, writes outputs to
        ``data/jobs/<job_id>/clips/clip_<idx>_edited.mp4``, and persists the
        updated ``file_path`` values back to ``clips.json``.

        A job with zero detected clips is treated as a valid completion; a
        warning is logged and an empty list is returned.

        Per-clip isolation: a failure on one clip is logged and skipped; the
        remaining clips are still processed. If *all* clips fail, a
        ``RuntimeError`` is raised so the ``process()`` wrapper marks the job
        failed.

        Note: ``edit_clip``'s own cache guard makes crash-resumed re-runs
        idempotent — already-rendered output files are detected and skipped
        automatically.

        Args:
            video_path: Absolute path to the downloaded source video.
            job: Job metadata dict.

        Returns:
            List of clip dicts that were successfully edited (``file_path``
            set), possibly empty if there were no clips to process.

        Raises:
            RuntimeError: If there were clips to edit but every one failed.
        """
        # Lazy import keeps the top-level module importable even when ffmpeg is
        # not installed (tests that mock edit_clip can still import job_runner).
        from processing.editor import edit_clip

        job_id = job["id"]
        clips = state.get_clips_for_job(job_id)

        if not clips:
            logger.info("Job %s: no clips to edit", job_id)
            return []

        # Resolve editing config from settings.yaml, falling back to safe defaults.
        settings = _load_settings()
        editing_cfg = settings.get("editing", {}) or {}
        default_profile = editing_cfg.get("default_profile", "default")
        gaming_cfg = settings.get("gaming", {}) or {}
        facecam_region = gaming_cfg.get("facecam_region")

        opts = job.get("options") or {}
        profile = opts.get("content_profile", default_profile)
        crop_bias = opts.get("crop_bias", "center")

        clips_dir = state.JOBS_DIR / job_id / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        succeeded: list[dict] = []

        for idx, clip in enumerate(clips):
            output_path = str(clips_dir / f"clip_{idx}_edited.mp4")
            try:
                edit_clip(
                    video_path,
                    clip["start_s"],
                    clip["end_s"],
                    output_path,
                    profile=profile,
                    crop_bias=crop_bias,
                    facecam_region=facecam_region,
                )
                clip["file_path"] = output_path
                succeeded.append(clip)
            except Exception as exc:  # noqa: BLE001 - per-clip isolation
                logger.error(
                    "Job %s: clip %d failed during editing, skipping: %s",
                    job_id,
                    idx,
                    exc,
                )

        # Persist updated file_path values for succeeded clips (clips list was
        # mutated in place, so failed clips retain their original state too).
        state.save_clips(job_id, clips)

        if clips and not succeeded:
            raise RuntimeError("all clips failed during editing")

        return succeeded

    def _caption(self, job: dict) -> list[dict]:
        """Burn word-level captions onto each edited clip.

        Captions come from per-clip Whisper word timestamps
        (``transcribe_words``) run on the edited clip file, which already
        starts at t=0 — so no transcript cache or offset is needed. The
        ``gaming`` and ``irl`` content profiles skip captioning entirely
        (``caption_burner.should_caption``), in which case this is a no-op and
        each clip keeps its edited ``file_path``.

        For captioned profiles, each clip's ``file_path`` (its edited mp4) is
        captioned to ``clips/clip_<idx>_captioned.mp4`` and ``file_path`` is
        updated. Per-clip failures are isolated; if every eligible clip fails
        the job is failed. ``caption_clip``'s cache guard keeps a crash-resumed
        re-run idempotent.

        Args:
            job: Job metadata dict.

        Returns:
            List of clip dicts that were successfully captioned (possibly
            empty: no clips, or a skipped profile).
        """
        from processing.caption_burner import caption_clip, should_caption
        from processing.transcriber import transcribe_words

        job_id = job["id"]

        settings = _load_settings()
        default_profile = (settings.get("editing") or {}).get("default_profile", "default")
        profile = (job.get("options") or {}).get("content_profile", default_profile)

        if not should_caption(profile):
            logger.info(
                "Job %s: content profile %r skips captioning", job_id, profile
            )
            return []

        clips = state.get_clips_for_job(job_id)
        if not clips:
            logger.info("Job %s: no clips to caption", job_id)
            return []

        clips_dir = state.JOBS_DIR / job_id / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        succeeded: list[dict] = []
        eligible = 0
        for idx, clip in enumerate(clips):
            edited_path = clip.get("file_path")
            if not edited_path:
                # Editing skipped this clip — nothing to caption.
                continue
            eligible += 1
            srt_path = str(clips_dir / f"clip_{idx}.srt")
            output_path = str(clips_dir / f"clip_{idx}_captioned.mp4")
            try:
                words = transcribe_words(edited_path)
                caption_clip(edited_path, words, srt_path, output_path, offset_s=0.0)
            except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
                logger.error(
                    "Job %s: clip %d failed during captioning, skipping: %s",
                    job_id,
                    idx,
                    exc,
                )
                continue
            clip["file_path"] = output_path
            succeeded.append(clip)

        # Persist updated file_path values (clips list mutated in place).
        state.save_clips(job_id, clips)

        if eligible and not succeeded:
            raise RuntimeError("all clips failed during captioning")

        return succeeded
