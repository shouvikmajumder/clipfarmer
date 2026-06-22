"""Job orchestration: polls for queued jobs and drives them through all stages.

``JobRunner`` owns the main processing loop. It is designed to run as a
long-lived process, continuously polling ``state.get_next_queued_job()`` and
routing each job through the ordered pipeline of stages. It supports
resumption: if a job's ``last_stage_completed`` is set, execution picks up
from the next uncompleted stage rather than starting over.
"""

from __future__ import annotations


class JobRunner:
    """Orchestrates end-to-end processing of queued jobs.

    Responsibilities:
    - Load the Whisper model once at startup to avoid reloading between jobs.
    - Poll for the next queued job in a tight loop (with configurable sleep).
    - Delegate each processing stage to the appropriate ``processing.*`` module.
    - Update job state in ``core.state`` before and after each stage.
    - Handle exceptions per-stage: mark the job failed and continue polling.
    """

    # Ordered list of (stage_name, private_method) pairs that define the
    # pipeline. The order here is the canonical execution order.
    STAGES: list[tuple[str, str]] = [
        ("downloading", "_download"),
        ("transcribing", "_transcribe"),
        ("detecting", "_detect_clips"),
        ("editing", "_edit"),
        ("captioning", "_add_captions"),
        ("formatting", "_format_clips"),
        ("posting", "_post_all"),
    ]

    def __init__(self) -> None:
        """Initialise the runner.

        Loads the Whisper model as specified in ``config/settings.yaml`` so
        that it is resident in memory for the lifetime of this process.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Start the main polling loop.

        Continuously calls ``state.get_next_queued_job()``. When a job is
        found, delegates to ``self.process(job)``. When no jobs are queued,
        sleeps for ``worker.job_poll_interval_s`` before checking again.
        Runs until the process is interrupted (KeyboardInterrupt is caught
        cleanly and causes a graceful shutdown log).
        """
        raise NotImplementedError

    def process(self, job: dict) -> None:
        """Run all pipeline stages for *job*, resuming from the last completed stage.

        Iterates ``STAGES`` starting from ``get_resume_index(job)``. For each
        stage, calls ``state.update_job_stage``, invokes the private stage
        method, then calls ``state.mark_stage_complete``. On any unhandled
        exception, calls ``state.mark_job_failed`` and returns early.
        On successful completion of all stages, calls ``state.mark_job_complete``.

        Args:
            job: Job metadata dict as returned by ``state.get_job``.
        """
        raise NotImplementedError

    def get_resume_index(self, job: dict) -> int:
        """Return the index into ``STAGES`` from which processing should start.

        If ``job["last_stage_completed"]`` is set, returns the index of the
        *next* stage after the completed one. If it is absent or ``None``,
        returns 0 (start from the beginning).

        Args:
            job: Job metadata dict.

        Returns:
            Integer index into ``STAGES``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Private stage methods
    # ------------------------------------------------------------------

    def _download(self, job: dict) -> None:
        """Download the source video for *job*.

        Delegates to ``processing.downloader.download`` and stores the
        returned file path back into the job record via ``state``.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError

    def _transcribe(self, job: dict) -> None:
        """Transcribe the downloaded video audio for *job*.

        Delegates to ``processing.transcriber.transcribe`` using the pre-loaded
        Whisper model. Stores the resulting segments via ``state``.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError

    def _detect_clips(self, job: dict) -> None:
        """Run clip detection on the transcript segments for *job*.

        Delegates to ``processing.clip_detector.detect_clips``. Persists
        discovered clips via ``state.save_clips``.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError

    def _edit(self, job: dict) -> None:
        """Crop and trim each detected clip to 9:16 vertical format.

        Iterates ``state.get_clips`` and delegates each to
        ``processing.editor.edit_clip``. Updates each clip record with the
        output path.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError

    def _add_captions(self, job: dict) -> None:
        """Burn captions into each edited clip.

        Delegates to ``processing.caption_burner.burn_captions`` per clip.
        Updates clip records with the captioned output path.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError

    def _format_clips(self, job: dict) -> None:
        """Final encode and size enforcement for each captioned clip.

        Delegates to ``processing.formatter.format_clip`` per clip. Enforces
        the 60-second cap and 50 MB file-size limit. Updates clip records
        with the final output path.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError

    def _post_all(self, job: dict) -> None:
        """Upload all finalised clips to the configured social platforms.

        Iterates the platform list from settings, calls the appropriate
        ``posting.*`` module for each clip/platform pair, and records
        success/failure via ``state.mark_post_success`` /
        ``state.mark_post_failed``.

        Args:
            job: Job metadata dict.
        """
        raise NotImplementedError
