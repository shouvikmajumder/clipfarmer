"""Entry point: python main.py [youtube_url]

Usage:
    python main.py <youtube_url>   # validate, enqueue, then run the worker
    python main.py                 # run the worker only (drain existing queue)

When given a URL, validates it and enqueues a processing job, then starts the
JobRunner loop which processes all queued jobs sequentially. When given no
arguments, skips the enqueue step and starts the JobRunner loop directly so
the worker can run standalone (e.g. alongside the web UI, which enqueues jobs
itself via core.state.enqueue_job).

Must be run with ``openclaw/`` as the working directory (e.g. ``cd openclaw
&& python main.py``), matching the rest of the codebase's import convention.
"""

import sys

from core.url_validator import validate_url
from core import state
from core.job_runner import JobRunner


def parse_args() -> str | None:
    """Parse CLI arguments and return the YouTube URL, if provided.

    Returns:
        The YouTube URL provided as the first positional argument, or
        ``None`` if no argument was given (worker-only mode).
    """
    if len(sys.argv) < 2:
        return None
    return sys.argv[1]


def main() -> None:
    """Main entry point.

    Parses the optional CLI URL argument. If a URL is provided, validates it
    and enqueues a job (exiting non-zero on validation failure rather than
    enqueueing an invalid URL). Either way, instantiates a JobRunner and
    starts the processing loop, which blocks forever.
    """
    youtube_url = parse_args()

    if youtube_url is not None:
        try:
            validate_url(youtube_url)
        except ValueError as exc:
            print(f"Invalid YouTube URL: {exc}", file=sys.stderr)
            sys.exit(1)

        job_id = state.enqueue_job(youtube_url)
        print(f"Enqueued job {job_id}")

    JobRunner().run()


if __name__ == "__main__":
    main()
