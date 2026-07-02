"""Entry point: python main.py [youtube_url] [--download-only]

Usage:
    python main.py <youtube_url>                  # download + detect clips
    python main.py <youtube_url> --download-only  # download only, skip detection
    python main.py                                # run worker only (drain queue)

Must be run with ``datapipeline/`` as the working directory.
"""

import argparse
import sys

from core.url_validator import validate_url
from core import state
from core.job_runner import JobRunner


def parse_args() -> tuple[str | None, bool]:
    parser = argparse.ArgumentParser(description="clipfarmer worker")
    parser.add_argument("url", nargs="?", help="YouTube URL to process")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the video but skip clip detection",
    )
    args = parser.parse_args()
    return args.url, args.download_only


def main() -> None:
    youtube_url, download_only = parse_args()

    if youtube_url is not None:
        try:
            validate_url(youtube_url)
        except ValueError as exc:
            print(f"Invalid YouTube URL: {exc}", file=sys.stderr)
            sys.exit(1)

        options = {"download_only": True} if download_only else {}
        job_id = state.enqueue_job(youtube_url, options=options)
        mode = " (download only)" if download_only else ""
        print(f"Enqueued job {job_id}{mode}")

    JobRunner().run()


if __name__ == "__main__":
    main()
