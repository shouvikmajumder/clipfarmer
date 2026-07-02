"""Minimal Flask server for the clipfarmer.bot web submission form.

Serves the single-page HTML form along with its CSS/JS assets, and exposes
one POST endpoint, ``/api/jobs``, that accepts a YouTube URL and enqueues it
via ``core.state.enqueue_job``.

This module is an *alternative* entry point to the existing CLI
(``python main.py <url>``). It does not implement any job processing logic
itself and is intentionally not wired into ``main.py`` — run it directly:

    cd datapipeline
    python interfaces/web/server.py

The import of ``enqueue_job`` is wrapped defensively in case ``core/state.py``
is ever missing or broken. If the import or the call fails, the endpoint
returns a clear 503 error instead of crashing the server.
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

# Running this file directly (``python interfaces/web/server.py``) puts only
# this file's own directory on sys.path, not the datapipeline/ package root, so
# ``core`` would not be importable regardless of the caller's cwd. Add the
# datapipeline/ root explicitly so the import below always works.
_OPENCLAW_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_OPENCLAW_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENCLAW_ROOT))

try:
    from core.state import enqueue_job
    from core import state
except ImportError:
    enqueue_job = None
    state = None

STATIC_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    """Serve the single-page web form."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_asset(filename):
    """Serve the form's CSS/JS assets (style.css, app.js) alongside index.html.

    Restricted to the known asset filenames so this can't be used as a
    generic directory-traversal-style file server for the rest of the repo.
    """
    if filename not in {"style.css", "app.js", "job.html", "job.js"}:
        return jsonify(error="Not found"), 404
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/jobs", methods=["POST"])
def create_job():
    """Accept a YouTube URL and enqueue it as a new processing job.

    Request body (JSON):
        {"url": "<youtube url>"}

    Responses:
        200 {"job_id": "<uuid>"}                  — job enqueued.
        400 {"error": "..."}                      — missing/invalid input.
        503 {"error": "..."}                      — job queue not available yet.
        500 {"error": "..."}                      — unexpected failure.
    """
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()

    if not url:
        return jsonify(error="A YouTube URL is required."), 400

    if enqueue_job is None:
        return (
            jsonify(
                error="Job queue is not available yet (core.state.enqueue_job "
                "is not implemented). Please try again shortly."
            ),
            503,
        )

    try:
        job_id = enqueue_job(url)
    except NotImplementedError:
        return (
            jsonify(
                error="Job queue is not available yet (core.state.enqueue_job "
                "is not implemented). Please try again shortly."
            ),
            503,
        )
    except Exception as exc:  # noqa: BLE001 - surface any backend failure to the UI
        return jsonify(error=f"Failed to enqueue job: {exc}"), 500

    return jsonify(job_id=job_id), 200


@app.route("/api/jobs/<job_id>")
def get_job_status(job_id):
    """Return the current status and clip count for an existing job.

    Responses:
        200 {"job_id", "state", "current_stage", "last_stage_completed",
             "youtube_id", "video_title", "clip_count"}
        404 {"error": "Job not found"}               — unknown job_id.
        503 {"error": "..."}                          — state module unavailable.
    """
    if state is None:
        return (
            jsonify(
                error="Job queue is not available yet (core.state is not "
                "implemented). Please try again shortly."
            ),
            503,
        )

    try:
        job = state.get_job(job_id)
    except FileNotFoundError:
        return jsonify(error="Job not found"), 404

    clips = state.get_clips(job_id)
    return (
        jsonify(
            job_id=job.get("id"),
            state=job.get("state"),
            current_stage=job.get("current_stage"),
            last_stage_completed=job.get("last_stage_completed"),
            youtube_id=job.get("youtube_id"),
            video_title=job.get("video_title"),
            clip_count=len(clips),
        ),
        200,
    )


@app.route("/jobs/<job_id>")
def job_page(job_id):
    """Serve the job results HTML page."""
    return send_from_directory(STATIC_DIR, "job.html")


@app.route("/jobs/<job_id>/video")
def job_video(job_id):
    """Stream the raw downloaded .mp4 for a job.

    Validates the job exists first so the job_id is confirmed to be a real UUID
    before any filesystem path is constructed. Tries the canonical
    ``raw/<youtube_id>.mp4`` path first, then falls back to the first sorted
    file found under the ``raw/`` directory.

    Responses:
        200 video/mp4 stream
        404 {"error": "Job not found"}      — unknown job_id.
        404 {"error": "Video not found"}    — job exists but no raw video yet.
        503 {"error": "..."}               — state module unavailable.
    """
    if state is None:
        return (
            jsonify(
                error="Job queue is not available yet (core.state is not "
                "implemented). Please try again shortly."
            ),
            503,
        )

    try:
        job = state.get_job(job_id)
    except FileNotFoundError:
        return jsonify(error="Job not found"), 404

    raw_dir = state.JOBS_DIR / job_id / "raw"
    youtube_id = job.get("youtube_id")

    video_path: Path | None = None
    if youtube_id:
        candidate = raw_dir / f"{youtube_id}.mp4"
        if candidate.is_file():
            video_path = candidate

    if video_path is None and raw_dir.is_dir():
        candidates = sorted(raw_dir.glob("*.*"))
        if candidates:
            video_path = candidates[0]

    if video_path is None:
        return jsonify(error="Video not found"), 404

    return send_file(video_path, mimetype="video/mp4", as_attachment=False, conditional=True)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
