"""Minimal Flask server for the OpenClaw web submission form.

Serves a single static HTML page (``index.html``) and exposes one POST
endpoint, ``/api/jobs``, that accepts a YouTube URL and enqueues it via
``core.state.enqueue_job``.

This module is an *alternative* entry point to the existing CLI
(``python main.py <url>``). It does not implement any job processing logic
itself and is intentionally not wired into ``main.py`` — run it directly:

    cd openclaw
    python interfaces/web/server.py

The import of ``enqueue_job`` is wrapped defensively: ``core/state.py`` is
being implemented in parallel and may not be ready yet. If the import or the
call fails, the endpoint returns a clear 503 error instead of crashing the
server.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

try:
    from core.state import enqueue_job
except ImportError:
    enqueue_job = None

STATIC_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    """Serve the single-page web form."""
    return send_from_directory(STATIC_DIR, "index.html")


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


if __name__ == "__main__":
    app.run(debug=True, port=5050)
