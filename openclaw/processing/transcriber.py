"""Audio transcription stage using mlx-whisper.

Runs mlx-whisper directly against the downloaded video file (mlx-whisper
shells out to ffmpeg internally to extract audio) to produce segment-level
transcription data. The resulting segments are the primary input for the
clip detection stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Path to settings.yaml relative to this file: openclaw/processing/transcriber.py -> openclaw/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

DEFAULT_WHISPER_MODEL = "medium"


def _load_whisper_model_name() -> str:
    """Read ``worker.whisper_model`` from ``config/settings.yaml``.

    Falls back to ``DEFAULT_WHISPER_MODEL`` if the file or key is missing.
    """
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
    except OSError:
        return DEFAULT_WHISPER_MODEL

    if not settings:
        return DEFAULT_WHISPER_MODEL

    return settings.get("worker", {}).get("whisper_model", DEFAULT_WHISPER_MODEL)


def transcribe(video_path: str) -> list[dict]:
    """Transcribe *video_path* using mlx-whisper and return timed segments.

    The Whisper model name is read from ``config/settings.yaml``
    (``worker.whisper_model``, default ``"medium"``) rather than hardcoded.

    Args:
        video_path: Absolute path to the raw downloaded video file.

    Returns:
        List of segment dicts, each with at minimum the shape::

            {
                "start": float,   # segment start time in seconds
                "end": float,     # segment end time in seconds
                "text": str,      # transcribed text for the segment
            }

        Additional native mlx-whisper segment fields (e.g. ``id``,
        ``avg_logprob``) are passed through unchanged.

    Raises:
        RuntimeError: If mlx-whisper inference fails.
    """
    import mlx_whisper

    model_name = _load_whisper_model_name()

    try:
        result: dict[str, Any] = mlx_whisper.transcribe(video_path, path_or_hf_repo=model_name)
    except Exception as exc:
        raise RuntimeError(
            f"mlx-whisper transcription failed for {video_path!r} "
            f"(model={model_name!r}): {exc}"
        ) from exc

    segments = result.get("segments") if result else None
    if segments is None:
        raise RuntimeError(
            f"mlx-whisper returned no segments for {video_path!r} (model={model_name!r})"
        )

    return [
        {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", "").strip(),
            **{k: v for k, v in seg.items() if k not in ("start", "end", "text")},
        }
        for seg in segments
    ]
