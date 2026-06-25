"""Audio transcription using mlx-whisper.

Used in two places in the pipeline:

1. **Captioning stage** — transcribes each final edited clip (already
   t=0-relative) to produce caption segments.  The captioning stage passes
   ``model_name`` explicitly (typically ``worker.clip_whisper_model``, a
   smaller/faster model).
2. **Legacy / external callers** — may call ``transcribe(video_path)``
   without a model override, in which case ``worker.whisper_model`` is read
   from ``config/settings.yaml`` as before.

mlx-whisper shells out to ffmpeg internally to extract audio, so the path
may point to any video or audio file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Path to settings.yaml relative to this file: openclaw/processing/transcriber.py -> openclaw/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_CLIP_WHISPER_MODEL = "base"


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


def _load_clip_whisper_model_name() -> str:
    """Read ``worker.clip_whisper_model`` from ``config/settings.yaml``.

    This is the smaller/faster model used when transcribing short final clips
    for caption burning.  Falls back to ``DEFAULT_CLIP_WHISPER_MODEL``
    (``"base"``) if the file or key is missing.
    """
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
    except OSError:
        return DEFAULT_CLIP_WHISPER_MODEL

    if not settings:
        return DEFAULT_CLIP_WHISPER_MODEL

    return settings.get("worker", {}).get("clip_whisper_model", DEFAULT_CLIP_WHISPER_MODEL)


def transcribe_words(video_path: str, model_name: str | None = None) -> list[dict]:
    """Transcribe *video_path* using mlx-whisper and return word-level timestamps.

    Used by the captioning stage to get per-word timing for SRT generation.
    The edited clip starts at t=0, so the returned word times are already
    clip-relative and require no offset adjustment.

    Args:
        video_path:  Absolute path to the video or audio file to transcribe.
        model_name:  Whisper model name/repo to use.  When ``None`` (default),
                     the model is read from ``config/settings.yaml``
                     (``worker.clip_whisper_model``, default ``"base"``).

    Returns:
        List of word dicts, each with the shape::

            {
                "word":  str,    # the word text (stripped)
                "start": float,  # word start time in seconds
                "end":   float,  # word end time in seconds
            }

        Words with missing or None start/end values are skipped.

    Raises:
        RuntimeError: If mlx-whisper inference fails or returns no segments.
    """
    import mlx_whisper

    resolved_model = model_name if model_name is not None else _load_clip_whisper_model_name()

    try:
        result: dict[str, Any] = mlx_whisper.transcribe(
            video_path,
            path_or_hf_repo=resolved_model,
            word_timestamps=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"mlx-whisper word transcription failed for {video_path!r} "
            f"(model={resolved_model!r}): {exc}"
        ) from exc

    segments = result.get("segments") if result else None
    if segments is None:
        raise RuntimeError(
            f"mlx-whisper returned no segments for {video_path!r} (model={resolved_model!r})"
        )

    words: list[dict] = []
    for seg in segments:
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            words.append(
                {
                    "word": str(w.get("word", "")).strip(),
                    "start": float(start),
                    "end": float(end),
                }
            )
    return words


def transcribe(video_path: str, model_name: str | None = None) -> list[dict]:
    """Transcribe *video_path* using mlx-whisper and return timed segments.

    Args:
        video_path:  Absolute path to the video or audio file to transcribe.
        model_name:  Whisper model name/repo to use.  When ``None`` (default),
                     the model is read from ``config/settings.yaml``
                     (``worker.whisper_model``, default ``"medium"``).  Pass an
                     explicit name (e.g. ``"base"``) to override — used by the
                     captioning stage which passes ``worker.clip_whisper_model``.

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

    resolved_model = model_name if model_name is not None else _load_whisper_model_name()

    try:
        result: dict[str, Any] = mlx_whisper.transcribe(video_path, path_or_hf_repo=resolved_model)
    except Exception as exc:
        raise RuntimeError(
            f"mlx-whisper transcription failed for {video_path!r} "
            f"(model={resolved_model!r}): {exc}"
        ) from exc

    segments = result.get("segments") if result else None
    if segments is None:
        raise RuntimeError(
            f"mlx-whisper returned no segments for {video_path!r} (model={resolved_model!r})"
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
