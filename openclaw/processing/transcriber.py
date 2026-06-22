"""Audio transcription stage using mlx-whisper.

Extracts audio from the downloaded video and runs mlx-whisper to produce
word- and segment-level transcription data. The resulting segments are the
primary input for the clip detection stage.
"""

from __future__ import annotations


def transcribe(job: dict, video_path: str) -> list[dict]:
    """Transcribe *video_path* using mlx-whisper and return timed segments.

    Audio is extracted via ffmpeg to a temporary WAV file before being passed
    to mlx-whisper. The Whisper model instance is expected to already be
    loaded (held by ``JobRunner``) and passed implicitly via the module-level
    model reference set during runner initialisation.

    Args:
        job: Job metadata dict. Used to resolve the model from config and for
             logging context (``job["id"]``).
        video_path: Absolute path to the raw downloaded video file.

    Returns:
        List of segment dicts, each with the shape::

            {
                "start": float,   # segment start time in seconds
                "end": float,     # segment end time in seconds
                "text": str,      # transcribed text for the segment
            }

    Raises:
        RuntimeError: If audio extraction or mlx-whisper inference fails.
    """
    raise NotImplementedError
