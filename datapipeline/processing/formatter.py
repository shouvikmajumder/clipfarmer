"""Final formatting stage: enforce duration and file-size limits.

Performs the last encode pass before posting:
1. Probes duration; if it exceeds *max_duration_s*, trims to exactly
   *max_duration_s* from the start.
2. Encodes (or re-encodes) to H.264/AAC MP4. If the resulting file exceeds
   *max_size_mb*, iteratively lowers the target bitrate and re-encodes,
   capped at ``MAX_SIZE_RETRIES`` attempts, raising if still over the limit
   after all retries are exhausted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Path to settings.yaml relative to this file: datapipeline/processing/formatter.py -> datapipeline/config
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

DEFAULT_MAX_CLIP_LENGTH_S = 60
MAX_SIZE_RETRIES = 3

# Bitrate (kbps) used on the first encode attempt; halved (down to a floor)
# on each retry that still exceeds max_size_mb.
INITIAL_VIDEO_BITRATE_KBPS = 4000
MIN_VIDEO_BITRATE_KBPS = 500
AUDIO_BITRATE_KBPS = 128


def _load_max_clip_length_s() -> int:
    """Read ``general.max_clip_length_s`` from ``config/settings.yaml``.

    Falls back to ``DEFAULT_MAX_CLIP_LENGTH_S`` if the file or key is missing.
    """
    try:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
    except OSError:
        return DEFAULT_MAX_CLIP_LENGTH_S

    if not settings:
        return DEFAULT_MAX_CLIP_LENGTH_S

    return settings.get("general", {}).get("max_clip_length_s", DEFAULT_MAX_CLIP_LENGTH_S)


def _probe_duration_s(path: str) -> float:
    """Return the duration in seconds of the media file at *path*.

    Raises:
        RuntimeError: If ffprobe fails.
    """
    import ffmpeg

    try:
        probe = ffmpeg.probe(path)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffprobe failed for {path!r}: {stderr}") from exc

    return float(probe["format"]["duration"])


def _encode(input_path: str, output_path: str, video_bitrate_kbps: int, duration_s: float | None) -> None:
    """Run a single ffmpeg encode pass, optionally trimming to *duration_s*.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    import ffmpeg

    kwargs = {
        "vcodec": "libx264",
        "acodec": "aac",
        "video_bitrate": f"{video_bitrate_kbps}k",
        "audio_bitrate": f"{AUDIO_BITRATE_KBPS}k",
        "movflags": "faststart",
    }
    if duration_s is not None:
        kwargs["t"] = duration_s

    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, **kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg final encode failed for {input_path!r}: {stderr}") from exc


def format_clip(
    input_path: str,
    output_path: str,
    max_duration_s: int | None = None,
    max_size_mb: int = 50,
) -> str:
    """Final-encode *input_path* enforcing duration and size limits.

    Args:
        input_path: Absolute path to the source (cropped + captioned) clip.
        output_path: Absolute path to write the final encoded clip to.
        max_duration_s: Maximum allowed clip duration in seconds. If
            ``None``, read from ``config/settings.yaml``
            (``general.max_clip_length_s``, default 60).
        max_size_mb: Maximum allowed output file size in megabytes
            (default 50).

    Returns:
        *output_path*, unchanged, for convenient chaining.

    Raises:
        RuntimeError: If ffmpeg/ffprobe fails, or if the output still
            exceeds *max_size_mb* after ``MAX_SIZE_RETRIES`` bitrate-reduction
            attempts.
    """
    if max_duration_s is None:
        max_duration_s = _load_max_clip_length_s()

    source_duration_s = _probe_duration_s(input_path)
    if source_duration_s <= 0:
        raise RuntimeError(
            f"format_clip: probed source duration for {input_path!r} is "
            f"invalid ({source_duration_s}s) -- refusing to encode a "
            f"degenerate clip"
        )
    trim_duration_s = min(source_duration_s, float(max_duration_s))

    max_size_bytes = max_size_mb * 1024 * 1024

    video_bitrate_kbps = INITIAL_VIDEO_BITRATE_KBPS
    attempts = 0
    while True:
        attempts += 1
        _encode(input_path, output_path, video_bitrate_kbps, trim_duration_s)

        output_size_bytes = Path(output_path).stat().st_size
        if output_size_bytes <= max_size_bytes:
            return output_path

        if attempts >= MAX_SIZE_RETRIES:
            raise RuntimeError(
                f"format_clip: output {output_path!r} still exceeds "
                f"{max_size_mb}MB ({output_size_bytes / (1024 * 1024):.2f}MB) "
                f"after {attempts} encode attempts"
            )

        video_bitrate_kbps = max(MIN_VIDEO_BITRATE_KBPS, video_bitrate_kbps // 2)
