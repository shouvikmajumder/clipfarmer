"""Video editing stage: trim and crop source footage to 9:16 vertical format.

Uses ``ffmpeg-python`` (lazily imported) to first trim the clip to
``[start_s, end_s]``, then crop/scale it to a 9:16 vertical frame. ``ffprobe``
is run first to check the input's aspect ratio: if it is already
approximately 9:16 (within ``ASPECT_TOLERANCE``), the crop step is skipped and
the source is simply trimmed + rescaled to the target resolution. Otherwise a
horizontal center-crop is applied before scaling (v1 — no face detection;
that is a stretch goal per the build plan).

On any ffmpeg/ffprobe failure this module raises; it does not retry or fall
back internally. ``core/job_runner.py`` is responsible for the
"skip this clip, continue others" retry/fallback policy described in
``plan_v4_trimmed.md`` section 11.
"""

from __future__ import annotations

# Target vertical resolution (9:16).
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT  # 0.5625

# How far off 9:16 the source aspect ratio may be before we still treat it as
# "already vertical" and skip the smart/center-crop step.
ASPECT_TOLERANCE = 0.02


def _probe_dimensions(input_path: str) -> tuple[int, int]:
    """Return ``(width, height)`` of the first video stream in *input_path*.

    Raises:
        RuntimeError: If ffprobe fails or no video stream is found.
    """
    import ffmpeg

    try:
        probe = ffmpeg.probe(input_path)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffprobe failed for {input_path!r}: {stderr}") from exc

    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"No video stream found in {input_path!r}")

    stream = video_streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    return width, height


def crop_to_vertical(input_path: str, start_s: float, end_s: float, output_path: str) -> str:
    """Trim *input_path* to ``[start_s, end_s]`` and crop/scale it to 9:16.

    Contract:
    1. Probe the input's video stream dimensions via ffprobe.
    2. If the source aspect ratio is already within ``ASPECT_TOLERANCE`` of
       9:16, skip cropping — just trim and rescale to ``1080x1920``.
    3. Otherwise, apply a horizontal center-crop (the widest centered region
       whose aspect ratio is exactly 9:16 relative to the source height),
       then scale to ``1080x1920``.
    4. Write the result to *output_path*, overwriting if it exists.

    Args:
        input_path: Absolute path to the source video file.
        start_s: Clip start time in seconds (within the source video).
        end_s: Clip end time in seconds (within the source video).
        output_path: Absolute path to write the cropped/trimmed output to.

    Returns:
        *output_path*, unchanged, for convenient chaining.

    Raises:
        ValueError: If ``end_s <= start_s``.
        RuntimeError: If ffprobe or ffmpeg fails (e.g. corrupt input, ffmpeg
            binary missing, unsupported codec). Callers are responsible for
            any retry/fallback policy — this function does not swallow
            errors.
    """
    import ffmpeg

    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")

    width, height = _probe_dimensions(input_path)
    source_aspect = width / height
    duration = end_s - start_s

    stream = ffmpeg.input(input_path, ss=start_s, t=duration)

    if abs(source_aspect - TARGET_ASPECT) <= ASPECT_TOLERANCE:
        # Already ~9:16 — skip smart/center crop, just rescale.
        video = stream.video.filter("scale", TARGET_WIDTH, TARGET_HEIGHT)
    else:
        # Center-crop to a 9:16 region of the source, then scale.
        crop_width = int(height * TARGET_ASPECT)
        if crop_width > width:
            # Source is narrower than 9:16 (unusually tall) — crop height instead.
            crop_height = int(width / TARGET_ASPECT)
            crop_height = min(crop_height, height)
            y_offset = max(0, (height - crop_height) // 2)
            video = stream.video.filter("crop", width, crop_height, 0, y_offset)
        else:
            x_offset = max(0, (width - crop_width) // 2)
            video = stream.video.filter("crop", crop_width, height, x_offset, 0)
        video = video.filter("scale", TARGET_WIDTH, TARGET_HEIGHT)

    audio = stream.audio

    try:
        (
            ffmpeg.output(video, audio, output_path)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg crop/trim failed for {input_path!r}: {stderr}") from exc

    return output_path
