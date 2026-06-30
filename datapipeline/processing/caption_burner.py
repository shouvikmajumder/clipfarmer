"""Caption burning stage: overlay subtitles onto a clip via ffmpeg.

Generates an SRT subtitle file from *segments* and burns it into the video
using ffmpeg's ``subtitles`` filter.

**Contract:** *segments* must already be sliced to the clip's time range and
re-based to start at 0 — i.e. each segment's ``start``/``end`` should be
relative to the beginning of *input_path*, not the original source video's
timeline. This keeps the function simple and dependency-free on any
clip/job metadata; callers (e.g. ``core/job_runner.py``) own the
slicing/offsetting step using the clip's ``start_s``/``end_s`` against the
full job transcript.

The intermediate ``.srt`` file is written next to *output_path* (same stem,
``.srt`` extension) and deleted after a successful burn. It is left in place
if burning fails, to aid debugging.
"""

from __future__ import annotations

from pathlib import Path


def _format_srt_timestamp(seconds: float) -> str:
    """Format *seconds* as an SRT timestamp: ``HH:MM:SS,mmm``."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: list[dict]) -> str:
    """Render *segments* (clip-relative start/end/text) to SRT-formatted text."""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start", 0.0) or 0.0
        end = seg.get("end", start) or start
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")  # blank line separator
    return "\n".join(lines) + ("\n" if lines else "")


def burn_captions(input_path: str, segments: list[dict], output_path: str) -> str:
    """Burn *segments* into *input_path* as captions and write to *output_path*.

    Contract: *segments* must already be sliced/offset to the clip's own
    timeline (each ``start``/``end`` relative to t=0 of *input_path*). See
    module docstring.

    Steps:
    1. Render *segments* to a temporary ``.srt`` file next to *output_path*.
    2. Burn it in via ffmpeg's ``subtitles`` filter.
    3. Delete the intermediate ``.srt`` file on success.

    Args:
        input_path: Absolute path to the source (already cropped/trimmed)
                    clip video.
        segments: Transcript segment dicts (``start``, ``end``, ``text``)
                  relative to the start of *input_path*. Segments with empty
                  text are skipped. May be empty, in which case the video is
                  passed through with no burned-in captions.
        output_path: Absolute path to write the captioned output to.

    Returns:
        *output_path*, unchanged, for convenient chaining.

    Raises:
        RuntimeError: If ffmpeg fails to burn the subtitles (e.g. missing
            binary, corrupt input, filter error). The intermediate ``.srt``
            file is left on disk in this case to aid debugging.
    """
    import ffmpeg

    srt_path = Path(output_path).with_suffix(".srt")
    srt_path.write_text(_segments_to_srt(segments), encoding="utf-8")

    # ffmpeg's subtitles filter requires an escaped path on some platforms
    # (colons in Windows drive letters, special chars). Posix paths with no
    # special characters pass through fine; escape conservatively regardless.
    escaped_srt_path = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")

    try:
        (
            ffmpeg.input(input_path)
            .filter("subtitles", escaped_srt_path)
            .output(output_path)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg caption burn failed for {input_path!r}: {stderr}") from exc

    srt_path.unlink(missing_ok=True)

    return output_path
