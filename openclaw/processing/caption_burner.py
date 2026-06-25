"""Caption burning stage: overlay word-level subtitles onto a clip via ffmpeg.

Captions are generated from PER-CLIP Whisper word timestamps produced by
``processing.transcriber.transcribe_words``.  Because each edited clip starts
at t=0, the word times returned by Whisper are already clip-relative and
``offset_s`` is always 0.0 in the normal pipeline.  The parameter is kept
explicit so the offset math is testable in isolation.

Profiles ``gaming`` and ``irl`` are skipped by the caller before reaching
``caption_clip``; ``should_caption`` encodes this policy so callers don't
hard-code the skip set.  Per-clip retry/skip policy lives in
``core/job_runner.py``.

The intermediate ``.srt`` file written next to *srt_path* is deleted after a
successful burn and left on disk on failure to aid debugging.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WORDS_PER_LINE: int = 6

# Profiles that are NOT captioned (case-insensitive comparison done in should_caption).
SKIP_CAPTION_PROFILES: set[str] = {"gaming", "irl"}

# FFmpeg subtitles filter force_style string for burned-in captions.
CAPTION_STYLE: str = (
    "FontName=Arial,FontSize=52,PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,Outline=3,Bold=1,Alignment=2,MarginV=80"
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def should_caption(profile: str) -> bool:
    """Return True if *profile* should have captions burned in.

    Profiles in ``SKIP_CAPTION_PROFILES`` (``gaming``, ``irl``) return False;
    everything else (``podcast``, ``interview``, ``default``, …) returns True.
    Comparison is case-insensitive.
    """
    return profile.lower() not in SKIP_CAPTION_PROFILES


def _format_srt_timestamp(seconds: float) -> str:
    """Format *seconds* as an SRT timestamp: ``HH:MM:SS,mmm``.

    Negative values are clamped to 0.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def group_words_into_lines(
    words: list[dict],
    max_words: int = MAX_WORDS_PER_LINE,
) -> list[list[dict]]:
    """Split *words* into consecutive chunks of at most *max_words*.

    Pure — no I/O.

    Args:
        words:     List of word dicts (each must have ``word``, ``start``,
                   ``end`` keys).
        max_words: Maximum number of words per line chunk.

    Returns:
        List of lists; each inner list is one line's worth of words.
    """
    if not words:
        return []
    return [words[i : i + max_words] for i in range(0, len(words), max_words)]


def words_to_srt(
    words: list[dict],
    offset_s: float = 0.0,
    max_words: int = MAX_WORDS_PER_LINE,
) -> str:
    """Build a valid SRT string from *words*.

    Each line group covers at most *max_words* words.  The start time of each
    block is ``first_word["start"] - offset_s`` and the end time is
    ``last_word["end"] - offset_s`` (both clamped to ≥ 0 by the formatter).

    In the normal pipeline ``offset_s`` is 0.0 because the clip starts at t=0
    and Whisper word times are already clip-relative.  The parameter is kept
    explicit so offset arithmetic can be unit-tested directly.

    Args:
        words:     List of word dicts with ``word``, ``start``, ``end`` keys.
        offset_s:  Seconds to subtract from each word's start/end timestamp.
        max_words: Maximum words per caption line.

    Returns:
        A valid SRT-formatted string.  Empty if *words* is empty or all line
        texts are blank after stripping.
    """
    lines = group_words_into_lines(words, max_words)
    blocks: list[str] = []
    index = 1

    for line in lines:
        if not line:
            continue
        text = " ".join(w.get("word", "").strip() for w in line).strip()
        if not text:
            continue

        start = line[0].get("start", 0.0) - offset_s
        end = line[-1].get("end", start) - offset_s

        blocks.append(str(index))
        blocks.append(
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}"
        )
        blocks.append(text)
        blocks.append("")  # blank line separator

        index += 1

    return "\n".join(blocks) + ("\n" if blocks else "")


def _escape_srt_path(path: str) -> str:
    """Escape *path* for use in the ffmpeg ``subtitles`` filter.

    The subtitles filter has its own mini-parser that treats backslash as an
    escape character and colon as a separator between the filename and filter
    options.  Escaping both characters ensures paths with Windows drive letters
    or special characters are handled correctly on all platforms.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:")


def build_burn_cmd(
    input_path: str,
    srt_path: str,
    output_path: str,
    style: str = CAPTION_STYLE,
) -> list[str]:
    """Return the FFmpeg argument list to burn subtitles into *input_path*.

    Pure — no I/O, no subprocesses.

    Subtitle burn-in works by modifying the raw video frames via the
    ``subtitles`` filter.  This means the video stream must be fully decoded
    and re-encoded — ``-c:v copy`` is impossible when any video filter is
    active.  Audio is unmodified so ``-c:a copy`` is safe.

    Args:
        input_path:  Absolute path to the source (already edited) clip.
        srt_path:    Absolute path to the ``.srt`` subtitle file.
        output_path: Absolute path to write the captioned output video.
        style:       FFmpeg ``force_style`` string for the subtitles filter.

    Returns:
        A list of strings ready to pass directly to ``subprocess.run``.
    """
    escaped = _escape_srt_path(srt_path)
    return [
        "ffmpeg",
        "-y",
        "-i", input_path,
        # Re-encode video: the subtitles filter modifies frames; -c:v copy is impossible.
        "-vf", f"subtitles={escaped}:force_style='{style}'",
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "copy",
        output_path,
    ]


# ---------------------------------------------------------------------------
# Impure entry point
# ---------------------------------------------------------------------------


def caption_clip(
    input_path: str,
    words: list[dict],
    srt_path: str,
    output_path: str,
    offset_s: float = 0.0,
) -> str:
    """Burn word-level captions into *input_path* and write to *output_path*.

    This is the only impure function in this module.  It is idempotent: if
    *output_path* already exists and is non-empty, the SRT is not written and
    FFmpeg is not invoked (safe to retry a partially-completed job).

    Per-clip retry/skip policy lives in ``core/job_runner.py``.

    Args:
        input_path:  Absolute path to the source (already edited) clip.
        words:       Word dicts from ``transcribe_words`` (``word``, ``start``,
                     ``end``); must be clip-relative (t=0 origin).
        srt_path:    Absolute path where the intermediate ``.srt`` file will
                     be written (and deleted on success).
        output_path: Absolute path to write the captioned output video.
        offset_s:    Seconds to subtract from word timestamps when building
                     the SRT.  Should be 0.0 in normal pipeline use (clips
                     start at t=0).

    Returns:
        *output_path*, unchanged, for convenient chaining.

    Raises:
        RuntimeError: If FFmpeg exits with a non-zero return code.  The
            intermediate ``.srt`` file is left on disk to aid debugging.
    """
    import subprocess

    # Cache guard: idempotent retry — skip everything if output already exists.
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(
            "caption_clip: cache hit, skipping FFmpeg for %r",
            output_path,
        )
        return output_path

    # Write the SRT subtitle file.
    srt_content = words_to_srt(words, offset_s)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    cmd = build_burn_cmd(input_path, srt_path, output_path)
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        # Leave srt_path on disk for debugging.
        raise RuntimeError(
            f"ffmpeg caption burn failed for {input_path!r}: {stderr}"
        )

    # Clean up the intermediate SRT on success.
    try:
        os.unlink(srt_path)
    except OSError:
        pass

    return output_path
