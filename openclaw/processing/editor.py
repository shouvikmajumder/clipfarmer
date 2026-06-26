"""Profile-based vertical-video editor: trim source footage and composite it
to 9:16 (608x1080) using one of several layout profiles.

Profiles
--------
- **podcast** / **default**: zoom + center-crop + scale; good for talking-head
  footage where the subject is centered.
- **interview**: same zoom/crop but lets the caller shift the crop window left
  or right so an off-center speaker stays in frame.
- **irl**: podcast pipeline plus a subtle contrast/saturation boost via the
  ``eq`` filter.
- **gaming**: blurred full-frame background with gameplay centred on top, plus
  a small facecam cutout overlaid at the bottom-right corner.

``build_filter_args`` and ``build_ffmpeg_cmd`` are pure functions — they build
CLI argument lists without touching the filesystem or spawning any process.
This makes them trivially unit-testable without a real FFmpeg binary.

``edit_clip`` is the only impure entry-point.  It is idempotent: if the output
file already exists and is non-empty it returns immediately without invoking
FFmpeg.  Per-clip retry/skip policy lives in the caller (``core/job_runner.py``),
matching the approach described in the v4 plan.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _ffmpeg_bin() -> str:
    """The ffmpeg binary to invoke (override with the ``OPENCLAW_FFMPEG`` env var)."""
    return os.environ.get("OPENCLAW_FFMPEG", "ffmpeg")

# ---------------------------------------------------------------------------
# Target resolution (9:16 at 1080p short-form vertical)
# ---------------------------------------------------------------------------
TARGET_WIDTH = 608
TARGET_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Aspect-safe 9:16 crop width:height expression.
# Clamps the crop region to the input's own dimensions via min(), so a source
# that is narrower than 9:16 (portrait/vertical input) crops by height instead
# of erroring out (crop width > input width). The commas inside min() are
# escaped \, so the filtergraph parser doesn't read them as filter separators
# (same technique as caption_burner.build_burn_cmd). x/y default to centre.
# ---------------------------------------------------------------------------
CROP_9_16_WH = r"min(iw\,ih*9/16):min(ih\,iw*16/9)"

# ---------------------------------------------------------------------------
# Horizontal crop-bias expressions for the interview profile.
# These are FFmpeg expression strings for the x-offset argument of the crop
# filter, where ow is the crop output width.
# ---------------------------------------------------------------------------
CROP_BIAS = {
    "left": "0",
    "center": "(iw-ow)/2",
    "right": "iw-ow",
}

# ---------------------------------------------------------------------------
# Default facecam region for the gaming profile, expressed as ratios of the
# source video's width/height.  Callers may override this per-clip.
# ---------------------------------------------------------------------------
DEFAULT_FACECAM_REGION: dict[str, float] = {
    "width_ratio": 0.25,
    "height_ratio": 0.35,
    "x_ratio": 0.75,
    "y_ratio": 0.65,
}


def build_filter_args(
    profile: str,
    crop_bias: str = "center",
    facecam_region: dict | None = None,
) -> list[str]:
    """Return FFmpeg filter CLI args for *profile*.

    For simple profiles the return value is ``["-vf", "<filter_chain>"]``.
    For the gaming profile it is
    ``["-filter_complex", "<fc>", "-map", "[vout]", "-map", "0:a?"]``.

    Profile lookup is case-insensitive.  ``None`` or any unknown string falls
    back to ``"default"``.

    Args:
        profile:       One of ``podcast``, ``interview``, ``gaming``, ``irl``,
                       ``default`` (case-insensitive).  Unknown values treated
                       as ``"default"``.
        crop_bias:     For the ``interview`` profile only.  One of ``"left"``,
                       ``"center"`` (default), or ``"right"``.  Unknown values
                       fall back to ``"center"``.
        facecam_region: For the ``gaming`` profile only.  Dict with keys
                        ``width_ratio``, ``height_ratio``, ``x_ratio``,
                        ``y_ratio`` (all floats, ratios of source dimensions).
                        Defaults to ``DEFAULT_FACECAM_REGION``.

    Returns:
        A list of strings ready to be spliced into an FFmpeg command.
    """
    # Normalise profile: None or unknown → "default"
    normalised = (profile or "default").lower().strip()
    known = {"podcast", "interview", "gaming", "irl", "default"}
    if normalised not in known:
        normalised = "default"

    if normalised in ("podcast", "default"):
        # Zoom in 15% to add headroom, then center-crop to exact 9:16 (the
        # aspect-safe expression clamps to the input so portrait sources don't
        # error), then scale to the 608x1080 target.
        vf = f"scale=iw*1.15:ih*1.15,crop={CROP_9_16_WH},scale=608:1080"
        return ["-vf", vf]

    if normalised == "interview":
        # Same zoom as podcast, but the crop x-offset is controlled by
        # crop_bias so an off-center speaker stays in frame.
        xbias = CROP_BIAS.get(crop_bias, CROP_BIAS["center"])
        vf = f"scale=iw*1.15:ih*1.15,crop={CROP_9_16_WH}:{xbias}:0,scale=608:1080"
        return ["-vf", vf]

    if normalised == "irl":
        # Podcast pipeline plus a subtle contrast/saturation boost via the eq
        # filter, which helps outdoor/run-and-gun footage look more polished.
        vf = (
            "scale=iw*1.15:ih*1.15,"
            f"crop={CROP_9_16_WH},"
            "scale=608:1080,"
            "eq=contrast=1.05:saturation=1.15"
        )
        return ["-vf", vf]

    # gaming — filter_complex with three layers:
    #   Layer 1 [bg]:   blurred full-frame fill so no black bars appear
    #   Layer 2 [base]: gameplay video fit to 608 width, centered on the fill
    #   Layer 3 [vout]: facecam sub-rectangle (~182 px wide) overlaid
    #                   bottom-right with a 20 px inset margin
    region = facecam_region if facecam_region is not None else DEFAULT_FACECAM_REGION
    w = region["width_ratio"]
    h = region["height_ratio"]
    x = region["x_ratio"]
    y = region["y_ratio"]

    # Build as a single semicolon-separated filter_complex string.
    # [bg]:   scale to at-least 608x1080, hard-crop to exactly 608x1080, then
    #         apply a Gaussian blur (sigma=20) so edges look intentional.
    # [fg]:   scale gameplay to 608 px wide, preserving aspect ratio.
    # overlay [base]: centre the gameplay over the blurred fill.
    # [cam]:  crop the facecam region from the original input, scale to ~182 px.
    # overlay [vout]: place the facecam at bottom-right with 20 px margins.
    fc = (
        f"[0:v]scale=608:1080:force_original_aspect_ratio=increase,"
        f"crop=608:1080,gblur=sigma=20[bg];"
        f"[0:v]scale=608:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
        f"[0:v]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale=182:-2[cam];"
        f"[base][cam]overlay=W-w-20:H-h-20[vout]"
    )
    return ["-filter_complex", fc, "-map", "[vout]", "-map", "0:a?"]


def build_ffmpeg_cmd(
    input_path: str,
    start_s: float,
    end_s: float,
    output_path: str,
    profile: str = "default",
    crop_bias: str = "center",
    facecam_region: dict | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Return the full FFmpeg command as a list of argument strings.

    Pure — no I/O, no subprocesses.

    Args:
        input_path:    Absolute path to the source video file.
        start_s:       Clip start time in seconds within the source video.
        end_s:         Clip end time in seconds within the source video.
        output_path:   Absolute path to write the processed output to.
        profile:       Layout profile (see ``build_filter_args``).
        crop_bias:     Horizontal crop bias for the ``interview`` profile.
        facecam_region: Facecam region overrides for the ``gaming`` profile.

    Returns:
        A list ready to pass directly to ``subprocess.run``.

    Raises:
        ValueError: If ``end_s <= start_s``.
    """
    if end_s <= start_s:
        raise ValueError(
            f"end_s ({end_s}) must be greater than start_s ({start_s})"
        )

    filter_args = build_filter_args(profile, crop_bias=crop_bias, facecam_region=facecam_region)
    duration = end_s - start_s

    return [
        ffmpeg_bin,
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        *filter_args,
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "aac",
        output_path,
    ]


def edit_clip(
    input_path: str,
    start_s: float,
    end_s: float,
    output_path: str,
    profile: str = "default",
    crop_bias: str = "center",
    facecam_region: dict | None = None,
) -> str:
    """Trim and reformat *input_path* to 9:16 using the given *profile*.

    This is the only impure function in this module.  It is idempotent: if
    *output_path* already exists and is non-empty, FFmpeg is skipped and
    *output_path* is returned immediately (safe to retry a partially-completed
    job).

    Args:
        input_path:    Absolute path to the source video file.
        start_s:       Clip start time in seconds.
        end_s:         Clip end time in seconds.
        output_path:   Absolute path to write the processed output to.
        profile:       Layout profile passed to ``build_filter_args``.
        crop_bias:     Horizontal crop bias for the ``interview`` profile.
        facecam_region: Facecam region overrides for the ``gaming`` profile.

    Returns:
        *output_path*, unchanged, for convenient chaining.

    Raises:
        ValueError:   If ``end_s <= start_s`` (propagated from
                      ``build_ffmpeg_cmd``).
        RuntimeError: If FFmpeg exits with a non-zero return code.
    """
    import subprocess

    # Cache guard: idempotent retry — skip FFmpeg if output already exists.
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(
            "edit_clip: cache hit, skipping FFmpeg for %r (profile=%r)",
            output_path,
            profile,
        )
        return output_path

    cmd = build_ffmpeg_cmd(
        input_path,
        start_s,
        end_s,
        output_path,
        profile=profile,
        crop_bias=crop_bias,
        facecam_region=facecam_region,
        ffmpeg_bin=_ffmpeg_bin(),
    )

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg editing failed for {input_path!r} (profile={profile!r}): {stderr}"
        )

    return output_path
