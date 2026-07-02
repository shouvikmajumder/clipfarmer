"""Formatting stage: mix royalty-free background music under a clip's speech.

Only some content profiles get music:

- **podcast**: music at 12% volume
- **irl**:     music at 15% volume
- **interview / gaming / default**: no music (the formatting stage is skipped
  for these by the caller).

Tracks live in ``assets/music/<profile>/*.mp3``; a random track is chosen per
clip.  If the profile's folder is missing or empty, the clip is copied to the
final path unchanged (a warning is logged) rather than failing the job.

``should_add_music``, ``music_volume``, ``pick_music_track`` and
``build_mix_cmd`` are pure / side-effect-light and unit-testable without
FFmpeg.  ``add_music`` is the only heavy entry point; it is idempotent via a
cache guard on the output file.  Per-clip retry/skip policy lives in
``core/job_runner.py``.

The mix is audio-only, so the video stream is copied (``-c:v copy``) — no
re-encode, unlike the caption stage.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile → music volume (fraction of original). Profiles absent here get no
# music and are skipped by the caller.
# ---------------------------------------------------------------------------
MUSIC_VOLUME_BY_PROFILE: dict[str, float] = {
    "podcast": 0.12,
    "irl": 0.15,
}

# Root of the music library: clipfarmer/assets/music/<profile>/*.mp3
MUSIC_DIR = Path(__file__).resolve().parent.parent / "assets" / "music"


def _ffmpeg_bin() -> str:
    """The ffmpeg binary to invoke (override with the ``CLIPFARMER_FFMPEG`` env var)."""
    return os.environ.get("CLIPFARMER_FFMPEG", "ffmpeg")


def should_add_music(profile: str) -> bool:
    """Return True if *profile* should have background music mixed in.

    True for ``podcast`` and ``irl``; False for ``interview``, ``gaming``,
    ``default`` and anything else.  Case-insensitive.
    """
    return profile.lower() in MUSIC_VOLUME_BY_PROFILE


def music_volume(profile: str) -> float | None:
    """Return the music volume fraction for *profile*, or ``None`` if it gets none."""
    return MUSIC_VOLUME_BY_PROFILE.get(profile.lower())


def pick_music_track(profile: str, music_dir: Path | None = None) -> str | None:
    """Return a random ``*.mp3`` path from the profile's music folder, or None.

    Returns ``None`` (caller falls back to copying the clip unchanged) when the
    folder is missing or contains no ``.mp3`` files.

    Args:
        profile:   Content profile (its folder is ``<music_dir>/<profile>``).
        music_dir: Root music directory; defaults to ``MUSIC_DIR``.
    """
    root = music_dir if music_dir is not None else MUSIC_DIR
    folder = root / profile.lower()
    if not folder.is_dir():
        return None
    tracks = sorted(str(p) for p in folder.glob("*.mp3"))
    if not tracks:
        return None
    return random.choice(tracks)


def build_mix_cmd(
    clip_path: str,
    music_path: str,
    output_path: str,
    duration_s: float,
    volume: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Return the FFmpeg command that mixes *music_path* under *clip_path*'s audio.

    Pure — no I/O, no subprocesses.

    filter_complex:
      - ``[1:a]atrim=end=<dur>,volume=<vol>[m]`` — trim music to the clip's
        duration and scale its volume down.
      - ``[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[aout]`` —
        mix speech + music, ending with the first input (the clip) so trailing
        music never extends the clip.

    The video stream is copied (``-map 0:v -c:v copy``); the mixed audio is
    encoded as AAC.

    Args:
        clip_path:   Input 0 — the (captioned/edited) clip with speech audio.
        music_path:  Input 1 — the background music track.
        output_path: Final output path.
        duration_s:  Clip duration in seconds (clip.end_s - clip.start_s).
        volume:      Music volume fraction (e.g. 0.12).
        ffmpeg_bin:  ffmpeg binary to invoke.
    """
    filter_complex = (
        f"[1:a]atrim=end={duration_s:.3f},volume={volume}[m];"
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )
    return [
        ffmpeg_bin,
        "-y",
        "-i", clip_path,
        # Loop the music input so a track shorter than the clip still covers it
        # (no trailing silence); atrim + amix=duration=first cap it to the clip.
        # -stream_loop is an input option: it applies only to the music input
        # that follows, not the clip above.
        "-stream_loop", "-1",
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-c:v", "copy",
        "-map", "[aout]",
        "-c:a", "aac",
        output_path,
    ]


def add_music(
    clip_path: str,
    output_path: str,
    profile: str,
    duration_s: float,
    music_dir: Path | None = None,
) -> str:
    """Mix background music under *clip_path* and write *output_path*.

    The only heavy entry point. Idempotent: if *output_path* already exists and
    is non-empty, returns it without invoking FFmpeg.

    If no music track is available for *profile* (missing/empty folder), the
    clip is copied to *output_path* unchanged and a warning is logged — the job
    is not failed.

    Args:
        clip_path:   Source clip (already captioned/edited).
        output_path: Final output path.
        profile:     Content profile (selects volume + music folder).
        duration_s:  Clip duration in seconds (clip.end_s - clip.start_s).
        music_dir:   Override the music root (for tests).

    Returns:
        *output_path*.

    Raises:
        RuntimeError: If FFmpeg exits non-zero while mixing.
    """
    import subprocess

    # Cache guard: idempotent retry.
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("add_music: cache hit, skipping FFmpeg for %r", output_path)
        return output_path

    track = pick_music_track(profile, music_dir=music_dir)
    if track is None:
        logger.warning(
            "add_music: no music track for profile %r (folder missing or empty); "
            "copying clip through unchanged.",
            profile,
        )
        shutil.copyfile(clip_path, output_path)
        return output_path

    volume = music_volume(profile)
    if volume is None:
        # Caller should not invoke add_music for a non-music profile, but guard.
        logger.warning("add_music: profile %r has no music volume; copying through.", profile)
        shutil.copyfile(clip_path, output_path)
        return output_path

    cmd = build_mix_cmd(
        clip_path, track, output_path, duration_s, volume, ffmpeg_bin=_ffmpeg_bin()
    )
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg music mix failed for {clip_path!r} (profile={profile!r}): {stderr}"
        )
    return output_path
