"""Tests for processing.format_music — background-music mixing stage.

All tests are pure: no real FFmpeg is invoked. Pure helpers are tested
directly; ``add_music`` is tested via a monkeypatched ``subprocess.run`` and
real temp files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from processing import format_music


# ---------------------------------------------------------------------------
# Profile gating + volumes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["podcast", "irl", "Podcast", "IRL"])
def test_should_add_music_true_for_podcast_and_irl(profile):
    assert format_music.should_add_music(profile) is True


@pytest.mark.parametrize("profile", ["interview", "gaming", "default", "unknown"])
def test_should_add_music_false_for_others(profile):
    assert format_music.should_add_music(profile) is False


def test_music_volume_per_profile():
    assert format_music.music_volume("podcast") == 0.12
    assert format_music.music_volume("irl") == 0.15
    # case-insensitive
    assert format_music.music_volume("PODCAST") == 0.12
    # non-music profiles get no volume
    assert format_music.music_volume("interview") is None
    assert format_music.music_volume("gaming") is None
    assert format_music.music_volume("default") is None


# ---------------------------------------------------------------------------
# Track selection
# ---------------------------------------------------------------------------


def test_pick_music_track_missing_folder_returns_none(tmp_path):
    # tmp_path has no "podcast" subfolder at all.
    assert format_music.pick_music_track("podcast", music_dir=tmp_path) is None


def test_pick_music_track_empty_folder_returns_none(tmp_path):
    (tmp_path / "podcast").mkdir()
    assert format_music.pick_music_track("podcast", music_dir=tmp_path) is None


def test_pick_music_track_returns_an_mp3(tmp_path):
    folder = tmp_path / "podcast"
    folder.mkdir()
    a = folder / "a.mp3"
    b = folder / "b.mp3"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    # a non-mp3 must be ignored
    (folder / "notes.txt").write_text("ignore me")

    result = format_music.pick_music_track("podcast", music_dir=tmp_path)
    assert result in {str(a), str(b)}


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def test_build_mix_cmd_structure():
    cmd = format_music.build_mix_cmd("clip.mp4", "track.mp3", "out.mp4", 20.0, 0.12)
    assert cmd[0] == "ffmpeg"
    assert cmd[:1] != ["/custom/ffmpeg"]
    # inputs in order: clip is 0, music is 1
    assert cmd.index("clip.mp4") < cmd.index("track.mp3")

    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "atrim=end=20.000" in fc
    assert "volume=0.12" in fc
    assert "amix=inputs=2:duration=first:dropout_transition=0" in fc

    # video passthrough, audio re-encoded to aac
    assert cmd[cmd.index("-map") + 1] == "0:v"
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "[aout]" in cmd
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"


def test_build_mix_cmd_volume_matches_profile():
    podcast = format_music.build_mix_cmd("c.mp4", "m.mp3", "o.mp4", 30.0, format_music.music_volume("podcast"))
    irl = format_music.build_mix_cmd("c.mp4", "m.mp3", "o.mp4", 30.0, format_music.music_volume("irl"))
    assert "volume=0.12" in podcast[podcast.index("-filter_complex") + 1]
    assert "volume=0.15" in irl[irl.index("-filter_complex") + 1]


def test_build_mix_cmd_respects_ffmpeg_bin():
    cmd = format_music.build_mix_cmd("c.mp4", "m.mp3", "o.mp4", 10.0, 0.12, ffmpeg_bin="/opt/ff/ffmpeg")
    assert cmd[0] == "/opt/ff/ffmpeg"


# ---------------------------------------------------------------------------
# add_music — fallback / cache / mix
# ---------------------------------------------------------------------------


def test_add_music_empty_folder_copies_clip_through(tmp_path, monkeypatch):
    """No music available → copy the clip unchanged, never invoke FFmpeg."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original-clip-bytes")
    out = tmp_path / "clip_0_final.mp4"

    run_mock = MagicMock(side_effect=AssertionError("subprocess.run must not be called"))
    monkeypatch.setattr(subprocess, "run", run_mock)

    # tmp_path has no podcast/ folder → pick_music_track returns None.
    result = format_music.add_music(str(clip), str(out), "podcast", 20.0, music_dir=tmp_path)

    assert result == str(out)
    assert out.read_bytes() == b"original-clip-bytes"
    run_mock.assert_not_called()


def test_add_music_cache_hit_skips_ffmpeg(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    out = tmp_path / "clip_0_final.mp4"
    out.write_bytes(b"already-rendered")

    run_mock = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = format_music.add_music(str(clip), str(out), "podcast", 20.0, music_dir=tmp_path)

    assert result == str(out)
    assert out.read_bytes() == b"already-rendered"  # untouched
    run_mock.assert_not_called()


def test_add_music_mixes_when_track_present(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    folder = tmp_path / "podcast"
    folder.mkdir()
    (folder / "track.mp3").write_bytes(b"music")
    out = tmp_path / "clip_0_final.mp4"

    run_mock = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = format_music.add_music(str(clip), str(out), "podcast", 25.0, music_dir=tmp_path)

    assert result == str(out)
    run_mock.assert_called_once()
    cmd = run_mock.call_args[0][0]
    assert str(clip) in cmd
    assert str(folder / "track.mp3") in cmd
    assert "volume=0.12" in cmd[cmd.index("-filter_complex") + 1]


def test_add_music_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    folder = tmp_path / "irl"
    folder.mkdir()
    (folder / "track.mp3").write_bytes(b"music")
    out = tmp_path / "clip_0_final.mp4"

    run_mock = MagicMock(return_value=MagicMock(returncode=1, stderr=b"boom"))
    monkeypatch.setattr(subprocess, "run", run_mock)

    with pytest.raises(RuntimeError, match="music mix failed"):
        format_music.add_music(str(clip), str(out), "irl", 25.0, music_dir=tmp_path)
