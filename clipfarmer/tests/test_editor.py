"""Tests for processing.editor — profile-based 9:16 clip editor.

All tests are pure: no real FFmpeg binary is invoked.  ``build_filter_args``
and ``build_ffmpeg_cmd`` are tested directly (pure functions).  ``edit_clip``
is tested via a monkeypatched ``subprocess.run``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from processing import editor


INPUT_PATH = "/tmp/fake_input.mp4"


# ---------------------------------------------------------------------------
# Cache-hit guard
# ---------------------------------------------------------------------------


def test_edit_clip_cache_hit_skips_ffmpeg(tmp_path, monkeypatch):
    """If output already exists and is non-empty, FFmpeg must not be called."""
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"not empty")  # simulate a pre-existing encoded file

    run_mock = MagicMock(side_effect=AssertionError("subprocess.run must not be called on cache hit"))
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = editor.edit_clip(INPUT_PATH, 0.0, 10.0, str(output_path))

    assert result == str(output_path)
    run_mock.assert_not_called()


def test_edit_clip_empty_file_is_not_a_cache_hit(tmp_path, monkeypatch):
    """A zero-byte output file must NOT be treated as a cache hit."""
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"")  # zero bytes — not a real cached file

    run_mock = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run_mock)

    editor.edit_clip(INPUT_PATH, 0.0, 10.0, str(output_path))

    run_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Profile filter strings — build_filter_args
# ---------------------------------------------------------------------------


def test_podcast_profile_contains_correct_filter_chain():
    args = editor.build_filter_args("podcast")
    assert args[0] == "-vf"
    vf = args[1]
    assert "scale=iw*1.15:ih*1.15" in vf
    assert r"crop=min(iw\,ih*9/16):min(ih\,iw*16/9)" in vf
    assert "scale=608:1080" in vf


def test_default_profile_identical_to_podcast():
    assert editor.build_filter_args("default") == editor.build_filter_args("podcast")


def test_default_profile_uses_vf_flag():
    args = editor.build_filter_args("default")
    assert args[0] == "-vf"
    assert len(args) == 2


def test_podcast_profile_uses_vf_flag():
    args = editor.build_filter_args("podcast")
    assert args[0] == "-vf"
    assert len(args) == 2


def test_interview_center_bias_contains_centered_expression():
    args = editor.build_filter_args("interview", crop_bias="center")
    assert args[0] == "-vf"
    assert "(iw-ow)/2" in args[1]


def test_interview_right_bias_contains_right_expression():
    args = editor.build_filter_args("interview", crop_bias="right")
    assert args[0] == "-vf"
    assert ":iw-ow:0" in args[1]


def test_interview_left_bias_contains_zero_offset():
    args = editor.build_filter_args("interview", crop_bias="left")
    assert args[0] == "-vf"
    # x-offset should be "0" immediately before ":0" (y-offset)
    assert ":0:0" in args[1]


def test_interview_unknown_bias_falls_back_to_center():
    args_unknown = editor.build_filter_args("interview", crop_bias="bogus")
    args_center = editor.build_filter_args("interview", crop_bias="center")
    assert args_unknown == args_center


def test_irl_profile_contains_eq_filter():
    args = editor.build_filter_args("irl")
    assert args[0] == "-vf"
    assert "eq=contrast=1.05:saturation=1.15" in args[1]


def test_irl_profile_contains_base_pipeline():
    args = editor.build_filter_args("irl")
    vf = args[1]
    assert "scale=iw*1.15:ih*1.15" in vf
    assert r"crop=min(iw\,ih*9/16):min(ih\,iw*16/9)" in vf
    assert "scale=608:1080" in vf


def test_gaming_profile_uses_filter_complex():
    args = editor.build_filter_args("gaming")
    assert args[0] == "-filter_complex"


def test_gaming_profile_contains_blur():
    args = editor.build_filter_args("gaming")
    fc = args[1]
    assert "gblur=sigma=20" in fc


def test_gaming_profile_contains_two_overlays():
    args = editor.build_filter_args("gaming")
    fc = args[1]
    assert fc.count("overlay=") == 2


def test_gaming_profile_maps_vout():
    args = editor.build_filter_args("gaming")
    assert "[vout]" in args
    assert args[args.index("[vout]") - 1] == "-map"


def test_gaming_profile_maps_audio():
    args = editor.build_filter_args("gaming")
    assert "0:a?" in args


def test_gaming_profile_custom_facecam_region():
    region = {"width_ratio": 0.3, "height_ratio": 0.4, "x_ratio": 0.6, "y_ratio": 0.5}
    args = editor.build_filter_args("gaming", facecam_region=region)
    fc = args[1]
    # Custom ratios should appear in the crop expression for the facecam layer
    assert "iw*0.3" in fc
    assert "ih*0.4" in fc
    assert "iw*0.6" in fc
    assert "ih*0.5" in fc


def test_gaming_profile_default_facecam_region():
    """With no custom region the DEFAULT_FACECAM_REGION values should appear."""
    args = editor.build_filter_args("gaming")
    fc = args[1]
    r = editor.DEFAULT_FACECAM_REGION
    assert f"iw*{r['width_ratio']}" in fc
    assert f"ih*{r['height_ratio']}" in fc


# ---------------------------------------------------------------------------
# Unknown / None profile fallback
# ---------------------------------------------------------------------------


def test_unknown_profile_falls_back_to_default():
    assert editor.build_filter_args("totally-unknown") == editor.build_filter_args("default")


def test_none_profile_falls_back_to_default():
    assert editor.build_filter_args(None) == editor.build_filter_args("default")


def test_profile_lookup_is_case_insensitive():
    assert editor.build_filter_args("PODCAST") == editor.build_filter_args("podcast")
    assert editor.build_filter_args("Gaming") == editor.build_filter_args("gaming")
    assert editor.build_filter_args("IRL") == editor.build_filter_args("irl")


# ---------------------------------------------------------------------------
# build_ffmpeg_cmd — structure and codec flags
# ---------------------------------------------------------------------------


def test_build_ffmpeg_cmd_includes_h264_videotoolbox():
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 0.0, 10.0, "/tmp/out.mp4")
    assert "h264_videotoolbox" in cmd


def test_build_ffmpeg_cmd_includes_bitrate():
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 0.0, 10.0, "/tmp/out.mp4")
    assert "-b:v" in cmd
    assert "8M" in cmd


def test_build_ffmpeg_cmd_includes_aac_audio():
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 0.0, 10.0, "/tmp/out.mp4")
    assert "-c:a" in cmd
    assert "aac" in cmd


def test_build_ffmpeg_cmd_includes_ss_trim():
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 5.5, 20.0, "/tmp/out.mp4")
    assert "-ss" in cmd
    ss_idx = cmd.index("-ss")
    assert cmd[ss_idx + 1] == "5.500"


def test_build_ffmpeg_cmd_includes_t_duration():
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 5.0, 20.0, "/tmp/out.mp4")
    assert "-t" in cmd
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "15.000"


def test_build_ffmpeg_cmd_raises_on_invalid_time_range():
    with pytest.raises(ValueError, match="end_s"):
        editor.build_ffmpeg_cmd(INPUT_PATH, 10.0, 5.0, "/tmp/out.mp4")


def test_build_ffmpeg_cmd_raises_when_end_equals_start():
    with pytest.raises(ValueError, match="end_s"):
        editor.build_ffmpeg_cmd(INPUT_PATH, 7.0, 7.0, "/tmp/out.mp4")


def test_build_ffmpeg_cmd_starts_with_ffmpeg_y():
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 0.0, 5.0, "/tmp/out.mp4")
    assert cmd[0] == "ffmpeg"
    assert cmd[1] == "-y"


def test_build_ffmpeg_cmd_ends_with_output_path():
    out = "/tmp/specific_output.mp4"
    cmd = editor.build_ffmpeg_cmd(INPUT_PATH, 0.0, 5.0, out)
    assert cmd[-1] == out


# ---------------------------------------------------------------------------
# edit_clip — FFmpeg execution path
# ---------------------------------------------------------------------------


def test_edit_clip_invokes_ffmpeg_when_no_output_exists(tmp_path, monkeypatch):
    output_path = str(tmp_path / "out.mp4")

    run_mock = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = editor.edit_clip(INPUT_PATH, 0.0, 10.0, output_path)

    assert result == output_path
    run_mock.assert_called_once()


def test_edit_clip_raises_runtime_error_on_ffmpeg_failure(tmp_path, monkeypatch):
    output_path = str(tmp_path / "out.mp4")

    failed_proc = MagicMock(returncode=1, stderr=b"encoder error")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=failed_proc))

    with pytest.raises(RuntimeError, match="ffmpeg editing failed"):
        editor.edit_clip(INPUT_PATH, 0.0, 10.0, output_path)


def test_edit_clip_error_message_includes_profile(tmp_path, monkeypatch):
    output_path = str(tmp_path / "out.mp4")

    failed_proc = MagicMock(returncode=1, stderr=b"bad codec")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=failed_proc))

    with pytest.raises(RuntimeError, match="gaming"):
        editor.edit_clip(INPUT_PATH, 0.0, 10.0, output_path, profile="gaming")
