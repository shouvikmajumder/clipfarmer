"""Tests for processing.caption_burner — word-level caption overlay via ffmpeg.

All tests are pure: no real FFmpeg binary or Whisper model is invoked.
Pure helpers (``should_caption``, ``words_to_srt``, ``group_words_into_lines``,
``build_burn_cmd``) are tested directly.  ``caption_clip`` is tested via a
monkeypatched ``subprocess.run``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from processing import caption_burner


INPUT_PATH = "/tmp/fake_input.mp4"


def _make_words(n: int, start_offset: float = 0.0) -> list[dict]:
    """Return *n* single-word dicts with sequential 0.5-second time slots."""
    return [
        {"word": f"w{i}", "start": start_offset + i * 0.5, "end": start_offset + i * 0.5 + 0.4}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# should_caption — profile skip logic
# ---------------------------------------------------------------------------


def test_gaming_is_not_captioned():
    assert caption_burner.should_caption("gaming") is False


def test_irl_is_not_captioned():
    assert caption_burner.should_caption("irl") is False


def test_podcast_is_captioned():
    assert caption_burner.should_caption("podcast") is True


def test_interview_is_captioned():
    assert caption_burner.should_caption("interview") is True


def test_default_is_captioned():
    assert caption_burner.should_caption("default") is True


def test_should_caption_is_case_insensitive_skip():
    assert caption_burner.should_caption("Gaming") is False
    assert caption_burner.should_caption("IRL") is False
    assert caption_burner.should_caption("GAMING") is False


def test_should_caption_is_case_insensitive_allow():
    assert caption_burner.should_caption("Podcast") is True
    assert caption_burner.should_caption("INTERVIEW") is True


# ---------------------------------------------------------------------------
# _format_srt_timestamp
# ---------------------------------------------------------------------------


def test_format_srt_timestamp_zero():
    assert caption_burner._format_srt_timestamp(0.0) == "00:00:00,000"


def test_format_srt_timestamp_simple():
    assert caption_burner._format_srt_timestamp(12.0) == "00:00:12,000"


def test_format_srt_timestamp_millis():
    assert caption_burner._format_srt_timestamp(12.5) == "00:00:12,500"


def test_format_srt_timestamp_minutes():
    assert caption_burner._format_srt_timestamp(61.25) == "00:01:01,250"


def test_format_srt_timestamp_hours():
    assert caption_burner._format_srt_timestamp(3661.0) == "01:01:01,000"


def test_format_srt_timestamp_negative_clamped_to_zero():
    assert caption_burner._format_srt_timestamp(-5.0) == "00:00:00,000"


# ---------------------------------------------------------------------------
# words_to_srt — offset arithmetic
# ---------------------------------------------------------------------------


def test_words_to_srt_with_offset_start_time():
    """start=12.0, offset=10.0 → block start should be 00:00:02,000."""
    words = [{"word": "hi", "start": 12.0, "end": 12.5}]
    srt = caption_burner.words_to_srt(words, offset_s=10.0)
    assert "00:00:02,000" in srt


def test_words_to_srt_with_offset_end_time():
    """start=12.0, end=12.5, offset=10.0 → block end should be 00:00:02,500."""
    words = [{"word": "hi", "start": 12.0, "end": 12.5}]
    srt = caption_burner.words_to_srt(words, offset_s=10.0)
    assert "00:00:02,500" in srt


def test_words_to_srt_no_offset_maps_start_correctly():
    """start=12.0, no offset → 00:00:12,000."""
    words = [{"word": "hi", "start": 12.0, "end": 12.5}]
    srt = caption_burner.words_to_srt(words, offset_s=0.0)
    assert "00:00:12,000" in srt


def test_words_to_srt_offset_block_contains_arrow():
    words = [{"word": "hi", "start": 12.0, "end": 12.5}]
    srt = caption_burner.words_to_srt(words, offset_s=10.0)
    assert "00:00:02,000 --> 00:00:02,500" in srt


def test_words_to_srt_empty_returns_empty_string():
    assert caption_burner.words_to_srt([]) == ""


def test_words_to_srt_index_is_1_based():
    words = [{"word": "hello", "start": 0.0, "end": 0.5}]
    srt = caption_burner.words_to_srt(words)
    lines = srt.strip().splitlines()
    assert lines[0] == "1"


# ---------------------------------------------------------------------------
# group_words_into_lines — chunking behaviour
# ---------------------------------------------------------------------------


def test_group_words_chunk_sizes_13_words():
    """13 words with MAX_WORDS_PER_LINE=6 → chunks [6, 6, 1]."""
    words = _make_words(13)
    groups = caption_burner.group_words_into_lines(words)
    assert [len(g) for g in groups] == [6, 6, 1]


def test_group_words_exactly_6_words_is_one_chunk():
    words = _make_words(6)
    groups = caption_burner.group_words_into_lines(words)
    assert len(groups) == 1
    assert len(groups[0]) == 6


def test_group_words_7_words_gives_two_chunks():
    words = _make_words(7)
    groups = caption_burner.group_words_into_lines(words)
    assert [len(g) for g in groups] == [6, 1]


def test_group_words_empty_returns_empty():
    assert caption_burner.group_words_into_lines([]) == []


def test_words_to_srt_13_words_produces_3_blocks():
    """13 words → 3 SRT blocks (indices 1, 2, 3)."""
    words = _make_words(13)
    srt = caption_burner.words_to_srt(words)
    # Count index lines: they are lines that are just a bare integer.
    index_lines = [ln for ln in srt.splitlines() if ln.strip().isdigit()]
    assert index_lines == ["1", "2", "3"]


def test_words_to_srt_13_words_has_3_arrow_lines():
    words = _make_words(13)
    srt = caption_burner.words_to_srt(words)
    arrow_lines = [ln for ln in srt.splitlines() if " --> " in ln]
    assert len(arrow_lines) == 3


# ---------------------------------------------------------------------------
# build_burn_cmd — structure and codec flags
# ---------------------------------------------------------------------------


def test_build_burn_cmd_contains_h264_videotoolbox():
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", "/tmp/out.mp4")
    assert "h264_videotoolbox" in cmd


def test_build_burn_cmd_contains_copy_audio():
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", "/tmp/out.mp4")
    assert "-c:a" in cmd
    idx = cmd.index("-c:a")
    assert cmd[idx + 1] == "copy"


def test_build_burn_cmd_contains_subtitles_vf():
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", "/tmp/out.mp4")
    vf_str = " ".join(cmd)
    assert "subtitles=" in vf_str


def test_build_burn_cmd_contains_force_style():
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", "/tmp/out.mp4")
    vf_str = " ".join(cmd)
    assert "force_style=" in vf_str


def test_build_burn_cmd_starts_ffmpeg_y():
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", "/tmp/out.mp4")
    assert cmd[0] == "ffmpeg"
    assert cmd[1] == "-y"


def test_build_burn_cmd_ends_with_output_path():
    out = "/tmp/captioned_out.mp4"
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", out)
    assert cmd[-1] == out


def test_build_burn_cmd_uses_vf_flag():
    cmd = caption_burner.build_burn_cmd(INPUT_PATH, "/tmp/sub.srt", "/tmp/out.mp4")
    assert "-vf" in cmd


# ---------------------------------------------------------------------------
# SRT format validity
# ---------------------------------------------------------------------------


def test_srt_block_has_index_line():
    words = [{"word": "hello", "start": 0.0, "end": 0.5}]
    srt = caption_burner.words_to_srt(words)
    lines = srt.strip().splitlines()
    assert lines[0].strip().isdigit()


def test_srt_block_has_timestamp_line_with_arrow():
    words = [{"word": "hello", "start": 0.0, "end": 0.5}]
    srt = caption_burner.words_to_srt(words)
    lines = srt.strip().splitlines()
    # Second line should be the timestamp line.
    assert " --> " in lines[1]


def test_srt_block_has_text_line():
    words = [{"word": "hello", "start": 0.0, "end": 0.5}]
    srt = caption_burner.words_to_srt(words)
    lines = srt.strip().splitlines()
    assert lines[2] == "hello"


# ---------------------------------------------------------------------------
# caption_clip — cache hit skips FFmpeg and SRT write
# ---------------------------------------------------------------------------


def test_caption_clip_cache_hit_skips_ffmpeg(tmp_path, monkeypatch):
    """Non-empty output_path must short-circuit before writing SRT or running FFmpeg."""
    output_path = tmp_path / "captioned.mp4"
    output_path.write_bytes(b"not empty")  # simulate a pre-existing encoded file

    srt_path = tmp_path / "sub.srt"

    run_mock = MagicMock(side_effect=AssertionError("subprocess.run must not be called on cache hit"))
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = caption_burner.caption_clip(
        INPUT_PATH,
        _make_words(3),
        str(srt_path),
        str(output_path),
    )

    assert result == str(output_path)
    run_mock.assert_not_called()
    # SRT file must NOT have been written on a cache hit.
    assert not srt_path.exists()


def test_caption_clip_no_cache_hit_calls_ffmpeg(tmp_path, monkeypatch):
    """When output does not exist, FFmpeg must be invoked."""
    output_path = str(tmp_path / "captioned.mp4")
    srt_path = str(tmp_path / "sub.srt")

    run_mock = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = caption_burner.caption_clip(
        INPUT_PATH,
        _make_words(3),
        srt_path,
        output_path,
    )

    assert result == output_path
    run_mock.assert_called_once()


def test_caption_clip_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    output_path = str(tmp_path / "captioned.mp4")
    srt_path = str(tmp_path / "sub.srt")

    failed_proc = MagicMock(returncode=1, stderr=b"filter error")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=failed_proc))

    with pytest.raises(RuntimeError, match="ffmpeg caption burn failed"):
        caption_burner.caption_clip(
            INPUT_PATH,
            _make_words(3),
            srt_path,
            output_path,
        )


def test_caption_clip_keeps_srt_on_failure(tmp_path, monkeypatch):
    """On FFmpeg failure the SRT file must remain on disk for debugging."""
    output_path = str(tmp_path / "captioned.mp4")
    srt_path = tmp_path / "sub.srt"

    failed_proc = MagicMock(returncode=1, stderr=b"error")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=failed_proc))

    with pytest.raises(RuntimeError):
        caption_burner.caption_clip(
            INPUT_PATH,
            _make_words(3),
            str(srt_path),
            output_path,
        )

    assert srt_path.exists()


def test_caption_clip_deletes_srt_on_success(tmp_path, monkeypatch):
    """On success the intermediate SRT must be cleaned up."""
    output_path = str(tmp_path / "captioned.mp4")
    srt_path = tmp_path / "sub.srt"

    run_mock = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run_mock)

    caption_burner.caption_clip(
        INPUT_PATH,
        _make_words(3),
        str(srt_path),
        output_path,
    )

    assert not srt_path.exists()


# ---------------------------------------------------------------------------
# Escaping + ffmpeg binary + subtitles-filter availability (fixes from smoke test)
# ---------------------------------------------------------------------------


def test_build_burn_cmd_escapes_force_style_commas():
    """force_style commas must be backslash-escaped (so the filtergraph parser
    doesn't read them as filter separators) and NOT wrapped in literal quotes."""
    cmd = caption_burner.build_burn_cmd("in.mp4", "sub.srt", "out.mp4")
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("subtitles=sub.srt:force_style=")
    assert r"\," in vf  # commas escaped
    assert "force_style='" not in vf  # no spurious single quotes
    # every comma in the value is part of an escaped "\," sequence
    assert vf.count(",") == vf.count(r"\,")


def test_build_burn_cmd_respects_ffmpeg_bin():
    cmd = caption_burner.build_burn_cmd("in.mp4", "sub.srt", "out.mp4", ffmpeg_bin="/opt/ff/ffmpeg")
    assert cmd[0] == "/opt/ff/ffmpeg"
    # default is plain "ffmpeg"
    assert caption_burner.build_burn_cmd("in.mp4", "sub.srt", "out.mp4")[0] == "ffmpeg"


def test_subtitles_filter_available_true(monkeypatch):
    fake = MagicMock(returncode=0, stdout=" T.. subtitles         V->V       Render text subtitles\n")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    assert caption_burner.subtitles_filter_available("ffmpeg") is True


def test_subtitles_filter_available_false_when_absent(monkeypatch):
    fake = MagicMock(returncode=0, stdout=" T.. scale            V->V       Scale\n T.. overlay          VV->V    Overlay\n")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    assert caption_burner.subtitles_filter_available("ffmpeg") is False


def test_subtitles_filter_available_false_on_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("no ffmpeg")))
    assert caption_burner.subtitles_filter_available("ffmpeg") is False
