"""Tests for processing.caption_burner — caption overlay via ffmpeg."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from processing import caption_burner

INPUT_PATH = "/tmp/fake_input.mp4"


class FakeFFmpegError(Exception):
    """Stand-in for ffmpeg.Error with a .stderr attribute."""

    def __init__(self, message: str = "boom", stderr: bytes | None = b"ffmpeg exploded"):
        super().__init__(message)
        self.stderr = stderr


@pytest.fixture
def mock_ffmpeg():
    """Inject a fake ``ffmpeg`` module mimicking the fluent API caption_burner.py uses:
    ``ffmpeg.input(...).filter("subtitles", ...).output(...).overwrite_output().run()``.
    """
    fake_module = ModuleType("ffmpeg")
    fake_module.Error = FakeFFmpegError

    filtered_stream = MagicMock(name="filtered_stream")
    output_node = MagicMock(name="output_node")
    output_node.overwrite_output.return_value = output_node
    output_node.run = MagicMock()
    filtered_stream.output.return_value = output_node

    input_stream = MagicMock(name="input_stream")
    input_stream.filter.return_value = filtered_stream

    fake_module.input = MagicMock(return_value=input_stream)

    sys.modules["ffmpeg"] = fake_module
    try:
        yield fake_module
    finally:
        del sys.modules["ffmpeg"]


def test_burn_captions_happy_path_returns_output_path(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello there"},
        {"start": 2.0, "end": 4.0, "text": "This is a clip"},
    ]

    result = caption_burner.burn_captions(INPUT_PATH, segments, output_path)

    assert result == output_path
    mock_ffmpeg.input.assert_called_once_with(INPUT_PATH)
    input_stream = mock_ffmpeg.input.return_value
    input_stream.filter.assert_called_once()
    assert input_stream.filter.call_args.args[0] == "subtitles"


def test_burn_captions_writes_and_cleans_up_srt_file(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    expected_srt_path = Path(output_path).with_suffix(".srt")
    segments = [{"start": 0.0, "end": 2.0, "text": "Hello there"}]

    caption_burner.burn_captions(INPUT_PATH, segments, output_path)

    # Cleaned up after a successful burn.
    assert not expected_srt_path.exists()


def test_burn_captions_srt_content_has_correct_timestamps_and_text(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    srt_path = Path(output_path).with_suffix(".srt")
    segments = [
        {"start": 0.0, "end": 1.5, "text": "Hello there"},
        {"start": 61.25, "end": 65.0, "text": "Second line"},
    ]

    # Capture the SRT content before cleanup by patching unlink to a no-op.
    original_unlink = Path.unlink

    captured = {}

    def fake_unlink(self, *args, **kwargs):
        if self == srt_path:
            captured["content"] = self.read_text()
        return original_unlink(self, *args, **kwargs)

    import unittest.mock as mock_module

    with mock_module.patch.object(Path, "unlink", fake_unlink):
        caption_burner.burn_captions(INPUT_PATH, segments, output_path)

    content = captured["content"]
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "Hello there" in content
    assert "00:01:01,250 --> 00:01:05,000" in content
    assert "Second line" in content


def test_burn_captions_skips_segments_with_empty_text(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    srt_path = Path(output_path).with_suffix(".srt")
    segments = [
        {"start": 0.0, "end": 1.0, "text": "   "},
        {"start": 1.0, "end": 2.0, "text": "Real text"},
    ]

    import unittest.mock as mock_module

    captured = {}
    original_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self == srt_path:
            captured["content"] = self.read_text()
        return original_unlink(self, *args, **kwargs)

    with mock_module.patch.object(Path, "unlink", fake_unlink):
        caption_burner.burn_captions(INPUT_PATH, segments, output_path)

    assert "Real text" in captured["content"]
    assert captured["content"].count("-->") == 1


def test_burn_captions_handles_empty_segments_list(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    result = caption_burner.burn_captions(INPUT_PATH, [], output_path)
    assert result == output_path


def test_burn_captions_raises_runtime_error_on_ffmpeg_failure_and_keeps_srt(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    srt_path = Path(output_path).with_suffix(".srt")
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

    filtered_stream = mock_ffmpeg.input.return_value.filter.return_value
    filtered_stream.output.return_value.overwrite_output.return_value.run.side_effect = (
        FakeFFmpegError("burn failed")
    )

    with pytest.raises(RuntimeError, match="ffmpeg caption burn failed"):
        caption_burner.burn_captions(INPUT_PATH, segments, output_path)

    # Intermediate .srt file is left on disk to aid debugging.
    assert srt_path.exists()
