"""Tests for processing.editor — 9:16 crop and trim logic."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from processing import editor

INPUT_PATH = "/tmp/fake_input.mp4"
OUTPUT_PATH = "/tmp/fake_output.mp4"


class FakeFFmpegError(Exception):
    """Stand-in for ffmpeg.Error with a .stderr attribute."""

    def __init__(self, message: str = "boom", stderr: bytes | None = b"ffmpeg exploded"):
        super().__init__(message)
        self.stderr = stderr


@pytest.fixture
def mock_ffmpeg():
    """Inject a fake ``ffmpeg`` module into sys.modules.

    ffmpeg-python may not be importable in all sandboxes, and even when it
    is, we don't want to shell out to a real ffmpeg binary. We build a fake
    module that mimics the small slice of the fluent API editor.py uses:
    ``ffmpeg.probe``, ``ffmpeg.input(...).video.filter(...).filter(...)``,
    ``ffmpeg.input(...).audio``, ``ffmpeg.output(...)``, ``.overwrite_output()``,
    ``.run()``, and the ``ffmpeg.Error`` exception class.
    """
    fake_module = ModuleType("ffmpeg")
    fake_module.Error = FakeFFmpegError
    fake_module.probe = MagicMock()

    # input(...) returns a stream-ish object with .video/.audio properties
    # that themselves support .filter(...) chaining and return self-like
    # mocks so chained calls don't blow up.
    input_stream = MagicMock(name="input_stream")
    video_stream = MagicMock(name="video_stream")
    video_stream.filter.return_value = video_stream
    audio_stream = MagicMock(name="audio_stream")
    input_stream.video = video_stream
    input_stream.audio = audio_stream

    fake_module.input = MagicMock(return_value=input_stream)

    output_node = MagicMock(name="output_node")
    output_node.overwrite_output.return_value = output_node
    output_node.run = MagicMock()
    fake_module.output = MagicMock(return_value=output_node)

    sys.modules["ffmpeg"] = fake_module
    try:
        yield fake_module
    finally:
        del sys.modules["ffmpeg"]


def _probe_result(width: int, height: int) -> dict:
    return {
        "streams": [
            {"codec_type": "video", "width": width, "height": height},
        ]
    }


def test_crop_to_vertical_raises_on_invalid_time_range(mock_ffmpeg):
    with pytest.raises(ValueError, match="end_s"):
        editor.crop_to_vertical(INPUT_PATH, start_s=10, end_s=5, output_path=OUTPUT_PATH)


def test_crop_to_vertical_skips_crop_when_already_vertical(mock_ffmpeg):
    """Source already ~9:16 (1080x1920) should skip the crop filter."""
    mock_ffmpeg.probe.return_value = _probe_result(1080, 1920)

    result = editor.crop_to_vertical(INPUT_PATH, start_s=0, end_s=10, output_path=OUTPUT_PATH)

    assert result == OUTPUT_PATH
    video_stream = mock_ffmpeg.input.return_value.video
    # Only "scale" should be applied, never "crop".
    crop_calls = [c for c in video_stream.filter.call_args_list if c.args[0] == "crop"]
    scale_calls = [c for c in video_stream.filter.call_args_list if c.args[0] == "scale"]
    assert crop_calls == []
    assert len(scale_calls) == 1
    assert scale_calls[0].args[1:] == (editor.TARGET_WIDTH, editor.TARGET_HEIGHT)


def test_crop_to_vertical_applies_center_crop_for_landscape_source(mock_ffmpeg):
    """A 16:9 landscape source should get a center-crop before scaling."""
    mock_ffmpeg.probe.return_value = _probe_result(1920, 1080)

    result = editor.crop_to_vertical(INPUT_PATH, start_s=5, end_s=15, output_path=OUTPUT_PATH)

    assert result == OUTPUT_PATH
    video_stream = mock_ffmpeg.input.return_value.video
    crop_calls = [c for c in video_stream.filter.call_args_list if c.args[0] == "crop"]
    scale_calls = [c for c in video_stream.filter.call_args_list if c.args[0] == "scale"]
    assert len(crop_calls) == 1
    assert len(scale_calls) == 1


def test_crop_to_vertical_passes_trim_window_to_ffmpeg_input(mock_ffmpeg):
    mock_ffmpeg.probe.return_value = _probe_result(1080, 1920)

    editor.crop_to_vertical(INPUT_PATH, start_s=12.5, end_s=37.5, output_path=OUTPUT_PATH)

    _, kwargs = mock_ffmpeg.input.call_args
    assert kwargs["ss"] == 12.5
    assert kwargs["t"] == 25.0


def test_crop_to_vertical_calls_overwrite_and_run(mock_ffmpeg):
    mock_ffmpeg.probe.return_value = _probe_result(1080, 1920)

    editor.crop_to_vertical(INPUT_PATH, start_s=0, end_s=10, output_path=OUTPUT_PATH)

    output_node = mock_ffmpeg.output.return_value
    output_node.overwrite_output.assert_called_once()
    output_node.overwrite_output.return_value.run.assert_called_once()


def test_crop_to_vertical_raises_runtime_error_on_probe_failure(mock_ffmpeg):
    mock_ffmpeg.probe.side_effect = FakeFFmpegError("probe failed")

    with pytest.raises(RuntimeError, match="ffprobe failed"):
        editor.crop_to_vertical(INPUT_PATH, start_s=0, end_s=10, output_path=OUTPUT_PATH)


def test_crop_to_vertical_raises_runtime_error_on_ffmpeg_run_failure(mock_ffmpeg):
    mock_ffmpeg.probe.return_value = _probe_result(1080, 1920)
    mock_ffmpeg.output.return_value.overwrite_output.return_value.run.side_effect = (
        FakeFFmpegError("encode failed")
    )

    with pytest.raises(RuntimeError, match="ffmpeg crop/trim failed"):
        editor.crop_to_vertical(INPUT_PATH, start_s=0, end_s=10, output_path=OUTPUT_PATH)


def test_crop_to_vertical_raises_on_no_video_stream(mock_ffmpeg):
    mock_ffmpeg.probe.return_value = {"streams": [{"codec_type": "audio"}]}

    with pytest.raises(RuntimeError, match="No video stream"):
        editor.crop_to_vertical(INPUT_PATH, start_s=0, end_s=10, output_path=OUTPUT_PATH)
