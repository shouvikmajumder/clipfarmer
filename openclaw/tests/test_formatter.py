"""Tests for processing.formatter — final encode and size enforcement."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from processing import formatter

INPUT_PATH = "/tmp/fake_input.mp4"


class FakeFFmpegError(Exception):
    """Stand-in for ffmpeg.Error with a .stderr attribute."""

    def __init__(self, message: str = "boom", stderr: bytes | None = b"ffmpeg exploded"):
        super().__init__(message)
        self.stderr = stderr


@pytest.fixture
def mock_ffmpeg():
    """Inject a fake ``ffmpeg`` module mimicking the fluent API formatter.py uses:
    ``ffmpeg.probe``, ``ffmpeg.input(...).output(...).overwrite_output().run()``.
    """
    fake_module = ModuleType("ffmpeg")
    fake_module.Error = FakeFFmpegError
    fake_module.probe = MagicMock()

    output_node = MagicMock(name="output_node")
    output_node.overwrite_output.return_value = output_node
    output_node.run = MagicMock()

    input_stream = MagicMock(name="input_stream")
    input_stream.output.return_value = output_node

    fake_module.input = MagicMock(return_value=input_stream)

    sys.modules["ffmpeg"] = fake_module
    try:
        yield fake_module
    finally:
        del sys.modules["ffmpeg"]


def _make_output_file(path: str, size_bytes: int) -> None:
    Path(path).write_bytes(b"0" * size_bytes)


def test_format_clip_happy_path_under_size_limit(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.return_value = {"format": {"duration": "30.0"}}
    mock_ffmpeg.input.return_value.output.side_effect = lambda *a, **kw: (
        _make_output_file(output_path, 10 * 1024 * 1024) or mock_ffmpeg.input.return_value.output.return_value
    )

    result = formatter.format_clip(INPUT_PATH, output_path, max_duration_s=60, max_size_mb=50)

    assert result == output_path
    assert Path(output_path).stat().st_size == 10 * 1024 * 1024
    # Only one encode attempt needed.
    assert mock_ffmpeg.input.call_count == 1


def test_format_clip_trims_when_source_duration_exceeds_max(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.return_value = {"format": {"duration": "90.0"}}

    def fake_output(*args, **kwargs):
        _make_output_file(output_path, 1024)
        return mock_ffmpeg.input.return_value.output.return_value

    mock_ffmpeg.input.return_value.output.side_effect = fake_output

    formatter.format_clip(INPUT_PATH, output_path, max_duration_s=60, max_size_mb=50)

    _, kwargs = mock_ffmpeg.input.return_value.output.call_args
    assert kwargs["t"] == 60.0


def test_format_clip_reads_max_duration_from_settings_when_not_provided(
    mock_ffmpeg, tmp_path, monkeypatch
):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("general:\n  max_clip_length_s: 45\n")
    monkeypatch.setattr(formatter, "SETTINGS_PATH", settings_path)

    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.return_value = {"format": {"duration": "90.0"}}

    def fake_output(*args, **kwargs):
        _make_output_file(output_path, 1024)
        return mock_ffmpeg.input.return_value.output.return_value

    mock_ffmpeg.input.return_value.output.side_effect = fake_output

    formatter.format_clip(INPUT_PATH, output_path, max_size_mb=50)

    _, kwargs = mock_ffmpeg.input.return_value.output.call_args
    assert kwargs["t"] == 45.0


def test_format_clip_retries_with_lower_bitrate_when_over_size(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.return_value = {"format": {"duration": "30.0"}}

    # First attempt: too big (60MB). Second attempt: fits (40MB).
    sizes = [60 * 1024 * 1024, 40 * 1024 * 1024]

    def fake_output(*args, **kwargs):
        size = sizes.pop(0)
        _make_output_file(output_path, size)
        return mock_ffmpeg.input.return_value.output.return_value

    mock_ffmpeg.input.return_value.output.side_effect = fake_output

    result = formatter.format_clip(INPUT_PATH, output_path, max_duration_s=60, max_size_mb=50)

    assert result == output_path
    assert Path(output_path).stat().st_size == 40 * 1024 * 1024
    assert mock_ffmpeg.input.call_count == 2

    # Bitrate should have been lowered on the second attempt.
    first_call_kwargs = mock_ffmpeg.input.return_value.output.call_args_list[0].kwargs
    second_call_kwargs = mock_ffmpeg.input.return_value.output.call_args_list[1].kwargs
    first_bitrate = int(first_call_kwargs["video_bitrate"].rstrip("k"))
    second_bitrate = int(second_call_kwargs["video_bitrate"].rstrip("k"))
    assert second_bitrate < first_bitrate


def test_format_clip_raises_after_max_retries_still_over_size(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.return_value = {"format": {"duration": "30.0"}}

    def fake_output(*args, **kwargs):
        _make_output_file(output_path, 60 * 1024 * 1024)  # always too big
        return mock_ffmpeg.input.return_value.output.return_value

    mock_ffmpeg.input.return_value.output.side_effect = fake_output

    with pytest.raises(RuntimeError, match="still exceeds"):
        formatter.format_clip(INPUT_PATH, output_path, max_duration_s=60, max_size_mb=50)

    assert mock_ffmpeg.input.call_count == formatter.MAX_SIZE_RETRIES


def test_format_clip_raises_runtime_error_on_ffmpeg_failure(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.return_value = {"format": {"duration": "30.0"}}
    mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run.side_effect = (
        FakeFFmpegError("encode crashed")
    )

    with pytest.raises(RuntimeError, match="ffmpeg final encode failed"):
        formatter.format_clip(INPUT_PATH, output_path, max_duration_s=60, max_size_mb=50)


def test_format_clip_raises_runtime_error_on_probe_failure(mock_ffmpeg, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    mock_ffmpeg.probe.side_effect = FakeFFmpegError("probe failed")

    with pytest.raises(RuntimeError, match="ffprobe failed"):
        formatter.format_clip(INPUT_PATH, output_path, max_duration_s=60, max_size_mb=50)
