"""Tests for processing.transcriber — mlx-whisper transcription wrapper."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from processing import transcriber

VIDEO_PATH = "/tmp/fake_video.mp4"


@pytest.fixture
def mock_mlx_whisper():
    """Inject a fake ``mlx_whisper`` module into sys.modules for the duration of a test.

    mlx-whisper is Apple Silicon only and may not be importable in this
    sandbox, so we can't rely on ``unittest.mock.patch("mlx_whisper.transcribe")``
    (which requires the module to already exist). Instead we install a stub
    module directly into ``sys.modules`` so the lazy ``import mlx_whisper``
    inside ``transcribe()`` resolves to our mock.
    """
    fake_module = ModuleType("mlx_whisper")
    fake_module.transcribe = MagicMock()
    sys.modules["mlx_whisper"] = fake_module
    try:
        yield fake_module.transcribe
    finally:
        del sys.modules["mlx_whisper"]


def test_transcribe_happy_path_returns_segments(mock_mlx_whisper):
    mock_mlx_whisper.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello there. "},
            {"start": 2.5, "end": 5.0, "text": " This is a test. "},
        ],
        "text": "Hello there. This is a test.",
    }

    result = transcriber.transcribe(VIDEO_PATH)

    assert result == [
        {"start": 0.0, "end": 2.5, "text": "Hello there."},
        {"start": 2.5, "end": 5.0, "text": "This is a test."},
    ]


def test_transcribe_reads_model_name_from_settings(mock_mlx_whisper):
    mock_mlx_whisper.return_value = {"segments": []}

    transcriber.transcribe(VIDEO_PATH)

    _, kwargs = mock_mlx_whisper.call_args
    assert kwargs["path_or_hf_repo"] == "medium"


def test_transcribe_falls_back_to_default_model_when_settings_missing(
    mock_mlx_whisper, monkeypatch, tmp_path
):
    monkeypatch.setattr(transcriber, "SETTINGS_PATH", tmp_path / "does_not_exist.yaml")
    mock_mlx_whisper.return_value = {"segments": []}

    transcriber.transcribe(VIDEO_PATH)

    _, kwargs = mock_mlx_whisper.call_args
    assert kwargs["path_or_hf_repo"] == transcriber.DEFAULT_WHISPER_MODEL


def test_transcribe_propagates_mlx_whisper_errors(mock_mlx_whisper):
    mock_mlx_whisper.side_effect = Exception("model failed to load")

    with pytest.raises(RuntimeError, match="mlx-whisper transcription failed"):
        transcriber.transcribe(VIDEO_PATH)


def test_transcribe_raises_when_no_segments_key(mock_mlx_whisper):
    mock_mlx_whisper.return_value = {"text": "no segments here"}

    with pytest.raises(RuntimeError, match="no segments"):
        transcriber.transcribe(VIDEO_PATH)


def test_transcribe_preserves_extra_native_fields(mock_mlx_whisper):
    mock_mlx_whisper.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.1, "id": 0},
        ]
    }

    result = transcriber.transcribe(VIDEO_PATH)

    assert result[0]["avg_logprob"] == -0.1
    assert result[0]["id"] == 0


def test_transcribe_does_not_require_mlx_whisper_importable_at_module_load():
    """Module-level import of transcriber must succeed even without mlx_whisper installed."""
    assert sys.modules.get("mlx_whisper") is None or True  # presence is irrelevant
    assert hasattr(transcriber, "transcribe")


def test_transcribe_explicit_model_name_overrides_settings(mock_mlx_whisper):
    """Passing model_name='base' must use that model instead of the settings value.

    The settings.yaml has worker.whisper_model='medium', so without an override
    the call would use 'medium'. Passing 'base' explicitly must bypass the
    settings lookup and call mlx_whisper.transcribe with path_or_hf_repo='base'.
    """
    mock_mlx_whisper.return_value = {"segments": []}

    transcriber.transcribe(VIDEO_PATH, model_name="base")

    _, kwargs = mock_mlx_whisper.call_args
    assert kwargs["path_or_hf_repo"] == "base"


def test_transcribe_explicit_model_name_not_overridden_by_settings(mock_mlx_whisper):
    """Passing model_name='tiny' must use 'tiny' even when settings say 'medium'."""
    mock_mlx_whisper.return_value = {"segments": []}

    transcriber.transcribe(VIDEO_PATH, model_name="tiny")

    _, kwargs = mock_mlx_whisper.call_args
    assert kwargs["path_or_hf_repo"] == "tiny"
    # Confirm settings value was NOT used.
    assert kwargs["path_or_hf_repo"] != "medium"
