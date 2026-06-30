"""Tests for processing.downloader — yt-dlp download wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from processing import downloader

JOB = {
    "id": "test-job-123",
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "youtube_id": "dQw4w9WgXcQ",
}


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Redirect downloader's JOBS_DIR to a temp directory for hermetic tests."""
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(downloader, "DATA_DIR", tmp_path)
    monkeypatch.setattr(downloader, "JOBS_DIR", jobs_dir)
    return jobs_dir


def _touch_output_file(jobs_dir: Path, job_id: str, name: str = "dQw4w9WgXcQ.mp4") -> Path:
    raw_dir = jobs_dir / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = raw_dir / name
    out_file.write_bytes(b"fake video bytes")
    return out_file


@patch("yt_dlp.YoutubeDL")
def test_download_happy_path_returns_expected_path(mock_youtube_dl_cls, _isolate_data_dir):
    jobs_dir = _isolate_data_dir
    expected_path = _touch_output_file(jobs_dir, JOB["id"])

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "dQw4w9WgXcQ", "ext": "mp4"}
    mock_ydl.prepare_filename.return_value = str(expected_path)
    mock_youtube_dl_cls.return_value.__enter__.return_value = mock_ydl

    result = downloader.download(JOB)

    assert result == str(expected_path.resolve())
    mock_ydl.extract_info.assert_called_once_with(JOB["youtube_url"], download=True)


@patch("yt_dlp.YoutubeDL")
def test_download_requests_capped_resolution(mock_youtube_dl_cls, _isolate_data_dir):
    jobs_dir = _isolate_data_dir
    expected_path = _touch_output_file(jobs_dir, JOB["id"])

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "dQw4w9WgXcQ", "ext": "mp4"}
    mock_ydl.prepare_filename.return_value = str(expected_path)
    mock_youtube_dl_cls.return_value.__enter__.return_value = mock_ydl

    downloader.download(JOB)

    # Inspect the ydl_opts passed to YoutubeDL() to confirm the resolution cap.
    call_args, _ = mock_youtube_dl_cls.call_args
    ydl_opts = call_args[0]
    assert "1080" in ydl_opts["format"]


@patch("yt_dlp.YoutubeDL")
def test_download_propagates_ytdlp_errors(mock_youtube_dl_cls, _isolate_data_dir):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = Exception("network unreachable")
    mock_youtube_dl_cls.return_value.__enter__.return_value = mock_ydl

    with pytest.raises(RuntimeError, match="yt-dlp download failed"):
        downloader.download(JOB)


@patch("yt_dlp.YoutubeDL")
def test_download_raises_when_output_file_missing(mock_youtube_dl_cls, _isolate_data_dir):
    jobs_dir = _isolate_data_dir
    missing_path = jobs_dir / JOB["id"] / "raw" / "dQw4w9WgXcQ.mp4"

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "dQw4w9WgXcQ", "ext": "mp4"}
    mock_ydl.prepare_filename.return_value = str(missing_path)
    mock_youtube_dl_cls.return_value.__enter__.return_value = mock_ydl

    with pytest.raises(RuntimeError, match="output file is missing"):
        downloader.download(JOB)
