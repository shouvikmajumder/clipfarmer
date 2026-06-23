"""Tests for core.url_validator — YouTube URL validation and preflight."""

from unittest.mock import patch

import pytest

from core.url_validator import (
    InvalidURLError,
    LiveStreamNotSupportedError,
    VideoTooLongError,
    VideoUnavailableError,
    validate_url,
)

VALID_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VALID_SHORT_URL = "https://youtu.be/dQw4w9WgXcQ"


def _make_info(**overrides):
    info = {
        "id": "dQw4w9WgXcQ",
        "title": "Test Video",
        "duration": 600,
        "is_live": False,
    }
    info.update(overrides)
    return info


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_valid_youtube_url_passes(mock_extract_info):
    mock_extract_info.return_value = _make_info()

    result = validate_url(VALID_URL)

    assert result["youtube_id"] == "dQw4w9WgXcQ"
    assert result["video_title"] == "Test Video"
    assert result["video_duration_s"] == 600


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_valid_short_youtube_url_passes(mock_extract_info):
    mock_extract_info.return_value = _make_info()

    result = validate_url(VALID_SHORT_URL)

    assert result["youtube_id"] == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://www.vimeo.com/12345",
        "https://example.com/watch?v=abc",
        "not a url at all",
        "",
        "ftp://youtube.com/watch?v=abc",
    ],
)
def test_non_youtube_url_rejected(bad_url):
    with pytest.raises(InvalidURLError):
        validate_url(bad_url)


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_live_stream_rejected(mock_extract_info):
    mock_extract_info.return_value = _make_info(is_live=True)

    with pytest.raises(LiveStreamNotSupportedError):
        validate_url(VALID_URL)


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_over_duration_video_rejected(mock_extract_info):
    # Hard limit is 21600s (6 hours) per settings.yaml; exceed it.
    mock_extract_info.return_value = _make_info(duration=21601)

    with pytest.raises(VideoTooLongError):
        validate_url(VALID_URL)


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_video_at_exact_limit_is_allowed(mock_extract_info):
    mock_extract_info.return_value = _make_info(duration=21600)

    result = validate_url(VALID_URL)

    assert result["video_duration_s"] == 21600


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_private_or_inaccessible_video_rejected(mock_extract_info):
    mock_extract_info.side_effect = Exception("Private video")

    with pytest.raises(VideoUnavailableError):
        validate_url(VALID_URL)


@patch("yt_dlp.YoutubeDL.YoutubeDL.extract_info")
def test_extract_info_returning_none_rejected(mock_extract_info):
    mock_extract_info.return_value = None

    with pytest.raises(VideoUnavailableError):
        validate_url(VALID_URL)
