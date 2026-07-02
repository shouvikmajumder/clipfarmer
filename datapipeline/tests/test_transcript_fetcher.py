"""Tests for processing.transcript_fetcher.fetch_transcript.

youtube-transcript-api is NOT installed. All tests inject a fake module into
sys.modules to simulate the library's behaviour at the lazy import boundary.

Two generation APIs are supported:
  - v0.6.x: ``YouTubeTranscriptApi.get_transcript(youtube_id)`` classmethod.
  - v1.x:   ``YouTubeTranscriptApi().fetch(youtube_id).to_raw_data()`` instance.

Tests verify:
  - ImportError => None (library absent).
  - Fake v0.6.x API returning raw entries => sorted list[dict] with start/end/text.
  - Fake v1.x API returning raw entries => same output.
  - Any API exception => None (non-fatal).
  - Empty raw response => None.
  - Output is sorted by start time.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from processing import transcript_fetcher


YOUTUBE_ID = "dQw4w9WgXcQ"

# A minimal raw transcript entry as returned by the library.
RAW_ENTRIES = [
    {"text": "Hello world", "start": 5.0, "duration": 2.0},
    {"text": "This is a test", "start": 0.0, "duration": 3.0},
    {"text": "Goodbye world", "start": 10.0, "duration": 2.5},
]


def _build_fake_api_v06(raw_entries: list[dict]) -> type:
    """Return a fake ``YouTubeTranscriptApi`` class that uses the v0.6.x classmethod API.

    The class has no ``fetch`` attribute (so _get_raw_transcript falls through
    to the ``get_transcript`` classmethod path).
    """
    class FakeYouTubeTranscriptApi:
        @classmethod
        def get_transcript(cls, youtube_id: str) -> list[dict]:
            return raw_entries

    return FakeYouTubeTranscriptApi


def _build_fake_api_v1(raw_entries: list[dict]) -> type:
    """Return a fake ``YouTubeTranscriptApi`` class that uses the v1.x instance API.

    The class has a ``fetch`` instance method that returns an object with
    ``to_raw_data()``.
    """
    class FakeFetched:
        def to_raw_data(self) -> list[dict]:
            return raw_entries

    class FakeYouTubeTranscriptApi:
        def fetch(self, youtube_id: str) -> FakeFetched:
            return FakeFetched()

    return FakeYouTubeTranscriptApi


def _inject_api_class(api_cls, monkeypatch) -> None:
    """Inject a fake youtube_transcript_api module with the given API class."""
    fake_module = ModuleType("youtube_transcript_api")
    fake_module.YouTubeTranscriptApi = api_cls
    # Also define exception classes so the import in fetch_transcript succeeds.
    fake_module.TranscriptsDisabled = type("TranscriptsDisabled", (Exception,), {})
    fake_module.NoTranscriptFound = type("NoTranscriptFound", (Exception,), {})
    fake_module.VideoUnavailable = type("VideoUnavailable", (Exception,), {})
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)


# ---------------------------------------------------------------------------
# ImportError => None
# ---------------------------------------------------------------------------


def test_fetch_transcript_returns_none_when_library_absent(monkeypatch):
    """When youtube-transcript-api is not installed, fetch_transcript returns None."""
    # Make sure the module is NOT in sys.modules (simulates absent library).
    monkeypatch.delitem(sys.modules, "youtube_transcript_api", raising=False)

    # Patch the import inside _get_raw_transcript so it raises ImportError.
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    with patch.object(transcript_fetcher, "fetch_transcript", wraps=None) as _:
        pass  # just verify the wrapping mechanism exists

    # The cleanest way: inject a module whose YouTubeTranscriptApi raises ImportError
    # at the point of import. We simulate this by removing the module from sys.modules
    # and patching the lazy import to raise.
    def _failing_import(name, *args, **kwargs):
        if name == "youtube_transcript_api":
            raise ImportError("youtube-transcript-api not installed")
        return __import__(name, *args, **kwargs)

    import builtins
    with patch.object(builtins, "__import__", side_effect=_failing_import):
        result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is None


# ---------------------------------------------------------------------------
# v0.6.x API path
# ---------------------------------------------------------------------------


def test_fetch_transcript_v06_returns_sorted_segments(monkeypatch):
    """v0.6.x classmethod API: returns sorted list[dict] with start/end/text."""
    api_cls = _build_fake_api_v06(RAW_ENTRIES)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    assert len(result) == 3
    # Should be sorted by start time.
    starts = [seg["start"] for seg in result]
    assert starts == sorted(starts)


def test_fetch_transcript_v06_segment_shape(monkeypatch):
    """Each segment has start (float), end (float = start+duration), text (str)."""
    api_cls = _build_fake_api_v06(RAW_ENTRIES)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    for seg in result:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert isinstance(seg["text"], str)


def test_fetch_transcript_v06_end_equals_start_plus_duration(monkeypatch):
    """end = start + duration for each segment."""
    raw = [{"text": "hello", "start": 3.0, "duration": 2.5}]
    api_cls = _build_fake_api_v06(raw)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(3.0)
    assert result[0]["end"] == pytest.approx(5.5)
    assert result[0]["text"] == "hello"


def test_fetch_transcript_v06_text_is_stripped(monkeypatch):
    """Whitespace is stripped from text."""
    raw = [{"text": "  hello world  ", "start": 0.0, "duration": 2.0}]
    api_cls = _build_fake_api_v06(raw)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    assert result[0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# v1.x API path
# ---------------------------------------------------------------------------


def test_fetch_transcript_v1_returns_sorted_segments(monkeypatch):
    """v1.x instance API: same output as v0.6.x — sorted by start."""
    api_cls = _build_fake_api_v1(RAW_ENTRIES)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    assert len(result) == 3
    starts = [seg["start"] for seg in result]
    assert starts == sorted(starts)


def test_fetch_transcript_v1_segment_shape(monkeypatch):
    """v1.x: segment has start, end, text keys with correct types."""
    api_cls = _build_fake_api_v1(RAW_ENTRIES)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    for seg in result:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg


# ---------------------------------------------------------------------------
# API exception => None
# ---------------------------------------------------------------------------


def test_fetch_transcript_api_exception_returns_none(monkeypatch):
    """Any exception from the library returns None (non-fatal)."""
    class FailingApi:
        @classmethod
        def get_transcript(cls, youtube_id: str) -> list[dict]:
            raise RuntimeError("API is down")

    _inject_api_class(FailingApi, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)
    assert result is None


def test_fetch_transcript_known_error_returns_none(monkeypatch):
    """TranscriptsDisabled, NoTranscriptFound, VideoUnavailable => None."""
    fake_module = ModuleType("youtube_transcript_api")

    class TranscriptsDisabled(Exception):
        pass

    class FailingApi:
        @classmethod
        def get_transcript(cls, youtube_id: str) -> list[dict]:
            raise TranscriptsDisabled("transcripts disabled")

    fake_module.YouTubeTranscriptApi = FailingApi
    fake_module.TranscriptsDisabled = TranscriptsDisabled
    fake_module.NoTranscriptFound = type("NoTranscriptFound", (Exception,), {})
    fake_module.VideoUnavailable = type("VideoUnavailable", (Exception,), {})

    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)
    assert result is None


# ---------------------------------------------------------------------------
# Empty response => None
# ---------------------------------------------------------------------------


def test_fetch_transcript_empty_raw_returns_none(monkeypatch):
    """An empty list returned by the API => None (triggers fallback)."""
    api_cls = _build_fake_api_v06([])
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)
    assert result is None


def test_fetch_transcript_all_empty_text_returns_none(monkeypatch):
    """Entries with empty text still produce segments (text=''), not None.

    The function only returns None for a *completely empty* raw list.  Empty
    text entries are still valid segments — they just have text=''.
    """
    raw = [{"text": "", "start": 0.0, "duration": 2.0}]
    api_cls = _build_fake_api_v06(raw)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)
    # A single empty-text entry still produces one segment (text is stripped to '').
    # After conversion, segments list has 1 entry => result is not None.
    assert result is not None
    assert len(result) == 1
    assert result[0]["text"] == ""


# ---------------------------------------------------------------------------
# Sorting guarantee
# ---------------------------------------------------------------------------


def test_fetch_transcript_output_sorted_by_start(monkeypatch):
    """Output must be sorted by start time regardless of input order."""
    raw_unsorted = [
        {"text": "third", "start": 30.0, "duration": 2.0},
        {"text": "first", "start": 0.0, "duration": 2.0},
        {"text": "second", "start": 15.0, "duration": 2.0},
    ]
    api_cls = _build_fake_api_v06(raw_unsorted)
    _inject_api_class(api_cls, monkeypatch)

    result = transcript_fetcher.fetch_transcript(YOUTUBE_ID)

    assert result is not None
    assert result[0]["text"] == "first"
    assert result[1]["text"] == "second"
    assert result[2]["text"] == "third"
