"""Tests for main.py — CLI entry point covering both URL and no-arg modes.

`core.state.enqueue_job` and `core.job_runner.JobRunner.run` are mocked in
every test, so these tests never touch the filesystem, yt-dlp, or block
forever on the worker loop.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import main
from core.url_validator import InvalidURLError


VALID_URL = "https://www.youtube.com/watch?v=abc123"
INVALID_URL = "not-a-youtube-url"


def test_valid_url_enqueues_and_runs(monkeypatch, capsys):
    """A valid URL argument is validated, enqueued, and the runner started."""
    monkeypatch.setattr(main.sys, "argv", ["main.py", VALID_URL])

    with patch.object(main, "validate_url", return_value={"youtube_id": "abc123"}) as mock_validate, \
         patch.object(main.state, "enqueue_job", return_value="job-123") as mock_enqueue, \
         patch.object(main, "JobRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        main.main()

    mock_validate.assert_called_once_with(VALID_URL)
    mock_enqueue.assert_called_once_with(VALID_URL)
    mock_runner_cls.assert_called_once_with()
    mock_runner.run.assert_called_once_with()

    captured = capsys.readouterr()
    assert "job-123" in captured.out


def test_invalid_url_exits_nonzero_and_does_not_enqueue(monkeypatch, capsys):
    """An invalid URL argument exits non-zero with a clear stderr message and
    never reaches enqueue_job or JobRunner."""
    monkeypatch.setattr(main.sys, "argv", ["main.py", INVALID_URL])

    with patch.object(
        main, "validate_url", side_effect=InvalidURLError(f"Not a recognised YouTube URL: {INVALID_URL!r}")
    ) as mock_validate, \
         patch.object(main.state, "enqueue_job") as mock_enqueue, \
         patch.object(main, "JobRunner") as mock_runner_cls:
        with pytest.raises(SystemExit) as exc_info:
            main.main()

    assert exc_info.value.code == 1
    mock_validate.assert_called_once_with(INVALID_URL)
    mock_enqueue.assert_not_called()
    mock_runner_cls.assert_not_called()

    captured = capsys.readouterr()
    assert "Invalid YouTube URL" in captured.err


def test_no_args_skips_enqueue_and_runs_worker(monkeypatch):
    """No-argument mode skips validation/enqueue entirely and runs the worker."""
    monkeypatch.setattr(main.sys, "argv", ["main.py"])

    with patch.object(main, "validate_url") as mock_validate, \
         patch.object(main.state, "enqueue_job") as mock_enqueue, \
         patch.object(main, "JobRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        main.main()

    mock_validate.assert_not_called()
    mock_enqueue.assert_not_called()
    mock_runner_cls.assert_called_once_with()
    mock_runner.run.assert_called_once_with()


def test_parse_args_returns_none_when_no_argument(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    assert main.parse_args() is None


def test_parse_args_returns_url_when_provided(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["main.py", VALID_URL])
    assert main.parse_args() == VALID_URL
