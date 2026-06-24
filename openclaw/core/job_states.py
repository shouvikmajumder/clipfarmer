"""Job state machine: valid states and allowed transitions.

``JobState`` is the single source of truth for what states a job may be in.
``VALID_TRANSITIONS`` encodes which state changes are legal, preventing bugs
where a job skips stages or moves backwards unexpectedly.

Pipeline stage order:
    downloading -> detecting -> editing -> captioning -> formatting -> posting -> complete
"""

from __future__ import annotations

from enum import Enum


class JobState(str, Enum):
    """Enumeration of all possible states a processing job may occupy."""

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DETECTING = "detecting"
    EDITING = "editing"
    CAPTIONING = "captioning"
    FORMATTING = "formatting"
    POSTING = "posting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


VALID_TRANSITIONS: dict[JobState, list[JobState]] = {
    JobState.QUEUED: [JobState.DOWNLOADING, JobState.FAILED, JobState.CANCELLED],
    JobState.DOWNLOADING: [JobState.DETECTING, JobState.FAILED, JobState.CANCELLED],
    JobState.DETECTING: [JobState.EDITING, JobState.FAILED, JobState.CANCELLED],
    JobState.EDITING: [JobState.CAPTIONING, JobState.FAILED, JobState.CANCELLED],
    JobState.CAPTIONING: [JobState.FORMATTING, JobState.FAILED, JobState.CANCELLED],
    JobState.FORMATTING: [JobState.POSTING, JobState.FAILED, JobState.CANCELLED],
    JobState.POSTING: [JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED],
    JobState.COMPLETE: [],
    JobState.FAILED: [JobState.QUEUED],  # Allow re-queuing a failed job.
    JobState.CANCELLED: [],
}


def assert_valid_transition(from_state: JobState, to_state: JobState) -> None:
    """Raise ValueError if the requested state transition is not permitted.

    Consults ``VALID_TRANSITIONS`` to determine legality. This should be
    called before any code that mutates a job's ``state`` field.

    Args:
        from_state: The job's current ``JobState``.
        to_state: The desired next ``JobState``.

    Raises:
        ValueError: With a descriptive message including both states when the
                    transition is not listed in ``VALID_TRANSITIONS``.
    """
    allowed = VALID_TRANSITIONS.get(from_state, [])
    if to_state not in allowed:
        raise ValueError(
            f"Invalid job state transition: {from_state.value!r} -> {to_state.value!r}. "
            f"Allowed transitions from {from_state.value!r}: "
            f"{[s.value for s in allowed] or 'none (terminal state)'}"
        )
