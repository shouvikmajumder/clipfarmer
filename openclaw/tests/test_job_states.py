"""Tests for core.job_states — JobState enum and transition guard.

Pipeline stage order (v5): QUEUED -> DOWNLOADING -> DETECTING -> EDITING ->
CAPTIONING -> FORMATTING -> POSTING -> COMPLETE (TRANSCRIBING was removed).
"""

import pytest

from core.job_states import JobState, VALID_TRANSITIONS, assert_valid_transition


# Linear pipeline order per v5 plan (TRANSCRIBING removed).
PIPELINE_ORDER = [
    JobState.QUEUED,
    JobState.DOWNLOADING,
    JobState.DETECTING,
    JobState.EDITING,
    JobState.CAPTIONING,
    JobState.FORMATTING,
    JobState.POSTING,
    JobState.COMPLETE,
]

NON_TERMINAL_STATES = [
    JobState.QUEUED,
    JobState.DOWNLOADING,
    JobState.DETECTING,
    JobState.EDITING,
    JobState.CAPTIONING,
    JobState.FORMATTING,
    JobState.POSTING,
]

TERMINAL_STATES = [JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED]


@pytest.mark.parametrize(
    "current,expected_next",
    list(zip(PIPELINE_ORDER[:-1], PIPELINE_ORDER[1:])),
)
def test_linear_pipeline_transitions_are_valid(current, expected_next):
    """Every forward step in the linear pipeline should be a legal transition."""
    assert_valid_transition(current, expected_next)  # should not raise


@pytest.mark.parametrize("state", NON_TERMINAL_STATES)
def test_failed_reachable_from_every_non_terminal_state(state):
    assert_valid_transition(state, JobState.FAILED)  # should not raise


@pytest.mark.parametrize("state", NON_TERMINAL_STATES)
def test_cancelled_reachable_from_every_non_terminal_state(state):
    assert_valid_transition(state, JobState.CANCELLED)  # should not raise


def test_failed_can_requeue():
    """A failed job can be re-queued for a retry."""
    assert_valid_transition(JobState.FAILED, JobState.QUEUED)  # should not raise


@pytest.mark.parametrize("terminal_state", TERMINAL_STATES)
def test_terminal_states_have_no_outgoing_transitions_except_failed_requeue(terminal_state):
    """COMPLETE and CANCELLED are dead ends; FAILED only permits re-queue."""
    allowed = VALID_TRANSITIONS[terminal_state]
    if terminal_state is JobState.FAILED:
        assert allowed == [JobState.QUEUED]
    else:
        assert allowed == []


def test_downloading_to_detecting_is_valid():
    """DOWNLOADING -> DETECTING must be valid in the v5 pipeline."""
    assert_valid_transition(JobState.DOWNLOADING, JobState.DETECTING)  # should not raise


def test_downloading_to_transcribing_is_invalid():
    """TRANSCRIBING no longer exists in the pipeline; any transition to it is invalid."""
    # The TRANSCRIBING state was removed. DOWNLOADING -> DETECTING is the new hop.
    # Since TRANSCRIBING is gone entirely from the enum, we verify DETECTING is
    # the direct successor of DOWNLOADING (already covered by the parametrized test
    # above) and that the VALID_TRANSITIONS for DOWNLOADING only contains valid v5 states.
    allowed = VALID_TRANSITIONS[JobState.DOWNLOADING]
    state_values = [s.value for s in allowed]
    assert "transcribing" not in state_values


def test_skipping_a_stage_is_invalid():
    """Jumping ahead in the pipeline (e.g. queued -> detecting) must raise."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.QUEUED, JobState.DETECTING)


def test_moving_backwards_is_invalid():
    """Moving backwards in the pipeline must raise."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.EDITING, JobState.DOWNLOADING)


def test_transition_from_complete_is_invalid():
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.COMPLETE, JobState.QUEUED)


def test_transition_from_cancelled_is_invalid():
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.CANCELLED, JobState.QUEUED)


def test_error_message_mentions_both_states():
    with pytest.raises(ValueError) as exc_info:
        assert_valid_transition(JobState.QUEUED, JobState.COMPLETE)
    message = str(exc_info.value)
    assert "queued" in message
    assert "complete" in message


def test_every_state_has_a_transition_entry():
    """VALID_TRANSITIONS must define a (possibly empty) entry for every JobState."""
    for state in JobState:
        assert state in VALID_TRANSITIONS


def test_transcribing_not_a_valid_job_state():
    """JobState.TRANSCRIBING must not exist in the v5 enum."""
    assert not hasattr(JobState, "TRANSCRIBING")
    state_values = [s.value for s in JobState]
    assert "transcribing" not in state_values
