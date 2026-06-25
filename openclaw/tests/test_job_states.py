"""Tests for core.job_states — JobState enum and transition guard."""

import pytest

from core.job_states import JobState, VALID_TRANSITIONS, assert_valid_transition


# Linear pipeline: queued → downloading → detecting → editing → captioning → complete.
PIPELINE_ORDER = [
    JobState.QUEUED,
    JobState.DOWNLOADING,
    JobState.DETECTING,
    JobState.EDITING,
    JobState.CAPTIONING,
    JobState.COMPLETE,
]

NON_TERMINAL_STATES = [
    JobState.QUEUED,
    JobState.DOWNLOADING,
    JobState.DETECTING,
    JobState.EDITING,
    JobState.CAPTIONING,
]

TERMINAL_STATES = [JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED]


@pytest.mark.parametrize(
    "current,expected_next",
    list(zip(PIPELINE_ORDER[:-1], PIPELINE_ORDER[1:])),
)
def test_linear_pipeline_transitions_are_valid(current, expected_next):
    """Every forward step in the linear pipeline should be a legal transition."""
    assert_valid_transition(current, expected_next)  # should not raise


def test_downloading_to_detecting_is_valid():
    """The new second pipeline stage must be reachable from downloading."""
    assert_valid_transition(JobState.DOWNLOADING, JobState.DETECTING)  # should not raise


def test_detecting_to_editing_is_valid():
    """Detection now flows into the editing stage rather than straight to complete."""
    assert_valid_transition(JobState.DETECTING, JobState.EDITING)  # should not raise


def test_editing_to_captioning_is_valid():
    """Editing now flows into the captioning stage rather than straight to complete."""
    assert_valid_transition(JobState.EDITING, JobState.CAPTIONING)  # should not raise


def test_captioning_to_complete_is_valid():
    """The captioning stage transitions cleanly to complete."""
    assert_valid_transition(JobState.CAPTIONING, JobState.COMPLETE)  # should not raise


def test_editing_to_complete_is_invalid():
    """Skipping the captioning stage (editing → complete) must now be rejected."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.EDITING, JobState.COMPLETE)


def test_detecting_to_complete_is_invalid():
    """Skipping ahead from detecting straight to complete must be rejected."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.DETECTING, JobState.COMPLETE)


def test_downloading_to_complete_is_invalid():
    """Skipping straight from downloading to complete must be rejected."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.DOWNLOADING, JobState.COMPLETE)


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


def test_skipping_a_stage_is_invalid():
    """Jumping ahead in the pipeline (e.g. queued -> complete) must raise."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.QUEUED, JobState.COMPLETE)


def test_moving_backwards_is_invalid():
    """Moving backwards in the pipeline must raise."""
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.COMPLETE, JobState.DOWNLOADING)


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
