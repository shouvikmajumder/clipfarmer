"""Tests for processing.clip_detector — segment scoring and clip selection."""

from processing.clip_detector import detect_clips, score_window


def _segment(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def test_detect_clips_empty_segments_returns_empty_list() -> None:
    assert detect_clips([], max_clips=3) == []


def test_detect_clips_max_clips_zero_returns_empty_list() -> None:
    segments = [_segment(0, 10, "hello there, what is going on here?")]
    assert detect_clips(segments, max_clips=0) == []


def test_detect_clips_low_signal_transcript_yields_no_clips() -> None:
    """Bland, low-energy filler text should score below the 0.45 threshold."""
    segments = [
        _segment(i * 5, i * 5 + 5, "um so anyway we continued the meeting as usual")
        for i in range(20)
    ]
    clips = detect_clips(segments, max_clips=3)
    assert clips == []


def test_detect_clips_high_signal_window_is_returned() -> None:
    """A window dense in all 5 signals should clear the threshold and be returned."""
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    segments = [_segment(0, 25, high_signal_text)]
    # pad timeline so the window has somewhere to end
    segments.append(_segment(25, 50, "nothing interesting happens here at all"))

    clips = detect_clips(segments, max_clips=3)
    assert len(clips) >= 1
    assert clips[0]["score"] >= 0.45
    assert clips[0]["start_s"] == 0


def test_detect_clips_respects_max_clips() -> None:
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    segments = []
    # Three separated high-signal blocks far enough apart to not overlap.
    for block_start in (0, 100, 200):
        segments.append(_segment(block_start, block_start + 25, high_signal_text))
        segments.append(_segment(block_start + 25, block_start + 50, "filler text here"))

    clips = detect_clips(segments, max_clips=2)
    assert len(clips) <= 2


def test_detect_clips_returned_clips_are_non_overlapping() -> None:
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    segments = []
    for block_start in (0, 30, 60, 90):
        segments.append(_segment(block_start, block_start + 25, high_signal_text))

    clips = detect_clips(segments, max_clips=3)
    for i, a in enumerate(clips):
        for b in clips[i + 1 :]:
            assert a["end_s"] <= b["start_s"] or b["end_s"] <= a["start_s"]


def test_detect_clips_sorted_by_score_descending() -> None:
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    mild_signal_text = "Why is this happening? This is interesting."
    segments = [
        _segment(0, 25, high_signal_text),
        _segment(50, 75, mild_signal_text * 3),
    ]
    clips = detect_clips(segments, max_clips=3)
    scores = [c["score"] for c in clips]
    assert scores == sorted(scores, reverse=True)


def test_score_window_empty_returns_zero() -> None:
    assert score_window([]) == 0.0


def test_score_window_range_is_bounded() -> None:
    text = "This is absolutely incredible! " * 50
    score = score_window([_segment(0, 30, text)])
    assert 0.0 <= score <= 1.0


def test_score_window_higher_for_high_signal_text() -> None:
    bland = score_window([_segment(0, 20, "we then proceeded to the next item on the agenda")])
    rich = score_window(
        [
            _segment(
                0,
                20,
                "This is absolutely incredible and shocking! The truth is nobody talks "
                "about this. Why does nobody talk about this? [laughter] [applause] "
                "Here's why this is insane!",
            )
        ]
    )
    assert rich > bland
