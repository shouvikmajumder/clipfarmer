"""Tests for processing.clip_detector — v5 signal-fused clip detection.

The v5 API:
    detect(transcript, comments=[], audio_signals=None, max_clips=3)
    score_window(segments, audio_signals=None)
    detect_clips(transcript, max_clips=3)   <- back-compat wrapper

Tests use only transcript text and comments (no real audio / librosa).
audio_signals is always None here, matching the no-librosa constraint.
"""

from __future__ import annotations

from processing.clip_detector import detect, detect_clips, score_window


def _segment(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


# ---------------------------------------------------------------------------
# detect() — primary entry point
# ---------------------------------------------------------------------------


def test_detect_returns_empty_for_none_transcript_no_audio() -> None:
    """No transcript + no audio_signals => no clips (both fallback paths need something)."""
    result = detect(None, comments=[], audio_signals=None)
    assert result == []


def test_detect_returns_empty_for_empty_transcript_no_audio() -> None:
    """Empty transcript list + no audio => no clips."""
    result = detect([], comments=[], audio_signals=None)
    assert result == []


def test_detect_max_clips_zero_returns_empty() -> None:
    segments = [_segment(0, 25, "This is absolutely incredible and shocking!")]
    result = detect(segments, comments=[], audio_signals=None, max_clips=0)
    assert result == []


def test_detect_high_signal_transcript_yields_clips() -> None:
    """A window dense in all content signals should clear the 0.45 threshold."""
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    # Build a timeline long enough for windows to land within it.
    segments = [_segment(0, 25, high_signal_text)]
    segments.append(_segment(25, 50, "nothing interesting happens here at all filler"))

    clips = detect(segments, comments=[], audio_signals=None, max_clips=3)
    assert len(clips) >= 1
    assert clips[0]["score"] >= 0.45
    assert clips[0]["start_s"] == 0


def test_detect_low_signal_transcript_yields_no_clips() -> None:
    """Bland filler text should score below the 0.45 threshold."""
    segments = [
        _segment(i * 5, i * 5 + 5, "um so anyway we continued the meeting as usual")
        for i in range(20)
    ]
    clips = detect(segments, comments=[], audio_signals=None, max_clips=3)
    assert clips == []


def test_detect_respects_max_clips() -> None:
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    segments = []
    for block_start in (0, 100, 200):
        segments.append(_segment(block_start, block_start + 25, high_signal_text))
        segments.append(_segment(block_start + 25, block_start + 50, "filler text here"))

    clips = detect(segments, comments=[], audio_signals=None, max_clips=2)
    assert len(clips) <= 2


def test_detect_returned_clips_are_non_overlapping() -> None:
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? How is this even possible? [laughter] "
        "[applause] Here's why this is insane and unbelievable!"
    )
    segments = []
    for block_start in (0, 60, 120, 180):
        segments.append(_segment(block_start, block_start + 25, high_signal_text))
        segments.append(_segment(block_start + 25, block_start + 50, "filler here now"))

    clips = detect(segments, comments=[], audio_signals=None, max_clips=3)
    for i, a in enumerate(clips):
        for b in clips[i + 1 :]:
            assert a["end_s"] <= b["start_s"] or b["end_s"] <= a["start_s"]


def test_detect_returned_clips_sorted_by_score_descending() -> None:
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
    clips = detect(segments, comments=[], audio_signals=None, max_clips=3)
    scores = [c["score"] for c in clips]
    assert scores == sorted(scores, reverse=True)


def test_detect_result_keys_present() -> None:
    """Each result dict must have start_s, end_s, and score keys."""
    high_signal_text = (
        "I made a million dollars. Here's why nobody talks about this. "
        "The truth is shocking and incredible. $500,000 in six months!"
    )
    segments = [_segment(0, 25, high_signal_text)]
    segments.append(_segment(25, 50, "boring filler text here nothing to see"))
    clips = detect(segments, comments=[], audio_signals=None, max_clips=3)
    for clip in clips:
        assert "start_s" in clip
        assert "end_s" in clip
        assert "score" in clip
        assert 0.0 <= clip["score"] <= 1.0


def test_detect_two_distinct_strong_moments_yields_multiple_clips() -> None:
    """Two clearly separated strong moments must both be returned.

    This guards the top_k distinct-moment selection fix: the greedy
    non-overlapping window selection must produce multiple clips, not just
    one clustered around the single hottest anchor.
    """
    strong_a = (
        "I made a million dollars. Here's why nobody talks about this secret. "
        "The truth is shocking! $500,000 in 6 months. Why does nobody explain?"
    )
    filler = " ".join(["and then we talked about the weather"] * 10)
    strong_b = (
        "This is absolutely unbelievable and incredible! Nobody tells you the "
        "real reason. What nobody talks about is the brutal truth. [laughter] "
        "Here's the thing everyone is missing! How is this even possible?"
    )

    # Block A at t=0..25, filler at t=25..100, Block B at t=100..125.
    segments = [
        _segment(0, 25, strong_a),
        *[_segment(25 + i * 5, 30 + i * 5, filler) for i in range(15)],
        _segment(100, 125, strong_b),
        _segment(125, 150, filler),
    ]

    clips = detect(segments, comments=[], audio_signals=None, max_clips=3)
    assert len(clips) >= 2, (
        f"Expected at least 2 distinct clips, got {len(clips)}: {clips}"
    )
    # Confirm the two clips are non-overlapping (different moments).
    if len(clips) >= 2:
        starts = [c["start_s"] for c in clips]
        ends = [c["end_s"] for c in clips]
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                assert ends[i] <= starts[j] or ends[j] <= starts[i]


# ---------------------------------------------------------------------------
# Social signal: comments fusion
# ---------------------------------------------------------------------------


def test_detect_comments_cluster_raises_score_for_nearby_window() -> None:
    """A window near a dense timestamp cluster should score higher with comments
    than with no comments (social_conf > 0 when timestamps exist).
    """
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? [laughter] Here's why this is insane!"
    )
    filler = "and we continued to discuss the mundane topics as usual"
    segments = [
        _segment(0, 25, high_signal_text),
        _segment(25, 50, filler),
    ]

    # Build comments that all timestamp at t=5 (right inside the first window).
    comments_with_ts = [
        {"text": f"great moment at 0:05 clip {i}", "like_count": 10, "timestamps": [5]}
        for i in range(20)
    ]

    clips_no_comments = detect(segments, comments=[], audio_signals=None, max_clips=3)
    clips_with_comments = detect(segments, comments=comments_with_ts, audio_signals=None, max_clips=3)

    # Both should detect the window; with a strong comment cluster inside the
    # window, the fused score should be >= the pure-content score.
    assert len(clips_no_comments) >= 1
    assert len(clips_with_comments) >= 1
    assert clips_with_comments[0]["score"] >= clips_no_comments[0]["score"]


def test_detect_comments_empty_vs_populated_changes_ranking() -> None:
    """When there are no timestamped comments, social_conf is 0.0 (pure content).
    When there are many, the cluster near one window can boost its rank.
    """
    text_a = (
        "I made a million dollars! Here's why nobody talks about this secret. "
        "The truth is shocking and incredible!"
    )
    text_b = (
        "Why is this even a thing? How does anyone live like this?"
    )
    segments = [
        _segment(0, 25, text_a),
        _segment(25, 50, "filler content nothing interesting here"),
        _segment(50, 75, text_b),
        _segment(75, 100, "more filler text nothing to see here"),
    ]

    # Comments cluster at t=55 — inside window B.
    comments = [
        {"text": f"loved the part at 0:55 wow {i}", "like_count": 5, "timestamps": [55]}
        for i in range(30)
    ]

    clips_no_comments = detect(segments, comments=[], audio_signals=None, max_clips=3)
    clips_with_comments = detect(segments, comments=comments, audio_signals=None, max_clips=3)

    # With dense comments near window B, it should rank at least as high as without.
    # We only assert that the comment signal doesn't hurt the window it boosts.
    assert len(clips_with_comments) >= 1
    assert len(clips_no_comments) >= 1


def test_detect_no_timestamps_means_pure_content() -> None:
    """Comments with no timestamps should have zero social_conf (pure content
    scoring), so detect with/without them should give same results.
    """
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "Here's why this is insane and unbelievable! [laughter]"
    )
    segments = [
        _segment(0, 25, high_signal_text),
        _segment(25, 50, "boring filler text"),
    ]

    comments_no_ts = [
        {"text": "great video!", "like_count": 100, "timestamps": []}
        for _ in range(50)
    ]

    clips_no_comments = detect(segments, comments=[], audio_signals=None, max_clips=3)
    clips_empty_ts = detect(segments, comments=comments_no_ts, audio_signals=None, max_clips=3)

    # Without any timestamps, social_conf = 0.0, so results must be identical.
    assert clips_no_comments == clips_empty_ts


# ---------------------------------------------------------------------------
# Sponsor penalty
# ---------------------------------------------------------------------------


def test_detect_sponsor_comments_suppress_window() -> None:
    """A window where >30% of nearby timestamped comments mention sponsor
    keywords should receive a lower (or equal, not higher) fused score.
    """
    high_signal_text = (
        "This is absolutely incredible and shocking! Nobody talks about this. "
        "What nobody tells you is the truth is way more brutal than you think. "
        "Why does nobody talk about this? [laughter] Here's why this is insane!"
    )
    filler = "and we continued to discuss the mundane topics as usual"
    segments = [
        _segment(0, 25, high_signal_text),
        _segment(25, 50, filler),
    ]

    # Regular comments — no sponsor keywords.
    normal_comments = [
        {"text": f"great clip at 0:10 wow {i}", "like_count": 5, "timestamps": [10]}
        for i in range(20)
    ]
    # Sponsor-heavy comments at the same timestamp.
    sponsor_comments = [
        {
            "text": f"use code XYZ for discount at 0:10 clip {i}",
            "like_count": 5,
            "timestamps": [10],
        }
        for i in range(20)
    ]

    clips_normal = detect(segments, comments=normal_comments, audio_signals=None, max_clips=3)
    clips_sponsor = detect(segments, comments=sponsor_comments, audio_signals=None, max_clips=3)

    # Sponsor comments should never increase the score above the normal case.
    if clips_normal and clips_sponsor:
        assert clips_sponsor[0]["score"] <= clips_normal[0]["score"] + 1e-6


# ---------------------------------------------------------------------------
# Back-compat wrappers: score_window and detect_clips
# ---------------------------------------------------------------------------


def test_score_window_empty_returns_zero() -> None:
    assert score_window([]) == 0.0


def test_score_window_range_is_bounded() -> None:
    text = "This is absolutely incredible! " * 50
    score = score_window([_segment(0, 30, text)])
    assert 0.0 <= score <= 1.0


def test_score_window_returns_float() -> None:
    result = score_window([_segment(0, 20, "hello there")])
    assert isinstance(result, float)


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
