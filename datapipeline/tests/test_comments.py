"""Tests for processing.comments — pure parsing, clustering, and scoring helpers.

No network calls, no mocking needed — all tested functions are pure.
"""

from __future__ import annotations

import pytest

from processing.comments import (
    cluster_timestamps,
    parse_timestamps,
    social_score,
    sponsor_penalty,
    timestamped_comment_count,
)


# ---------------------------------------------------------------------------
# parse_timestamps
# ---------------------------------------------------------------------------


class TestParseTimestamps:
    def test_mm_ss_format(self):
        """1:23 => 83 seconds."""
        assert parse_timestamps("great moment at 1:23") == [83]

    def test_hh_mm_ss_format(self):
        """1:02:45 => 3765 seconds."""
        assert parse_timestamps("see 1:02:45 for the highlight") == [3765]

    def test_multiple_timestamps_in_order(self):
        """Multiple timestamps returned in left-to-right order."""
        result = parse_timestamps("great moment at 1:23 and also 2:05")
        assert result == [83, 125]

    def test_zero_zero_format(self):
        """0:00 => 0 seconds."""
        assert parse_timestamps("starts at 0:00") == [0]

    def test_no_timestamps_returns_empty(self):
        assert parse_timestamps("no timestamps here") == []

    def test_empty_string_returns_empty(self):
        assert parse_timestamps("") == []

    def test_invalid_minutes_rejected(self):
        """99:99 has minutes >= 60 and must be rejected."""
        # The regex uses [0-5]?\d for minutes (0-59), so 99:99 won't match.
        result = parse_timestamps("99:99 is invalid")
        assert result == []

    def test_invalid_seconds_rejected(self):
        """Minutes=0, seconds=60 must be rejected."""
        # 0:60 — the regex only matches seconds [0-5]\d (00-59), so 60 won't match.
        result = parse_timestamps("0:60 is invalid")
        assert result == []

    def test_boundary_valid_minute_59(self):
        """0:59 => 59 seconds (valid boundary)."""
        assert parse_timestamps("clip at 0:59") == [59]

    def test_hh_mm_ss_hours_field(self):
        """2:00:00 => 7200 seconds."""
        assert parse_timestamps("see 2:00:00") == [7200]

    def test_multiple_timestamps_mixed_formats(self):
        """Mix of mm:ss and h:mm:ss."""
        result = parse_timestamps("at 0:30 and then 1:00:00 again")
        assert result == [30, 3600]

    def test_duplicate_timestamps_both_returned(self):
        """Duplicate mentions returned as separate entries."""
        result = parse_timestamps("see 1:00 and again 1:00")
        assert result == [60, 60]


# ---------------------------------------------------------------------------
# cluster_timestamps
# ---------------------------------------------------------------------------


class TestClusterTimestamps:
    def _make_comment(self, *timestamps: int) -> dict:
        return {"text": "", "like_count": 0, "timestamps": list(timestamps)}

    def test_empty_comments_returns_empty(self):
        assert cluster_timestamps([]) == []

    def test_comments_with_no_timestamps_returns_empty(self):
        comments = [{"text": "nice", "like_count": 0, "timestamps": []}]
        assert cluster_timestamps(comments) == []

    def test_single_timestamp_one_cluster(self):
        comments = [self._make_comment(60)]
        clusters = cluster_timestamps(comments)
        assert len(clusters) == 1
        assert clusters[0]["count"] == 1
        assert clusters[0]["center_s"] == 60.0
        assert clusters[0]["start_s"] == 60.0
        assert clusters[0]["end_s"] == 60.0

    def test_close_timestamps_form_one_cluster(self):
        """Timestamps within window_s=15 of the cluster start are merged."""
        comments = [
            self._make_comment(100),
            self._make_comment(108),
            self._make_comment(114),
        ]
        clusters = cluster_timestamps(comments, window_s=15.0)
        assert len(clusters) == 1
        assert clusters[0]["count"] == 3

    def test_far_timestamps_form_separate_clusters(self):
        """Timestamps > window_s apart form separate clusters."""
        comments = [
            self._make_comment(10),
            self._make_comment(200),
        ]
        clusters = cluster_timestamps(comments, window_s=15.0)
        assert len(clusters) == 2

    def test_sorted_by_count_descending(self):
        """Result is sorted by count descending (busiest cluster first)."""
        comments = [
            # Three timestamps near 300 (larger cluster).
            self._make_comment(300),
            self._make_comment(305),
            self._make_comment(310),
            # One timestamp near 100.
            self._make_comment(100),
        ]
        clusters = cluster_timestamps(comments, window_s=15.0)
        counts = [c["count"] for c in clusters]
        assert counts == sorted(counts, reverse=True)

    def test_cluster_center_is_mean(self):
        """center_s is the arithmetic mean of timestamps in the cluster."""
        comments = [self._make_comment(100), self._make_comment(110)]
        clusters = cluster_timestamps(comments, window_s=15.0)
        assert len(clusters) == 1
        assert clusters[0]["center_s"] == pytest.approx(105.0)

    def test_cluster_start_and_end_bounds(self):
        """start_s is the earliest, end_s is the latest in the cluster."""
        comments = [self._make_comment(50), self._make_comment(55), self._make_comment(62)]
        clusters = cluster_timestamps(comments, window_s=15.0)
        assert len(clusters) == 1
        assert clusters[0]["start_s"] == 50.0
        assert clusters[0]["end_s"] == 62.0

    def test_cluster_result_keys(self):
        """Each cluster dict must have center_s, count, start_s, end_s."""
        comments = [self._make_comment(100)]
        clusters = cluster_timestamps(comments)
        required_keys = {"center_s", "count", "start_s", "end_s"}
        for cluster in clusters:
            assert required_keys.issubset(cluster.keys())


# ---------------------------------------------------------------------------
# social_score
# ---------------------------------------------------------------------------


class TestSocialScore:
    def _make_cluster(self, center_s: float, count: int) -> dict:
        return {
            "center_s": center_s,
            "count": count,
            "start_s": center_s,
            "end_s": center_s,
        }

    def test_empty_clusters_returns_zero(self):
        assert social_score([], 0.0, 60.0) == 0.0

    def test_cluster_center_inside_window_scores_proportionally(self):
        """A cluster with count=max_count and center inside window => 1.0."""
        clusters = [self._make_cluster(30.0, 10)]
        score = social_score(clusters, 0.0, 60.0, max_count=10)
        assert score == pytest.approx(1.0)

    def test_cluster_center_outside_window_scores_zero(self):
        """A cluster whose center is outside the window => 0.0."""
        clusters = [self._make_cluster(100.0, 10)]
        score = social_score(clusters, 0.0, 60.0, max_count=10)
        assert score == 0.0

    def test_busiest_cluster_window_scores_highest(self):
        """The busiest cluster's window should score 1.0 (the maximum)."""
        clusters = [
            self._make_cluster(30.0, 20),   # busiest
            self._make_cluster(90.0, 5),
        ]
        score_busy = social_score(clusters, 0.0, 60.0, max_count=20)
        assert score_busy == pytest.approx(1.0)

    def test_less_busy_cluster_scores_less_than_busiest(self):
        clusters = [
            self._make_cluster(30.0, 20),   # busiest (max)
            self._make_cluster(90.0, 5),
        ]
        score_busy = social_score(clusters, 0.0, 60.0, max_count=20)
        score_less = social_score(clusters, 61.0, 120.0, max_count=20)
        assert score_less < score_busy

    def test_normalised_by_max_count(self):
        """Score is count / max_count for the best cluster in window."""
        clusters = [self._make_cluster(30.0, 5)]
        score = social_score(clusters, 0.0, 60.0, max_count=10)
        assert score == pytest.approx(0.5)

    def test_max_count_auto_derived_when_none(self):
        """When max_count=None, it is derived from the clusters list."""
        clusters = [
            self._make_cluster(30.0, 8),
            self._make_cluster(90.0, 4),
        ]
        # Best cluster in [0..60] has count=8; auto max_count=8 => score 1.0.
        score = social_score(clusters, 0.0, 60.0, max_count=None)
        assert score == pytest.approx(1.0)

    def test_score_in_unit_interval(self):
        clusters = [self._make_cluster(30.0, 7), self._make_cluster(90.0, 3)]
        for window_start, window_end in [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]:
            s = social_score(clusters, window_start, window_end, max_count=10)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# sponsor_penalty
# ---------------------------------------------------------------------------


class TestSponsorPenalty:
    def _make_comment(self, text: str, *timestamps: int) -> dict:
        return {"text": text, "like_count": 0, "timestamps": list(timestamps)}

    def test_no_nearby_comments_returns_one(self):
        """No timestamped comments near the window => no penalty (1.0)."""
        comments = [self._make_comment("use code XYZ", 200)]
        clusters = []
        result = sponsor_penalty(comments, clusters, 0.0, 60.0)
        assert result == pytest.approx(1.0)

    def test_few_sponsor_comments_returns_one(self):
        """Fewer than 30% sponsor comments nearby => no penalty."""
        comments = [
            self._make_comment("great video!", 10),
            self._make_comment("amazing content!", 15),
            self._make_comment("use code XYZ for discount", 12),  # 1/3 = 33%?
        ]
        # 1 out of 3 sponsor is 33%, just above 30%; let's test exactly at boundary:
        # 1/4 = 25% < 30% => no penalty.
        comments_with_enough_normal = [
            self._make_comment("great video!", 10),
            self._make_comment("amazing content!", 15),
            self._make_comment("loved this part!", 20),
            self._make_comment("use code XYZ", 12),  # 1/4 = 25%
        ]
        result = sponsor_penalty(comments_with_enough_normal, [], 0.0, 60.0)
        assert result == pytest.approx(1.0)

    def test_majority_sponsor_comments_returns_penalty(self):
        """More than 30% nearby timestamped comments with sponsor keyword => 0.3."""
        comments = [
            self._make_comment("use code XYZ for 10% off", 10),
            self._make_comment("sponsored segment starts here", 15),
            self._make_comment("check out this offer", 12),
            self._make_comment("great video!", 14),  # 3/4 = 75% > 30%
        ]
        result = sponsor_penalty(comments, [], 0.0, 60.0)
        assert result == pytest.approx(0.3)

    def test_sponsor_penalty_with_custom_keywords(self):
        """Custom sponsor_keywords override the defaults."""
        comments = [
            self._make_comment("totally organic content here", 10),
            self._make_comment("totally organic content here", 15),
            self._make_comment("totally organic promo happening", 12),
            self._make_comment("totally organic content here", 13),
        ]
        # Default keywords wouldn't match "promo happening" alone; add "promo".
        result_no_kw = sponsor_penalty(comments, [], 0.0, 60.0, sponsor_keywords=["promo"])
        # 1/4 = 25% < 30% => 1.0.
        result = sponsor_penalty(
            comments, [], 0.0, 60.0,
            sponsor_keywords=["totally organic promo"],
        )
        # "totally organic promo" matches exactly one comment (1/4 = 25%).
        assert result == pytest.approx(1.0)

    def test_empty_comments_returns_one(self):
        result = sponsor_penalty([], [], 0.0, 60.0)
        assert result == pytest.approx(1.0)

    def test_comments_outside_proximity_ignored(self):
        """Comments with timestamps far from the window are excluded."""
        # Window [0..60], proximity=5 => search band [-5..65].
        # Timestamp at 200 is outside the band => not a nearby comment.
        comments = [
            self._make_comment("use code XYZ", 200),
            self._make_comment("use code XYZ", 201),
        ]
        result = sponsor_penalty(comments, [], 0.0, 60.0, proximity_s=5.0)
        assert result == pytest.approx(1.0)

    def test_penalty_multiplier_in_valid_range(self):
        """sponsor_penalty always returns a value in (0, 1]."""
        comments = [self._make_comment("use code XYZ sponsored", ts) for ts in range(10)]
        result = sponsor_penalty(comments, [], 0.0, 60.0)
        assert 0.0 < result <= 1.0


# ---------------------------------------------------------------------------
# timestamped_comment_count
# ---------------------------------------------------------------------------


class TestTimestampedCommentCount:
    def test_empty_list(self):
        assert timestamped_comment_count([]) == 0

    def test_comments_with_no_timestamps(self):
        comments = [
            {"text": "nice", "like_count": 0, "timestamps": []},
            {"text": "great", "like_count": 5, "timestamps": []},
        ]
        assert timestamped_comment_count(comments) == 0

    def test_counts_all_timestamps_across_comments(self):
        comments = [
            {"text": "at 1:00 and 2:00", "like_count": 0, "timestamps": [60, 120]},
            {"text": "at 3:00", "like_count": 0, "timestamps": [180]},
        ]
        assert timestamped_comment_count(comments) == 3

    def test_single_timestamp_comment(self):
        comments = [{"text": "see 0:30", "like_count": 0, "timestamps": [30]}]
        assert timestamped_comment_count(comments) == 1

    def test_mix_of_ts_and_no_ts(self):
        comments = [
            {"text": "at 1:00", "like_count": 0, "timestamps": [60]},
            {"text": "nice video", "like_count": 0, "timestamps": []},
            {"text": "at 2:00 and 3:00", "like_count": 0, "timestamps": [120, 180]},
        ]
        assert timestamped_comment_count(comments) == 3
