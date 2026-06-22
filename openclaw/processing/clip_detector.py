"""Clip detection stage: score transcript segments and select the best clips.

Applies five heuristic signals to each transcript segment to produce a
composite score, then returns the top-N non-overlapping clips that meet the
minimum score threshold defined in ``config/settings.yaml``.

Signals (weights are illustrative — final values in settings):
1. Sentiment intensity (positive/negative peak, not neutral)
2. Keyword density (pre-defined high-engagement word list)
3. Speech rate (words-per-second; too slow or too fast penalised)
4. Segment length fit (prefers segments close to the target clip duration)
5. Laughter / reaction cues in surrounding text
"""

from __future__ import annotations


def detect_clips(segments: list[dict], options: dict) -> list[dict]:
    """Score all transcript segments and return the top-N clips.

    Applies ``score_segment`` to each element of *segments*, filters out
    those below ``options["min_clip_score"]`` (or the settings default),
    then performs a greedy non-overlapping selection to return the best
    ``options["max_clips_per_job"]`` clips.

    Args:
        segments: List of transcript segment dicts with ``start``, ``end``,
                  and ``text`` keys.
        options: Per-job options dict that may override global config values
                 (``min_clip_score``, ``max_clips_per_job``,
                 ``max_clip_length_s``).

    Returns:
        List of clip dicts, each with the shape::

            {
                "clip_id": str,    # UUID
                "start_s": float,
                "end_s": float,
                "score": float,
            }

        Ordered from highest to lowest score. May be an empty list if no
        segment meets the threshold.
    """
    raise NotImplementedError


def score_segment(segment: dict) -> float:
    """Compute a composite engagement score for a single transcript segment.

    Applies all five signals and returns a value in [0.0, 1.0] representing
    the estimated virality / engagement potential of the segment.

    Args:
        segment: Transcript segment dict with at minimum ``start``, ``end``,
                 and ``text`` keys.

    Returns:
        Float score in the range [0.0, 1.0].
    """
    raise NotImplementedError
