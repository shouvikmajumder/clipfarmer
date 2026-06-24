"""Audio (transcript-independent) signals for the v5 clip-detection pipeline.

Loads and analyses a media file *once* via librosa, then exposes per-window
scoring methods so many candidate windows can be evaluated cheaply from the
same in-memory arrays.

librosa and numpy are lazy-imported inside ``AudioSignals.from_file`` so that
importing this module at the top level never fails when those packages are
absent — matching the lazy-import pattern used by ``processing.transcriber``
for mlx-whisper.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# Defaults that mirror the values in config/settings.yaml so the module
# behaves correctly even if the config file is missing.
_DEFAULT_DRAMATIC_PAUSE_MIN_S = 0.6
_DEFAULT_SAMPLE_RATE = 16_000
# RMS frames whose energy falls below this percentile are considered silence.
_SILENCE_PERCENTILE = 10


def _load_dramatic_pause_min_s() -> float:
    """Read ``detection.dramatic_pause_min_s`` from ``config/settings.yaml``.

    Falls back to ``_DEFAULT_DRAMATIC_PAUSE_MIN_S`` if the file or key is
    missing.
    """
    try:
        with open(SETTINGS_PATH) as fh:
            data = yaml.safe_load(fh)
    except OSError:
        return _DEFAULT_DRAMATIC_PAUSE_MIN_S
    if not data:
        return _DEFAULT_DRAMATIC_PAUSE_MIN_S
    return data.get("detection", {}).get(
        "dramatic_pause_min_s", _DEFAULT_DRAMATIC_PAUSE_MIN_S
    )


# ---------------------------------------------------------------------------
# AudioSignals
# ---------------------------------------------------------------------------

class AudioSignals:
    """Precomputed audio feature arrays for a single media file.

    Load once with ``AudioSignals.from_file``, then call the per-window
    methods many times without re-reading the file.

    All scoring methods return a float in [0.0, 1.0] and are robust against
    degenerate inputs (empty arrays, zero-length windows) — they always
    return ``0.0`` (or ``[]`` for ``silence_boundaries``) rather than raising.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        energy_envelope: "np.ndarray",  # type: ignore[name-defined]
        frame_times: "np.ndarray",  # type: ignore[name-defined]
        zcr_envelope: "np.ndarray",  # type: ignore[name-defined]
        silence_gaps: list[tuple[float, float]],
        duration_s: float,
        dramatic_pause_min_s: float,
    ) -> None:
        self._energy = energy_envelope       # shape (n_frames,), normalised [0, 1]
        self._times = frame_times            # shape (n_frames,), seconds
        self._zcr = zcr_envelope             # shape (n_frames,), normalised [0, 1]
        self._silence_gaps = silence_gaps    # list of (start_s, end_s)
        self._duration_s = duration_s
        self._pause_min_s = dramatic_pause_min_s

    @classmethod
    def from_file(
        cls,
        media_path: str,
        dramatic_pause_min_s: float | None = None,
    ) -> "AudioSignals":
        """Load *media_path* and precompute all feature arrays.

        librosa and numpy are imported here (lazy) so the module-level import
        of ``audio_analyzer`` never requires them to be installed.

        Args:
            media_path: Absolute path to a video or audio file.  librosa
                calls ffmpeg internally for non-WAV formats.
            dramatic_pause_min_s: Minimum gap duration (in seconds) to count
                as a dramatic pause.  Defaults to the value in
                ``config/settings.yaml`` (``detection.dramatic_pause_min_s``).

        Returns:
            A fully populated ``AudioSignals`` instance.

        Raises:
            RuntimeError: If librosa cannot load the file.
        """
        import numpy as np  # lazy import
        import librosa  # lazy import

        if dramatic_pause_min_s is None:
            dramatic_pause_min_s = _load_dramatic_pause_min_s()

        # ------------------------------------------------------------------
        # Load audio
        # ------------------------------------------------------------------
        try:
            y, sr = librosa.load(media_path, sr=_DEFAULT_SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise RuntimeError(
                f"audio_analyzer: librosa could not load {media_path!r}: {exc}"
            ) from exc

        duration_s = float(librosa.get_duration(y=y, sr=sr))

        # ------------------------------------------------------------------
        # RMS energy envelope
        # ------------------------------------------------------------------
        rms_raw = librosa.feature.rms(y=y)  # shape (1, n_frames)
        rms = rms_raw[0]                     # shape (n_frames,)
        rms_max = rms.max()
        if rms_max > 0:
            energy_envelope = rms / rms_max
        else:
            energy_envelope = np.zeros_like(rms)

        frame_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=512
        )

        # ------------------------------------------------------------------
        # Zero-crossing rate envelope (used for laughter heuristic)
        # ------------------------------------------------------------------
        zcr_raw = librosa.feature.zero_crossing_rate(y)  # shape (1, n_frames)
        zcr = zcr_raw[0]
        zcr_max = zcr.max()
        if zcr_max > 0:
            zcr_envelope = zcr / zcr_max
        else:
            zcr_envelope = np.zeros_like(zcr)

        # ------------------------------------------------------------------
        # Silence gap detection
        # ------------------------------------------------------------------
        # Use the 10th-percentile RMS value as the silence floor.
        silence_floor = float(np.percentile(rms, _SILENCE_PERCENTILE))
        # Guard: if the floor is effectively zero (very quiet recording),
        # use a small absolute threshold to avoid treating everything as silence.
        silence_floor = max(silence_floor, 1e-5)

        is_silent = rms <= silence_floor
        silence_gaps: list[tuple[float, float]] = []
        in_gap = False
        gap_start = 0.0

        for i, silent in enumerate(is_silent):
            t = float(frame_times[i])
            if silent and not in_gap:
                gap_start = t
                in_gap = True
            elif not silent and in_gap:
                gap_end = t
                if gap_end - gap_start >= dramatic_pause_min_s:
                    silence_gaps.append((gap_start, gap_end))
                in_gap = False

        # Close an open gap at the end of the file.
        if in_gap and len(frame_times) > 0:
            gap_end = float(frame_times[-1])
            if gap_end - gap_start >= dramatic_pause_min_s:
                silence_gaps.append((gap_start, gap_end))

        return cls(
            energy_envelope=energy_envelope,
            frame_times=frame_times,
            zcr_envelope=zcr_envelope,
            silence_gaps=silence_gaps,
            duration_s=duration_s,
            dramatic_pause_min_s=dramatic_pause_min_s,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        """Total duration of the loaded media file in seconds."""
        return self._duration_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _window_mask(self, start_s: float, end_s: float) -> "np.ndarray":  # type: ignore[name-defined]
        """Boolean mask of frame indices that fall within [start_s, end_s]."""
        return (self._times >= start_s) & (self._times < end_s)

    # ------------------------------------------------------------------
    # Per-window scoring methods
    # ------------------------------------------------------------------

    def energy(self, start_s: float, end_s: float) -> float:
        """Mean normalised RMS energy over the window [start_s, end_s].

        Args:
            start_s: Window start in seconds.
            end_s: Window end in seconds.

        Returns:
            Float in [0.0, 1.0].
        """
        try:
            import numpy as np
            mask = self._window_mask(start_s, end_s)
            if not mask.any():
                return 0.0
            return float(np.clip(self._energy[mask].mean(), 0.0, 1.0))
        except Exception:
            return 0.0

    def laughter(self, start_s: float, end_s: float) -> float:
        """Coarse audio proxy for laughter within the window.

        **Note — coarse proxy:** laughter in audio shares characteristics with
        other high-energy, noisy speech bursts.  The primary laughter signal
        for transcript-available jobs is the ``[Laughter]`` tag detected by
        ``clip_detector``.  This method is the fallback for the no-transcript
        path and should be treated as a low-confidence indicator.

        The heuristic: a laughter burst shows up as a region of
        *simultaneously* high energy and high zero-crossing rate (rapid
        oscillation).  We compute the mean of the product
        ``energy * zcr_normalised`` over the window; because each component is
        already normalised to [0, 1], the product rewards only frames where
        *both* are elevated.

        Args:
            start_s: Window start in seconds.
            end_s: Window end in seconds.

        Returns:
            Float in [0.0, 1.0].
        """
        try:
            import numpy as np
            mask = self._window_mask(start_s, end_s)
            if not mask.any():
                return 0.0
            combined = self._energy[mask] * self._zcr[mask]
            return float(np.clip(combined.mean(), 0.0, 1.0))
        except Exception:
            return 0.0

    def dramatic_pause(self, start_s: float, end_s: float) -> float:
        """Score the presence of dramatic pauses in or just before the window.

        Counts detected silence gaps (>= ``dramatic_pause_min_s``) whose
        midpoint falls within [start_s, end_s) or within the one-pause-length
        run-up before the window start (so a pause-then-statement pattern
        where the pause falls just before the window is still credited).

        Normalisation: ``min(1.0, count / 3.0)`` — three or more pauses
        saturates to 1.0.

        Args:
            start_s: Window start in seconds.
            end_s: Window end in seconds.

        Returns:
            Float in [0.0, 1.0].
        """
        try:
            lookback = self._pause_min_s
            count = sum(
                1
                for (gs, ge) in self._silence_gaps
                if (start_s - lookback) <= (gs + ge) / 2.0 < end_s
            )
            return min(1.0, count / 3.0)
        except Exception:
            return 0.0

    def silence_boundaries(self, start_s: float, end_s: float) -> list[float]:
        """Return silence-gap midpoints within the window.

        These midpoints mark natural cut-points for boundary refinement in
        the no-transcript path — a sentence boundary is likely close to a
        significant pause.

        Args:
            start_s: Window start in seconds.
            end_s: Window end in seconds.

        Returns:
            List of midpoint timestamps (floats, in seconds) for gaps whose
            midpoint falls within [start_s, end_s).  Empty list if none.
        """
        try:
            midpoints = []
            for (gs, ge) in self._silence_gaps:
                mid = (gs + ge) / 2.0
                if start_s <= mid < end_s:
                    midpoints.append(mid)
            return midpoints
        except Exception:
            return []
