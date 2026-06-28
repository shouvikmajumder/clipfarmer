#!/usr/bin/env python3
"""
ClipFarmer clip detection debug script.
Usage: python debug_clip_detection.py <YOUTUBE_URL> [--profile default] [--max-clips 3]

Standalone integration runner for the clip detection layer. Runs the REAL
detection code against a REAL YouTube video (real network + real audio
analysis) and prints a sectioned PASS / WARN / FAIL report.

NOTE: This is wired to the actually-implemented v5 API, not the richer
hypothetical surface (ClipCandidate / per-signal classes / content-type
profiles / scene_change). Concretely:
  - audio:      processing.audio_analyzer.AudioSignals.from_file(path)
  - transcript: processing.transcript_fetcher.fetch_transcript(id)  -> segment dicts
  - comments:   processing.comments.fetch_comments + cluster_timestamps
  - signals:    processing.signals.* (module functions, not classes)
  - detect:     processing.clip_detector.detect(transcript, comments, audio_signals, max_clips)
detect() returns plain {"start_s","end_s","score"} dicts, so this script
recomputes the per-signal breakdown itself for the per-clip display.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the in-repo `processing` package importable. It lives under clipfarmer/.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CLIPFARMER = ROOT / "clipfarmer"
if str(CLIPFARMER) not in sys.path:
    sys.path.insert(0, str(CLIPFARMER))

# ---------------------------------------------------------------------------
# Output formatting (ANSI, stdlib only)
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

COUNTS = {"pass": 0, "warn": 0, "fail": 0}
TIMINGS: dict[str, float] = {}


def ok(msg):
    COUNTS["pass"] += 1
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")


def warn(msg):
    COUNTS["warn"] += 1
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")


def fail(msg):
    COUNTS["fail"] += 1
    print(f"  {RED}✗ FAIL{RESET}  {msg}")


def info(msg):
    print(f"  {CYAN}→{RESET}      {msg}")


def section(title):
    print(f"\n{BOLD}{'─' * 60}\n  {title}\n{'─' * 60}{RESET}")


def bar(score: float, width: int = 20) -> str:
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Shared helpers (real signal recomputation)
# ---------------------------------------------------------------------------
def _window_text(transcript, start, end) -> str:
    """Concatenated text of transcript segments overlapping [start, end)."""
    if not transcript:
        return ""
    segs = [
        s for s in transcript
        if (s.get("start") or 0.0) < end and (s.get("end") or 0.0) > start
    ]
    return " ".join((s.get("text") or "") for s in segs).strip()


def compute_signals(start, end, transcript, audio):
    """Recompute the 8 v5 signals for a window, mirroring clip_detector.

    Returns (window_text, first_sentence, {signal_name: score}). Used for both
    the Section 7 dry run and the per-clip breakdown in Section 8.
    """
    from processing import signals as sig
    from processing.clip_detector import REACTION_MARKER_RE

    text = _window_text(transcript, start, end)
    has_t = bool(text)
    fs = sig.first_sentence(text) if has_t else ""

    scores: dict[str, float] = {}
    scores["hook_quality"] = sig.hook_quality(fs) if has_t else 0.0
    scores["coherence"] = sig.standalone_coherence(fs) if has_t else 0.0
    scores["sentiment"] = sig.sentiment(text) if has_t else 0.0
    scores["specificity"] = sig.specificity(text) if has_t else 0.0
    scores["curiosity"] = sig.curiosity(text) if has_t else 0.0

    if audio is not None:
        audio_laugh = audio.laughter(start, end)
        text_laugh = min(1.0, len(REACTION_MARKER_RE.findall(text)) / 2.0)
        scores["audio_energy"] = audio.energy(start, end)
        scores["laughter"] = max(text_laugh, audio_laugh)
        scores["dramatic_pause"] = audio.dramatic_pause(start, end)
    else:
        scores["audio_energy"] = 0.0
        scores["laughter"] = 0.0
        scores["dramatic_pause"] = 0.0

    return text, fs, scores


SIGNAL_ORDER = [
    "hook_quality", "coherence", "sentiment", "specificity",
    "curiosity", "audio_energy", "laughter", "dramatic_pause",
]
TRANSCRIPT_SIGNALS = {"hook_quality", "coherence", "sentiment", "specificity", "curiosity"}

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})")


# ===========================================================================
# Section 1 — dependency check
# ===========================================================================
def check_dependencies():
    section("1. DEPENDENCY CHECK")

    hard_failed = False

    # (name, import, version_getter, required_for_detection)
    def probe(label, importer, required):
        nonlocal hard_failed
        try:
            ver = importer()
            ok(f"{label}{(' — ' + str(ver)) if ver else ''}")
        except Exception as e:  # noqa: BLE001 - debug tool, surface reason
            if required:
                hard_failed = True
                fail(f"{label} — {e}")
            else:
                warn(f"{label} not available ({e})")

    probe("yt-dlp", lambda: __import__("yt_dlp").version.__version__, True)
    probe("librosa", lambda: __import__("librosa").__version__, True)
    probe("numpy", lambda: __import__("numpy").__version__, True)
    # Optional: detection degrades gracefully without these.
    probe("youtube-transcript-api (transcript signal)",
          lambda: (__import__("youtube_transcript_api"), "")[1], False)
    probe("google-api-python-client (comment signal)",
          lambda: (__import__("googleapiclient"), "")[1], False)
    # Used elsewhere in the pipeline, not by detection itself.
    probe("opencv-python (editor stage)", lambda: __import__("cv2").__version__, False)
    probe("mlx-whisper (captioning stage)",
          lambda: (__import__("mlx_whisper"), "")[1], False)

    # ffmpeg on PATH
    import shutil
    if shutil.which("ffmpeg"):
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        ok(f"ffmpeg found — {version_line}")
    else:
        hard_failed = True
        fail("ffmpeg not found on PATH")

    # Processing layer imports cleanly (real names)
    try:
        from processing.clip_detector import detect, REACTION_MARKER_RE  # noqa: F401
        from processing.transcript_fetcher import fetch_transcript  # noqa: F401
        from processing.comments import (  # noqa: F401
            fetch_comments, cluster_timestamps, social_score,
            sponsor_penalty, timestamped_comment_count,
        )
        from processing.audio_analyzer import AudioSignals  # noqa: F401
        from processing import signals  # noqa: F401
        ok("processing layer imports cleanly")
    except ImportError as e:
        hard_failed = True
        fail(f"processing layer import error: {e}")

    if hard_failed:
        print(f"\n  {RED}✗ Stopping — fix the above failures before running the rest of this script.{RESET}")
        sys.exit(1)


# ===========================================================================
# Section 2 — URL validation + metadata pre-flight
# ===========================================================================
def validate_url(url):
    section("2. URL VALIDATION")

    m = YOUTUBE_ID_RE.search(url)
    if not m:
        fail(f"Could not extract video ID from: {url}")
        sys.exit(1)
    youtube_id = m.group(1)
    ok(f"Video ID: {youtube_id}")

    import yt_dlp
    t0 = time.time()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        meta = ydl.extract_info(url, download=False)
    TIMINGS["Metadata"] = time.time() - t0

    duration = meta.get("duration") or 0
    ok(f"Title: {meta.get('title', 'unknown')}")
    ok(f"Duration: {duration // 60}m {duration % 60}s")
    ok(f"Channel: {meta.get('channel', 'unknown')}")
    ok(f"View count: {meta.get('view_count', 0):,}")
    info(f"Metadata fetched in {TIMINGS['Metadata']:.1f}s")

    if duration > 7200:
        fail("Video is over 2 hours — use a shorter video for debugging")
        sys.exit(1)
    if duration > 3600:
        warn("Video is over 60 minutes — debug run will take a while")

    return youtube_id, meta.get("title", "unknown"), float(duration)


# ===========================================================================
# Section 3 — download
# ===========================================================================
def download_video(url, youtube_id, tmpdir, skip_path):
    section("3. DOWNLOAD")

    if skip_path:
        p = Path(skip_path)
        if not p.exists():
            fail(f"--skip-download path does not exist: {p}")
            sys.exit(1)
        size_mb = p.stat().st_size / 1e6
        ok(f"Using existing file: {p.name} ({size_mb:.1f} MB)")
        TIMINGS["Download"] = 0.0
        return p

    info(f"Scratch dir: {tmpdir}")
    import yt_dlp
    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": str(tmpdir / f"{youtube_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    t0 = time.time()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    TIMINGS["Download"] = time.time() - t0

    mp4_files = list(tmpdir.glob(f"{youtube_id}*.mp4"))
    if not mp4_files:
        fail("Download produced no mp4 file")
        sys.exit(1)
    raw_video_path = mp4_files[0]
    size_mb = raw_video_path.stat().st_size / 1e6
    ok(f"Downloaded: {raw_video_path.name} ({size_mb:.1f} MB) in {TIMINGS['Download']:.1f}s")
    return raw_video_path


# ===========================================================================
# Section 4 — audio analysis
# ===========================================================================
def analyze_audio_section(raw_video_path, video_duration_s):
    section("4. AUDIO ANALYSIS")

    from processing.audio_analyzer import AudioSignals
    import numpy as np

    t0 = time.time()
    try:
        audio = AudioSignals.from_file(str(raw_video_path))
        TIMINGS["Audio analyze"] = time.time() - t0
    except Exception as e:  # noqa: BLE001 - surface in debug tool
        TIMINGS["Audio analyze"] = time.time() - t0
        fail(f"Audio analysis failed: {e}")
        fail("Cannot continue without audio features")
        sys.exit(1)

    ok(f"Audio analyzed in {TIMINGS['Audio analyze']:.1f}s")
    ok(f"Duration: {audio.duration_s:.1f}s")

    # AudioSignals exposes per-window methods, not a raw RMS array. Probe a
    # normalized per-second energy envelope so we can show loudest moments.
    secs = int(audio.duration_s)
    rms = np.array([audio.energy(t, t + 1) for t in range(secs)]) if secs > 0 else np.array([])
    ok(f"Per-second energy samples: {len(rms)} (probed via AudioSignals.energy)")

    if len(rms):
        baseline = float(np.median(rms[rms > 0])) if np.any(rms > 0) else 0.0
    else:
        baseline = 0.0
    ok(f"Baseline (median normalized energy): {baseline:.4f}")

    pauses = audio.silence_boundaries(0.0, audio.duration_s)
    ok(f"Dramatic pauses detected: {len(pauses)} (≥ silence threshold)")
    if pauses:
        info(f"First 5 pauses at: {[fmt_ts(t) for t in pauses[:5]]}")

    if len(rms) < 10:
        warn("Very few energy samples — audio extraction may have partially failed")

    if len(rms):
        top5_idx = np.argsort(rms)[-5:][::-1]
        info("5 loudest seconds (potential highlights):")
        for idx in top5_idx:
            ratio = (rms[idx] / baseline) if baseline > 0 else 0.0
            print(f"    {fmt_ts(int(idx))}  energy={rms[idx]:.4f}  ({ratio:.1f}× baseline)")

    return audio


# ===========================================================================
# Section 5 — transcript
# ===========================================================================
def fetch_transcript_section(youtube_id, video_duration_s):
    section("5. TRANSCRIPT FETCH")

    from processing.transcript_fetcher import fetch_transcript

    t0 = time.time()
    transcript = fetch_transcript(youtube_id)
    TIMINGS["Transcript"] = time.time() - t0

    if transcript is None:
        warn(f"No transcript available ({TIMINGS['Transcript']:.1f}s) — detection will use audio + social signals only")
        warn("This is handled gracefully. Not a failure.")
        return None

    word_count = sum(len((s.get("text") or "").split()) for s in transcript)
    ok(f"Transcript fetched in {TIMINGS['Transcript']:.1f}s — {len(transcript)} segments (~{word_count} words)")

    info("First 5 segments with timestamps:")
    for s in transcript[:5]:
        text = (s.get("text") or "").strip()
        print(f"    {s.get('start', 0.0):6.2f}s  '{text[:60]}'")
    info("Last 3 segments:")
    for s in transcript[-3:]:
        text = (s.get("text") or "").strip()
        print(f"    {s.get('start', 0.0):6.2f}s  '{text[:60]}'")

    last_end = transcript[-1].get("end") or 0.0
    coverage = (last_end / video_duration_s) if video_duration_s else 0.0
    if coverage < 0.7:
        warn(f"Transcript only covers {coverage * 100:.0f}% of video duration")
    else:
        ok(f"Transcript covers {coverage * 100:.0f}% of video duration")

    return transcript


# ===========================================================================
# Section 6 — comments
# ===========================================================================
def fetch_comments_section(youtube_id):
    section("6. COMMENT FETCHER")

    from processing.comments import (
        fetch_comments, cluster_timestamps, sponsor_penalty,
        timestamped_comment_count,
    )

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        warn("YOUTUBE_API_KEY not set — skipping comment fetch")
        warn("Set it with: export YOUTUBE_API_KEY=your_key")
        TIMINGS["Comments"] = 0.0
        return [], []

    t0 = time.time()
    comments = fetch_comments(youtube_id, api_key=api_key)
    clusters = cluster_timestamps(comments)
    TIMINGS["Comments"] = time.time() - t0

    if not comments:
        warn(f"No comments fetched ({TIMINGS['Comments']:.1f}s) — comments disabled, no API quota, or bad key")
        return comments, clusters

    ts_total = timestamped_comment_count(comments)
    ok(f"Fetched {len(comments)} comment(s) in {TIMINGS['Comments']:.1f}s ({ts_total} timestamp mentions)")

    if not clusters:
        warn("No timestamp clusters found — viewers didn't timestamp notable moments")
        return comments, clusters

    ok(f"Found {len(clusters)} timestamp cluster(s)")
    info("Top 3 clusters by timestamp count:")
    for i, c in enumerate(clusters[:3]):
        sp = sponsor_penalty(comments, clusters, c["start_s"], c["end_s"])
        sponsor_flag = "YES ⚠" if sp < 1.0 else "no"
        print(
            f"    #{i + 1}  {fmt_ts(c['center_s'])}  "
            f"count={c['count']}  "
            f"span={fmt_ts(c['start_s'])}–{fmt_ts(c['end_s'])}  "
            f"sponsor={sponsor_flag}"
        )

    return comments, clusters


# ===========================================================================
# Section 7 — signal dry run
# ===========================================================================
def signal_dry_run(transcript, comments, clusters, audio, video_duration_s):
    section("7. SIGNAL DRY RUN")

    if clusters:
        anchor = clusters[0]["center_s"]
        test_start = max(0.0, anchor - 20.0)
    else:
        test_start = video_duration_s * 0.4
    test_end = min(test_start + 60.0, audio.duration_s if audio else test_start + 60.0)

    info(f"Test window: {fmt_ts(test_start)} → {fmt_ts(test_end)}")

    text, fs, scores = compute_signals(test_start, test_end, transcript, audio)
    info(f"Words in window: {len(text.split())}")
    if fs:
        info(f'First sentence: "{fs[:100]}"')

    print()
    print(f"  {'Signal':<18} {'Score':>6}  {'Bar':<20}  Notes")
    print(f"  {'─' * 18} {'─' * 6}  {'─' * 20}  {'─' * 18}")
    for name in SIGNAL_ORDER:
        score = scores.get(name, 0.0)
        note = ""
        if not transcript and name in TRANSCRIPT_SIGNALS:
            note = "(no transcript)"
        elif audio is None and name in ("audio_energy", "laughter", "dramatic_pause"):
            note = "(no audio)"
        print(f"  {name:<18} {score:>6.3f}  {bar(score):<20}  {note}")

    return scores


# ===========================================================================
# Section 8 — full detection run
# ===========================================================================
def _cluster_in_window(clusters, start, end):
    return any(start <= c["center_s"] <= end for c in clusters)


def print_candidate(i, total, clip, transcript, comments, clusters, audio):
    start = clip["start_s"]
    end = clip["end_s"]
    duration = end - start
    score = clip["score"]
    has_social = _cluster_in_window(clusters, start, end)
    anchor_label = "(comment cluster)" if has_social else "(window midpoint)"

    text, fs, scores = compute_signals(start, end, transcript, audio)
    snippet = (fs or text or "").strip().replace("\n", " ")
    if len(snippet) > 52:
        snippet = snippet[:49] + "..."

    W = 58

    def line(s=""):
        print(f"  │ {s:<{W}} │")

    print(f"  ┌─{'─' * W}─┐")
    line(f"CLIP {i + 1} of {total}")  # no ANSI here so the box border stays aligned
    line(f"Start:  {fmt_ts(start)}   End:  {fmt_ts(end)}   Duration: {duration:.0f}s")
    line(f"Anchor: {fmt_ts((start + end) / 2)}   {anchor_label}")
    line(f"Final score:   {score:.3f}")
    line(f"Has transcript: {'YES' if text else 'no'}    Has social signal: {'YES' if has_social else 'no'}")
    line()
    line("Signal breakdown (recomputed for display):")
    for name in SIGNAL_ORDER:
        sc = scores.get(name, 0.0)
        line(f"  {name:<16} {sc:.3f}  {bar(sc, 18)}")
    line()
    line(f'Snippet: "{snippet}"')
    print(f"  └─{'─' * W}─┘")


def full_detection_run(transcript, comments, clusters, audio, args):
    section("8. FULL DETECTION RUN")

    from processing.clip_detector import detect

    t0 = time.time()
    candidates = detect(
        transcript,
        comments=comments,
        audio_signals=audio,
        max_clips=args.max_clips,
    )
    TIMINGS["Detection"] = time.time() - t0

    if not candidates:
        fail(f"detect() returned empty list ({TIMINGS['Detection']:.1f}s)")
        return candidates

    ok(f"Detection completed in {TIMINGS['Detection']:.1f}s — {len(candidates)} clip(s) found")
    print()
    for i, clip in enumerate(candidates):
        print_candidate(i, len(candidates), clip, transcript, comments, clusters, audio)
        print()

    if args.profile != "default":
        warn(f"--profile {args.profile}: content-type profiles aren't implemented; "
             f"the detector auto-selects default vs no_transcript weights. Ignored.")

    return candidates


# ===========================================================================
# Section 9 — validation
# ===========================================================================
def validate_candidates(candidates, transcript, audio, video_duration_s):
    section("9. VALIDATION")

    if not candidates:
        warn("No candidates to validate")
        return

    for i, c in enumerate(candidates):
        prefix = f"Clip {i + 1}"
        score = c["score"]
        start, end = c["start_s"], c["end_s"]

        if 0.0 <= score <= 1.0:
            ok(f"{prefix}: score in [0,1] ({score:.3f})")
        else:
            fail(f"{prefix}: score out of range ({score})")

        if start < end:
            ok(f"{prefix}: start < end ({start:.1f}s < {end:.1f}s)")
        else:
            fail(f"{prefix}: start >= end!")

        duration = end - start
        if 15 <= duration <= 90:
            ok(f"{prefix}: duration reasonable ({duration:.0f}s)")
        else:
            warn(f"{prefix}: duration unusual ({duration:.0f}s — expected 15-90s)")

        if 0 <= start and end <= video_duration_s + 5:
            ok(f"{prefix}: within video bounds")
        else:
            fail(f"{prefix}: outside video duration!")

        # Recomputed signal breakdown must all be in range.
        _, _, scores = compute_signals(start, end, transcript, audio)
        bad = {k: v for k, v in scores.items() if not 0.0 <= v <= 1.0}
        if not bad:
            ok(f"{prefix}: all recomputed signal scores in [0,1]")
        else:
            fail(f"{prefix}: signal scores out of range: {bad}")

        # Snippet availability
        snippet = _window_text(transcript, start, end)
        if snippet and len(snippet) > 5:
            ok(f"{prefix}: transcript snippet present")
        elif not transcript:
            info(f"{prefix}: no snippet (no transcript available — expected)")
        else:
            warn(f"{prefix}: snippet empty despite transcript present")

    # Overlap check between consecutive clips
    if len(candidates) > 1:
        ordered = sorted(candidates, key=lambda c: c["start_s"])
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            overlap = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
            total = max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"])
            iou = overlap / total if total > 0 else 0.0
            if iou < 0.5:
                ok(f"Clips at {fmt_ts(a['start_s'])} and {fmt_ts(b['start_s'])}: no significant overlap (IoU={iou:.2f})")
            else:
                fail(f"Clips at {fmt_ts(a['start_s'])} and {fmt_ts(b['start_s'])}: overlapping! (IoU={iou:.2f}) — dedup may have failed")


# ===========================================================================
# Section 10 — summary
# ===========================================================================
def summary(total_wall):
    section("10. SUMMARY")

    total = COUNTS["pass"] + COUNTS["warn"] + COUNTS["fail"]
    print(f"  Total checks:  {total}")
    print(f"  {GREEN}✓ Passed:      {COUNTS['pass']}{RESET}")
    print(f"  {YELLOW}⚠ Warned:      {COUNTS['warn']}{RESET}")
    print(f"  {RED}✗ Failed:      {COUNTS['fail']}{RESET}")
    print()

    if COUNTS["fail"] > 0:
        print(f"  {BOLD}{RED}RESULT: PARTIAL — fix failures above before using this in production{RESET}")
    elif COUNTS["warn"] > 0:
        print(f"  {BOLD}{YELLOW}RESULT: PASS WITH WARNINGS — review warnings above{RESET}")
    else:
        print(f"  {BOLD}{GREEN}RESULT: ALL PASS — clip detection layer is working correctly{RESET}")

    print()
    print(f"  Total wall time: {total_wall:.1f}s")
    print("  Breakdown:")
    for label in ("Metadata", "Download", "Audio analyze", "Transcript", "Comments", "Detection"):
        if label in TIMINGS:
            print(f"    {label + ':':<16}{TIMINGS[label]:>6.1f}s")


# ===========================================================================
# main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="ClipFarmer clip detection debug runner")
    parser.add_argument("url", help="YouTube URL to test against")
    parser.add_argument("--profile", default="default",
                        choices=["podcast", "sports", "gaming", "default"],
                        help="Content profile (informational; not implemented in v5)")
    parser.add_argument("--max-clips", type=int, default=3,
                        help="Maximum number of clips to return")
    parser.add_argument("--skip-download", metavar="PATH",
                        help="Skip download, use this existing video file instead")
    args = parser.parse_args()

    print(f"""
{BOLD}╔{'═' * 54}╗
║     ClipFarmer Clip Detection — Debug Runner           ║
╚{'═' * 54}╝{RESET}

  URL:       {args.url}
  Profile:   {args.profile}
  Max clips: {args.max_clips}""")

    import tempfile
    import shutil

    tmpdir = Path(tempfile.mkdtemp(prefix="clipfarmer_debug_"))
    wall0 = time.time()

    try:
        check_dependencies()
        youtube_id, video_title, video_duration_s = validate_url(args.url)
        raw_video_path = download_video(args.url, youtube_id, tmpdir, args.skip_download)
        audio = analyze_audio_section(raw_video_path, video_duration_s)
        transcript = fetch_transcript_section(youtube_id, video_duration_s)
        comments, clusters = fetch_comments_section(youtube_id)
        signal_dry_run(transcript, comments, clusters, audio, video_duration_s)
        candidates = full_detection_run(transcript, comments, clusters, audio, args)
        validate_candidates(candidates, transcript, audio, video_duration_s)
        summary(time.time() - wall0)
    except KeyboardInterrupt:
        print("\n  Interrupted — cleaning up...")
    finally:
        # Don't delete a user-supplied --skip-download file's directory.
        if not args.skip_download:
            shutil.rmtree(tmpdir, ignore_errors=True)
            info(f"Cleaned up {tmpdir}")

    print(f"\n{DIM}  Run again with:{RESET}")
    print(f'  python debug_clip_detection.py "{args.url}" --profile {args.profile}')


if __name__ == "__main__":
    main()
