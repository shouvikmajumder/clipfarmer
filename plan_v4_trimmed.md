# clipfarmer.bot v4 — Trimmed Build Plan
> Core goal: **URL in → process → clips out → auto-post to YouTube Shorts, TikTok, Instagram Reels**
> Supersedes v3. Everything that doesn't serve that goal is removed.

---

## 1. What Changed From v3

The v3 plan was built around two optional UI layers (Telegram bot, Flask web UI) and treated posting as opt-in with confirmation buttons per clip. Neither of those is the core goal.

**v4 removes entirely:**
- Telegram bot (`interfaces/telegram_bot.py`, `delivery/telegram_delivery.py`, `alerts/`)
- Flask web UI (`interfaces/web_ui.py`, `delivery/web_delivery.py`)
- Opt-in posting gates — posting is now automatic on job completion
- `assets/music/` — optional background music is v5+
- `events` DB table — replaced by Python file logging
- `progress_message_id` DB column — Telegram-only artifact
- `scratch/` external drive convention — hardware concern, not code
- launchd plist — production deployment concern, not a build blocker

**v4 adds nothing new** — it's the v3 processing core with the UX wrapper stripped off.

**Entry point:** `python main.py <youtube_url>` — a single CLI call.

---

## 2. Project Structure

```
clipfarmer/
├── main.py                      # CLI: python main.py <youtube_url>
├── requirements.txt
├── pytest.ini
├── .gitignore
├── config/
│   ├── settings.yaml
│   └── .env.example
├── core/
│   ├── __init__.py
│   ├── state.py                 # JSON read/write — all state ops live here
│   ├── job_runner.py            # Worker loop: drains the queue one job at a time
│   ├── job_states.py            # JobState enum + valid transitions
│   └── url_validator.py         # YT URL regex + yt-dlp preflight check
├── processing/
│   ├── __init__.py
│   ├── downloader.py            # yt-dlp download to data/jobs/<id>/
│   ├── transcriber.py           # mlx-whisper → timestamped transcript
│   ├── clip_detector.py         # 5-signal scorer → ranked clip windows
│   ├── editor.py                # ffmpeg crop to 9:16 + smart/center crop
│   ├── caption_burner.py        # Burn captions into video via ffmpeg
│   └── formatter.py             # Final encode, enforce <60s, <50MB
├── posting/
│   ├── __init__.py
│   ├── youtube.py               # YouTube Data API v3 → Shorts upload
│   ├── tiktok.py                # TikTok Content Posting API upload
│   └── instagram.py             # Instagram Graph API → Reels upload
├── data/
│   ├── clipfarmer.db
│   ├── jobs/                    # Per-job working dirs, cleaned after success
│   └── logs/
└── tests/
    ├── test_state.py
    ├── test_job_states.py
    ├── test_url_validator.py
    ├── test_downloader.py
    ├── test_transcriber.py
    ├── test_clip_detector.py
    ├── test_editor.py
    ├── test_caption_burner.py
    ├── test_formatter.py
    ├── test_job_runner.py
    └── test_posting.py
```

---

## 3. JSON Storage

All state is stored as JSON files co-located with each job's working directory. No database dependency — easy to inspect, manually fix, and later migrate to SQLite.

**Directory layout under `data/`:**
```
data/
├── jobs/
│   └── <job_id>/
│       ├── job.json        # job metadata + state
│       ├── clips.json      # list of clip records produced from this job
│       ├── posts.json      # list of post records for clips in this job
│       ├── raw/            # downloaded video file(s)
│       └── clips/          # processed output clip files
└── logs/
```

**job.json shape:**
```json
{
  "id": "<uuid>",
  "youtube_url": "https://...",
  "youtube_id": "abc123",
  "video_title": "...",
  "video_duration_s": 1200,
  "submitted_at": "2026-01-01T00:00:00",
  "state": "queued",
  "current_stage": null,
  "last_stage_completed": null,
  "error_message": null,
  "retry_count": 0,
  "options": {"max_clips": 3}
}
```

**clips.json shape** (list):
```json
[
  {
    "id": "<uuid>",
    "job_id": "<job_uuid>",
    "file_path": "data/jobs/<job_id>/clips/clip_0.mp4",
    "score": 0.78,
    "start_s": 142,
    "end_s": 198,
    "transcript_snippet": "Here's the thing nobody tells you...",
    "status": "ready",
    "created_at": "2026-01-01T00:05:00"
  }
]
```

**posts.json shape** (list):
```json
[
  {
    "id": "<uuid>",
    "clip_id": "<clip_uuid>",
    "platform": "youtube",
    "status": "queued",
    "posted_at": null,
    "post_url": null,
    "error": null,
    "retry_count": 0
  }
]
```

All writes go through `core/state.py`, which reads the relevant JSON file, mutates it in memory, and atomically writes it back (write to `.tmp`, then `os.replace`). Since the worker processes one job at a time there are no concurrent write conflicts.

---

## 4. Job State Machine

**Scope note:** this pipeline's only job is to turn a submitted URL into a
downloaded video file on disk, queued so many videos can be submitted without
blocking on each other. Clip curation (transcription, clip scoring, cropping,
captioning, formatting) and posting are a **separate downstream workflow**,
built later once we've done more research into generating good clips. They
are not wired into this state machine.

```
queued
  │
  ▼
pre-flight            ← yt-dlp extract_info(download=False): validate URL,
  │                     check accessibility, get duration
  ├─► cancelled       ← bad URL, private, live stream, or >6 hours
  │
  ▼
downloading
  │
  ▼
complete              ← video file is on disk, ready for the clip-curation
                          workflow to pick up later
```

Jobs are processed one at a time, FIFO, oldest `submitted_at` first. On
worker restart, any job not in `complete / failed / cancelled` resumes by
re-running the downloading stage (yt-dlp's own download is idempotent/
resumable, so re-running it is safe).

---

## 5. Worker (job_runner.py)

```python
class JobRunner:
    def run(self):
        while True:
            job = state.get_next_queued_job()
            if not job:
                time.sleep(5)
                continue
            try:
                self.process(job)
            except Exception as e:
                state.mark_job_failed(job.id, str(e))
                logger.error(f"Job {job.id} failed: {e}")

    def process(self, job):
        metadata = validate_url(job.youtube_url)   # cancels job on bad input
        state.update_job_metadata(job.id, **metadata)
        state.update_job_stage(job.id, 'downloading')
        download(job)                               # processing/downloader.py
        state.mark_stage_complete(job.id, 'downloading')
        state.mark_job_complete(job.id)
```

Single worker, sequential FIFO queue — one download at a time, in submission
order. No transcription model to load, no clip processing in this loop.
Downloaded files are **kept** (not cleaned up) since they're the deliverable
the next workflow reads from.

---

## 6. Clip Detector — 5 Universal Signals (deferred, needs more research)

> **Not built in this phase.** This is a placeholder design for the future
> clip-curation workflow, kept here for reference. It is not implemented or
> called by `job_runner.py`. Before building it for real we need more
> research into what actually makes a good clip — the weights/signals below
> are a starting hypothesis, not a finished design.

| Signal | Weight | Rationale |
|---|---|---|
| Sentiment magnitude | 0.30 | Strong emotion is engaging across all content |
| Speech pace | 0.20 | Energy spikes are universal |
| Question density | 0.15 | Questions hook viewers universally |
| Generic keyword hits | 0.20 | "The truth is", "nobody talks about", "here's why" |
| Laughter/reaction markers | 0.15 | Whisper's `[laughter]`, `[applause]` tags |

Outputs ranked clip windows `(start_s, end_s, score)`. Score threshold: 0.45. Max 3 clips per job.

---

## 7. Posting — Automatic, All Platforms (deferred, not part of this pipeline)

> **Not built in this phase.** This pipeline stops at "video downloaded to
> disk." Clip detection, editing, captioning, formatting, and posting become
> their own downstream workflow once we've researched how to reliably
> generate good clips. The sections below describe that future workflow's
> intended shape; they are not implemented or wired into `job_runner.py`.

After `formatting`, `post_all` iterates every clip and posts it to all three platforms without user confirmation. Each platform gets its own row in `posts`. Failures are retried up to 3× with exponential backoff; after that the post is marked `failed` and the clip is still kept in DB.

```python
PLATFORMS = ['youtube', 'tiktok', 'instagram']

def post_all(self, job):
    clips = state.get_clips_for_job(job.id)
    for clip in clips:
        for platform in PLATFORMS:
            post_id = state.create_post_record(clip.id, platform)
            try:
                url = post_to_platform(platform, clip)
                state.mark_post_success(post_id, url)
            except Exception as e:
                state.mark_post_failed(post_id, str(e))
```

---

## 8. Entry Point (main.py)

```python
# Usage: python main.py <youtube_url>   (queue one video and run the worker)
#        python main.py                 (just run the worker, draining the queue)
import sys
from core import state
from core.job_runner import JobRunner

if __name__ == '__main__':
    if len(sys.argv) > 1:
        job_id = state.enqueue_job(sys.argv[1])
        print(f"Job queued: {job_id}")
    runner = JobRunner()
    runner.run()                # blocks, processing the FIFO queue one job at a time
```

Multiple videos can be queued (via repeated CLI calls or the web UI) before
or while the worker is running — they process sequentially in submission
order.

---

## 9. Settings (config/settings.yaml)

```yaml
general:
  max_clip_length_s: 60
  min_clip_score: 0.45
  max_clips_per_job: 3
  max_video_duration_hard_limit_s: 21600   # 6 hours
  min_free_disk_gb: 5
  data_dir: data/

worker:
  whisper_model: medium
  job_poll_interval_s: 5

posting:
  platforms: [youtube, tiktok, instagram]
  max_retries: 3
```

---

## 10. Environment Variables (.env.example)

```
# YouTube Data API v3
YOUTUBE_CLIENT_ID=
YOUTUBE_CLIENT_SECRET=
YOUTUBE_REFRESH_TOKEN=

# TikTok Content Posting API
TIKTOK_CLIENT_KEY=
TIKTOK_CLIENT_SECRET=
TIKTOK_ACCESS_TOKEN=

# Instagram Graph API
INSTAGRAM_ACCESS_TOKEN=
INSTAGRAM_ACCOUNT_ID=
```

---

## 11. Failure Handling

| Failure | Retry | Outcome |
|---|---|---|
| Invalid / private / live URL | 0 | Job cancelled immediately with reason |
| Download fails | 2× | Job failed |
| Transcription fails | 1× | Job failed |
| FFmpeg crash on one clip | 1× with center-crop fallback | That clip skipped; others continue |
| Disk full | 0 | Worker halts; logs error |
| Whisper OOM | 0 | Worker halts; logs error |
| Platform post fails | 3× exp backoff | Post marked failed; clip kept in DB |
| Worker crashes mid-job | launchd restart | Resume from `last_stage_completed + 1` |

---

## 12. Build Sequence

One commit per module. Test before committing.

**Phase A — Scaffold (1 commit):**
- Full directory tree, empty `__init__.py` files, stub functions with docstrings
- `requirements.txt`, `.gitignore`, `pytest.ini`, `.env.example`, `config/settings.yaml`

**Phase B — Download queue (current scope):**
1. `core/state.py` — JSON read/write for job state (queued/downloading/complete/failed/cancelled)
2. `core/job_states.py` — `JobState` enum (`QUEUED`, `DOWNLOADING`, `COMPLETE`, `FAILED`, `CANCELLED`), valid transition map
3. `core/url_validator.py` — regex check + yt-dlp `extract_info` preflight
4. `processing/downloader.py` — yt-dlp download to `data/jobs/<id>/raw/`
5. `core/job_runner.py` — FIFO worker: pre-flight → download → complete, crash recovery, disk-space check

Deliverable: submit any number of YouTube URLs, the worker downloads them
one at a time into `data/jobs/<id>/raw/`, and a job is `complete` once its
file is on disk and ready for the next workflow to read.

**Phase C — Clip curation + posting (later, needs research first):**
- `processing/transcriber.py`, `clip_detector.py`, `editor.py`,
  `caption_burner.py`, `formatter.py` — exist in the repo as a design
  sketch only; not wired into anything yet.
- `posting/youtube.py`, `tiktok.py`, `instagram.py` — stubs, untouched.
- This becomes its own workflow that reads completed jobs' downloaded files,
  once we've researched what actually makes a good clip.

---

## 13. Definition of Done (Phase B scope)

- [ ] `python main.py <url>` queues a video and the worker downloads it without errors
- [ ] Multiple URLs queued back-to-back all process sequentially, FIFO, without blocking submission
- [ ] Worker survives `kill -9` mid-download and resumes correctly
- [ ] Bad URLs (private, live, non-YT) rejected cleanly at pre-flight
- [ ] Downloaded file is kept on disk (not cleaned up) once the job is `complete`
- [ ] All pytest suites pass
