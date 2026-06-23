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
openclaw/
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
│   ├── openclaw.db
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
transcribing
  │
  ▼
detecting
  │
  ▼
editing
  │
  ▼
captioning
  │
  ▼
formatting
  │
  ▼
posting               ← auto-post all clips to all platforms
  │
  ▼
complete
```

Every transition writes `last_stage_completed` before advancing. On worker restart, any job not in `complete / failed / cancelled` resumes from `last_stage_completed + 1`.

---

## 5. Worker (job_runner.py)

```python
class JobRunner:
    def __init__(self):
        self.whisper = mlx_whisper.load_model('medium')  # load once at startup

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
        stages = [
            ('downloading',  self.download),
            ('transcribing', self.transcribe),
            ('detecting',    self.detect_clips),
            ('editing',      self.edit),
            ('captioning',   self.add_captions),
            ('formatting',   self.format_clips),
            ('posting',      self.post_all),
        ]
        start_idx = self.get_resume_index(job)
        for i, (stage_name, fn) in enumerate(stages[start_idx:], start=start_idx):
            state.update_job_stage(job.id, stage_name)
            fn(job)
            state.mark_stage_complete(job.id, stage_name)
        state.mark_job_complete(job.id)
```

Single worker, one job at a time. Whisper stays loaded in RAM between jobs. Raw download deleted after job completes.

---

## 6. Clip Detector — 5 Universal Signals

| Signal | Weight | Rationale |
|---|---|---|
| Sentiment magnitude | 0.30 | Strong emotion is engaging across all content |
| Speech pace | 0.20 | Energy spikes are universal |
| Question density | 0.15 | Questions hook viewers universally |
| Generic keyword hits | 0.20 | "The truth is", "nobody talks about", "here's why" |
| Laughter/reaction markers | 0.15 | Whisper's `[laughter]`, `[applause]` tags |

Outputs ranked clip windows `(start_s, end_s, score)`. Score threshold: 0.45. Max 3 clips per job.

---

## 7. Posting — Automatic, All Platforms

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
# Usage: python main.py <youtube_url>
import sys
from core.url_validator import validate_url
from core import state
from core.job_runner import JobRunner

if __name__ == '__main__':
    url = sys.argv[1]
    validate_url(url)           # raises on bad input
    job_id = state.enqueue_job(url)
    print(f"Job queued: {job_id}")
    runner = JobRunner()
    runner.run()                # blocks until queue is drained
```

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

**Phase B — Core + Processing:**
1. `core/state.py` — all SQLite ops (create tables, enqueue, read/write job/clip/post state)
2. `core/job_states.py` — `JobState` enum, valid transition map, `assert_valid_transition()`
3. `core/url_validator.py` — regex check + yt-dlp `extract_info` preflight
4. `processing/downloader.py` — yt-dlp download to `data/jobs/<id>/raw/`
5. `processing/transcriber.py` — mlx-whisper → `[{start, end, text}]`
6. `processing/clip_detector.py` — 5-signal scorer → ranked `(start_s, end_s, score)` list
7. `processing/editor.py` — ffprobe aspect check, smart crop or center crop to 9:16
8. `processing/caption_burner.py` — ffmpeg `subtitles` filter burn-in
9. `processing/formatter.py` — final encode, enforce 60s cap and 50MB size limit
10. `core/job_runner.py` — wires stages 4–9, crash recovery, disk-space check

**Phase C — Posting:**
11. `posting/youtube.py` — YouTube Data API v3, resumable upload to Shorts
12. `posting/tiktok.py` — TikTok Content Posting API
13. `posting/instagram.py` — Instagram Graph API Reels upload
14. Wire `post_all()` into `job_runner.py` after formatting

**Phase D — Entry point + integration:**
15. `main.py` — CLI glue, end-to-end smoke test

---

## 13. Definition of Done

- [ ] `python main.py <url>` runs end-to-end without errors
- [ ] 3 clips produced for a 30-minute test video
- [ ] All 3 clips auto-posted to YouTube Shorts, TikTok, and Instagram Reels
- [ ] Worker survives `kill -9` mid-job and resumes correctly
- [ ] Bad URLs (private, live, non-YT) rejected cleanly
- [ ] All pytest suites pass
- [ ] Raw downloads cleaned up after job completion
