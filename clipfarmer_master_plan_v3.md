# ClipFarmer v3 Master Build Plan
> Request-driven architecture. Supersedes v1 and v2. This is the document to hand to Claude Code.

---

## 1. Paradigm Shift

ClipFarmer is no longer a 24/7 autonomous content discovery system. It is now an **on-demand clip processing service** with two entry points: a Telegram bot you message YouTube links to, and a manual web UI where you paste links into a form. Both feed the same processing pipeline. Results return through whichever interface the user requested from (with optional opt-in posting at the end).

This eliminates entire classes of risk: no more channel monitoring failures, no scheduled cron windows, no shadowban risk from automated posting, no need to discover what's new. You decide what to clip. The system clips it.

---

## 2. All Finalized Decisions

| Category | Decision |
|---|---|
| Architecture | Request-driven, two entry points |
| Primary input | Telegram bot |
| Secondary input | Local Flask web UI |
| Posting | Opt-in per clip, never automatic |
| Worker model | Single process, one job at a time |
| Job queue | SQLite-backed, persistent, resumable |
| Whisper model | medium (good balance of accuracy + speed on M4 base RAM) |
| Telegram bot mode | Long-polling (no public webhook needed) |
| Auth | Hardcoded whitelist of Telegram user IDs |
| Niche-specific tuning | Removed for v1 — universal scorer for all content |
| Music | Optional toggle per job — default on for v1 |
| Channel monitoring | Removed entirely |
| State recovery | Resume from last completed stage on crash |
| Tech stack | Python 3.12 + python-telegram-bot + Flask |
| Backup | Time Machine (full disk) + DB pushed to GitHub |
| Dev approach | Full scaffold first, then fill in modules |
| Testing | pytest written alongside every module |
| Git workflow | One commit per module |

---

## 3. New Project Structure

```
clipfarmer/
├── main.py                      # Launches bot + worker + web UI concurrently
├── requirements.txt
├── pytest.ini
├── .gitignore
├── config/
│   ├── settings.yaml
│   └── .env.example
├── core/
│   ├── __init__.py
│   ├── state.py                 # SQLite read/write
│   ├── job_runner.py            # The worker loop
│   ├── job_states.py            # Enum + state machine logic
│   └── url_validator.py         # YT URL parsing + pre-flight checks
├── interfaces/
│   ├── __init__.py
│   ├── telegram_bot.py          # python-telegram-bot handler
│   └── web_ui.py                # Flask app
├── processing/                  # Layer 3 — unchanged
│   ├── __init__.py
│   ├── downloader.py
│   ├── transcriber.py
│   ├── clip_detector.py
│   ├── editor.py
│   ├── caption_burner.py
│   └── formatter.py
├── delivery/
│   ├── __init__.py
│   ├── telegram_delivery.py     # Sends clips back via TG
│   └── web_delivery.py          # Serves clips via web UI
├── posting/                     # Optional, opt-in
│   ├── __init__.py
│   ├── tiktok.py
│   ├── instagram.py
│   └── youtube.py
├── alerts/
│   ├── __init__.py
│   └── telegram_alerts.py
├── assets/
│   └── music/                   # Royalty-free CC0 tracks
├── data/
│   ├── clipfarmer.db
│   ├── jobs/                    # Per-job working subdirectories
│   └── logs/
├── scratch/                     # USB-C drive mount point
└── tests/
    └── ...
```

**Key removals from v2:**
- `core/orchestrator.py` (replaced by `job_runner.py`)
- `core/scheduler.py` (no more cron)
- `pipelines/` directory entirely
- `core/source_monitor.py`
- `config/channels.yaml`

---

## 4. SQLite Schema (rewritten for v3)

```sql
-- Every job a user submits (one job = one YT link)
CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,         -- UUID
    youtube_url     TEXT NOT NULL,
    youtube_id      TEXT,
    video_title     TEXT,
    video_duration_s INTEGER,
    submitted_by    TEXT NOT NULL,            -- 'telegram:<user_id>' or 'webui'
    submitted_at    TEXT DEFAULT (datetime('now')),
    state           TEXT NOT NULL DEFAULT 'queued',
    -- queued -> downloading -> transcribing -> detecting -> editing
    --  -> captioning -> formatting -> complete | failed | cancelled
    current_stage   TEXT,
    last_stage_completed TEXT,                -- For resume on crash
    error_message   TEXT,
    retry_count     INTEGER DEFAULT 0,
    options_json    TEXT,                     -- {music: true, max_clips: 3}
    progress_message_id INTEGER               -- Telegram msg ID to edit for progress
);

-- Clips produced from a completed job
CREATE TABLE clips (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    score        REAL,
    start_s      INTEGER,
    end_s        INTEGER,
    transcript_snippet TEXT,                  -- For display + alt-text
    status       TEXT DEFAULT 'ready',
    -- ready -> approved | rejected | posted
    created_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

-- Post records (only when user opts in to post)
CREATE TABLE posts (
    id           TEXT PRIMARY KEY,
    clip_id      TEXT NOT NULL,
    platform     TEXT NOT NULL,
    status       TEXT DEFAULT 'queued',
    posted_at    TEXT,
    post_url     TEXT,
    error        TEXT,
    retry_count  INTEGER DEFAULT 0,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

-- Event log
CREATE TABLE events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT,
    job_id     TEXT,
    message    TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

**Why this matters for crash recovery:** Every stage transition writes `last_stage_completed`. On worker startup, scan for any job not in a terminal state (`complete`/`failed`/`cancelled`) and resume from `last_stage_completed + 1`. No work is ever lost.

---

## 5. Job State Machine

```
                  ┌─────────┐
                  │ queued  │
                  └────┬────┘
                       │
              ┌────────▼─────────┐
              │ pre-flight check │ ← validates URL, gets metadata
              └────────┬─────────┘
                  ┌────┴────┐
                  ▼         ▼
            cancelled    downloading
            (bad URL,         │
             too long, etc.)  ▼
                          transcribing
                              │
                              ▼
                          detecting clips
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
                          complete  ──► user sees clips
                              │
                              ▼
                          (user opt-in)
                              │
                              ▼
                          posting (per clip, per platform)
```

Every transition is atomic in SQLite. Worker reads job, processes one stage, commits the new state, repeats.

---

## 6. The Telegram Bot

**Authentication:**
```python
ALLOWED_USER_IDS = [int(x) for x in os.environ['ALLOWED_TELEGRAM_USER_IDS'].split(',')]

async def auth_filter(update):
    return update.effective_user.id in ALLOWED_USER_IDS
```
Unauthorized messages get silently ignored. No "unauthorized" reply (which would confirm the bot exists).

**Commands:**
| Command | Behavior |
|---|---|
| `/start` | Welcome message + usage instructions |
| `/help` | Same as /start |
| `/queue` | Shows current queue + your jobs |
| `/cancel <job_id>` | Cancels a queued or in-progress job |
| Any YouTube URL | Submits as new job |

**The conversation flow:**
```
User: https://youtube.com/watch?v=abc123

Bot:  ✓ Got it. Checking video...
      [bot edits the message as state changes]

Bot:  ✓ "The 3-Hour Rogan Episode" (3h 12m)
      ⚠ Long video — will take ~25 min
      Position in queue: 1
      [Cancel] button

Bot:  [edits message every 30s]
      🔄 Downloading... 45%
      🔄 Transcribing audio (this is the slow part)...
      🔄 Found 3 clip candidates
      🔄 Editing clip 1 of 3
      🔄 Adding captions to clip 2 of 3
      ✓ Done! Sending clips...

Bot:  [sends each clip as separate video message]
      Clip 1/3 (Score: 0.78)
      "Here's the thing nobody tells you about..."
      [Post to TikTok] [Post to IG] [Post to YT] [Discard]

      Clip 2/3 (Score: 0.71) ...
      Clip 3/3 (Score: 0.65) ...
```

**Progress updates use `edit_message`, not new messages.** This avoids spam and keeps the chat clean.

**Critical edge case handling:**
- URL is not YouTube → "❌ Only YouTube links are supported"
- Video is age-restricted → "❌ This video requires authentication"
- Video is over 4 hours → "⚠ Video is 4h+ — confirm to proceed? [Yes] [No]"
- Video is a live stream → "❌ Live streams not supported, wait until it's archived"
- Bot was offline when message sent → on restart, scan unprocessed messages and reply "Sorry, I was offline — please resend"

---

## 7. The Web UI

A single-page Flask app on `http://localhost:5000` (M4 hostname accessible from local network if desired).

**Pages:**
| Route | Function |
|---|---|
| `/` | Form: paste YT URL, options checkboxes (music on/off, max clips), Submit |
| `/job/<id>` | Live progress view (SSE polling for state updates) |
| `/clips/<job_id>` | Grid of finished clips with inline video players + actions |
| `/queue` | Current queue + history |
| `/clip/<id>/download` | Direct file download |
| `/clip/<id>/post/<platform>` | Trigger optional posting |

**Live progress via Server-Sent Events:**
```python
@app.route('/job/<job_id>/stream')
def stream_progress(job_id):
    def event_stream():
        last_state = None
        while True:
            state = get_job_state(job_id)
            if state != last_state:
                yield f"data: {json.dumps(state)}\n\n"
                last_state = state
            if state['state'] in ('complete', 'failed', 'cancelled'):
                break
            time.sleep(2)
    return Response(event_stream(), mimetype='text/event-stream')
```

SSE is simpler than WebSockets, fully Pythonic, and good enough for once-per-2-second updates.

---

## 8. The Worker (job_runner.py)

The heart of v3. Replaces the old orchestrator entirely.

```python
class JobRunner:
    """Single-process worker that drains the job queue."""

    def __init__(self):
        # Load Whisper model ONCE at startup (~3GB RAM)
        # Keep it in memory between jobs to avoid cold-start cost
        self.whisper = mlx_whisper.load_model('medium')

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
                alerts.send_critical(f"Job {job.id} failed: {e}")

    def process(self, job):
        """Run a job through all stages, with state checkpoints."""
        stages = [
            ('downloading',   self.download),
            ('transcribing',  self.transcribe),
            ('detecting',     self.detect_clips),
            ('editing',       self.edit),
            ('captioning',    self.add_captions),
            ('formatting',    self.format_clips),
        ]

        # Resume support: skip stages already completed
        start_idx = self.get_resume_index(job)

        for i, (stage_name, fn) in enumerate(stages[start_idx:], start=start_idx):
            state.update_job_stage(job.id, stage_name)
            self.send_progress_update(job, stage_name)
            fn(job)
            state.mark_stage_complete(job.id, stage_name)

        state.mark_job_complete(job.id)
        self.deliver(job)
```

**Why a single worker?** The M4 base has 16GB RAM. Whisper (3GB) + concurrent FFmpeg (2GB+ per job) + Python overhead means running 2 jobs in parallel will swap and thermal throttle. A queue with one worker is simpler, predictable, and uses the M4's hardware encoder fully.

---

## 9. Niche-Agnostic Clip Detector

Since the user supplies arbitrary links, we can't tune the scorer per niche. The v3 scorer uses a universal signal set that works across content types:

**5 signals (same weights as v2):**
| Signal | Weight | Why universal |
|---|---|---|
| Sentiment magnitude | 0.30 | Strong emotion is engaging regardless of topic |
| Speech pace | 0.20 | Energy spikes across all content types |
| Question density | 0.15 | Questions hook viewers universally |
| Generic keyword hits | 0.20 | "The truth is", "nobody talks about", "here's why" |
| Laughter/reaction markers | 0.15 | Whisper tags `[laughter]`, `[applause]` |

The keyword list shrinks to ~15 universally-engaging phrases instead of niche-specific. Future v4 work: auto-detect content type from the source channel and apply niche tuning.

---

## 10. Failure Handling Matrix (Updated)

| Failure type | Retry | On final failure | User notification |
|---|---|---|---|
| Invalid URL | 0 | Reject immediately | Telegram reply: "❌ Not a valid YouTube URL" |
| Video private/age-gated | 0 | Reject in pre-flight | Telegram reply: "❌ This video is not publicly accessible" |
| Live stream | 0 | Reject in pre-flight | Telegram reply: "❌ Live streams not supported" |
| Download fails mid-stream | 2x with backoff | Mark `failed`, keep error | Telegram reply: "❌ Download failed: <reason>" |
| Transcription fails | 1x | Mark `failed` | Telegram reply: "❌ Transcription failed — try a shorter clip" |
| FFmpeg crash (crop/caption) | 1x with fallback (center-crop) | Skip just that clip, others continue | Reply when complete: "⚠ Clip 2 failed but 1 and 3 are ready" |
| Disk full | 0 | Halt worker, alert | Telegram alert: "🚨 Mac mini disk full — fix before resuming" |
| Whisper OOM | 0 | Halt worker | Telegram alert: "🚨 Out of memory — restart needed" |
| Telegram send fails | 3x with backoff | Fall back to web UI delivery | Email/log alert |
| Platform post fails | 3x exponential backoff | Keep clip, alert | "❌ TikTok posting failed: <reason>" |
| Worker process crashes | launchd auto-restart | Resume in-flight job from last_stage_completed | Telegram: "⚙ Worker restarted, resuming your job" |

---

## 11. Bugs and Edge Cases Specifically Designed Around

Each of these is a real risk this redesign introduces — listed with the mitigation built into the spec:

| Bug | Mitigation |
|---|---|
| User pastes non-YT link | `url_validator.py` regex-checks YT URLs before queueing |
| Video age-restricted or geo-blocked | Pre-flight `extract_info(download=False)` catches this in <2s before committing to download |
| Video is a 6-hour stream | Pre-flight checks duration, asks user to confirm if >3 hours |
| User sends 3 links rapidly | Each becomes its own job, queued FIFO. Bot replies with queue position |
| Web UI + Telegram both submit at same instant | SQLite handles atomic writes; jobs get unique UUIDs |
| Worker crashes mid-job | State machine + `last_stage_completed` allows precise resume |
| Whisper model takes 30s to load each job | Loaded once at worker startup, kept in RAM |
| Source video already 9:16 (vertical) | `editor.py` runs `ffprobe` first; if input ≈ 9:16, skip smart crop, just rescale |
| No face detected anywhere | Fall back to center crop, log warning, continue |
| Disk fills up with raw downloads | After each job completes successfully, raw download is deleted. Worker also checks free space at startup, halts if < 5GB |
| User chats during a 25-min job | Bot continues to respond to `/queue` and `/cancel` — these go through the bot, not the worker |
| Telegram clip file > 50MB | All clips are <60s @ 1080p ≈ 8-12MB. Hard cap in formatter to ensure this |
| User loses Telegram bot token | Token in `.env` only, never committed. Quick rotation supported |
| Bot username discovered by stranger | `ALLOWED_USER_IDS` whitelist silently ignores all others |
| Non-English video | Whisper auto-detects language; scorer works on word counts but keyword hits drop to 0. Quality may degrade — log warning |
| User wants to retry a failed job | `/retry <job_id>` command queues a fresh attempt |
| Worker restarts mid-Telegram-delivery | Delivery is the last step after `complete` — if it fails, clips stay in DB, user can re-trigger via `/sendclips <job_id>` |

---

## 12. Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| Bot framework | python-telegram-bot (long-polling mode) |
| Web framework | Flask + Jinja2 |
| Live updates (web) | Server-Sent Events (SSE) |
| Video download | yt-dlp (Python API) |
| Transcription | mlx-whisper (medium model) |
| Video processing | ffmpeg-python (wrapper around FFmpeg + VideoToolbox) |
| Face detection | OpenCV Haar cascade (fast, lightweight) |
| State | SQLite via stdlib `sqlite3` |
| Testing | pytest |
| Process supervision | launchd (Mac native) |
| Logging | Python `logging` + rotating file handler |
| Backup | Time Machine + GitHub for DB/config |

---

## 13. Hardware (unchanged from v2)

- Mac mini M4 base — $599
- USB-C flash drive 512GB — ~$35
- External drive for Time Machine — ~$60-80
- Total: ~$700-715

---

## 14. Settings Config (settings.yaml)

```yaml
general:
  scratch_dir: /Volumes/ClipFarmerScratch
  max_clip_length_s: 60
  min_clip_score: 0.45
  max_clips_per_job: 3
  add_music_by_default: true
  max_video_duration_warning_s: 10800   # 3 hours
  max_video_duration_hard_limit_s: 21600 # 6 hours - reject above this
  min_free_disk_gb_to_start: 5

worker:
  whisper_model: medium
  ffmpeg_concurrent_jobs: 1               # single worker, one job
  job_poll_interval_s: 5

telegram:
  bot_token_env_var: TELEGRAM_BOT_TOKEN
  allowed_user_ids_env_var: ALLOWED_TELEGRAM_USER_IDS
  progress_update_interval_s: 30

web_ui:
  port: 5000
  host: 127.0.0.1
  enable_lan_access: false
```

---

## 15. Build Phases

**Phase A — Scaffold (one pass):**
- [ ] Full directory structure from Section 3
- [ ] Empty `__init__.py`, stub functions with docstrings
- [ ] `requirements.txt` with all pinned deps
- [ ] `.gitignore`, `pytest.ini`, `.env.example`
- [ ] Initial commit: "Project scaffold"

**Phase B — Fill in modules in this order:**
1. `core/state.py` (DB operations) + tests
2. `core/job_states.py` (state machine) + tests
3. `core/url_validator.py` (URL parsing + preflight) + tests
4. `processing/downloader.py` + tests
5. `processing/transcriber.py` + tests
6. `processing/clip_detector.py` + tests
7. `processing/editor.py` + tests
8. `processing/caption_burner.py` + tests
9. `processing/formatter.py` + tests
10. `core/job_runner.py` (wires processing/* together) + tests
11. `delivery/telegram_delivery.py` + tests
12. `interfaces/telegram_bot.py` (mocked job_runner first) + tests
13. `delivery/web_delivery.py` + tests
14. `interfaces/web_ui.py` + tests
15. `main.py` (launches bot + worker + web UI concurrently)
16. `alerts/telegram_alerts.py`
17. `posting/youtube.py` (easiest API)
18. `posting/tiktok.py`
19. `posting/instagram.py`

Each module: build → pytest → manual sanity check → commit + push → next module.

---

## 16. Definition of "v3 done"

- [ ] Scaffold complete, all modules stubbed
- [ ] All Layer 3 processing modules built + tests passing
- [ ] End-to-end flow works from web UI: paste link → see 3 finished clips
- [ ] End-to-end flow works from Telegram: send link → receive 3 clip videos
- [ ] State machine survives a worker kill mid-job (test by `kill -9` during processing)
- [ ] Authentication whitelist confirmed working
- [ ] Pre-flight rejection working for all bad URL cases
- [ ] Progress updates editing in place (not spamming new messages)
- [ ] Optional posting to YouTube Shorts works end-to-end
- [ ] launchd plist running the system on boot
- [ ] Daily DB backup to GitHub running

v3 done = you can paste any YouTube link into Telegram from anywhere, get 3 polished clips delivered to your phone within 5-30 minutes, and tap a button to post any of them to YouTube Shorts. Everything beyond that (TikTok/IG posting, multi-niche tuning, etc.) is v4.
