# annemusic

YouTube → karaoke video. Paste a link, get back the same video with the vocals
removed and burned-in karaoke lyrics that fill in sync with the music.

## Architecture

```
Frontend (Firebase Hosting, static HTML)
   │ POST /jobs {youtube_url, username}
   ▼
API (Cloud Run, FastAPI)  ──enqueue──►  Cloud Tasks  ──►  Worker (Cloud Run)
   │                                                           │
   └──► Upstash Redis  ◄───── status updates ──────────────────┘
                                                               │
                                                          Upload .mp4
                                                               ▼
                                                      GCS: annemusic-videos/
```

See `.claude/plans/glowing-gathering-ocean.md` for the full design doc.

## Local dev

```bash
# 1. Create Upstash Redis DB (dashboard), copy REST URL + token into .env.
cp infra/env.example .env
# edit .env

# 2. API
cd api
pip install -r requirements.txt
export $(cat ../.env | xargs) && uvicorn main:app --reload --port 8000

# 3. Worker (separate terminal) — uses SKIP_AUTH=1 to bypass OIDC
cd worker
pip install -r requirements.txt
brew install ffmpeg   # or apt-get install ffmpeg
export $(cat ../.env | xargs) SKIP_AUTH=1 && uvicorn main:app --reload --port 8001

# 4. Wire the API to the local worker instead of Cloud Tasks
# Add to .env: WORKER_URL=http://localhost:8001, TASKS_QUEUE=  (empty)
# API detects empty TASKS_QUEUE and POSTs directly instead of enqueueing.

# 5. Smoke test
curl -X POST localhost:8000/users -H 'content-type: application/json' -d '{"username":"test"}'
curl -X POST localhost:8000/jobs  -H 'content-type: application/json' \
  -d '{"username":"test","youtube_url":"https://www.youtube.com/watch?v=<short clip>"}'
# Returned job_id — poll:
curl localhost:8000/jobs/<job_id>

# 6. Frontend (just open web/index.html in a browser; edit the API_URL at the top)
```

## Deploy to GCP

```bash
# One-time
bash infra/setup.sh

# Each release
bash infra/deploy-worker.sh
bash infra/deploy-api.sh

# Frontend
cd web && firebase deploy
```

## Known v1 caveats

- **yt-dlp + GCP IPs**: YouTube sometimes 403s datacenter IP ranges. If it happens, export cookies from a browser and mount as a Secret Manager secret (see `worker/pipeline.py` for the hook).
- **Cold start**: worker image is ~5 GB (models baked in). First request after idle adds ~15s. Set `--min-instances 1` on the worker to avoid it (costs ~$30/mo).
- **CPU-only**: a 3-min song takes ~5–10 min end to end. Flip on Cloud Run GPU (`--gpu 1 --gpu-type nvidia-l4`) to cut to ~30s; no code changes needed.
