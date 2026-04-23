# annemusic

Family-scale karaoke-video generator. User uploads a music video; the system returns a manifest pointing at (a) the original video, (b) a vocals-removed instrumental track, and (c) an ASS subtitle file that the browser overlays in real time with per-word fill animation via JASSUB. A "record-along" path lets users sing with the video and mixes their recording back over the instrumental.

**Status:** MVP (monolithic) shipped 2026-04-23 on Cloud Run. Currently restructuring into stages with explicit contracts. Full plan: `/Users/mtahasylmz/.claude/plans/glowing-gathering-ocean.md`.

## Architecture

```
Browser (apps/web, React + JASSUB renderer)
   │ POST /uploads → V4 signed PUT URL → PUT to GCS
   │ POST /jobs → trigger workflow
   ▼
apps/api (Cloud Run, TS)  ──── triggers ────►  apps/orchestrator (Cloud Run, TS)
                                                   │
                                               @upstash/workflow
                                                   │  context.call() (one per stage,
                                                   │   up to 12h, retries via QStash)
                                                   ▼
                          ┌───────────────────────────┬───────────────────────┐
                          │                           │                       │
                  stages/separate           stages/transcribe          stages/align
                  (Python, demucs           (Python, Whisper           (Python, whisperx
                   or BS-RoFormer)           + LRCLIB lookup)           wav2vec2)
                          │                           │                       │
                          └──────────┬────────────────┴───────────────────────┘
                                     ▼
                               stages/compose  (TS — ASS + manifest JSON, no re-encode)
                                     │
                                     ▼
                          GCS: manifest.json, lyrics.ass, vocals.wav, instrumental.wav
                                     │
                                     ▼
                  Browser reads manifest; JASSUB overlays ASS on <video>
                  whose audio is the instrumental. Optional record-along:
                    MediaRecorder → GCS → POST /record-mix
                    → stages/record-mix (Python, ffmpeg + RubberBand)
```

**Why this shape:** contracts-first means each stage is replaceable without touching the others. `context.call()` in Upstash Workflow holds HTTP calls on their infra for up to 12 h with retries — stages stay stateless request/response servers with no long-poll quirks. Same orchestrator code runs against `@upstash/qstash-cli dev` locally or prod QStash.

## Repo layout

```
packages/
  contracts/         Zod schemas + generated JSON Schema. Single source of truth.
  shared-ts/         (phase B) logger, upstash stream client, env helpers.
  shared-py/         (phase B) same, Python.
stages/
  separate/          (phase D1) vocals/instrumental split. Python.
  transcribe/        (phase D2) text + segment timings. Python. LRCLIB lookup.
  align/             (phase D3) segment → per-word timings. Python.
  compose/           (phase D4) words + URIs → .ass + manifest. TS.
  record-mix/        (phase D5) user recording + instrumental → mix. Python.
apps/
  orchestrator/      (phase C) @upstash/workflow, one workflow per pipeline run.
  api/               (phase C) signed PUT URLs, /jobs, /users — TS.
  web/               (phase C) React + JASSUB frontend.
tools/
  logs/              (phase B) pnpm logs CLI — tails Upstash Redis Streams.
infra/
  setup.sh           one-time GCP bootstrap (kept from MVP).
  deploy-stage.sh    (phase E) parametric deploy of any stage or app.
  bucket-cors.json   browser-PUT CORS config for the bucket.
  wipe.sh            nuke all dev state (Redis + GCS).
  env.example        every env var each service reads.
```

## Current phase status

- ✅ **Phase A** — Clean slate, monorepo scaffold, `packages/contracts` with 24 Zod schemas exported as JSON Schema.
- 🔜 **Phase B** — Shared logger + Redis-Streams log tail CLI.
- **Phase C** — Orchestrator + API + web skeleton, every stage returns stubs. End-to-end wiring proven without ML.
- **Phase D** — Port ML logic per stage (separate / transcribe / align / compose / record-mix). Each stage is safe to work on in its own agent session once C is live.
- **Phase E** — Deploy to Cloud Run.

## Key project scars (do not re-learn)

1. **YouTube blocks GCP ASN.** `yt-dlp` works from residential IPs but not Cloud Run / Compute Engine (AS15169). The project pivoted from "paste a URL" to "upload a file." Don't reintroduce server-side YouTube fetching without a residential proxy or PO-Token sidecar.
2. **Cloud Run secret mounts are read-only.** Anything that needs to write-back to a mounted secret (e.g. session cookie refresh) silently fails. If a future stage needs mutable secrets, copy to `/tmp` at job start.
3. **Cloud Build caches nothing by default.** `infra/cloudbuild.yaml` (coming in Phase E) reuses the MVP's `--cache-from :latest` pattern; code-only rebuilds are ~2 min, not 10.
4. **whisperx 3.1.6's VAD model URL (S3) is dead.** `whisperx.load_model()` therefore must not be called. The MVP (and the `transcribe` stage plan) uses `faster_whisper.WhisperModel` for transcription and only `whisperx.align()` for the wav2vec2 forced-alignment step. If upgrading whisperx past 3.3, re-verify this.
5. **`faster-whisper==1.0.3`** is pinned because whisperx 3.1.6 doesn't pass `hotwords` to `TranscriptionOptions`. If bumping faster-whisper, either bump whisperx together or inject `asr_options={"hotwords": None}`.
6. **New GCP projects don't auto-grant Cloud Build the builder role on the Compute Engine default SA.** `infra/setup.sh` handles this; without it, `gcloud builds submit` 403s.
7. **GCS signed PUT URLs inherit the signer's IAM.** The API SA needs `storage.objects.create` on the bucket AND `iam.serviceAccounts.tokenCreator` on itself (for `signBlob` from Cloud Run's metadata-server credentials).
8. **Bucket CORS is required** for browser PUT (`infra/bucket-cors.json`).
9. **Local testing beats cloud rebuilds.** Every stage will have a `bench/` directory with fixtures; reproduce locally before redeploying. The MVP lost several hours iterating in Cloud Run on issues a 30-second local run would have surfaced.

## Day-to-day

```bash
# Install + build contracts
pnpm install
pnpm contracts:build

# Tail logs (once phase B lands)
pnpm logs --stage "*" --follow

# Bring everything up locally (fills in as phases land)
pnpm dev

# Wipe dev state (Redis + GCS)
bash infra/wipe.sh --yes
```

## Contracts are the spine

Every stage validates both request and response against its schema. If you're about to invent a new field, add it to `packages/contracts/src/<stage>.ts` first, then `pnpm contracts:build` to regenerate the JSON Schema the Python stages consume. Never hand-edit `packages/contracts/json-schema/*.json`.
