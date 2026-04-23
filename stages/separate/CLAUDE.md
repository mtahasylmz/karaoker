# stages/separate

Vocal / instrumental source separation. Input: a source video/audio file.
Output: `vocals.wav` + `no_vocals.wav` plus the sample rate and model name.

## Contract

- Request:  `packages/contracts/json-schema/separate_request.json`
- Response: `packages/contracts/json-schema/separate_response.json`

Fields that matter most: `source_uri` (gs:// or file:// in dev), optional
`model` override. Result URIs are `gs://{bucket}/stages/separate/{job_id}/{vocals,no_vocals}.wav`.

## Model

Default: **`mel_band_roformer_kim`** (Mel-Band RoFormer, Kimberley Jensen
checkpoint) via `audio-separator`. Switched from htdemucs on 2026-04-23
after the bench showed ~+2.6 dB vocals SDR and ~+2.6 dB instrumental SDR
for lower wall-clock on M4. See `bench/RESEARCH.md` and
`bench/results/summary.md`.

Set via env `SEPARATE_MODEL` or the request's `model` field. Recognized:

| model slug | route | notes |
|---|---|---|
| `mel_band_roformer_kim` | audio-separator | primary, SOTA quality |
| `bs_roformer_ep317` | audio-separator | close 2nd on quality, ~2× slower |
| `htdemucs` | demucs CLI subprocess | documented fallback |
| `htdemucs_ft` | demucs CLI subprocess | 4× slower than htdemucs, marginal gain |

Unknown slugs raise at pipeline entry. The `audio-separator` package
handles both RoFormer variants; htdemucs family still shells out to
`python -m demucs --two-stems=vocals` to keep the proven MVP path intact.

## Bench

See `bench/` for the SOTA memo (`RESEARCH.md`), harness (`run_bench.py`,
`fetch_fixtures.py`), and current results (`results/summary.md`).

## Local dev

This stage has its **own uv lockfile** (`stages/separate/uv.lock`) rather
than being a workspace member — `audio-separator>=0.44` requires numpy>=2,
while `stages/transcribe` and `stages/align` are pinned to numpy<2 via
whisperx 3.1.6 / faster-whisper 1.0.3. One workspace lock can't hold both.

```bash
# One-time: install deps into stages/separate/.venv
cd stages/separate
uv sync

# Run against local fs (no GCS needed):
export DEV_FS_ROOT=/tmp/annemusic-dev
export GCS_BUCKET=whatever         # only used to format uri strings
mkdir -p $DEV_FS_ROOT/uploads
cp /path/to/some-clip.mp4 $DEV_FS_ROOT/uploads/demo.mp4

# Default mel_band_roformer_kim:
.venv/bin/python -m separate.main

# Fire a request (job_id must match ^[a-f0-9]{12,32}$):
curl -sS -X POST http://127.0.0.1:8101/process \
  -H 'content-type: application/json' \
  -d '{"job_id":"a1b2c3d4e5f6","source_uri":"gs://mock/uploads/demo.mp4"}' | jq

# Fallback path:
SEPARATE_MODEL=htdemucs .venv/bin/python -m separate.main
```

First run per model downloads weights to `/tmp/audio-separator-models/`
(RoFormer: ~900 MB for BS, ~913 MB for Mel) or `~/.cache/torch/hub`
(htdemucs: ~320 MB). Subsequent runs hit the cache.

## Docker (prod)

`docker build` will bake the RoFormer checkpoint into the image so Cloud
Run cold starts don't re-download ~900 MB. Mirror the UVR-community
weights to our own GCS bucket to insulate from their host churn (see
RESEARCH.md §4 "Weight hosting"). `infra/deploy-stage.sh stages/separate`
in Phase E.
