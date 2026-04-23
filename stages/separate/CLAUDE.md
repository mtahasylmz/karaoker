# stages/separate

Vocal / instrumental source separation. Input: a source video/audio file.
Output: `vocals.wav` + `no_vocals.wav` plus the sample rate and model name.

## Contract

- Request:  `packages/contracts/json-schema/separate_request.json`
- Response: `packages/contracts/json-schema/separate_response.json`

Fields that matter most: `source_uri` (gs:// or file:// in dev), optional
`model` override. Result URIs are `gs://{bucket}/stages/separate/{job_id}/{vocals,no_vocals}.wav`.

## Model

Default: `htdemucs` (the MVP model, reliable, CPU-friendly). Set via env
`SEPARATE_MODEL` or the request's `model` field. Candidates to bench later:

- `htdemucs-ft` — fine-tuned demucs, modestly better at vocal isolation
- `bs-roformer` — SOTA on MUSDB-HQ benchmarks, larger + slower
- `mel-roformer` — close second to BS, arguably cleaner high-frequency vocals

Bench harness lives under `bench/`. Populate with 3–5 fixtures (rock, pop,
electronic, rap, a cappella-leaning) and measure SDR + wall-clock per model.

## Local dev

```bash
# One-time: create uv venv, install demucs + torch CPU + shared-py
uv sync --package annemusic-stage-separate

# Run against local fs (no GCS needed):
export DEV_FS_ROOT=/tmp/annemusic-dev
export GCS_BUCKET=whatever         # only used to format uri strings
# Put a test file where the stage expects the source:
mkdir -p $DEV_FS_ROOT/uploads
cp /path/to/some-clip.mp4 $DEV_FS_ROOT/uploads/demo.mp4

uv run --package annemusic-stage-separate python -m separate

# Fire a request:
curl -sS -X POST http://127.0.0.1:8101/process \
  -H 'content-type: application/json' \
  -d '{"job_id":"abc123def456","source_uri":"gs://mock/uploads/demo.mp4"}' | jq
```

First run downloads the demucs htdemucs weights (~320 MB) into
`~/.cache/torch/hub`. Subsequent runs hit the cache.

## Docker (prod)

`docker build` bakes the weights into the image so Cloud Run cold starts
don't re-download. `infra/deploy-stage.sh stages/separate` in Phase E.

## Bench

`bench/README.md` lists candidate models and the SDR measurement protocol.
Add MUSDB-HQ 30 s excerpts to `fixtures/` (not committed; gitignored) for
repeatable scoring.
