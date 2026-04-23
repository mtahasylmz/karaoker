# stages/record-mix

Post-pipeline step. Takes a user recording (webm/opus from browser
`MediaRecorder`) plus the instrumental from `stages/separate` and returns a
merged mp3. Optional pitch correction is in the contract; v1 is only
`autotune="off"` + `gain_db` adjustment.

Not in the main workflow — the API hits this stage directly after the user
finishes recording ("sing-along").

## Contract

- Request:  `packages/contracts/json-schema/record_mix_request.json`
- Response: `packages/contracts/json-schema/record_mix_response.json`

## Autotune modes

- `off` (v1) — straight `amix` of recording + instrumental, no DSP.
- `smooth` (v1: pass-through with warn; v2: RubberBand gentle pitch
  smoothing, no scale detection).
- `snap` (not implemented) — scale-detecting snap-to-nearest-note. Out of
  scope per plan. Returns 500.

## Local dev

```bash
uv sync --all-packages

export DEV_FS_ROOT=/tmp/annemusic-dev
export GCS_BUCKET=whatever
uv run --package annemusic-stage-record-mix python -m record_mix.main
# listens on :8105

curl -sS -X POST http://127.0.0.1:8105/process \
  -H 'content-type: application/json' \
  -d '{"job_id":"abc123def456","recording_uri":"gs://whatever/recordings/demo.webm","instrumental_uri":"gs://whatever/stages/separate/abc123def456/no_vocals.wav","autotune":"off","gain_db":0}' | jq
```

## v2 notes

- **RubberBand for smooth**: the `rubberband-cli` binary can shift pitch
  without changing tempo. For vocal smoothing, a light formant-preserving
  shift (< 5 cents) produces the "autotune-lite" feel. CLI install via
  `brew install rubberband` / `apt install rubberband-cli`.
- **Scale detection for snap**: `librosa.pyin` for per-frame F0, cluster
  into target scale (C major by default, detectable from instrumental),
  round each F0 to nearest scale degree, re-synth via PSOLA. Significant
  work; parked in `NOTES.md`.
