# stages/align

Segment-level transcription → word-level timings. Takes the output of
`stages/transcribe` plus the vocals wav and runs wav2vec2 forced alignment
via `whisperx.align()`.

## Contract

- Request:  `packages/contracts/json-schema/align_request.json`
- Response: `packages/contracts/json-schema/align_response.json`

## Model

Per-language wav2vec2 checkpoints auto-selected by whisperx. English →
`facebook/wav2vec2-base-960h`. Turkish → `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`.
Full mapping in `whisperx.alignment.DEFAULT_ALIGN_MODELS_HF` / `_TORCH`.

If the language has no align model, the stage logs a warning and falls back
to evenly splitting each segment's tokens across its duration — downstream
stages still get something usable.

## `vocal_activity` pass-through

Transcribe emits `vocal_activity: [{start, end, kind}]` on every response
(contract-required), and the orchestrator forwards it here. This stage
does not consume the signal (wav2vec2 only needs audio + text) — it
forwards the array verbatim into its response so `stages/compose` can
render instrumental-break UI (placeholders, countdowns) without
re-downloading the vocals audio. Required on both request and response;
no synthesis on either side.

## Local dev

```bash
uv sync --all-packages

export DEV_FS_ROOT=/tmp/annemusic-dev
export GCS_BUCKET=whatever
uv run --package annemusic-stage-align python -m align.main
# listens on :8103

curl -sS -X POST http://127.0.0.1:8103/process \
  -H 'content-type: application/json' \
  -d @- <<'JSON' | jq
{
  "job_id": "abc123def456",
  "vocals_uri": "gs://whatever/stages/separate/abc123def456/vocals.wav",
  "language": "en",
  "segments": [
    {"text":"never gonna give you up","start":25.1,"end":27.5}
  ]
}
JSON
```

First run downloads the language-specific wav2vec2 (~360 MB for English).
Cached under `~/.cache/torch/hub` afterwards.

## Bench

`bench/README.md` compares whisperx (this default), NVIDIA NeMo Forced
Aligner, and Montreal Forced Aligner on alignment MAE across per-language
fixtures. SOTA-worthy bump: switch to NeMo if MAE wins by >30 ms on the
Turkish fixture.
