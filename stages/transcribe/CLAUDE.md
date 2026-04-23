# stages/transcribe

Takes the vocals stem from `stages/separate` (plus optional `title`/`artist`/
`known_lyrics` hints) and returns segment-level text with start/end. Two
sources, one contract:

- **LRCLIB fast path** — if `title` + `artist` resolve to a synced LRC on
  [lrclib.net](https://lrclib.net), parse the `[mm:ss.xx]` lines into
  segments and skip Whisper entirely. Quality ceiling: perfect lyrics +
  human timings (crowdsourced). `source: "lrclib"` in the response.
- **Whisper fallback** — `faster-whisper` with built-in Silero VAD. The MVP
  proved `whisperx.load_model` is unusable (dead S3 VAD URL); only
  `whisperx.align()` is called, later, in `stages/align`. `source: "whisper"`.

If `known_lyrics` is supplied it becomes Whisper's `initial_prompt` (first
200 chars — more biases the model). Worth it for Turkish / rare languages.

## Contract

- Request:  `packages/contracts/json-schema/transcribe_request.json`
- Response: `packages/contracts/json-schema/transcribe_response.json`

## Model

Default: `WHISPER_MODEL=small` (env var). Bump to `medium` / `large-v3` for
better accuracy on non-English (especially Turkish) at ~2–4× wall time.
`WHISPER_COMPUTE_TYPE` controls ctranslate2 quantization (`int8`, `int8_float16`,
`float16`). Default `int8` for CPU.

`faster-whisper==1.0.3` pinned. Later versions added required fields to
`TranscriptionOptions` that whisperx 3.1.6 doesn't pass — if this stage ever
imports whisperx (it currently doesn't), the MVP's `asr_options={"hotwords": None}`
shim applies. See the top-level `CLAUDE.md` scar #4.

## Local dev

```bash
uv sync --package annemusic-stage-transcribe

export DEV_FS_ROOT=/tmp/annemusic-dev
export GCS_BUCKET=whatever
uv run --package annemusic-stage-transcribe python -m transcribe.main
# listens on :8102

# Reuse the vocals.wav produced by the separate stage in the same dev job:
curl -sS -X POST http://127.0.0.1:8102/process \
  -H 'content-type: application/json' \
  -d '{"job_id":"abc123def456","vocals_uri":"gs://whatever/stages/separate/abc123def456/vocals.wav","title":"Never Gonna Give You Up","artist":"Rick Astley"}' | jq
```

## Bench

`bench/README.md` compares Whisper variants (small / medium / large-v3) and
cheap alternatives (NVIDIA Canary-1B, Parakeet-TDT) on a per-language WER.
Keep fixtures under 60 s so bench runs stay interactive.
