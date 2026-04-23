# stages/compose

Word timings + media URIs → `.ass` subtitles + `manifest.json`. No video
re-encode. The browser plays the original video (`<video src=video_url>`)
with the `.ass` overlaid via JASSUB and the audio swapped for the
instrumental.

## Contract

- Request:  `packages/contracts/json-schema/compose_request.json`
- Response: `packages/contracts/json-schema/compose_response.json`
- Manifest: `packages/contracts/json-schema/playback_manifest.json`

`style` fields are all optional; sensible defaults (yellow sung / white
unsung, Arial 72, bottom-center, 8 words/line, 1.5 s gap) live in
`src/ass.ts:DEFAULT_STYLE`.

## Deterministic, no heavy deps

Pure TS. Ported from MVP `worker/ass_builder.py`. If you change line
grouping / fill math, add a regression test against the MVP's
`tests/test_ass_builder.py` expectations.

## Local dev

```bash
pnpm --filter @annemusic/stage-compose install
pnpm --filter @annemusic/stage-compose dev
# listens on :8104

curl -sS -X POST http://127.0.0.1:8104/process \
  -H 'content-type: application/json' \
  -d @- <<'JSON' | jq
{
  "job_id": "abc123def456",
  "words": [
    {"text":"never","start":25.1,"end":25.4},
    {"text":"gonna","start":25.4,"end":25.7},
    {"text":"give","start":25.7,"end":25.9},
    {"text":"you","start":25.9,"end":26.1},
    {"text":"up","start":26.1,"end":26.5}
  ],
  "video_uri": "gs://whatever/uploads/abc123def456.mp4",
  "instrumental_uri": "gs://whatever/stages/separate/abc123def456/no_vocals.wav",
  "language": "en"
}
JSON
```

In dev (`DEV_FS_ROOT` set), both `.ass` and `manifest.json` land under
`$DEV_FS_ROOT/stages/compose/<job_id>/`.
