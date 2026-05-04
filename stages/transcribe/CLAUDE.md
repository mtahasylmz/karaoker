# stages/transcribe

Produces segment-level text with start/end timings **plus `vocal_activity`
regions** marking where the singer is silent (instrumental breaks). The
latter drives karaoke UI during gaps so no stale word highlight sits frozen
on screen.

Backend is routed per-language by `shared.flow_for(language)` (mirrors
`packages/contracts/src/flows.ts`):

- **Qwen3-ASR-1.7B** on the original mix (`source_uri`) for languages in
  its supported set — Apache 2.0, trained on singing/music incl. Turkish.
  `source: "qwen3"` in the response.
- **faster-whisper** on the vocals stem (`vocals_uri`) for everything else,
  and as today's fallback until the Qwen3 backend is wired
  (`_QWEN3_AVAILABLE` in `pipeline.py`). `source: "whisper"`.

Either way `vocal_activity` comes from RMS envelope on the **vocals stem**
(`vad.detect`). The ASR model may eat the mix, but VAD always runs on the
isolated stem — absence of energy there is ground truth for instrumental
breaks.

If `known_lyrics` is supplied it becomes the ASR's `initial_prompt` /
bias text (first 200 chars — more biases the model). Worth it for Turkish
and rare vocabulary; it's a prompt, not a transcription replacement.

## Contract

- Request:  `packages/contracts/json-schema/transcribe_request.json`
- Response: `packages/contracts/json-schema/transcribe_response.json`

`vocal_activity: [{start, end, kind: "vocals"|"instrumental"}]` is an
ordered, non-overlapping array covering `[0, audio_duration]`. Consumers
treat any unreported time as "unknown."

## Model

### Target (in progress): Qwen3-ASR-1.7B

- Apache 2.0. Released 2026-01-29 by Alibaba. Supports 52 languages incl.
  Turkish. Marketed as "speech/music/**song** recognition" — only open
  release that explicitly trains on singing.
- Serves via vLLM in prod (day-0 support), `transformers` for local dev.
- Deploy target: **Cloud Run + NVIDIA L4 GPU** (GA, scales to zero,
  ~$0.67/hr). Regions: `europe-west1`/`europe-west4` for low TR latency.
- Integration is Phase D2b (open). CLAUDE.md will drop the Whisper sections
  once the swap lands.

### Current (legacy): faster-whisper

Still in `pipeline.py` as the fallback until Qwen3 swap lands. `small` is
being dropped as default — we'll only ship `large-v3-turbo` if we keep
Whisper at all.

`faster-whisper==1.0.3` pinned (see top-level `CLAUDE.md` scar #4).

### Rejected alternatives

- ElevenLabs Scribe / Deepgram / AudioShake — closed-source, not our
  direction.
- NVIDIA Parakeet / Canary-Qwen — no Turkish support.
- Mistral Voxtral — no Turkish in the 13 core languages.

### Env knobs (Qwen3 + device)

- `TRANSCRIBE_DEVICE` — override the device picker (`cuda:0` / `mps` /
  `cpu`). Default: cuda > mps > cpu, auto-detected at load time.
- `QWEN3_MODEL` — HF repo ID (default `Qwen/Qwen3-ASR-1.7B`).
- `QWEN3_MAX_NEW_TOKENS` — passed to `Qwen3ASRModel.from_pretrained`
  (default `512`). Raise for very long audio per the Qwen README.

The Qwen3 forced aligner lives in `stages/align`; this stage emits a
single coarse segment spanning the audio and leaves per-word timing to
align. `known_lyrics` biasing is a no-op on the qwen3 path (context
biasing is DashScope-cloud-only, not in the local `qwen-asr` package) —
one `info` log fires per request when `known_lyrics` is set while
routing to qwen3.

## VAD (`vad.py`)

Short-time RMS envelope on the vocals stem + hysteresis thresholding.
Thresholds tuned for Demucs output: on-threshold −40 dBFS, off-threshold
−46 dBFS, 250 ms smoothing, min 0.3 s vocal region, min 1.5 s instrumental
region. The 2506.15514 ALT paper showed this same trick beats Whisper's
native 30 s windowing for long-form segmentation. Purely signal-processing,
no ML deps beyond numpy + soundfile.

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
  -d '{"job_id":"abc123def456","vocals_uri":"gs://whatever/stages/separate/abc123def456/vocals.wav","source_uri":"gs://whatever/uploads/abc123def456.mp4","language":"en"}' | jq
```

## Bench

`bench/README.md` compares the candidate ASR models on a per-language WER +
end-to-end alignment success on Turkish + English fixtures from
`sample-music/`. Primary comparison is Qwen3-ASR-1.7B vs faster-whisper
`large-v3-turbo`; `alt-eval` from the Jam-ALT authors provides the
readability-aware WER metric. Keep fixtures under 60 s so bench runs stay
interactive.
