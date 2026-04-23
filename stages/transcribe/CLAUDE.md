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

### Phase D2b complete — Qwen3-ASR backend wired, `_QWEN3_AVAILABLE=True`

- Default checkpoint: `Qwen/Qwen3-ASR-Flash` (override via `QWEN_MODEL` env).
  Apache 2.0, trained on singing incl. Turkish.
- Inference lives in `src/transcribe/qwen3.py`: tries `qwen3_asr_toolkit`
  first (Alibaba's wrapper, natively handles long-form audio + context
  bias + timestamps), falls back to raw `transformers` (`AutoProcessor` +
  `AutoModelForCausalLM.generate`). The raw path emits a single segment
  covering the clip — `stages/align` produces word-level timings downstream
  either way.
- Deps live behind the `qwen3` extra in `pyproject.toml` (`uv sync
  --extra qwen3`). Kept as an extra — not base deps — because
  `faster-whisper==1.0.3` needs `numpy<2` (scar #5) and the transcribe
  stage must stay installable inside the workspace lockfile that
  `stages/align` also participates in (`torch==2.2.2` pin there forced the
  `torch>=2.2` floor, not `>=2.4`).
- Device selection via `QWEN_DEVICE=cpu|mps|cuda|auto`. Mac dev uses
  `cpu` or `mps`; prod Dockerfile uses `cuda` on Cloud Run + NVIDIA L4.
- Runtime failures (OOM, missing weights, bad API shape) degrade to
  faster-whisper on the vocals stem with a warn log — the stage never
  500s because Qwen3 is unhappy.

### Legacy / fallback: faster-whisper

Still in `pipeline.py` as (a) the primary backend for languages outside
`_QWEN_TRANSCRIBE_LANGS` in `shared.flows` and (b) the runtime fallback
when Qwen3 load/inference fails. `small` is the default size; bump to
`large-v3-turbo` if we decide to keep Whisper beyond the fallback role.

`faster-whisper==1.0.3` pinned (see top-level `CLAUDE.md` scar #4).

### Rejected alternatives

- ElevenLabs Scribe / Deepgram / AudioShake — closed-source, not our
  direction.
- NVIDIA Parakeet / Canary-Qwen — no Turkish support.
- Mistral Voxtral — no Turkish in the 13 core languages.

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
