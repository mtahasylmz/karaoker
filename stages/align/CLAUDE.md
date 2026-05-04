# stages/align

Segment-level transcription → word-level timings. Takes the output of
`stages/transcribe` plus the vocals wav and runs either
Qwen3-ForcedAligner-0.6B (Apache-2.0, 11 languages, GPU only) or
whisperx's wav2vec2 forced alignment. Backend per job is picked by
`shared.flows.flow_for(language).align`; see *Backends & routing* below.

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

## Backends & routing

`shared.flow_for(language).align` picks between `"qwen3"` (for the 11 Qwen
languages: en, zh, yue, fr, de, it, ja, ko, pt, ru, es) and `"whisperx"`
(everything else). The routing table is mirrored in
`packages/contracts/src/flows.ts` and `packages/shared-py/shared/flows.py`;
they must stay in sync by hand.

`_resolve_backend` gates qwen3 on both `torch.cuda.is_available()` AND
`qwen-asr` being importable. If either is missing the job logs
`backend_downgrade` and runs whisperx for the whole job — the stage still
returns a valid response, just without Qwen's gains.

Long songs are split into ≤300 s windows by `plan_chunks`, preferring to
cut at instrumental regions from `vocal_activity` (so chunks never
bisect a phrase). Qwen3's 5-minute-per-call cap is the hard constraint;
whisperx is chunk-friendly for free.

Per-chunk failures cascade:
`qwen3` → `whisperx` (for that chunk only) → `even-split`. Ops can see
what actually ran by reading `diagnostics.{backend, chunk_count,
cuda_available, qwen3_fallback_chunks, whisperx_fallback_chunks}` on the
response.

`qwen-asr[vllm]` is NOT in `pyproject.toml` because its `torchaudio==2.9.1`
pin conflicts with the whisperx 3.1.6 / torch 2.2.2 base. The GPU Cloud
Run Dockerfile installs it in a second pip step on top of the base image.

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
  ],
  "vocal_activity": [
    {"start":0.0,"end":25.1,"kind":"instrumental"}
  ]
}
JSON
```

First run downloads the language-specific wav2vec2 (~360 MB for English).
Cached under `~/.cache/torch/hub` afterwards. On a GPU host with
`qwen-asr[vllm]` installed and `language` in the Qwen-aligned set, the
first run also pulls Qwen3-ForcedAligner-0.6B (~1.2 GB).

## Bench

See `bench/README.md`. Drop `.wav`/`.txt`/`.json` fixture triples under
`bench/fixtures/<lang>/`; the runner prints per-fixture word-onset MAE
(median + mean + count of per-word errors >500 ms) for either backend.
`fixtures/` is gitignored so audio never lands in the repo.

Excluded by license: ctc-forced-aligner (MMS CC-BY-NC), CrisperWhisper
(CC-BY-NC), SOFA, MFA pretrained models. Qwen3-ForcedAligner and the
whisperx wav2vec2 defaults are Apache-2.0.
