# stages/record-mix — bench

Exercise the record-mix pipeline over 3–4 representative parameter combos on
a small, committed-script-not-committed-artifact fixture set. Tracks:

- **wall-clock** — time from `pipeline.run()` start to mix upload.
- **output size / duration** — mp3 bytes + `ffprobe` duration (should match
  the instrumental ±0.3 s).
- **diagnostics** — the `applied` + `skipped` dicts returned by the pipeline.

Used as the "local iteration beats cloud rebuilds" loop from scar #9 — run
this before every Cloud Run redeploy.

## Fixtures

`fetch_fixtures.py` pulls one short public-domain music clip, runs it through
Demucs once (htdemucs, two-stems=vocals) to produce matching `vocals.wav` +
`instrumental.wav`, and writes a synthetic "user recording" by mixing the
extracted vocal with a dash of added noise + a 55-ms offset (so alignment has
something to find).

Outputs under `bench/fixtures/`:
- `recording.webm`   — synthetic user take (webm/opus, matches browser MediaRecorder)
- `vocals.wav`       — demucs vocals stem (aligned reference for GCC-PHAT)
- `instrumental.wav` — demucs `no_vocals.wav`

## Setup (one-time)

```bash
# From repo root:
uv sync --package annemusic-stage-record-mix
uv run --package annemusic-stage-record-mix python stages/record-mix/bench/fetch_fixtures.py
```

First run downloads the source clip + htdemucs weights (~320 MB cached to
`~/.cache/torch/hub`).

## Run

```bash
# From repo root:
uv run --package annemusic-stage-record-mix python stages/record-mix/bench/run_bench.py
```

Writes:
- `bench/results/results.json`
- `bench/results/summary.md`

Each row is one `(combo_name, wall_s, size_kb, duration_s, diagnostics)` tuple.

## Combos

| name | knobs |
|---|---|
| `baseline` | `clean_bleed=false, reverb_wet=0, duck_db=0, presence_db=0, gain_db=0` |
| `defaults` | stage defaults (reverb_wet=0, duck_db=4, presence_db=2) |
| `heavy`    | `clean_bleed=true, reverb_wet=0.4, duck_db=10, presence_db=5` |
| `edge_gains` | `vocal_gain_db=+12, instrumental_gain_db=-12, gain_db=+12` |

Extend the `COMBOS` list in `run_bench.py` to add more.

## When to re-bench

- Touching the filter graph in `filter_chain.py`.
- Bumping the `demucs` pin in `pyproject.toml`.
- Replacing `assets/plate_ir.wav`.
