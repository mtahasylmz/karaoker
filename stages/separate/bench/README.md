# stages/separate — bench

Compare candidate vocal-separation models on real tracks with ground-truth
stems. Measures **vocals SDR**, **instrumental SDR**, and **wall-clock / RTF**
(real-time factor = wall_clock / audio_duration).

See [`RESEARCH.md`](RESEARCH.md) for the SOTA memo that picked the candidates.

## Candidates (current)

| slug | audio-separator file | notes |
|---|---|---|
| `htdemucs` | `htdemucs.yaml` | production floor |
| `htdemucs_ft` | `htdemucs_ft.yaml` | 4× slower per-stem fine-tune |
| `mel_band_roformer_kim` | `vocals_mel_band_roformer.ckpt` | Kimberley Jensen checkpoint |
| `bs_roformer_ep317` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | viperx checkpoint |

Extend `MODELS` in `run_bench.py` to add more (e.g. `scnet_xl_ihf`).

## Setup (one-time)

```bash
# From stages/separate/bench/
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install 'audio-separator[cpu]' musdb museval soundfile numpy

# Fetch the MUSDB18 7-track sample as fixtures (auto-downloads, ~50 MB).
# Writes ../fixtures/{name}/{mixture,vocals,no_vocals}.wav
python fetch_fixtures.py
```

The sample is low-res MUSDB, not MUSDB18-HQ — SDR deltas across models are
meaningful, but absolute numbers will sit below published leaderboard values.
For publication-grade numbers grab MUSDB18-HQ (~30 GB of WAV) and point
`FIXTURES_DIR` at its `test/` subset.

## Run

```bash
# All models × all fixtures, fresh run:
python run_bench.py --clean

# One model, one fixture:
python run_bench.py --models mel_band_roformer_kim --fixtures Al_James_-_Schoolboy_Facination
```

Results:
- `results/{model}/{fixture}/` — separated stems
- `results/results.json` — flat list of dataclass rows
- `results/summary.md` — ranked table, easy to paste

Incremental: the summary is re-written after every model so a crash mid-run
still leaves partial results on disk.

## SDR metric

`museval.evaluate` with `win=hop=1s` — classic cSDR (chunk-level SDR). We
median across the per-chunk values per track, then median across tracks per
model. Median is robust to the occasional silent chunk that produces an
`inf`. For a single-number uSDR (global) you can pass `win=None, hop=None`
into museval; keeping cSDR here because it's what the MUSDB leaderboard
historically reported and what `demucs` benchmarks use.

## Compute notes (local dev)

- **M-series Mac:** runs on MPS with CPU fallback. RoFormer models push
  ~6–8 GB RAM; 16 GB is tight but fine for 30 s fixtures.
- **T4 / L4:** expected 10–20× faster than M4 CPU/MPS for RoFormer.
- First run per model downloads weights (~600 MB for Mel-RoFormer, ~900 MB
  for BS-RoFormer, ~320 MB for htdemucs) into the audio-separator cache.

## When to re-bench

- Bumping a model checkpoint in `stages/separate` pipeline.
- Adding a new candidate from `RESEARCH.md` §3.
- Re-measuring on actual Cloud Run hardware before a production switch.
