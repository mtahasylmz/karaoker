# transcribe/bench

End-to-end comparison of the two ASR backends on the three clips in
`sample-music/` at the repo root. The harness calls `pipeline.run()` with a
monkeypatched `download_file` that copies local files instead of hitting
GCS — so it exercises the real flow-routing + VAD + response assembly code,
not just the ASR call.

## One-time setup: stage the vocals files

Each fixture needs `bench/fixtures/<slug>.vocals.wav` on disk. Two options:

**(a) Real Demucs output (preferred).** Run `stages/separate` once on each
mp4, then copy the resulting `vocals.wav` into `bench/fixtures/` with the
name listed in `compare.py`.

**(b) ffmpeg karaoke subtraction (quick + inaccurate).** Good enough to
exercise the VAD code path but bad for WER comparisons:

```bash
ffmpeg -i "sample-music/Billie Eilish - Happier Than Ever (Official Music Video).mp4" \
  -af "pan=mono|c0=0.5*c0+-0.5*c1" -ar 16000 -ac 1 \
  stages/transcribe/bench/fixtures/billie-eilish-happier-than-ever.vocals.wav
```

## One-time setup: hand-write the ground-truth transcripts

Each fixture needs `bench/fixtures/<slug>.txt`. Start lines with `#` to add
source comments (e.g. the lyrics website used) — those lines are stripped
before the WER computation. Example:

```
# source: https://genius.com/...
We fell in love in October
That's why I love fall
...
```

## Run

```bash
uv sync --package annemusic-stage-transcribe --extra qwen3
ANNEMUSIC_SAMPLE_MUSIC=/abs/path/to/sample-music \
  uv run --package annemusic-stage-transcribe \
  python stages/transcribe/bench/compare.py
```

The harness prints a markdown table with one row per (fixture × backend) and
exits non-zero if the Qwen3 WER is more than 5 percentage points worse than
whisper on any fixture.

WER is computed via `alt-eval` when installed (the Jam-ALT authors'
readability-aware metric); otherwise it falls back to Levenshtein-over-tokens
on lowercased/punctuation-stripped text. For apples-to-apples comparison
across runs, install `alt-eval`:

```bash
uv pip install alt-eval
```
