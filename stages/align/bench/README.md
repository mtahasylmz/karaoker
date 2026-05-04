# stages/align/bench

Word-onset MAE bench for the align stage. Measures how close each backend's
word start times are to a ground-truth annotation, in milliseconds.

## Running

```bash
# auto: let shared.flows.flow_for(language) pick the backend
uv run --package annemusic-stage-align python stages/align/bench/runner.py --backend=auto

# force a backend
uv run --package annemusic-stage-align python stages/align/bench/runner.py --backend=whisperx
uv run --package annemusic-stage-align python stages/align/bench/runner.py --backend=qwen3  # needs CUDA

# only run one language's fixtures
uv run --package annemusic-stage-align python stages/align/bench/runner.py --lang=en
```

With no fixtures present the runner prints `no fixtures found` and exits 0.

## Fixture layout

Drop triples under `fixtures/<language-code>/`. The language code is the
folder name and is what we pass through to `pipeline._align_{qwen3,whisperx}`.

```
bench/fixtures/
  en/
    rick.wav        # 16 kHz mono preferred (any ffmpeg-decodable works)
    rick.txt        # either one plain-text line spanning the whole file,
                    # or TSV "<start>\t<end>\t<text>" segments per line
    rick.json       # OPTIONAL ground-truth word onsets:
                    # {"words":[{"text":"never","start":25.12}, ...]}
  tr/
    …
```

`fixtures/` is gitignored — do not commit binary audio or annotations to the
repo.

## Yardstick

Current public SOTA on the DALI lyric-alignment benchmark is around **41 ms
median / 216 ms mean** word-onset MAE. Treat `>500 ms median` on a language
as a red flag; check the backend, the audio quality (make sure we're
benching the vocals stem, not the mix), and whether the `.txt` segmentation
matches what `stages/transcribe` would emit for the same file.

## Excluded aligners (discussion only — do NOT install)

- `ctc-forced-aligner` PyPI package default MMS weights → CC-BY-NC-4.0
- CrisperWhisper → CC-BY-NC
- SOFA → research-only
- MFA (Montreal Forced Aligner) pretrained acoustic models → non-commercial

Qwen3-ForcedAligner-0.6B and the whisperx wav2vec2 defaults
(`facebook/wav2vec2-base-960h`, `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`,
etc.) are Apache-2.0 and fine for production. If you want to add a new
backend for bench comparison only, guard the import and isolate the
dependency; do not promote it to `pyproject.toml` without license review.
