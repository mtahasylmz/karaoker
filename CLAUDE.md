# annemusic

Karaoke pipeline: music video in → instrumental audio + per-word-timed `.ass`
subtitles out. **No UI, no services** — a Python CLI on a CUDA box (RTX-class;
RunPod SSH available for real-GPU runs). Everything before the 2026-06 fresh
start lives on `main` history; don't rebuild it, harvest from it.

## Mindset

1. **Ponytail** (`/ponytail`, always on): laziest thing that works. Stdlib over
   dependency, one file over five, delete over add, YAGNI.
2. **SwarmForge six-pack** (below): spec → user approval → architect → coder →
   cleaner+hardener → QA. No code before an approved spec.

## Target shape

```
video.mp4 → separate (roformer/demucs) → transcribe (qwen3-asr, LID first)
          → align (qwen3-forced-aligner / whisperx) → lyrics.ass + manifest.json
```

## Hard-won ML facts (validated on A40, 2026-06 — do not re-learn)

- **qwen3 aligner: pack chunks to ~120 s** (`QWEN_CHUNK_TARGET_S`). Quality
  collapses near its 300 s cap — 57% degenerate word spans at 285 s, near-clean
  at 60–130 s. 300 is the hard limit, not a packing size.
- **Repair, don't reject:** aligner output has 1–2 wild artifacts per chunk
  (zero-length ties, 30 s "words", words stamped past the audio). Nudge/clamp/drop
  the offender; reject a chunk only when anomalies are systemic (>20%).
- **Held sung notes run 5–10 s** — speech-derived word-span caps reject real lyrics.
- **hf_transfer must be installed** wherever `HF_HUB_ENABLE_HF_TRANSFER=1`
  (RunPod images set it globally); missing package = hard download failure.
- **whisperx 3.1.6: never call `load_model`** (dead VAD URL). faster-whisper for
  ASR, `whisperx.align()` only for wav2vec2 alignment.
- **No language hint → detect first:** faster-whisper LID on a ~30 s vocals-stem
  window anchored at the first VAD vocal region (song intros poison LID at 0 s).
  Route qwen3 vs whisper off the detection (≥0.5 prob).
- **Qwen3-ASR eats the full mix** (trained with BGM), wants English-name language
  args, takes `context=` for known-lyrics biasing. Whisper wants the vocals stem.
- **RMS-VAD on the vocals stem** is ground truth for instrumental breaks;
  thresholds tuned for stem output (−40/−46 dBFS hysteresis).
- **Subprocesses: `sys.executable`, never `"python"`.** torchaudio ≥2.9 needs
  `torchcodec` to save audio. qwen-asr 0.0.6 throws internal NameError on ~4-word
  chunks — absorb via per-chunk fallback.
- **Separation bench (M4, 2026-04):** mel_band_roformer_kim > bs_roformer >
  htdemucs (~+2.6 dB SDR both stems); via `audio-separator`, needs numpy≥2.

# CLAUDE.md — SwarmForge Six-Pack, Orchestrated by Claude

This repo follows the SwarmForge discipline (github.com/unclebob/swarmforge,
six-pack) without the tmux plumbing. Claude is the orchestrator: it plays the
**specifier inline with the user** and runs the other five roles as **isolated
subagents** (Workflow/Agent tool). Do not simulate six hats in one context —
role isolation is the product, not an implementation detail.

If `swarmforge/roles/*.prompt` and `swarmforge/constitution/` exist in this
repo, those full charters override the condensed ones below.

## Pipeline (strict, one-directional)

specifier (inline) → architect → coder → cleaner + hardener (batch, may run together) → QA → merge

- **Approval gate:** no code until the user explicitly approves the spec.
- Never reorder or skip stages. Failures loop back to the coder, not forward.
- The git log is the handoff ledger: each role commits with a role-tagged message.

## Independence rule (the point of the roles)

Each subagent receives ONLY: its charter, the approved spec, and the diff.
Never the conversation history, never another role's rationale. QA is prompted
adversarially ("find where this violates the spec"), never confirmationally.

## Roles (condensed charters)

**Specifier — Claude, inline with the user.**
Turn intent into Gherkin in `features/*.feature` plus an end-to-end QA
procedure in `qa/<feature>.qa.md`. Gherkin follows
github.com/unclebob/Acceptance-Pipeline-Specification: scenario names are
`<feature>-<index>`, parameters for anything that varies, prune example-table
columns that don't improve acceptance mutation, hoist repeated setup into
`Background`. E2E means through the user interface only — no project API; CLI
flags count as UI when user-facing. Ask questions to settle ambiguity, then
STOP for approval.

**Architect — subagent.**
Plans the slice: module boundaries, what stays testable, small adapter seams
around environmentally unsuitable code. Output is guidance for the coder, not code.

**Coder — subagent, isolated git worktree.**
TDD: failing unit test first (one that a plausible wrong implementation would
fail), then minimum code to pass. Keeps the acceptance pipeline green
(gherkin-parser → project generator → generated tests). Regex-capture step
handlers by default; separate literal handlers only for genuinely different
behavior. Generated acceptance tests stay separate from unit tests. Does NOT
touch the QA suite, and does not run mutation/CRAP/DRY.

**Cleaner — subagent, batch.** DRY and CRAP cleanup on touched code.

**Hardener — subagent, batch.** Mutation testing; kill surviving mutants with
better tests; property tests where they pay. Runs Gherkin acceptance mutation.

**QA — subagent, adversarial.**
Final independent verification: executable QA suite through the UI, acceptance
+ unit + property tests, then CRAP + DRY. Reproduce failures before changing
code; QA-owned fixes stay minimal. If the QA suite contradicts the Gherkin or
unit tests, STOP and ask — never silently change behavior.

## Repo conventions

- `features/` — Gherkin specs
- `qa/` — per-feature `*.qa.md` procedures + one executable suite (e.g. `qa/suite.ts`)
- `acceptance/` — APS pipeline: parser wiring, generator, runtime, step handlers
- Unit tests beside source (`*.test.*`), property tests `*.prop.*`

## Quality gates (all green before merge)

unit, acceptance, qa suite, property, code mutation, Gherkin acceptance
mutation, CRAP, DRY. Tool mapping is per-language — establish it at repo setup
and record it here. Known-good TS mapping: Stryker (mutation), jscpd (DRY),
a small CRAP script over coverage output.

### Python tool mapping (established 2026-07, architect)

| Gate | Tool / command |
|---|---|
| unit | `uv run pytest` (testpaths = `annemusic/`, files `*_test.py`) |
| acceptance | `uv run python acceptance/generate.py && uv run pytest acceptance/generated` |
| property | `uv run pytest -o python_files='*_prop.py' annemusic` (hypothesis; separate from unit run) |
| coverage | pytest-cov → `--cov=annemusic --cov-report=json` |
| code mutation | mutmut; `[tool.mutmut] paths_to_mutate` = pure modules only (`core.py`, `vad.py`, `ass.py`) — never `backends.py`/`cli.py` (subprocess/GPU wrappers; unit tests can't kill those mutants). Runner must exec pytest DIRECTLY (`python -m pytest`), never `uv run pytest`: mutmut's timeout kills only the runner process, an orphaned pytest child keeps the stdout pipe open, and every infinite-loop mutant then stalls the sweep (untested pile-up + a live mutant left on disk) |
| CRAP | `tools/crap.py`: radon `cc --json` × coverage json, CRAP = c²·(1−cov)³ + c, threshold 30 |
| DRY | `npx --yes jscpd@4 annemusic --format python` (v4 flag is `--format`, not `--languages`) |
| Gherkin acceptance mutation | APS Babashka `gherkin-mutator --level soft --generated-dir acceptance/generated --runner-worker "uv run python acceptance/mutation_worker.py"` (persistent ndjson worker per mutator-spec; generated tests load mutated IR via `APS_IR`); needs the disk-cached acceptance runner to be affordable |

APS tools: Babashka `gherkin-parser`/`gherkin-mutator` from
github.com/unclebob/Acceptance-Pipeline-Specification, fetched fresh (clone
to a gitignored `.aps/`), never vendored or reimplemented. `bb` is installed.

## Bootstrap (first feature in a fresh repo)

1. Specifier writes the first spec; user approves.
2. Coder's first task includes standing up the APS acceptance pipeline
   (use the APS `gherkin-parser`; never reimplement it) and the package
   scripts for every quality gate.
3. QA's first task includes making `qa/suite.ts` (or equivalent) executable
   end-to-end against the real UI.
