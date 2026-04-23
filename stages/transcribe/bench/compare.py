"""Compare the two ASR backends end-to-end on `sample-music/` fixtures.

Runs `pipeline.run()` with a monkeypatched `download_file` that copies from
local disk instead of GCS, once per (fixture × backend). Reports WER against
hand-written ground-truth transcripts and wall-clock time. Exits non-zero if
the Qwen3 WER regresses more than 5 percentage points vs whisper on any
fixture.

Usage:

    # One-time: stage vocals files for each fixture under bench/fixtures/
    # Either run stages/separate on each .mp4 and copy the vocals.wav over,
    # or (quick + dirty) use the ffmpeg karaoke-subtraction fallback that
    # README.md documents.

    uv sync --package annemusic-stage-transcribe --extra qwen3
    ANNEMUSIC_SAMPLE_MUSIC=/abs/path/to/sample-music \\
      uv run --package annemusic-stage-transcribe \\
      python stages/transcribe/bench/compare.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make the transcribe package importable when running this file directly.
_HERE = Path(__file__).resolve().parent
_STAGE = _HERE.parent
sys.path.insert(0, str(_STAGE / "src"))

from transcribe import pipeline  # noqa: E402


@dataclass(frozen=True)
class Fixture:
    slug: str
    mp4_name: str       # file inside sample-music/
    vocals_name: str    # file inside bench/fixtures/ (pre-extracted)
    language: str


FIXTURES = [
    Fixture(
        slug="billie-eilish-happier-than-ever",
        mp4_name="Billie Eilish - Happier Than Ever (Official Music Video).mp4",
        vocals_name="billie-eilish-happier-than-ever.vocals.wav",
        language="en",
    ),
    Fixture(
        slug="dur-leyla-pulse-of-anatolia",
        mp4_name="Dur Leyla (Anatolian Psychedelic Rock) - Pulse of Anatolia.mp4",
        vocals_name="dur-leyla-pulse-of-anatolia.vocals.wav",
        language="tr",
    ),
    Fixture(
        slug="ahmet-kaya-icimde-olen-biri",
        mp4_name="İçimde Ölen Biri (Ahmet Kaya).mp4",
        vocals_name="ahmet-kaya-icimde-olen-biri.vocals.wav",
        language="tr",
    ),
]

REGRESSION_THRESHOLD_PP = 5.0  # percentage points


def _sample_music_root() -> Path:
    env = os.environ.get("ANNEMUSIC_SAMPLE_MUSIC")
    if env:
        return Path(env).resolve()
    return _STAGE.parents[1] / "sample-music"


def _fixtures_dir() -> Path:
    return _HERE / "fixtures"


def _normalize(text: str) -> str:
    """Minimal pre-alt-eval normalizer: lowercase, strip common punctuation,
    collapse whitespace. alt-eval has its own canonicalizer; this is a fallback
    for when alt-eval isn't installed."""
    import re
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate in [0, 1]. Tries alt-eval first, falls back to a
    Levenshtein-over-tokens computation."""
    try:
        from alt_eval import compute_metrics  # type: ignore
        result = compute_metrics([reference], [hypothesis])
        # alt-eval returns a dict with WER as a percentage or fraction
        # depending on version — normalize both to [0, 1].
        for key in ("WER", "wer", "word_error_rate"):
            if key in result:
                val = float(result[key])
                return val / 100.0 if val > 1.0 else val
    except Exception:
        pass

    ref = _normalize(reference).split()
    hyp = _normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # Standard Levenshtein over tokens.
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / float(m)


def _run_one(fx: Fixture, backend: str, sample_root: Path) -> dict:
    """Runs pipeline.run() with download_file monkeypatched to copy local files."""
    mp4 = sample_root / fx.mp4_name
    vocals_wav = _fixtures_dir() / fx.vocals_name
    if not mp4.exists():
        raise FileNotFoundError(f"sample-music file missing: {mp4}")
    if not vocals_wav.exists():
        raise FileNotFoundError(f"vocals fixture missing: {vocals_wav} (see bench/README.md)")

    # Point the dispatcher at either qwen3 or whisper without changing the
    # flow-routing code. Whisper requires vocals; qwen3 takes the mix.
    if backend == "qwen3":
        pipeline._QWEN3_AVAILABLE = True
        lang_override = fx.language  # qwen3-supported languages all present in FIXTURES
    elif backend == "whisper":
        pipeline._QWEN3_AVAILABLE = False
        lang_override = fx.language
    else:
        raise ValueError(backend)

    # Redirect download_file to copy local files into the pipeline's tmp dir.
    original_download = pipeline.download_file

    def fake_download(object_path: str, local_path):
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        # object_path for mix is derived from a fake gs:// URI we pass below;
        # we dispatch on whether it ends in vocals.wav vs the mp4 name.
        if object_path.endswith(".wav"):
            shutil.copy(vocals_wav, local)
        else:
            shutil.copy(mp4, local)
        return local

    pipeline.download_file = fake_download
    try:
        job_id = f"bench{fx.slug[:8].replace('-', '')}"[:16]
        t0 = time.time()
        out = pipeline.run(
            job_id=job_id,
            vocals_uri=f"gs://bench/stages/separate/{job_id}/vocals.wav",
            source_uri=f"gs://bench/uploads/{job_id}/{fx.mp4_name}",
            language=lang_override,
        )
        wall = time.time() - t0
    finally:
        pipeline.download_file = original_download

    hypothesis = " ".join(s["text"] for s in out["segments"])
    ref_path = _fixtures_dir() / f"{fx.slug}.txt"
    if not ref_path.exists():
        raise FileNotFoundError(f"ground-truth transcript missing: {ref_path}")
    reference = ref_path.read_text(encoding="utf-8")
    # Skip commented-out source lines at the top of the ref file.
    reference = "\n".join(l for l in reference.splitlines() if not l.lstrip().startswith("#"))

    return {
        "fixture": fx.slug,
        "language": fx.language,
        "backend": backend,
        "source": out["source"],
        "detected_language": out["language"],
        "wer": _wer(reference, hypothesis),
        "wall_s": wall,
    }


def main() -> int:
    sample_root = _sample_music_root()
    if not sample_root.exists():
        print(f"sample-music not found at {sample_root}", file=sys.stderr)
        print("Set ANNEMUSIC_SAMPLE_MUSIC=/abs/path/to/sample-music", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for fx in FIXTURES:
        for backend in ("whisper", "qwen3"):
            try:
                rows.append(_run_one(fx, backend, sample_root))
            except Exception as e:
                print(f"[{fx.slug}/{backend}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                rows.append({
                    "fixture": fx.slug, "language": fx.language, "backend": backend,
                    "source": "error", "detected_language": "", "wer": float("nan"),
                    "wall_s": float("nan"),
                })

    _print_table(rows)
    return _exit_code(rows)


def _print_table(rows: list[dict]) -> None:
    print()
    print("| fixture | language | backend | source | detected | WER | wall (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        wer = f"{r['wer']*100:.1f}%" if r["wer"] == r["wer"] else "—"
        wall = f"{r['wall_s']:.1f}" if r["wall_s"] == r["wall_s"] else "—"
        print(f"| {r['fixture']} | {r['language']} | {r['backend']} | "
              f"{r['source']} | {r['detected_language']} | {wer} | {wall} |")
    print()


def _exit_code(rows: list[dict]) -> int:
    by_fixture: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["wer"] != r["wer"]:
            continue
        by_fixture.setdefault(r["fixture"], {})[r["backend"]] = r["wer"]

    for fixture, wers in by_fixture.items():
        w = wers.get("whisper")
        q = wers.get("qwen3")
        if w is None or q is None:
            continue
        regression_pp = (q - w) * 100.0
        if regression_pp > REGRESSION_THRESHOLD_PP:
            print(
                f"REGRESSION: {fixture}: qwen3 WER {q*100:.1f}% vs whisper {w*100:.1f}% "
                f"({regression_pp:+.1f}pp > {REGRESSION_THRESHOLD_PP}pp threshold)",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
