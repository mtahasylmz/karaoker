"""Align bench runner — per-language word-onset MAE.

Usage:
    uv run --package annemusic-stage-align python stages/align/bench/runner.py \
        --backend=auto
    uv run --package annemusic-stage-align python stages/align/bench/runner.py \
        --backend=whisperx --lang=en

Fixture layout (drop triples under stages/align/bench/fixtures/{lang}/):
    fixtures/
      en/
        song_a.wav        # 16 kHz mono recommended; ffmpeg-decodable regardless
        song_a.txt        # one line per segment: "<start>\t<end>\t<text>"
                          # or a single plain-text line spanning the whole file
        song_a.json       # OPTIONAL ground-truth onsets:
                          #   {"words":[{"text":"...","start":0.42}, ...]}

Backends:
    --backend=qwen3    force Qwen3-ForcedAligner (needs CUDA)
    --backend=whisperx force whisperx/wav2vec2
    --backend=auto     use shared.flows.flow_for(language)

Yardstick:
    DALI SOTA ≈ 41 ms median / 216 ms mean word-onset MAE.
    >500 ms median on a language → subpar; dig in.

Excluded aligners (for comparison discussion only, do NOT add as deps):
    - ctc-forced-aligner default MMS weights (CC-BY-NC-4.0)
    - CrisperWhisper (CC-BY-NC)
    - SOFA (research-only)
    - MFA pretrained models (non-commercial)
    Qwen3-ForcedAligner + whisperx wav2vec2 defaults are both Apache-2.0.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parent / "fixtures"
FAIL_THRESHOLD_MS = 500.0


def _segments_from_txt(txt_path: Path, audio_seconds: float) -> list[dict]:
    """Parse a .txt into segments. Accepts two shapes:

    1. Tab-separated: "<start>\t<end>\t<text>" one per line.
    2. Single plain-text line → one segment spanning the whole audio.
    """
    raw = txt_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    tsv_rows: list[dict] = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 3:
            try:
                tsv_rows.append(
                    {"start": float(parts[0]), "end": float(parts[1]),
                     "text": "\t".join(parts[2:]).strip()}
                )
                continue
            except ValueError:
                pass
        tsv_rows = []
        break
    if tsv_rows:
        return tsv_rows
    return [{"start": 0.0, "end": audio_seconds, "text": raw.replace("\n", " ")}]


def _load_ground_truth(json_path: Path | None) -> list[dict] | None:
    if json_path is None or not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        words = data.get("words") if isinstance(data, dict) else None
        if isinstance(words, list):
            return [w for w in words if "start" in w]
    except Exception:
        pass
    return None


def _audio_seconds(wav_path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 300.0  # fallback: plan_chunks's max — bench only, not production


def _mae_ms(pred_words: list[dict], gt_words: list[dict]) -> tuple[float, float, int]:
    """Median + mean onset error in ms + count of per-word errors > 500 ms.

    Alignment heuristic: pair predicted[i] with GT[i] by position. If lengths
    differ we truncate to the shorter — avoids hallucinating word matches.
    """
    n = min(len(pred_words), len(gt_words))
    if n == 0:
        return float("nan"), float("nan"), 0
    errs = [
        abs(float(pred_words[i]["start"]) - float(gt_words[i]["start"])) * 1000.0
        for i in range(n)
    ]
    fails = sum(1 for e in errs if e > FAIL_THRESHOLD_MS)
    return statistics.median(errs), statistics.fmean(errs), fails


def _iter_fixtures(lang_filter: str | None):
    if not BENCH_ROOT.exists():
        return
    for lang_dir in sorted(p for p in BENCH_ROOT.iterdir() if p.is_dir()):
        if lang_filter and lang_dir.name != lang_filter:
            continue
        for wav in sorted(lang_dir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                continue
            json_path = wav.with_suffix(".json")
            yield lang_dir.name, wav, txt, (json_path if json_path.exists() else None)


def _run_once(lang: str, wav: Path, segments: list[dict], backend: str) -> list[dict]:
    """Call into pipeline internals with a preloaded audio array.

    Bypasses pipeline.run's GCS download step — bench reads the wav
    directly. Backend resolution respects the --backend flag: auto routes
    through shared.flows.flow_for(language), everything else is forced.
    """
    from align import pipeline

    if backend == "auto":
        resolved = pipeline._resolve_backend("bench", pipeline.flow_for(lang).align)
    elif backend == "qwen3":
        resolved = "qwen3" if pipeline._QWEN3_IMPORT_OK and pipeline._cuda_available() else "whisperx"
        if resolved != "qwen3":
            print(f"  ! qwen3 requested but unavailable → falling back to whisperx")
    elif backend == "whisperx":
        resolved = "whisperx"
    else:
        raise SystemExit(f"unknown --backend: {backend}")

    import whisperx

    audio = whisperx.load_audio(str(wav))
    chunks = pipeline.plan_chunks(segments, [], max_seconds=pipeline._QWEN_MAX_SECONDS)
    words: list[dict] = []
    # Bench uses one temp dir per fixture so Qwen3 chunk wav slices don't pile up.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="align-bench-") as tmp_s:
        tmp = Path(tmp_s)
        for i, chunk in enumerate(chunks):
            if resolved == "qwen3":
                try:
                    words += pipeline._align_qwen3(tmp, audio, chunk, lang, idx=i)
                    continue
                except Exception as e:
                    print(f"  ! qwen3 chunk {i} fallback: {type(e).__name__}: {e}")
            try:
                words += pipeline._align_whisperx(audio, chunk, lang)
            except Exception as e:
                print(f"  ! whisperx chunk {i} fallback: {type(e).__name__}: {e}")
                words += pipeline._synthesize_words(chunk)
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("auto", "qwen3", "whisperx"), default="auto")
    parser.add_argument("--lang", default=None, help="only run fixtures for this language (folder name)")
    args = parser.parse_args()

    fixtures = list(_iter_fixtures(args.lang))
    if not fixtures:
        print("no fixtures found")
        print(f"  expected layout: {BENCH_ROOT}/<lang>/<name>.{{wav,txt,json}}")
        return 0

    print(f"# align bench — backend={args.backend}  fixtures={len(fixtures)}")
    print(f"{'lang':<5} {'fixture':<30} {'backend':<10} "
          f"{'median_ms':>10} {'mean_ms':>10} {'fails>500':>10} {'wall_s':>8}  words")
    for lang, wav, txt, json_path in fixtures:
        audio_seconds = _audio_seconds(wav)
        segments = _segments_from_txt(txt, audio_seconds)
        gt = _load_ground_truth(json_path)
        t0 = time.time()
        try:
            pred_words = _run_once(lang, wav, segments, args.backend)
        except Exception as e:  # don't let one fixture kill the report
            print(f"{lang:<5} {wav.stem:<30} ERROR: {type(e).__name__}: {e}")
            continue
        wall = time.time() - t0
        if gt is not None:
            med_ms, mean_ms, fails = _mae_ms(pred_words, gt)
            med_s = f"{med_ms:>10.1f}"
            mean_s = f"{mean_ms:>10.1f}"
            fails_s = f"{fails:>10d}"
        else:
            med_s, mean_s, fails_s = f"{'—':>10}", f"{'—':>10}", f"{'—':>10}"
        print(f"{lang:<5} {wav.stem:<30} {args.backend:<10} "
              f"{med_s} {mean_s} {fails_s} {wall:>8.2f}  {len(pred_words)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
