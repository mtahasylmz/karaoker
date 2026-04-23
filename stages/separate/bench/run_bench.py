"""Run candidate separation models over fixtures, record SDR + wall-clock.

Each fixture in ../fixtures/{name}/ must contain:
  mixture.wav   — input the separator runs against
  vocals.wav    — ground-truth vocals stem
  no_vocals.wav — ground-truth instrumental stem

For each (model, fixture) we time `Separator.separate(mixture)` once,
read back the predicted vocals + instrumental, and compute uSDR (signal-to-
distortion ratio, single number per track, stereo-averaged) against the
ground truth via museval's bss_eval_sources.

Results land in ./results/{model}/{fixture}/ and are summarized in
./results/results.json + ./results/summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

BENCH_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCH_DIR.parent / "fixtures"
RESULTS_DIR = BENCH_DIR / "results"

# Model registry — (display name, audio-separator filename).
# Keys are slugs used for output directories.
MODELS: dict[str, str] = {
    "htdemucs":              "htdemucs.yaml",
    "htdemucs_ft":           "htdemucs_ft.yaml",
    "mel_band_roformer_kim": "vocals_mel_band_roformer.ckpt",
    "bs_roformer_ep317":     "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}


@dataclass
class Result:
    model: str
    fixture: str
    wall_clock_s: float
    audio_duration_s: float
    rtf: float  # real-time factor: wall_clock / audio_duration
    vocals_sdr_db: float | None
    instrumental_sdr_db: float | None
    sample_rate: int
    error: str | None = None


_STEM_ALIASES = (
    "Vocals", "Instrumental",
    "Drums", "Bass", "Other", "Guitar", "Piano",
)


def run_model(model_key: str, model_file: str, fixtures: list[Path]) -> list[Result]:
    from audio_separator.separator import Separator  # heavy import, lazy

    model_out = RESULTS_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    sep = Separator(
        output_dir=str(model_out),  # init-only in audio-separator; shared per model
        output_format="WAV",
        log_level=30,
    )
    print(f"[{model_key}] loading {model_file} ...")
    t_load = time.perf_counter()
    sep.load_model(model_file)
    print(f"[{model_key}] loaded in {time.perf_counter()-t_load:.1f}s")

    results: list[Result] = []
    for fx in fixtures:
        name = fx.name
        mixture = fx / "mixture.wav"
        if not mixture.exists():
            continue
        audio_dur = _duration(mixture)

        # Force fixture-scoped output basenames so runs don't collide
        # (audio-separator uses input filename by default → all fixtures write
        # to "mixture_(Vocals)_<model>.wav" and overwrite each other).
        custom_names = {alias: f"{name}_{alias.lower().replace(' ', '_')}"
                        for alias in _STEM_ALIASES}

        print(f"[{model_key}] {name}: separating...")
        t0 = time.perf_counter()
        try:
            sep.separate(str(mixture), custom_output_names=custom_names)
            wall = time.perf_counter() - t0
            vocals_path = model_out / f"{name}_vocals.wav"
            if not vocals_path.exists():
                raise RuntimeError(f"no vocals output at {vocals_path}; "
                                   f"got {sorted(p.name for p in model_out.glob(f'{name}_*'))}")
            instr_path = _build_instrumental(model_out, name, mixture)
            vocals_sdr = _sdr(fx / "vocals.wav", vocals_path)
            instr_sdr = _sdr(fx / "no_vocals.wav", instr_path)
            results.append(Result(
                model=model_key, fixture=name,
                wall_clock_s=wall, audio_duration_s=audio_dur,
                rtf=wall / audio_dur,
                vocals_sdr_db=vocals_sdr, instrumental_sdr_db=instr_sdr,
                sample_rate=_sr(mixture),
            ))
            print(f"[{model_key}] {name}: {wall:.1f}s (rtf={wall/audio_dur:.2f}) "
                  f"voc_sdr={vocals_sdr:.2f} inst_sdr={instr_sdr:.2f}")
        except Exception as e:
            wall = time.perf_counter() - t0
            print(f"[{model_key}] {name}: FAILED after {wall:.1f}s: {e}")
            results.append(Result(
                model=model_key, fixture=name,
                wall_clock_s=wall, audio_duration_s=audio_dur,
                rtf=wall / audio_dur if audio_dur else 0.0,
                vocals_sdr_db=None, instrumental_sdr_db=None,
                sample_rate=_sr(mixture),
                error=f"{type(e).__name__}: {e}",
            ))
    return results


def _build_instrumental(model_out: Path, name: str, mixture: Path) -> Path:
    """Return a path to the instrumental stem for this fixture.

    Prefer an explicit 'instrumental' output (RoFormer 2-stem models).
    Fall back to summing htdemucs' drums+bass+other+piano+guitar.
    Last resort: mixture minus vocals (waveform subtraction)."""
    explicit = model_out / f"{name}_instrumental.wav"
    if explicit.exists():
        return explicit

    parts = [model_out / f"{name}_{s}.wav"
             for s in ("drums", "bass", "other", "guitar", "piano")]
    parts = [p for p in parts if p.exists()]
    if len(parts) >= 2:
        acc = None
        sr = None
        for p in parts:
            a, s = sf.read(p, dtype="float32", always_2d=True)
            if acc is None:
                acc = a.copy(); sr = s
            else:
                n = min(len(acc), len(a))
                acc = acc[:n] + a[:n]
        out = model_out / f"{name}_instrumental.wav"
        sf.write(out, acc, sr, subtype="FLOAT")
        return out

    vocals = model_out / f"{name}_vocals.wav"
    mix, sr = sf.read(mixture, dtype="float32", always_2d=True)
    voc, _ = sf.read(vocals, dtype="float32", always_2d=True)
    n = min(len(mix), len(voc))
    out = model_out / f"{name}_instrumental.wav"
    sf.write(out, mix[:n] - voc[:n], sr, subtype="FLOAT")
    return out


def _sdr(reference: Path, estimate: Path) -> float:
    """uSDR: one SDR per stereo stem, averaged over channels.
    museval.evaluate() expects (nsrc, nsampl, nchan). We pass nsrc=1."""
    import museval
    ref, sr_r = sf.read(reference, dtype="float32", always_2d=True)
    est, sr_e = sf.read(estimate, dtype="float32", always_2d=True)
    if sr_r != sr_e:
        raise RuntimeError(f"sample rate mismatch: {sr_r} vs {sr_e}")
    # Align lengths (separator output may be padded by frame rounding)
    n = min(ref.shape[0], est.shape[0])
    ref = ref[:n]; est = est[:n]
    sdr, _, _, _ = museval.evaluate(
        ref[np.newaxis, ...], est[np.newaxis, ...],
        win=sr_r * 1, hop=sr_r * 1,  # 1 s chunks → chunk-level SDR
    )
    # sdr shape: (nsrc=1, nframes) — take median to match museval's usual reporting
    vals = sdr[0]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def _duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def _sr(path: Path) -> int:
    return sf.info(path).samplerate


def write_summary(results: list[Result]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )

    # Aggregate per model
    by_model: dict[str, list[Result]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    lines = ["# Bench results\n",
             "| model | n | vocals SDR (median) | instr SDR (median) | avg RTF | errors |",
             "|---|---:|---:|---:|---:|---:|"]
    for m, rs in by_model.items():
        ok = [r for r in rs if r.error is None and r.vocals_sdr_db is not None]
        errs = sum(1 for r in rs if r.error)
        if ok:
            v = float(np.median([r.vocals_sdr_db for r in ok]))
            i = float(np.median([r.instrumental_sdr_db for r in ok]))
            rtf = float(np.mean([r.rtf for r in ok]))
            lines.append(f"| `{m}` | {len(ok)} | {v:.2f} | {i:.2f} | {rtf:.2f} | {errs} |")
        else:
            lines.append(f"| `{m}` | 0 | — | — | — | {errs} |")

    lines.append("\n## Per-fixture detail\n")
    lines.append("| model | fixture | wall (s) | rtf | voc SDR | inst SDR | error |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in results:
        err = r.error or ""
        v = f"{r.vocals_sdr_db:.2f}" if r.vocals_sdr_db is not None else "—"
        i = f"{r.instrumental_sdr_db:.2f}" if r.instrumental_sdr_db is not None else "—"
        lines.append(
            f"| `{r.model}` | {r.fixture} | {r.wall_clock_s:.1f} | {r.rtf:.2f} | {v} | {i} | {err} |"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    help=f"subset of {list(MODELS.keys())}")
    ap.add_argument("--fixtures", nargs="+", default=None,
                    help="fixture names (subfolder names under ../fixtures); default all")
    ap.add_argument("--clean", action="store_true",
                    help="wipe ./results before running")
    args = ap.parse_args()

    if args.clean and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    all_fixtures = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())
    if args.fixtures:
        wanted = set(args.fixtures)
        all_fixtures = [p for p in all_fixtures if p.name in wanted]
    if not all_fixtures:
        raise SystemExit(f"no fixtures under {FIXTURES_DIR}. Run fetch_fixtures.py first.")
    print(f"fixtures: {[p.name for p in all_fixtures]}")

    # PyTorch on Apple Silicon: let MPS fall back to CPU for unsupported ops.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    results: list[Result] = []
    for key in args.models:
        if key not in MODELS:
            print(f"skip unknown model: {key}")
            continue
        try:
            results.extend(run_model(key, MODELS[key], all_fixtures))
        except Exception as e:
            print(f"[{key}] model setup failed: {type(e).__name__}: {e}")
            for fx in all_fixtures:
                results.append(Result(
                    model=key, fixture=fx.name,
                    wall_clock_s=0.0, audio_duration_s=_duration(fx / "mixture.wav"),
                    rtf=0.0, vocals_sdr_db=None, instrumental_sdr_db=None,
                    sample_rate=_sr(fx / "mixture.wav"),
                    error=f"setup: {type(e).__name__}: {e}",
                ))
        write_summary(results)  # incremental, so partial runs still produce output

    print(f"\nwrote {RESULTS_DIR/'results.json'} and {RESULTS_DIR/'summary.md'}")


if __name__ == "__main__":
    main()
