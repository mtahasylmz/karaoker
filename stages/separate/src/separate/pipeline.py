"""Separate vocals from instrumental.

Primary path: Mel-Band RoFormer (Kimberley Jensen checkpoint) via
`audio-separator`. See `stages/separate/bench/RESEARCH.md` for why.

Fallback path: htdemucs / htdemucs_ft via `python -m demucs` subprocess.
Flip with `SEPARATE_MODEL=htdemucs` (or via the request's `model` field).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from shared import create_logger, download_file, upload_file, object_path_from_gs_uri

log = create_logger("separate")


# audio-separator model filenames for the RoFormer candidates.
# Keys are the user-facing slugs stored in SEPARATE_MODEL / request.model.
AS_MODEL_FILES: dict[str, str] = {
    "mel_band_roformer_kim": "vocals_mel_band_roformer.ckpt",
    "bs_roformer_ep317":     "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}
DEMUCS_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s"}
DEFAULT_MODEL = "mel_band_roformer_kim"


def run(job_id: str, source_uri: str, model: str | None = None) -> dict:
    """Download source, separate, upload vocals+instrumental. Returns dict
    matching SeparateResponse contract (less stage/job_id/timing wrappers)."""
    started = int(time.time() * 1000)
    active_model = (model or os.environ.get("SEPARATE_MODEL") or DEFAULT_MODEL).strip()
    log.info(job_id, "starting", {"model": active_model, "source": source_uri})

    with tempfile.TemporaryDirectory(prefix=f"separate-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)

        # 1) pull source file (mp4/mov/webm/mkv) to tmp
        source_obj = object_path_from_gs_uri(source_uri)
        ext = Path(source_obj).suffix or ".mp4"
        local_source = tmp / f"source{ext}"
        log.debug(job_id, "downloading source", {"object": source_obj})
        download_file(source_obj, local_source)

        # 2) ffmpeg → stereo 44.1k wav
        audio = tmp / "audio.wav"
        log.debug(job_id, "extracting audio", {})
        _run_cmd([
            "ffmpeg", "-y", "-i", str(local_source),
            "-vn", "-ar", "44100", "-ac", "2",
            str(audio),
        ])

        # 3) separate
        if active_model in AS_MODEL_FILES:
            vocals_src, instr_src = _separate_audio_separator(
                job_id, active_model, audio, tmp / "stems",
            )
        elif active_model in DEMUCS_MODELS:
            vocals_src, instr_src = _separate_demucs(
                job_id, active_model, audio, tmp / "demucs",
            )
        else:
            raise RuntimeError(
                f"unknown SEPARATE_MODEL={active_model!r}; "
                f"expected one of {sorted(AS_MODEL_FILES) + sorted(DEMUCS_MODELS)}"
            )

        # 4) upload outputs under a stable stage-owned path
        vocals_obj = f"stages/separate/{job_id}/vocals.wav"
        instr_obj = f"stages/separate/{job_id}/no_vocals.wav"
        upload_file(vocals_obj, vocals_src, content_type="audio/wav")
        upload_file(instr_obj, instr_src, content_type="audio/wav")

    finished = int(time.time() * 1000)
    bucket = os.environ.get("GCS_BUCKET", "")
    result = {
        "job_id": job_id,
        "stage": "separate",
        "started_at": started,
        "finished_at": finished,
        "duration_ms": finished - started,
        "vocals_uri": f"gs://{bucket}/{vocals_obj}",
        "instrumental_uri": f"gs://{bucket}/{instr_obj}",
        "sample_rate": 44100,
        "model_used": active_model,
    }
    log.info(job_id, "done", {"duration_ms": result["duration_ms"], "model": active_model})
    return result


def _separate_audio_separator(
    job_id: str, model: str, audio: Path, out_dir: Path,
) -> tuple[Path, Path]:
    """RoFormer path via the `audio-separator` library."""
    # Import lazily so the subprocess demucs path stays usable even if
    # audio-separator can't load (e.g. missing GPU runtime in some envs).
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(
        output_dir=str(out_dir),
        output_format="WAV",
        log_level=30,
    )
    log.info(job_id, "audio-separator loading", {"model": model, "file": AS_MODEL_FILES[model]})
    sep.load_model(AS_MODEL_FILES[model])

    # Mel-Band RoFormer emits ("vocals", "other"); BS-RoFormer emits
    # ("vocals", "instrumental"). audio-separator's custom_output_names keys
    # are case-insensitive, so covering both stem vocabularies here lands
    # the non-vocal stem at no_vocals.wav regardless of which model is active.
    custom_names = {
        "Vocals":       "vocals",
        "Other":        "no_vocals",
        "Instrumental": "no_vocals",
    }
    log.info(job_id, "audio-separator running", {"model": model})
    sep.separate(str(audio), custom_output_names=custom_names)

    vocals = out_dir / "vocals.wav"
    no_vocals = out_dir / "no_vocals.wav"
    if not vocals.exists() or not no_vocals.exists():
        raise RuntimeError(
            f"audio-separator outputs missing: {sorted(p.name for p in out_dir.iterdir())}"
        )
    return vocals, no_vocals


def _separate_demucs(
    job_id: str, model: str, audio: Path, out_dir: Path,
) -> tuple[Path, Path]:
    """Demucs fallback path — subprocess to the packaged `demucs` CLI."""
    log.info(job_id, "demucs running", {"model": model})
    _run_cmd([
        "python", "-m", "demucs",
        "--two-stems=vocals",
        "-n", model,
        "-o", str(out_dir),
        str(audio),
    ])

    model_subdir = out_dir / model
    stem_dir = next(model_subdir.iterdir())  # demucs uses input filename as subdir
    vocals_src = stem_dir / "vocals.wav"
    instrumental_src = stem_dir / "no_vocals.wav"
    if not vocals_src.exists() or not instrumental_src.exists():
        raise RuntimeError(f"demucs output missing: {list(stem_dir.iterdir())}")
    return vocals_src, instrumental_src


def _run_cmd(cmd: list[str]) -> None:
    log.debug(None, "exec", {"cmd": " ".join(cmd[:3]) + "..."})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd[:3])}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
