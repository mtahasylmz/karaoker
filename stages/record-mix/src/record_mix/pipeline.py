"""Mix a user recording with the instrumental, honouring every contract knob.

v2 pipeline stages (all optional based on request fields):
    download → GCC-PHAT align → Demucs bleed cleanup → re-align →
    two-pass loudnorm (vocal + instrumental) → composed filter chain
    (presence EQ, convolution reverb, sidechain ducking, master gain, limiter)
    → mp3 encode → upload.

The heavy lifting lives in sibling modules so each piece is unit-testable:
  - align_sync.py    — GCC-PHAT
  - bleed.py         — Demucs invocation
  - loudnorm.py      — two-pass loudnorm
  - filter_chain.py  — composable ffmpeg fragments
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from shared import (
    create_logger,
    download_file,
    upload_file,
    object_path_from_gs_uri,
)

from . import align_sync, bleed, filter_chain, loudnorm

log = create_logger("record-mix")


MIX_DEFAULTS: dict[str, float] = {
    "vocal_gain_db": 0.0,
    "instrumental_gain_db": 0.0,
    "reverb_wet": 0.0,
    "duck_db": 4.0,
    "presence_db": 2.0,
}

PLATE_IR = Path(__file__).resolve().parent.parent.parent / "assets" / "plate_ir.wav"

# Re-align after Demucs only when the shift is worth applying. Below this we
# treat the second correlation as noise-level drift.
REALIGN_MIN_SHIFT_MS = 0.5


def run(
    job_id: str,
    recording_uri: str,
    instrumental_uri: str,
    vocals_uri: str | None = None,
    autotune: str = "off",
    clean_bleed: bool = True,
    gain_db: float = 0.0,
    mix: dict[str, Any] | None = None,
) -> dict:
    started = int(time.time() * 1000)
    mix_params = {**MIX_DEFAULTS, **(mix or {})}
    log.info(
        job_id,
        "starting",
        {
            "autotune": autotune,
            "clean_bleed": clean_bleed,
            "gain_db": gain_db,
            "mix": mix_params,
            "has_vocals_uri": vocals_uri is not None,
        },
    )

    if autotune == "snap":
        raise ValueError("autotune='snap' not implemented in v1 (scale detection out of scope)")

    skipped: list[str] = []
    applied: dict[str, Any] = {
        "vocal_gain_db": mix_params["vocal_gain_db"],
        "instrumental_gain_db": mix_params["instrumental_gain_db"],
        "presence_db": mix_params["presence_db"],
        "duck_db": mix_params["duck_db"],
        "reverb_wet": mix_params["reverb_wet"],
        "master_gain_db": gain_db,
        "clean_bleed": False,
        "realigned_after_cleanup": False,
    }

    if autotune == "smooth":
        log.warn(job_id, "autotune=smooth is pass-through in v1; v2 adds RubberBand", {})
        skipped.append("autotune=smooth")

    with tempfile.TemporaryDirectory(prefix=f"record-mix-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)

        # ---- 1. Download inputs ----------------------------------------
        rec_obj = object_path_from_gs_uri(recording_uri)
        inst_obj = object_path_from_gs_uri(instrumental_uri)
        local_rec_in = tmp / f"recording_in{Path(rec_obj).suffix or '.webm'}"
        local_inst = tmp / "instrumental.wav"
        download_file(rec_obj, local_rec_in)
        download_file(inst_obj, local_inst)

        local_vocals: Path | None = None
        if vocals_uri:
            voc_obj = object_path_from_gs_uri(vocals_uri)
            local_vocals = tmp / "vocals.wav"
            download_file(voc_obj, local_vocals)

        # ---- 2. Decode recording → mono 48k WAV for DSP ----------------
        rec_decoded = tmp / "recording.wav"
        _run(["ffmpeg", "-y", "-i", str(local_rec_in),
              "-ac", "1", "-ar", "48000", "-f", "wav", str(rec_decoded)])

        # ---- 3. GCC-PHAT alignment (optional) --------------------------
        aligned = _align(job_id, rec_decoded, local_inst, local_vocals, tmp, skipped, applied)

        # ---- 4. Demucs bleed cleanup (optional) ------------------------
        pre_mix_vocal = aligned
        if clean_bleed:
            cleaned = bleed.clean_bleed(aligned, tmp / "demucs")
            applied["clean_bleed"] = True
            pre_mix_vocal = cleaned
            # Re-align the cleaned stem — Demucs adds 1–3 ms phase shift.
            if local_vocals is not None:
                try:
                    sig = align_sync.load_mono_48k(cleaned)
                    ref = align_sync.load_mono_48k(local_vocals)
                    lag_s, snr = align_sync.gcc_phat(sig, ref)
                    if snr >= align_sync.SNR_ACCEPT_DB and abs(lag_s) * 1000 >= REALIGN_MIN_SHIFT_MS:
                        shifted = tmp / "vocal_post_clean_aligned.wav"
                        _shift_audio(cleaned, shifted, lag_s)
                        pre_mix_vocal = shifted
                        applied["realigned_after_cleanup"] = True
                        applied["alignment_offset_ms_post_clean"] = lag_s * 1000.0
                except Exception as e:
                    log.warn(job_id, "post-clean realignment failed; using un-realigned stem",
                             {"err": str(e)})

        # ---- 5. Two-pass loudnorm measurement --------------------------
        vocal_meas = loudnorm.measure(pre_mix_vocal, targets=loudnorm.VOCAL_TARGETS)
        inst_meas = loudnorm.measure(local_inst, targets=loudnorm.INSTRUMENTAL_TARGETS)
        applied["loudnorm"] = {
            "vocal": _loudnorm_summary(vocal_meas),
            "instrumental": _loudnorm_summary(inst_meas),
        }
        vocal_ln = loudnorm.second_pass_filter(vocal_meas, targets=loudnorm.VOCAL_TARGETS)
        inst_ln = loudnorm.second_pass_filter(inst_meas, targets=loudnorm.INSTRUMENTAL_TARGETS)

        # ---- 6. Compose filter graph -----------------------------------
        reverb_wet = float(mix_params["reverb_wet"])
        has_reverb = reverb_wet > 0 and PLATE_IR.exists()
        if reverb_wet > 0 and not PLATE_IR.exists():
            log.warn(job_id, "reverb_wet>0 but plate_ir.wav missing; skipping reverb",
                     {"expected": str(PLATE_IR)})
            skipped.append("reverb_asset_missing")

        fragments: list[str] = []
        fragments.append(filter_chain.vocal_chain(
            in_label="0:a", out_label="v_post_eq",
            loudnorm_filter=vocal_ln,
            vocal_gain_db=mix_params["vocal_gain_db"],
            presence_db=mix_params["presence_db"],
        ))
        vocal_pre_duck_label = "v_post_eq"
        if has_reverb:
            fragments.append(filter_chain.reverb_chain(
                in_label="v_post_eq", out_label="v_post_rev",
                ir_input="2:a", reverb_wet=reverb_wet,
            ))
            vocal_pre_duck_label = "v_post_rev"

        fragments.append(filter_chain.instrumental_chain(
            in_label="1:a", out_label="i_pre",
            loudnorm_filter=inst_ln,
            instrumental_gain_db=mix_params["instrumental_gain_db"],
        ))
        fragments.append(filter_chain.ducking_chain(
            vocal_label=vocal_pre_duck_label, instrumental_label="i_pre",
            vocal_out="v_mix", ducked_out="i_ducked",
            duck_db=mix_params["duck_db"],
        ))
        fragments.append(filter_chain.bus_chain(
            vocal_label="v_mix", instrumental_label="i_ducked",
            out_label="out", master_gain_db=gain_db,
        ))
        graph = filter_chain.assemble(fragments)

        # ---- 7. Encode + upload ----------------------------------------
        mix_local = tmp / "mix.mp3"
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(pre_mix_vocal), "-i", str(local_inst)]
        if has_reverb:
            ffmpeg_cmd += ["-i", str(PLATE_IR)]
        ffmpeg_cmd += [
            "-filter_complex", graph,
            "-map", "[out]",
            "-c:a", "libmp3lame", "-q:a", "2",
            str(mix_local),
        ]
        _run(ffmpeg_cmd)

        mix_obj = f"stages/record-mix/{job_id}/mix.mp3"
        upload_file(mix_obj, mix_local, content_type="audio/mpeg")

    finished = int(time.time() * 1000)
    bucket = os.environ.get("GCS_BUCKET", "")
    result = {
        "job_id": job_id,
        "stage": "record-mix",
        "started_at": started,
        "finished_at": finished,
        "duration_ms": finished - started,
        "mix_uri": f"gs://{bucket}/{mix_obj}",
        "diagnostics": {"applied": applied, "skipped": skipped},
    }
    log.info(job_id, "done",
             {"duration_ms": result["duration_ms"], "skipped": skipped, "applied": applied})
    return result


# ------------------------------------------------------------------------
# Alignment helper
# ------------------------------------------------------------------------

def _align(
    job_id: str,
    rec_wav: Path,
    instrumental_wav: Path,
    vocals_wav: Path | None,
    tmp: Path,
    skipped: list[str],
    applied: dict[str, Any],
) -> Path:
    """Return path to the time-aligned recording. Updates applied/skipped in place."""
    sig = align_sync.load_mono_48k(rec_wav)

    lag_s: float | None = None
    snr_db: float | None = None
    ref_used: str | None = None

    if vocals_wav is not None:
        ref = align_sync.load_mono_48k(vocals_wav)
        candidate_lag, candidate_snr = align_sync.gcc_phat(sig, ref)
        if candidate_snr >= align_sync.SNR_ACCEPT_DB:
            lag_s, snr_db, ref_used = candidate_lag, candidate_snr, "vocals"

    if lag_s is None:
        ref = align_sync.load_mono_48k(instrumental_wav)
        candidate_lag, candidate_snr = align_sync.gcc_phat(sig, ref)
        if candidate_snr >= align_sync.SNR_ACCEPT_DB:
            lag_s, snr_db, ref_used = candidate_lag, candidate_snr, "instrumental"

    if lag_s is None:
        log.warn(job_id, "GCC-PHAT peak below SNR threshold; skipping alignment",
                 {"threshold_db": align_sync.SNR_ACCEPT_DB})
        skipped.append("align_sync_no_signal")
        return rec_wav

    applied["alignment_offset_ms"] = lag_s * 1000.0
    applied["alignment_snr_db"] = snr_db
    applied["alignment_ref"] = ref_used
    log.info(job_id, "align_sync locked", {
        "offset_ms": applied["alignment_offset_ms"],
        "snr_db": snr_db,
        "ref": ref_used,
    })

    aligned = tmp / "recording_aligned.wav"
    _shift_audio(rec_wav, aligned, lag_s)
    return aligned


def _shift_audio(src: Path, dst: Path, lag_s: float) -> None:
    """Apply a positive (trim) or negative (delay) offset via ffmpeg."""
    ms = lag_s * 1000.0
    if ms > 0:
        af = f"atrim=start={lag_s:.6f},asetpts=PTS-STARTPTS"
    else:
        delay = int(round(abs(ms)))
        af = f"adelay={delay}:all=1"
    _run(["ffmpeg", "-y", "-i", str(src), "-af", af,
          "-ar", "48000", "-ac", "1", str(dst)])


def _loudnorm_summary(meas: dict) -> dict:
    """Pick the fields worth reporting in diagnostics (strings from ffmpeg → floats)."""
    def _f(key: str) -> float:
        try:
            return float(meas.get(key))
        except (TypeError, ValueError):
            return float("nan")
    return {
        "input_i": _f("input_i"),
        "input_tp": _f("input_tp"),
        "input_lra": _f("input_lra"),
        "input_thresh": _f("input_thresh"),
        "target_offset": _f("target_offset"),
    }


# ------------------------------------------------------------------------
# Subprocess helper — preserved from v1 for consistency across the repo.
# ------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    log.debug(None, "exec", {"cmd": " ".join(cmd[:3]) + "..."})
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {' '.join(cmd[:3])}\n"
            f"stderr: {r.stderr[-2000:]}"
        )
