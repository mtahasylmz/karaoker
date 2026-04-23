"""Mix a user recording with the instrumental, honoring tunable knobs.

Every field of the extended `record_mix_request` contract is read here.
What's fully implemented today: per-channel gain, presence EQ, sidechain
ducking, master trim, output limiter. What's accepted-but-not-yet-wired:
GCC-PHAT sync via `vocals_uri`, Demucs bleed cleanup via `clean_bleed`,
convolution reverb via `mix.reverb_wet`, RubberBand pitch for
`autotune="smooth"`. Skipped features emit a warn log and are surfaced
in the response `diagnostics.skipped` list so callers can see what
actually ran.
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

log = create_logger("record-mix")


MIX_DEFAULTS: dict[str, float] = {
    "vocal_gain_db": 0.0,
    "instrumental_gain_db": 0.0,
    "reverb_wet": 0.0,
    "duck_db": 4.0,
    "presence_db": 2.0,
}


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

    skipped: list[str] = []

    if autotune == "snap":
        raise ValueError("autotune='snap' not implemented in v1 (scale detection out of scope)")
    if autotune == "smooth":
        log.warn(job_id, "autotune=smooth is pass-through in v1; v2 adds RubberBand", {})
        skipped.append("autotune=smooth")

    if clean_bleed:
        log.warn(job_id, "clean_bleed=true but Demucs bleed cleanup not wired yet", {})
        skipped.append("clean_bleed")

    if vocals_uri:
        log.warn(job_id, "vocals_uri provided but GCC-PHAT sync not wired yet", {})
        skipped.append("gcc_phat_sync")

    if mix_params["reverb_wet"] > 0:
        log.warn(
            job_id,
            "reverb_wet>0 but convolution reverb (afir + plate IR) not wired yet",
            {"reverb_wet": mix_params["reverb_wet"]},
        )
        skipped.append("reverb")

    with tempfile.TemporaryDirectory(prefix=f"record-mix-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)
        rec_obj = object_path_from_gs_uri(recording_uri)
        inst_obj = object_path_from_gs_uri(instrumental_uri)
        local_rec = tmp / f"recording{Path(rec_obj).suffix or '.webm'}"
        local_inst = tmp / "instrumental.wav"
        download_file(rec_obj, local_rec)
        download_file(inst_obj, local_inst)

        mix_local = tmp / "mix.mp3"
        filter_chain = _build_filter_chain(
            vocal_gain_db=mix_params["vocal_gain_db"],
            instrumental_gain_db=mix_params["instrumental_gain_db"],
            presence_db=mix_params["presence_db"],
            duck_db=mix_params["duck_db"],
            master_gain_db=gain_db,
        )
        _run([
            "ffmpeg", "-y",
            "-i", str(local_rec),
            "-i", str(local_inst),
            "-filter_complex", filter_chain,
            "-map", "[out]",
            "-c:a", "libmp3lame", "-q:a", "2",
            str(mix_local),
        ])

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
        "diagnostics": {
            "applied": {
                "vocal_gain_db": mix_params["vocal_gain_db"],
                "instrumental_gain_db": mix_params["instrumental_gain_db"],
                "presence_db": mix_params["presence_db"],
                "duck_db": mix_params["duck_db"],
                "master_gain_db": gain_db,
            },
            "skipped": skipped,
        },
    }
    log.info(job_id, "done", {"duration_ms": result["duration_ms"], "skipped": skipped})
    return result


def _build_filter_chain(
    *,
    vocal_gain_db: float,
    instrumental_gain_db: float,
    presence_db: float,
    duck_db: float,
    master_gain_db: float,
) -> str:
    # Ratio mapping is approximate: threshold is ~-26 dBFS; ratio grows
    # linearly with requested ducking amount. Real gain reduction depends
    # on signal level, so duck_db is a "taste" knob, not a dB guarantee.
    duck_ratio = max(1.0, 1.0 + duck_db * 0.5)
    return (
        f"[0:a]volume={vocal_gain_db}dB,"
        f"highpass=f=80,"
        f"equalizer=f=4000:width_type=q:w=1:g={presence_db},"
        f"acompressor=threshold=-18dB:ratio=3:attack=5:release=80"
        f"[v_pre];"
        f"[v_pre]asplit=2[v_mix][v_sc];"
        f"[1:a]volume={instrumental_gain_db}dB[i_pre];"
        f"[i_pre][v_sc]sidechaincompress="
        f"threshold=0.05:ratio={duck_ratio:.3f}:attack=20:release=250"
        f"[i_ducked];"
        f"[v_mix][i_ducked]amix=inputs=2:duration=longest:dropout_transition=2,"
        f"volume={master_gain_db}dB,"
        f"alimiter=limit=0.94"
        f"[out]"
    )


def _run(cmd: list[str]) -> None:
    log.debug(None, "exec", {"cmd": " ".join(cmd[:3]) + "..."})
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {' '.join(cmd[:3])}\n"
            f"stderr: {r.stderr[-2000:]}"
        )
