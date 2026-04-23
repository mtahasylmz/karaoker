"""Mix a user recording with the instrumental, optional post-processing.

v1: `autotune: "off"` is the only thing fully wired. `"smooth"` passes
through with a warn (the infra works; the DSP is intentional scope creep
for v2 — see NOTES.md). `"snap"` is rejected — scale-detecting snap-to-note
is explicitly out of v1 per the plan.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from shared import (
    create_logger,
    download_file,
    upload_file,
    object_path_from_gs_uri,
)

log = create_logger("record-mix")


def run(
    job_id: str,
    recording_uri: str,
    instrumental_uri: str,
    autotune: str = "off",
    gain_db: float = 0.0,
) -> dict:
    started = int(time.time() * 1000)
    log.info(job_id, "starting", {"autotune": autotune, "gain_db": gain_db})

    if autotune == "snap":
        raise ValueError("autotune='snap' not implemented in v1 (scale detection out of scope)")
    if autotune == "smooth":
        log.warn(job_id, "autotune=smooth is pass-through in v1; v2 adds RubberBand", {})

    with tempfile.TemporaryDirectory(prefix=f"record-mix-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)
        rec_obj = object_path_from_gs_uri(recording_uri)
        inst_obj = object_path_from_gs_uri(instrumental_uri)
        local_rec = tmp / f"recording{Path(rec_obj).suffix or '.webm'}"
        local_inst = tmp / "instrumental.wav"
        download_file(rec_obj, local_rec)
        download_file(inst_obj, local_inst)

        # Mix via ffmpeg. amix normalizes levels; we then apply the gain the
        # user asked for. Output mp3 for compactness and browser playback.
        mix_local = tmp / "mix.mp3"
        filter_chain = "[0:a]volume=1.0[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=longest:dropout_transition=2"
        if gain_db != 0.0:
            filter_chain += f",volume={gain_db}dB"
        _run([
            "ffmpeg", "-y",
            "-i", str(local_rec),
            "-i", str(local_inst),
            "-filter_complex", filter_chain,
            "-c:a", "libmp3lame", "-q:a", "2",
            str(mix_local),
        ])

        # Upload under a stable path per job.
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
    }
    log.info(job_id, "done", {"duration_ms": result["duration_ms"]})
    return result


def _run(cmd: list[str]) -> None:
    log.debug(None, "exec", {"cmd": " ".join(cmd[:3]) + "..."})
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {' '.join(cmd[:3])}\n"
            f"stderr: {r.stderr[-2000:]}"
        )
