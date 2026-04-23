"""Run record-mix pipeline over 4 parameter combos; write results.json + summary.md.

Uses DEV_FS_ROOT so the pipeline's shared.gcs helpers short-circuit to the
local filesystem — no GCS creds needed. Each combo's output mp3 lands under
bench/results/<combo>/mix.mp3 for listening.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BENCH_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCH_DIR / "fixtures"
RESULTS_DIR = BENCH_DIR / "results"

# Everything under this root is treated as gs://test-bucket/<object_path>.
BUCKET = "test-bucket"


COMBOS: list[dict[str, Any]] = [
    {"name": "baseline", "clean_bleed": False, "autotune": "off", "gain_db": 0.0,
     "mix": {"vocal_gain_db": 0, "instrumental_gain_db": 0, "reverb_wet": 0,
             "duck_db": 0, "presence_db": 0}},
    {"name": "defaults", "clean_bleed": False, "autotune": "off", "gain_db": 0.0,
     "mix": {}},
    {"name": "heavy", "clean_bleed": True, "autotune": "off", "gain_db": 0.0,
     "mix": {"vocal_gain_db": 0, "instrumental_gain_db": 0, "reverb_wet": 0.4,
             "duck_db": 10, "presence_db": 5}},
    {"name": "edge_gains", "clean_bleed": False, "autotune": "off", "gain_db": 12.0,
     "mix": {"vocal_gain_db": 12, "instrumental_gain_db": -12, "reverb_wet": 0,
             "duck_db": 4, "presence_db": 2}},
]


@dataclass
class Row:
    combo: str
    wall_s: float
    mix_bytes: int
    duration_s: float
    diagnostics: dict


def _duration_s(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _stage_fixtures(root: Path) -> None:
    """Copy fixtures into DEV_FS_ROOT so pipeline.download_file() can find them."""
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "separate" / "bench_job").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "recording.webm", root / "uploads" / "recording.webm")
    shutil.copy(FIXTURES_DIR / "vocals.wav", root / "stages" / "separate" / "bench_job" / "vocals.wav")
    shutil.copy(FIXTURES_DIR / "instrumental.wav", root / "stages" / "separate" / "bench_job" / "instrumental.wav")


def _run_combo(combo: dict[str, Any], job_id: str, fs_root: Path) -> Row:
    from record_mix import pipeline
    t0 = time.perf_counter()
    result = pipeline.run(
        job_id=job_id,
        recording_uri=f"gs://{BUCKET}/uploads/recording.webm",
        instrumental_uri=f"gs://{BUCKET}/stages/separate/bench_job/instrumental.wav",
        vocals_uri=f"gs://{BUCKET}/stages/separate/bench_job/vocals.wav",
        autotune=combo["autotune"],
        clean_bleed=combo["clean_bleed"],
        gain_db=combo["gain_db"],
        mix=combo["mix"],
    )
    wall = time.perf_counter() - t0

    mix_object = result["mix_uri"].split(f"{BUCKET}/", 1)[1]
    mix_path = fs_root / mix_object
    dest_dir = RESULTS_DIR / combo["name"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(mix_path, dest_dir / "mix.mp3")

    return Row(
        combo=combo["name"],
        wall_s=wall,
        mix_bytes=mix_path.stat().st_size,
        duration_s=_duration_s(mix_path),
        diagnostics=result["diagnostics"],
    )


def _write_outputs(rows: list[Row]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2, default=float)
    )
    lines = [
        "# record-mix bench\n",
        "| combo | wall (s) | mix (KB) | duration (s) | skipped |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        skipped = ",".join(r.diagnostics.get("skipped", [])) or "—"
        lines.append(
            f"| `{r.combo}` | {r.wall_s:.2f} | {r.mix_bytes/1024:.1f} | "
            f"{r.duration_s:.2f} | {skipped} |"
        )
    lines.append("\n## Applied knobs per combo\n")
    for r in rows:
        lines.append(f"### `{r.combo}`")
        lines.append("```json")
        lines.append(json.dumps(r.diagnostics.get("applied", {}), indent=2, default=float))
        lines.append("```")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    if not (FIXTURES_DIR / "recording.webm").exists():
        raise SystemExit(
            f"fixtures missing under {FIXTURES_DIR}. "
            f"Run fetch_fixtures.py first."
        )
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    fs_root = RESULTS_DIR / "fs"
    fs_root.mkdir()
    _stage_fixtures(fs_root)

    os.environ["DEV_FS_ROOT"] = str(fs_root)
    os.environ["GCS_BUCKET"] = BUCKET

    rows: list[Row] = []
    for i, combo in enumerate(COMBOS):
        job_id = f"{i:012x}" + "a" * 4  # 16 hex chars, per contract
        print(f"[{combo['name']}] running …")
        try:
            row = _run_combo(combo, job_id, fs_root)
            rows.append(row)
            print(f"  wall={row.wall_s:.2f}s size={row.mix_bytes/1024:.1f}KB "
                  f"dur={row.duration_s:.2f}s skipped={row.diagnostics.get('skipped', [])}")
        except Exception as e:
            print(f"[{combo['name']}] FAILED: {type(e).__name__}: {e}")
            rows.append(Row(combo=combo["name"], wall_s=0.0, mix_bytes=0,
                            duration_s=0.0, diagnostics={"error": str(e)}))
        _write_outputs(rows)  # incremental

    print(f"\nwrote {RESULTS_DIR/'results.json'} and {RESULTS_DIR/'summary.md'}")


if __name__ == "__main__":
    main()
