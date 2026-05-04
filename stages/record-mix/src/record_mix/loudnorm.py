"""Two-pass EBU R128 loudnorm: measurement pass + deterministic second pass.

Pass 1 runs `loudnorm=...:print_format=json` against a null sink. ffmpeg prints
a JSON block to stderr at the end that includes `input_i`, `input_tp`,
`input_lra`, `input_thresh`, and `target_offset`. We parse that block and feed
the measurements back in as `measured_*` on pass 2, with `linear=true` so
pass 2 applies a single deterministic gain curve.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


VOCAL_TARGETS = {"I": -16.0, "TP": -1.5, "LRA": 11.0}
INSTRUMENTAL_TARGETS = {"I": -14.0, "TP": -1.5, "LRA": 11.0}

_REQUIRED_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def measure(input_wav: Path, *, targets: dict[str, float] = VOCAL_TARGETS) -> dict:
    """Run ffmpeg's loudnorm measurement pass; return the parsed measurements."""
    af = (
        f"loudnorm=I={targets['I']}:TP={targets['TP']}:LRA={targets['LRA']}"
        ":print_format=json"
    )
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(input_wav), "-af", af, "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"loudnorm measure failed ({r.returncode}): stderr={r.stderr[-2000:]}"
        )
    return _parse_measurement(r.stderr)


def second_pass_filter(
    measured: dict,
    *,
    targets: dict[str, float] = VOCAL_TARGETS,
) -> str:
    """Build the pass-2 loudnorm filter string with measured_* params baked in."""
    return (
        f"loudnorm=I={targets['I']}:TP={targets['TP']}:LRA={targets['LRA']}"
        f":measured_I={float(measured['input_i'])}"
        f":measured_TP={float(measured['input_tp'])}"
        f":measured_LRA={float(measured['input_lra'])}"
        f":measured_thresh={float(measured['input_thresh'])}"
        f":offset={float(measured['target_offset'])}"
        f":linear=true"
        f":print_format=summary"
    )


def _parse_measurement(stderr: str) -> dict:
    # ffmpeg prints several JSON-ish lines; the loudnorm block is the last
    # balanced `{ ... }` chunk. Grab the largest trailing one.
    blocks = re.findall(r"\{[^{}]*\}", stderr, flags=re.DOTALL)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if all(k in data for k in _REQUIRED_KEYS):
            return data
    raise RuntimeError(
        f"loudnorm: could not parse measurement JSON from ffmpeg stderr; "
        f"tail={stderr[-500:]}"
    )


__all__ = [
    "VOCAL_TARGETS",
    "INSTRUMENTAL_TARGETS",
    "measure",
    "second_pass_filter",
]
