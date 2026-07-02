"""APS gherkin-mutator persistent runner worker.

Protocol (mutator-spec.md): newline-delimited JSON on stdin/stdout. Each job
supplies a mutated IR (``feature_json``) and the generated-tests dir; the
worker runs the generated entry points against that IR (via the APS_IR env
var the generated tests read) and replies with one JSON outcome line.

Invoke:  bb gherkin-mutator --feature features/pipeline.feature \\
             --generated-dir acceptance/generated --level soft \\
             --runner-worker "uv run python acceptance/mutation_worker.py"

Heavy CLI invocations are deduplicated by acceptance/runtime.py's disk cache,
which is what makes mutation sweeps affordable (~7 min per unique run).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _timeout_seconds(spec: str | None) -> float | None:
    if not spec:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", str(spec).strip())
    if not m:
        return None
    value = float(m.group(1))
    return value * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[m.group(2) or "s"]


def run_job(job: dict) -> dict:
    started = time.time_ns()
    generated = job.get("generated_dir") or str(ROOT / "acceptance" / "generated")
    env = dict(os.environ, APS_IR=str(Path(job["feature_json"]).resolve()))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", generated, "-q", "-x", "-p", "no:cacheprovider"],
            capture_output=True, text=True, env=env, cwd=ROOT,
            timeout=_timeout_seconds(job.get("timeout")),
        )
        outcome = "test_success" if proc.returncode == 0 else "test_failure"
        output, error = (proc.stdout + proc.stderr)[-4000:], ""
    except subprocess.TimeoutExpired:
        outcome, output, error = "infrastructure_error", "", "test run timed out"
    except Exception as e:  # tests could not be started/evaluated
        outcome, output, error = "infrastructure_error", "", f"{type(e).__name__}: {e}"
    return {
        "id": job.get("id"),
        "outcome": outcome,
        "output": output,
        "error": error,
        "duration": time.time_ns() - started,
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = run_job(json.loads(line))
        print(json.dumps(response), flush=True)  # stdout is protocol-only


if __name__ == "__main__":
    main()
