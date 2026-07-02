"""CRAP gate: radon cc --json x coverage.py json.

CRAP(f) = comp(f)^2 * (1 - cov(f))^3 + comp(f)

comp = radon cyclomatic complexity, cov = line-coverage fraction over the
function's span. Produce the coverage report first:

    uv run pytest --cov=annemusic --cov-report=json
    uv run python tools/crap.py [--threshold 30] [--coverage coverage.json]

Excluded: *_test.py / *_prop.py (test code) and annemusic/backends.py plus
annemusic/cli.py's __main__ shim — backends is the adapter shell around
GPU/subprocess seams, exercised by the acceptance suite rather than unit
tests, so unit line coverage would misstate its risk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXCLUDE_SUFFIXES = ("_test.py", "_prop.py", "backends.py")


def span_coverage(file_cov: dict, start: int, end: int) -> float:
    executed = set(file_cov["executed_lines"])
    missing = set(file_cov["missing_lines"])
    statements = [n for n in range(start, end + 1) if n in executed or n in missing]
    if not statements:
        return 1.0
    return sum(1 for n in statements if n in executed) / len(statements)


def crap(comp: float, cov: float) -> float:
    return comp * comp * (1.0 - cov) ** 3 + comp


def collect(paths: list[str], coverage_path: Path) -> list[tuple]:
    cov_files = json.loads(coverage_path.read_text())["files"]
    cc = json.loads(
        subprocess.run(
            ["radon", "cc", "-j", *paths], capture_output=True, text=True, check=True
        ).stdout
    )
    rows = []
    for path, blocks in cc.items():
        if path.endswith(EXCLUDE_SUFFIXES):
            continue
        file_cov = cov_files.get(path)
        if file_cov is None:
            continue
        for b in blocks:
            cov = span_coverage(file_cov, b["lineno"], b["endline"])
            rows.append(
                (crap(b["complexity"], cov), b["complexity"], cov, path, b["name"])
            )
    return sorted(rows, reverse=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["annemusic"])
    ap.add_argument("--coverage", default="coverage.json")
    ap.add_argument("--threshold", type=float, default=30.0)
    args = ap.parse_args(argv)

    rows = collect(args.paths or ["annemusic"], Path(args.coverage))
    bad = [r for r in rows if r[0] > args.threshold]
    for score, comp, cov, path, name in rows[:15]:
        flag = " <-- OVER" if score > args.threshold else ""
        print(f"{score:7.1f}  cc={comp:<3d} cov={cov:4.0%}  {path}:{name}{flag}")
    print(f"max CRAP {rows[0][0]:.1f} over {len(rows)} functions "
          f"(threshold {args.threshold:g})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
