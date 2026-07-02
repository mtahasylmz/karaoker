"""Generate executable acceptance tests from features/*.feature.

Uses the APS Babashka ``gherkin-parser`` (never a reimplementation): clones
github.com/unclebob/Acceptance-Pipeline-Specification into gitignored .aps/
on first use, parses each feature to IR JSON, and emits one pytest file per
feature under acceptance/generated/ (kept separate from unit tests).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APS = ROOT / ".aps"
APS_URL = "https://github.com/unclebob/Acceptance-Pipeline-Specification"
GENERATED = ROOT / "acceptance" / "generated"


def ensure_aps() -> None:
    if not (APS / "bb.edn").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", APS_URL, str(APS)], check=True
        )


def parse(feature: Path) -> dict:
    ir_path = GENERATED / f"{feature.stem}.ir.json"
    subprocess.run(
        ["bb", "gherkin-parser", str(feature.resolve()), str(ir_path.resolve())],
        cwd=str(APS), check=True,
    )
    return json.loads(ir_path.read_text())


def _slug(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()


def emit(ir: dict) -> str:
    lines = [
        f'"""Generated from features/{ir["name"]}.feature — DO NOT EDIT.',
        "",
        "Regenerate with: uv run python acceptance/generate.py",
        '"""',
        "",
        "import pytest",
        "",
        "from acceptance.steps import run_step",
        "",
        "",
        f"BACKGROUND = {json.dumps([[s['keyword'], s['text']] for s in ir.get('background', [])], indent=4)}",
        "",
        "",
        "def _scenario(steps, params=None):",
        "    ctx = {}",
        "    for keyword, text in BACKGROUND + steps:",
        "        run_step(ctx, keyword, text, params or {})",
        "",
    ]
    for sc in ir["scenarios"]:
        steps = json.dumps(
            [[s["keyword"], s["text"]] for s in sc["steps"]], indent=4
        )
        name = f"test_{_slug(sc['name'])}"
        lines.append("")
        if sc.get("examples"):
            keys = sorted(sc["examples"][0])
            rows = [tuple(ex[k] for k in keys) for ex in sc["examples"]]
            lines += [
                f"@pytest.mark.parametrize({json.dumps(','.join(keys))}, {json.dumps(rows)})",
                f"def {name}({', '.join(keys)}):",
                f"    _scenario({steps}, params={{{', '.join(f'{k!r}: {k}' for k in keys)}}})",
                "",
            ]
        else:
            lines += [f"def {name}():", f"    _scenario({steps})", ""]
    return "\n".join(lines)


def main() -> None:
    ensure_aps()
    GENERATED.mkdir(parents=True, exist_ok=True)
    features = sorted((ROOT / "features").glob("*.feature"))
    if not features:
        sys.exit("no features/*.feature found")
    for feature in features:
        ir = parse(feature)
        out = GENERATED / f"test_{feature.stem}.py"
        out.write_text(emit(ir))
        print(f"generated {out.relative_to(ROOT)} "
              f"({len(ir['scenarios'])} scenarios)")


if __name__ == "__main__":
    main()
