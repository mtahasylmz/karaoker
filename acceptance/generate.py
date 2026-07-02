"""APS acceptance entrypoint generator (+ convenience driver).

APS form (github.com/unclebob/Acceptance-Pipeline-Specification):

    python acceptance/generate.py <json-ir> <generated-test-output>

reads parser-produced JSON IR and writes one pytest entry-point file whose
test functions are tied to scenario STRUCTURE (names + example row count),
never to literal example values: at runtime the tests load whatever IR the
``APS_IR`` env var points at (default: the IR beside the generated file), so
the gherkin-mutator can re-run the same entry points against mutated IR.
Also writes ``metadata/<feature-metadata-name>.json`` with an
``implementation_hash`` over the generated files.

No arguments: parse every features/*.feature with the APS Babashka
``gherkin-parser`` (cloned into gitignored .aps/ on first use — never
reimplemented) and generate into acceptance/generated/.
"""

from __future__ import annotations

import hashlib
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


def parse(feature: Path, ir_path: Path) -> None:
    ensure_aps()
    subprocess.run(
        ["bb", "gherkin-parser", str(feature.resolve()), str(ir_path.resolve())],
        cwd=str(APS), check=True,
    )


def _slug(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()


def _metadata_name(feature_path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", feature_path.lower()).strip("-") + ".json"


def emit(ir: dict, ir_path: Path) -> str:
    header = f'''"""Generated acceptance entry points for feature {ir["name"]!r} — DO NOT EDIT.

Regenerate with: uv run python acceptance/generate.py

Entry points are tied to scenario structure only; step text and example
values come from the IR loaded at runtime (env APS_IR overrides, for
gherkin-mutator runs against mutated IR).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, {str(ROOT)!r})

from acceptance.steps import run_step  # noqa: E402

_DEFAULT_IR = Path(__file__).parent / {ir_path.name!r}
IR = json.loads(Path(os.environ.get("APS_IR", _DEFAULT_IR)).read_text())
SCENARIOS = {{s["name"]: s for s in IR["scenarios"]}}
BACKGROUND = IR.get("background", [])


def _scenario(name, example_index=None):
    sc = SCENARIOS[name]
    examples = sc.get("examples") or []
    params = examples[example_index] if example_index is not None else {{}}
    ctx = {{}}
    for step in BACKGROUND + sc["steps"]:
        run_step(ctx, step["keyword"], step["text"], params)

'''
    blocks: list[str] = []
    for sc in ir["scenarios"]:
        name = f"test_{_slug(sc['name'])}"
        examples = sc.get("examples") or []
        if examples:
            indices = list(range(len(examples)))
            blocks.append(
                f"@pytest.mark.parametrize(\"example_index\", {indices})\n"
                f"def {name}(example_index):\n"
                f"    _scenario({sc['name']!r}, example_index)\n"
            )
        else:
            blocks.append(
                f"def {name}():\n    _scenario({sc['name']!r})\n"
            )
    return header + "\n\n".join(blocks)


def generate(ir_path: Path, out: Path) -> Path:
    """The APS entrypoint-generator: <json-ir> -> generated pytest file."""
    ir = json.loads(ir_path.read_text())
    if out.suffix == ".py":
        out_file, out_dir = out, out.parent
    else:
        out_dir = out
        out_file = out / f"test_{_slug(ir['name'])}.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Generated tests default to an IR copy beside them.
    local_ir = out_dir / ir_path.name
    if ir_path.resolve() != local_ir.resolve():
        local_ir.write_text(json.dumps(ir, indent=2))
    content = emit(ir, local_ir)
    out_file.write_text(content)

    feature_path = f"features/{ir['name']}.feature"
    digest = hashlib.sha256(content.encode()).hexdigest()
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    (meta_dir / _metadata_name(feature_path)).write_text(json.dumps({
        "schema_version": 1,
        "feature_path": feature_path,
        "ir_path": str(local_ir.relative_to(ROOT)) if local_ir.is_relative_to(ROOT) else str(local_ir),
        "implementation_hash": f"sha256:{digest}",
        "hash_scope": "generated_files",
        "generated_files": [
            str(out_file.relative_to(ROOT)) if out_file.is_relative_to(ROOT) else str(out_file)
        ],
    }, indent=2))
    return out_file


def main(argv: list[str]) -> int:
    if len(argv) == 2:  # APS command shape: <json-ir> <generated-test-output>
        out = generate(Path(argv[0]), Path(argv[1]))
        print(f"generated {out}")
        return 0
    if argv:
        print("usage: generate.py [<json-ir> <generated-test-output>]", file=sys.stderr)
        return 2
    features = sorted((ROOT / "features").glob("*.feature"))
    if not features:
        print("no features/*.feature found", file=sys.stderr)
        return 1
    GENERATED.mkdir(parents=True, exist_ok=True)
    for feature in features:
        ir_path = GENERATED / f"{feature.stem}.ir.json"
        parse(feature, ir_path)
        out = generate(ir_path, GENERATED)
        print(f"generated {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
