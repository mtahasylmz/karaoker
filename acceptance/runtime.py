"""Memoized CLI runner for acceptance tests.

One real pipeline run is ~7 min on this machine, so identical invocations are
collapsed two ways:

- in-process memo: scenarios sharing a When (pipeline-1/-2/-4) execute once;
- disk cache under .acceptance-cache/ (successful, no-precondition runs
  only): repeated suites and gherkin-mutation sweeps become warm-cache reads.

Invocations are list-argv (never shell): the fixture filename has spaces,
parens and unicode.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".acceptance-cache"
CLI = ROOT / ".venv" / "bin" / "annemusic"

_memo: dict[str, "Result"] = {}


@dataclass
class Result:
    exit_code: int
    stderr: str
    cwd: Path  # directory the CLI ran in; artifacts resolve against this


def fixture() -> Path | None:
    val = os.environ.get("ANNEMUSIC_FIXTURE")
    return Path(val) if val else None


def fixture_lyrics() -> Path | None:
    fx = fixture()
    return fx.parent / "lyrics.txt" if fx else None


def _tokens(command: str) -> list[str]:
    toks = shlex.split(command)
    assert toks and toks[0] == "annemusic", f"unexpected command: {command}"
    fx = fixture()
    return [str(fx) if t == "$ANNEMUSIC_FIXTURE" and fx else t for t in toks[1:]]


def _key(tokens: list[str], with_lyrics: bool) -> str:
    fx = fixture()
    parts: list = [tokens]
    if fx and fx.exists():
        parts.append(fx.stat().st_mtime)
    lyr = fixture_lyrics()
    if with_lyrics and lyr and lyr.exists():
        parts.append(("lyrics", lyr.stat().st_mtime))
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()


def run_cli(command: str, pre_files: list[str] | None = None) -> Result:
    """Run one CLI invocation in a fresh working directory.

    ``pre_files``: relative paths to create before running (Given steps).
    Runs with preconditions or failures are never disk-cached.
    """
    tokens = _tokens(command)
    with_lyrics = "--lyrics" in tokens
    cacheable = not pre_files
    key = _key(tokens, with_lyrics) + ("" if cacheable else ":pre")

    if cacheable and key in _memo:
        return _memo[key]

    slot = CACHE_DIR / key
    if cacheable and (slot / "meta.json").exists():
        meta = json.loads((slot / "meta.json").read_text())
        result = Result(meta["exit_code"], meta["stderr"], slot / "cwd")
        _memo[key] = result
        return result

    cwd = Path(tempfile.mkdtemp(prefix="annemusic-acceptance-"))
    for rel in pre_files or []:
        p = cwd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    lyr = fixture_lyrics()
    if with_lyrics and lyr and lyr.exists():
        shutil.copyfile(lyr, cwd / "lyrics.txt")

    proc = subprocess.run(
        [str(CLI), *tokens], cwd=cwd, capture_output=True, text=True
    )
    result = Result(proc.returncode, proc.stderr, cwd)

    if cacheable:
        _memo[key] = result
        if proc.returncode == 0:
            slot.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cwd, slot / "cwd", dirs_exist_ok=True)
            (slot / "meta.json").write_text(
                json.dumps({"exit_code": proc.returncode, "stderr": proc.stderr})
            )
    return result
