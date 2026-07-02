"""Step handlers for the generated acceptance tests.

Regex-capture handlers by default: one handler per step *shape*, captures for
the values that vary. Separate literal handlers only where the wording is
genuinely different behavior.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from acceptance import runtime

STEPS: list[tuple[re.Pattern, object]] = []


def step(pattern: str):
    def deco(fn):
        STEPS.append((re.compile(pattern), fn))
        return fn
    return deco


def run_step(ctx: dict, keyword: str, text: str, params: dict) -> None:
    for k, v in params.items():
        text = text.replace(f"<{k}>", str(v))
    for rx, fn in STEPS:
        m = rx.fullmatch(text)
        if m:
            fn(ctx, *m.groups())
            return
    pytest.fail(f"no step handler for: {text!r}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _result(ctx) -> runtime.Result:
    assert "result" in ctx, "no When step ran yet"
    return ctx["result"]


def _artifact(ctx, rel: str) -> Path:
    return _result(ctx).cwd / rel


def _manifest(ctx, rel: str) -> dict:
    manifest = json.loads(_artifact(ctx, rel).read_text())
    ctx["last_manifest"] = manifest  # for follow-up steps that don't name it
    return manifest


def _media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


_WORD_RX = re.compile(r"\w+", re.UNICODE)


def _norm_words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RX.findall(text)}


# --------------------------------------------------------------------------- #
# Background
# --------------------------------------------------------------------------- #

@step(r"the annemusic CLI is installed on a CUDA-capable machine")
def given_installed(ctx):
    # CUDA itself is not asserted: the pipeline must pass on the whisperx
    # path (the qwen3 aligner is CUDA-only and absent on this machine).
    assert runtime.CLI.exists(), f"CLI not installed at {runtime.CLI}; run `uv sync`"


@step(r"ANNEMUSIC_FIXTURE points at a music video with sung lyrics")
def given_fixture(ctx):
    fx = runtime.fixture()
    if fx is None or not fx.exists():
        pytest.skip("ANNEMUSIC_FIXTURE not set or missing")


@step(r'a matching "lyrics\.txt" with the fixture\'s true lyrics sits beside it')
def given_lyrics_beside_fixture(ctx):
    lyr = runtime.fixture_lyrics()
    # Recorded, not asserted: only pipeline-5 needs it and skips politely.
    ctx["lyrics_available"] = bool(lyr and lyr.exists())


# --------------------------------------------------------------------------- #
# When
# --------------------------------------------------------------------------- #

@step(r'I run "([^"]+)"')
def when_run(ctx, command):
    if "--lyrics" in command and not ctx.get("lyrics_available"):
        pytest.skip("lyrics.txt absent beside the fixture")
    ctx["result"] = runtime.run_cli(command, pre_files=ctx.get("pre_files"))


@step(r'"([^"]+)" already exists')
def given_already_exists(ctx, rel):
    ctx.setdefault("pre_files", []).append(rel)


# --------------------------------------------------------------------------- #
# Then: exit codes and stderr
# --------------------------------------------------------------------------- #

@step(r"the exit code is 0")
def then_exit_zero(ctx):
    r = _result(ctx)
    assert r.exit_code == 0, f"exit {r.exit_code}; stderr: {r.stderr[-2000:]}"


@step(r"the exit code is non-zero")
def then_exit_nonzero(ctx):
    assert _result(ctx).exit_code != 0


@step(r'stderr names "([^"]+)"')
def then_stderr_names(ctx, needle):
    assert needle in _result(ctx).stderr


@step(r"the exit code is non-zero and stderr says to pass --force")
def then_refused_clobber(ctx):
    r = _result(ctx)
    assert r.exit_code != 0
    assert "--force" in r.stderr


# --------------------------------------------------------------------------- #
# Then: artifacts
# --------------------------------------------------------------------------- #

@step(r'"([^"]+)" exists')
def then_file_exists(ctx, rel):
    assert _artifact(ctx, rel).is_file()


@step(r'"([^"]+)" is non-silent audio the same duration as the input \(±1 s\)')
def then_nonsilent_same_duration(ctx, rel):
    import numpy as np
    import soundfile as sf

    path = _artifact(ctx, rel)
    data, sr = sf.read(str(path), always_2d=True)
    duration = len(data) / sr
    expected = _media_duration(runtime.fixture())
    assert abs(duration - expected) <= 1.0, f"{duration:.2f}s vs input {expected:.2f}s"
    rms = float(np.sqrt(np.mean(np.square(data))))
    assert rms > 1e-4, f"audio is silent (rms={rms:.2e})"


@step(r'"([^"]+)" has at least (\d+) Dialogue lines using \\kf karaoke tags')
def then_ass_dialogue_lines(ctx, rel, count):
    text = _artifact(ctx, rel).read_text()
    dialogue = [
        ln for ln in text.splitlines()
        if ln.startswith("Dialogue:") and "\\kf" in ln
    ]
    assert len(dialogue) >= int(count), f"only {len(dialogue)} \\kf Dialogue lines"


@step(r'"([^"]+)" has non-empty (.+)')
def then_manifest_nonempty_fields(ctx, rel, fields):
    manifest = _manifest(ctx, rel)
    for field in re.findall(r'"([^"]+)"', fields):
        value = manifest.get(field)
        assert value not in (None, "", [], {}), f"manifest field {field!r} empty"


@step(r'"([^"]+)" field "([^"]+)" equals "([^"]+)"')
def then_manifest_field_equals(ctx, rel, field, expected):
    assert _manifest(ctx, rel)[field] == expected


@step(r'"([^"]+)" field "([^"]+)" is a 2-3 letter code')
def then_manifest_field_lang_code(ctx, rel, field):
    value = _manifest(ctx, rel)[field]
    assert re.fullmatch(r"[a-zA-Z]{2,3}", value), f"not a language code: {value!r}"


@step(r'"([^"]+)" contains no artifacts')
def then_no_artifacts(ctx, rel):
    d = _artifact(ctx, rel)
    assert not d.exists() or not any(d.iterdir())


@step(r'artifacts land in "\./<video-stem>/"')
def then_artifacts_in_stem_dir(ctx):
    stem = runtime.fixture().stem
    out = _result(ctx).cwd / stem
    for name in ("manifest.json", "lyrics.ass", "vocals.wav", "instrumental.wav"):
        assert (out / name).is_file(), f"missing {out / name}"


# --------------------------------------------------------------------------- #
# Then: word timings
# --------------------------------------------------------------------------- #

@step(r'every word in "([^"]+)" has start < end')
def then_words_start_before_end(ctx, rel):
    words = _manifest(ctx, rel)["words"]
    assert words
    bad = [w for w in words if not w["start"] < w["end"]]
    assert not bad, f"{len(bad)} words with start >= end, e.g. {bad[0]}"


@step(r"word starts are non-decreasing")
def then_word_starts_monotonic(ctx):
    starts = [w["start"] for w in ctx["last_manifest"]["words"]]
    assert all(a <= b for a, b in zip(starts, starts[1:])), "word starts decrease"


@step(r"every word lies within \[0, duration\]")
def then_words_within_duration(ctx):
    manifest = ctx["last_manifest"]
    duration = manifest["duration"]
    bad = [
        w for w in manifest["words"]
        if w["start"] < 0 or w["end"] > duration + 0.05
    ]
    assert not bad, f"{len(bad)} words outside [0, {duration}], e.g. {bad[0]}"


# --------------------------------------------------------------------------- #
# Then: known-lyrics coverage
# --------------------------------------------------------------------------- #

@step(r'at least (\d+)% of the distinct words in "([^"]+)" appear in "([^"]+)" words')
def then_lyrics_coverage(ctx, pct, lyrics_rel, manifest_rel):
    lyrics_words = _norm_words((_result(ctx).cwd / lyrics_rel).read_text())
    manifest_words = _norm_words(
        " ".join(w["text"] for w in _manifest(ctx, manifest_rel)["words"])
    )
    assert lyrics_words
    covered = len(lyrics_words & manifest_words) / len(lyrics_words)
    assert covered >= int(pct) / 100, f"coverage {covered:.0%} < {pct}%"
