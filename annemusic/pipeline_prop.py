"""Property tests (hypothesis) for the pure pipeline logic.

Run separately from the unit gate:

    uv run pytest -o python_files='*_prop.py' annemusic
"""

from __future__ import annotations

import re

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from annemusic import ass, core, vad

# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #

_TIME = st.floats(allow_nan=False, allow_infinity=False)


@st.composite
def segment_lists(draw) -> list[dict]:
    """Ordered, non-overlapping ASR-style segments with word-token text."""
    n = draw(st.integers(1, 10))
    t = draw(st.floats(0, 10))
    segs = []
    for i in range(n):
        start = t + draw(st.floats(0, 8))
        end = start + draw(st.floats(0.05, 60))
        n_words = draw(st.integers(1, 6))
        segs.append({
            "start": start,
            "end": end,
            "text": " ".join(f"w{i}x{j}" for j in range(n_words)),
        })
        t = end
    return segs


@st.composite
def activities(draw) -> list[dict]:
    """Contiguous alternating vocal-activity regions covering [0, 1000]."""
    edges = draw(st.lists(st.floats(0.5, 999.5), max_size=8, unique=True))
    bounds = [0.0, *sorted(edges), 1000.0]
    offset = draw(st.integers(0, 1))
    return [
        {
            "start": a,
            "end": b,
            "kind": ("vocals", "instrumental")[(i + offset) % 2],
        }
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
    ]


@st.composite
def aligner_items(draw) -> list[dict]:
    """Chunk-relative aligner output, anomalies (ties, monsters) included."""
    n = draw(st.integers(1, 30))
    items = []
    for i in range(n):
        start = draw(st.floats(-1, 40))
        items.append({
            "text": f"w{i}",
            "start": start,
            "end": start + draw(st.floats(0, 35)),
        })
    return items


@st.composite
def word_lists(draw) -> list[dict]:
    """Ordered manifest-style words with strictly positive spans."""
    n = draw(st.integers(0, 25))
    t = 0.0
    words = []
    for i in range(n):
        start = t + draw(st.floats(0, 4))
        end = start + draw(st.floats(0.02, 8))
        words.append({"text": f"w{i}", "start": start, "end": end})
        t = end
    return words


# --------------------------------------------------------------------------- #
# core: chunk planning
# --------------------------------------------------------------------------- #


@given(segment_lists(), activities(), st.floats(30, 300), st.floats(10, 300))
def test_plan_chunks_conserves_segments_and_respects_target(
    segs, va, max_s, target_s
):
    chunks = core.plan_chunks(segs, va, max_seconds=max_s, target_seconds=target_s)
    assert [s for c in chunks for s in c] == segs  # conservation, in order
    assert all(chunks)  # no empty chunk
    target = min(target_s, max_s)
    for c in chunks:
        window = c[-1]["end"] - c[0]["start"]
        assert window <= target or len(c) == 1  # only a lone segment may exceed


@given(segment_lists(), activities())
def test_split_at_vad_breaks_conserves_words(segs, va):
    out = core.split_at_vad_breaks(segs, va)
    original = " ".join(s["text"] for s in segs).split()
    assert " ".join(s["text"] for s in out).split() == original
    for s in out:
        assert s["end"] >= s["start"]


# --------------------------------------------------------------------------- #
# core: repair policy — repaired output is sane or the chunk is rejected
# --------------------------------------------------------------------------- #


@given(aligner_items(), st.floats(0, 100), st.floats(1, 300))
def test_repair_words_output_is_sane_or_rejected(items, chunk_start, chunk_len):
    chunk_end = chunk_start + chunk_len
    try:
        words = core.repair_words(items, chunk_start, chunk_end)
    except core.SanityError:
        return  # rejection is a legal outcome
    assert words
    lo, hi = chunk_start - 0.05, chunk_end + 0.05
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)  # non-decreasing (pipeline-2)
    for w in words:
        assert lo <= w["start"] <= hi
        assert w["start"] <= w["end"] <= hi
        assert w["end"] - w["start"] <= core.MAX_WORD_SPAN_S + 1e-9


@given(segment_lists())
def test_synthesize_words_conserves_tokens_in_order(segs):
    words = core.synthesize_words(segs)
    assert [w["text"] for w in words] == " ".join(s["text"] for s in segs).split()
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(w["end"] > w["start"] for w in words)


@given(word_lists())
def test_clean_words_is_idempotent(words):
    once = core.clean_words(words)
    assert core.clean_words(once) == once


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
               min_size=1, max_size=4))
def test_flow_routing_is_consistent_and_case_insensitive(lang):
    flow = core.flow_for(lang)
    assert flow == core.flow_for(lang.upper())
    assert flow.transcribe_input == core.input_for_backend(flow.transcribe)
    assert flow.transcribe in ("qwen3", "whisper")
    assert flow.align in ("qwen3", "whisperx")


# --------------------------------------------------------------------------- #
# ass: formatting stability
# --------------------------------------------------------------------------- #


@given(st.floats(-5, 100_000, allow_nan=False))
def test_fmt_time_parses_back_to_the_same_centiseconds(t):
    s = ass.fmt_time(t)
    m = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)\.(\d\d)", s)
    assert m, s
    h, mi, sec, cs = map(int, m.groups())
    assert ((h * 60 + mi) * 60 + sec) * 100 + cs == max(0, round(t * 100))


@given(word_lists(), st.integers(1, 10), st.floats(0.1, 5))
def test_group_lines_conserves_words_and_bounds(words, max_words, gap):
    lines = ass.group_lines(words, max_words, gap)
    assert [w for ln in lines for w in ln] == words
    assert all(1 <= len(ln) <= max_words for ln in lines)
    for ln in lines:
        for a, b in zip(ln, ln[1:]):
            assert b["start"] - a["end"] <= gap + 1e-9  # no in-line dead gaps


@given(word_lists())
def test_build_ass_emits_one_kf_dialogue_per_grouped_line(words):
    doc = ass.build_ass(words)
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    expected = len(ass.group_lines(
        words, ass.DEFAULT_STYLE["max_words_per_line"], ass.DEFAULT_STYLE["break_gap"]
    ))
    assert len(dialogue) == expected
    assert all("\\kf" in ln for ln in dialogue)


# --------------------------------------------------------------------------- #
# vad: regions partition the audio
# --------------------------------------------------------------------------- #


def _blocky_audio(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Random silence / noise-floor / singing blocks, like a separated stem."""
    out = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(n, i + int(rng.integers(sr // 10, sr)))
        amp = float(rng.choice([0.0, 0.002, 0.3]))
        out[i:j] = amp * np.sin(
            2 * np.pi * 440 * np.arange(j - i) / sr, dtype=np.float64
        ).astype(np.float32)
        i = j
    return out


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 60_000), st.integers(0, 2**32 - 1))
def test_detect_regions_partition_the_audio(n, seed):
    sr = 16_000
    data = _blocky_audio(np.random.default_rng(seed), n, sr)
    regions = vad.detect_regions(data, sr)
    if n == 0:
        assert regions == []
        return
    assert regions[0]["start"] == 0.0
    assert regions[-1]["end"] == n / float(sr)
    for a, b in zip(regions, regions[1:]):
        assert a["end"] == b["start"]  # contiguous
        assert a["kind"] != b["kind"]  # coalesced
    for r in regions:
        assert r["end"] >= r["start"]
        assert r["kind"] in ("vocals", "instrumental")
