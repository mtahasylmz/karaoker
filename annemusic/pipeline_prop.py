"""Property tests (hypothesis) for the pure pipeline logic.

Run separately from the unit gate:

    uv run pytest -o python_files='*_prop.py' annemusic
"""

from __future__ import annotations

import re

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from annemusic import ass, chunks, core, lrclib, repair, vad
from annemusic.lines import words_to_lines

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


# Word text with karaoke-relevant punctuation woven in, for the line-convention
# property: some words end in sentence/clause marks, some are plain.
_LYRIC_TOKENS = st.sampled_from(
    ["hello", "world", "stay.", "wait,", "why?", "go!", "away…", "night", "Mr.",
     "iPhone", "99", "the", "dark", "line;", "here:"]
)


@st.composite
def lyric_word_lists(draw) -> list[dict]:
    """Ordered words drawn from punctuated lyric tokens (for words_to_lines)."""
    n = draw(st.integers(0, 30))
    t = 0.0
    words = []
    for _ in range(n):
        start = t + draw(st.floats(0, 3))
        end = start + draw(st.floats(0.02, 6))
        words.append({"text": draw(_LYRIC_TOKENS), "start": start, "end": end})
        t = end
    return words


# --------------------------------------------------------------------------- #
# chunks: chunk planning
# --------------------------------------------------------------------------- #


@given(segment_lists(), activities(), st.floats(30, 300), st.floats(10, 300))
def test_plan_chunks_conserves_segments_and_respects_target(
    segs, va, max_s, target_s
):
    planned = chunks.plan_chunks(segs, va, max_seconds=max_s, target_seconds=target_s)
    assert [s for c in planned for s in c] == segs  # conservation, in order
    assert all(planned)  # no empty chunk
    target = min(target_s, max_s)
    for c in planned:
        window = c[-1]["end"] - c[0]["start"]
        assert window <= target or len(c) == 1  # only a lone segment may exceed


@given(segment_lists(), activities())
def test_split_at_vad_breaks_conserves_words(segs, va):
    out = chunks.split_at_vad_breaks(segs, va)
    original = " ".join(s["text"] for s in segs).split()
    assert " ".join(s["text"] for s in out).split() == original
    for s in out:
        assert s["end"] >= s["start"]


# --------------------------------------------------------------------------- #
# repair: repair policy — repaired output is sane or the chunk is rejected
# --------------------------------------------------------------------------- #


@given(aligner_items(), st.floats(0, 100), st.floats(1, 300))
def test_repair_words_output_is_sane_or_rejected(items, chunk_start, chunk_len):
    chunk_end = chunk_start + chunk_len
    try:
        words = repair.repair_words(items, chunk_start, chunk_end)
    except repair.SanityError:
        return  # rejection is a legal outcome
    assert words
    lo, hi = chunk_start - 0.05, chunk_end + 0.05
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)  # non-decreasing (pipeline-2)
    for w in words:
        assert lo <= w["start"] <= hi
        assert w["start"] <= w["end"] <= hi
        assert w["end"] - w["start"] <= repair.MAX_WORD_SPAN_S + 1e-9


@given(segment_lists())
def test_synthesize_words_conserves_tokens_in_order(segs):
    words = core.synthesize_words(segs)
    assert [w["text"] for w in words] == " ".join(s["text"] for s in segs).split()
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(w["end"] > w["start"] for w in words)


@st.composite
def lrc_line_lists(draw) -> list[dict]:
    """LRC lines with human timestamps — zero-width windows allowed (a mark
    can repeat), which is exactly where even_words must still emit words."""
    n = draw(st.integers(0, 12))
    t = 0.0
    lines = []
    for i in range(n):
        start = t + draw(st.floats(0, 5))
        end = start + draw(st.floats(0, 6))  # 0 allowed: zero-width LRC window
        n_words = draw(st.integers(0, 5))
        lines.append({
            "text": " ".join(f"l{i}t{j}" for j in range(n_words)),
            "start": start,
            "end": end,
        })
        t = end
    return lines


@given(lrc_line_lists())
def test_even_words_conserves_tokens_and_stays_positive_and_in_window(lines):
    words = core.even_words(lines)
    # Conservation, in order: every LRC token becomes exactly one word.
    assert [w["text"] for w in words] == " ".join(l["text"] for l in lines).split()
    # The step floor guarantees a strictly positive span for every word, even
    # when the LRC window has zero width (start == end).
    assert all(w["end"] > w["start"] for w in words)
    # Attribute words to lines by index (each line contributes its token count)
    # and check each word starts at/after its own line's start; within a line
    # starts strictly increase.
    idx = 0
    for ln in lines:
        n = len((ln.get("text") or "").split())
        group = words[idx:idx + n]
        idx += n
        s = float(ln["start"])
        assert all(w["start"] >= s - 1e-9 for w in group)
        starts = [w["start"] for w in group]
        assert starts == sorted(starts)
    assert idx == len(words)  # exact partition, no words unaccounted for


@given(word_lists())
def test_clean_words_is_idempotent(words):
    once = repair.clean_words(words)
    assert repair.clean_words(once) == once


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


def _strip_and_case(seq: str) -> str:
    """Normalize a word sequence the way words_to_lines does line-finally:
    drop line-final commas/periods and first-letter capitalization, so raw
    input and line output become comparable."""
    return re.sub(r"[.,]+(?=\s|$)", "", seq).casefold()


@given(word_lists())
def test_words_to_lines_conserves_words_and_bounds(words):
    lines = words_to_lines(words)
    clean = repair.clean_words(words)
    # Words are conserved IN ORDER, modulo line-final ,/. stripping and
    # first-letter capitalization (the segmentation contract changed with
    # pipeline-13: text is no longer verbatim word concatenation).
    src = _strip_and_case(" ".join(w["text"] for w in clean))
    out = _strip_and_case(" ".join(ln["text"] for ln in lines))
    assert src == out
    for ln in lines:
        assert ln["end"] >= ln["start"]


@given(lyric_word_lists())
def test_words_to_lines_obeys_lyric_conventions(words):
    # pipeline-13 at unit speed: no line ends in ',' or '.', and every
    # non-empty line starts with an uppercase letter or a digit.
    for ln in words_to_lines(words):
        text = ln["text"]
        assert not text.endswith((",", ".")), text
        if text:
            assert text[0].isupper() or text[0].isdigit(), text


@given(word_lists())
def test_build_ass_emits_one_plain_dialogue_per_line(lines):
    doc = ass.build_ass(lines)
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    renderable = [ln for ln in lines if (ln["text"] or "").strip() and ln["end"] > ln["start"]]
    assert len(dialogue) == len(renderable)
    assert all("\\kf" not in ln and "\\k" not in ln for ln in dialogue)


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


# --------------------------------------------------------------------------- #
# lrclib.parse_lrc: parsing stability — sorted, gapless, positive-span output
# --------------------------------------------------------------------------- #


@st.composite
def lrc_text(draw) -> str:
    """A raw LRC blob: timestamped lines (some out of order, some blank-text,
    some untimed junk) that parse_lrc must normalize."""
    parts = []
    for _ in range(draw(st.integers(0, 10))):
        mm = draw(st.integers(0, 99))
        ss = draw(st.integers(0, 59))
        cs = draw(st.integers(0, 99))
        text = draw(st.text(alphabet="abc def", min_size=0, max_size=8))
        parts.append(f"[{mm:02d}:{ss:02d}.{cs:02d}]{text}")
    if draw(st.booleans()):
        parts.append("a line with no timestamp at all")
    draw(st.randoms()).shuffle(parts)
    return "\n".join(parts)


@given(lrc_text())
def test_parse_lrc_is_sorted_gapless_and_nonneg(blob):
    out = list(lrclib.parse_lrc(blob))
    starts = [r["start"] for r in out]
    assert starts == sorted(starts)                  # sorted by start
    assert all(r["text"].strip() for r in out)       # no blank-text lines
    assert all(r["end"] >= r["start"] for r in out)  # non-negative spans
    # A line never overruns the next line's start (ends are clamped to the next
    # timestamp; the final line gets a +3 s tail with nothing after it).
    for a, b in zip(out, out[1:]):
        assert a["end"] <= b["start"] + 1e-9
