"""Unit tests for the pure pipeline logic in core.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from annemusic import core
from annemusic.core import (
    SanityError,
    build_manifest,
    flow_for,
    input_for_backend,
    plan_chunks,
    repair_words,
    resolve_out_dir,
    split_at_vad_breaks,
    synthesize_words,
)


def _seg(start: float, end: float, text: str = "x") -> dict:
    return {"start": start, "end": end, "text": text}


def _instr(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "instrumental"}


def _vox(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "vocals"}


def _flatten(chunks: list[list[dict]]) -> list[dict]:
    return [s for c in chunks for s in c]


# --------------------------------------------------------------------------- #
# flow routing
# --------------------------------------------------------------------------- #


def test_flow_no_language_is_whisper_vocals():
    assert flow_for(None) == ("whisper", "vocals", "whisperx")


def test_flow_turkish_transcribes_qwen3_aligns_whisperx():
    # tr is in Qwen3-ASR's 30 languages but NOT in the aligner's 11.
    assert flow_for("tr") == ("qwen3", "mix", "whisperx")


def test_flow_english_is_full_qwen3():
    assert flow_for("en") == ("qwen3", "mix", "qwen3")


def test_flow_unknown_language_falls_back():
    assert flow_for("xx") == ("whisper", "vocals", "whisperx")


def test_flow_is_case_insensitive():
    assert flow_for("TR").transcribe == "qwen3"


def test_input_for_backend():
    assert input_for_backend("qwen3") == "mix"
    assert input_for_backend("whisper") == "vocals"


# --------------------------------------------------------------------------- #
# plan_chunks
# --------------------------------------------------------------------------- #


def test_plan_empty_input():
    assert plan_chunks([], []) == []


def test_plan_single_chunk_when_short():
    segments = [_seg(i * 2.0, i * 2.0 + 1.5) for i in range(10)]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks == [segments]


def test_plan_splits_at_instrumental():
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [_instr(180.0, 182.0)], max_seconds=300.0)
    assert len(chunks) == 2
    assert chunks[0] == segments[:3]
    assert chunks[1] == segments[3:]
    assert _flatten(chunks) == segments


def test_plan_splits_at_last_boundary_without_vad():
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks[0] == segments[:5]
    assert chunks[1] == segments[5:]


def test_plan_target_seconds_packs_smaller_than_max():
    # The hard-won A40 fact: pack to ~120 s even though the cap is 300 s.
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [], max_seconds=300.0, target_seconds=120.0)
    for c in chunks:
        assert c[-1]["end"] - c[0]["start"] <= 120.0 or len(c) == 1
    assert _flatten(chunks) == segments


def test_plan_single_oversize_segment_kept_whole():
    segments = [_seg(0.0, 400.0, "long")]
    assert plan_chunks(segments, [], max_seconds=300.0) == [segments]


def test_plan_oversize_then_normal():
    segments = [_seg(0.0, 400.0, "long"), _seg(400.0, 410.0, "short")]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks == [segments[:1], segments[1:]]


def test_plan_preserves_order_and_count():
    segments = [_seg(i * 30.0, i * 30.0 + 25.0) for i in range(40)]
    va = [_instr(150.0, 155.0), _instr(600.0, 610.0), _vox(0.0, 150.0)]
    chunks = plan_chunks(segments, va, max_seconds=300.0)
    assert _flatten(chunks) == segments
    for c in chunks:
        assert c[-1]["end"] - c[0]["start"] <= 300.0 or len(c) == 1


# --------------------------------------------------------------------------- #
# split_at_vad_breaks
# --------------------------------------------------------------------------- #


def test_split_passthrough_single_region():
    out = split_at_vad_breaks([_seg(20.0, 80.0, "alpha beta gamma")], [_vox(10.0, 90.0)])
    assert out == [_seg(20.0, 80.0, "alpha beta gamma")]


def test_split_passthrough_no_vocals_regions():
    segs = [_seg(0.0, 50.0, "hello world")]
    assert split_at_vad_breaks(segs, [_instr(0.0, 50.0)]) == segs


def test_split_single_coarse_segment_at_two_regions():
    # The real qwen3-transcribe shape: one segment spanning the whole song.
    va = [
        _instr(0.0, 20.0),
        _vox(20.0, 80.0),
        _instr(80.0, 120.0),
        _vox(120.0, 220.0),
        _instr(220.0, 222.0),
    ]
    words = [f"w{i}" for i in range(160)]
    out = split_at_vad_breaks([_seg(0.0, 222.0, " ".join(words))], va)
    assert len(out) == 2
    assert (out[0]["start"], out[0]["end"]) == (20.0, 80.0)
    assert (out[1]["start"], out[1]["end"]) == (120.0, 220.0)
    # Proportional: 60/160s of vocal time -> round(160*0.375)=60 words.
    assert len(out[0]["text"].split()) == 60
    assert out[0]["text"].split() + out[1]["text"].split() == words


def test_split_clips_segment_to_regions():
    out = split_at_vad_breaks(
        [_seg(0.0, 100.0, "a b c d e f")], [_vox(10.0, 40.0), _vox(60.0, 90.0)]
    )
    assert len(out) == 2
    assert (out[0]["start"], out[0]["end"]) == (10.0, 40.0)
    assert (out[1]["start"], out[1]["end"]) == (60.0, 90.0)
    assert " ".join(s["text"] for s in out).split() == ["a", "b", "c", "d", "e", "f"]


# --------------------------------------------------------------------------- #
# repair_words — the qwen3 sanity policy: repair isolated, reject systemic
# --------------------------------------------------------------------------- #

# All tests use chunk window [2.0, 10.0]; item times are relative to the slice.


def _repair(items):
    return repair_words(items, chunk_start=2.0, chunk_end=10.0)


def _item(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_zero_length_span_is_nudged_not_rejected():
    words = _repair([
        _item("hello", 0.50, 1.00),
        _item("world", 3.93, 3.93),  # the live A40 case: tied pair
        _item("again", 5.00, 5.40),
    ])
    assert [w["text"] for w in words] == ["hello", "world", "again"]
    assert words[1]["end"] == pytest.approx(words[1]["start"] + 0.02)


def test_words_are_rebased_to_chunk_start():
    words = _repair([_item("hello", 0.50, 1.00)])
    assert words[0]["start"] == pytest.approx(2.50)
    assert words[0]["end"] == pytest.approx(3.00)


def test_decreasing_span_still_rejects():
    with pytest.raises(SanityError, match="decreasing"):
        _repair([_item("hello", 0.50, 1.00), _item("world", 4.00, 3.50)])


def test_word_starting_outside_window_is_dropped():
    words = _repair([
        _item("hello", 0.50, 1.00),
        _item("ghost", 9.60, 9.90),  # +2.0 -> starts at 11.6 > hi(10.05)
        _item("world", 5.00, 5.40),
    ])
    assert [w["text"] for w in words] == ["hello", "world"]


def test_word_ending_past_window_is_clamped():
    words = _repair([
        _item("hello", 0.50, 1.00),
        _item("tail", 7.50, 8.50),  # -> [9.5, 10.5]; hi = 10.05
    ])
    assert words[1]["end"] == pytest.approx(10.05)


def test_tail_pull_in_never_retracts_below_previous_start():
    # Two words crowded at the window edge: the second ends past hi and gets
    # pulled in; its start must not drop below the first's (pipeline-2).
    words = repair_words(
        [
            _item("early", 1.0, 1.5),
            _item("edge", 10.04, 10.05),   # tie at the very edge
            _item("tail", 10.045, 10.4),   # ends past hi=10.05 -> pulled in
        ],
        chunk_start=0.0,
        chunk_end=10.0,
    )
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)


def test_all_words_outside_rejects():
    with pytest.raises(SanityError, match="survived|systemic"):
        _repair([_item("ghost", 9.60, 9.90)])


def test_held_note_span_accepted_verbatim():
    # 7 s held sung note is real singing, not garbage.
    words = _repair([_item("oh", 0.10, 0.50), _item("what", 0.60, 7.60)])
    assert words[1]["end"] - words[1]["start"] == pytest.approx(7.0)


def test_absurd_span_clamped():
    words = repair_words(
        [_item("loooong", 0.10, 30.0), _item("word", 31.0, 31.5)],
        chunk_start=2.0,
        chunk_end=36.0,
    )
    assert words[0]["end"] - words[0]["start"] == pytest.approx(core.MAX_WORD_SPAN_S)


def test_systemic_repairs_reject():
    items = [_item(f"w{i}", 0.5 + i * 0.1, 0.5 + i * 0.1) for i in range(20)]
    with pytest.raises(SanityError, match="systemic"):
        _repair(items)


def test_missing_fields_reject():
    with pytest.raises(SanityError, match="missing"):
        _repair([{"text": "hello", "start": 0.5, "end": None}])


# --------------------------------------------------------------------------- #
# synthesize_words
# --------------------------------------------------------------------------- #


def test_synthesize_even_split():
    words = synthesize_words([_seg(10.0, 14.0, "a b c d")])
    assert [w["text"] for w in words] == ["a", "b", "c", "d"]
    assert words[0]["start"] == pytest.approx(10.0)
    assert words[-1]["end"] == pytest.approx(14.0)
    assert all(w["end"] - w["start"] == pytest.approx(1.0) for w in words)


def test_synthesize_skips_empty_and_degenerate():
    assert synthesize_words([_seg(0.0, 5.0, "  "), _seg(5.0, 5.0, "x")]) == []


# --------------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------------- #


def test_manifest_has_exactly_the_spec_keys():
    m = build_manifest("tr", 240.0, [_item("a", 1.0, 2.0)], [_vox(0.0, 240.0)])
    assert set(m) == {"language", "duration", "words", "vocal_activity"}
    assert m["language"] == "tr"
    assert m["duration"] == 240.0


def test_manifest_drops_degenerate_words():
    words = [
        _item("good", 1.0, 2.0),
        _item("", 2.0, 3.0),          # empty text
        _item("backwards", 5.0, 5.0),  # zero length
    ]
    m = build_manifest("en", 10.0, words, [])
    assert [w["text"] for w in m["words"]] == ["good"]


# --------------------------------------------------------------------------- #
# CLI preflight decisions
# --------------------------------------------------------------------------- #


def test_resolve_out_dir_default_is_video_stem():
    assert resolve_out_dir("some/dir/My Song (Live).mp4", None) == Path("My Song (Live)")


def test_resolve_out_dir_explicit():
    assert resolve_out_dir("video.mp4", "out") == Path("out")
