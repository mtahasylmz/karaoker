"""Unit tests for the pure pipeline logic in core.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from annemusic.core import (
    build_manifest,
    even_words,
    flow_for,
    input_for_backend,
    resolve_out_dir,
    synthesize_words,
)


def _seg(start: float, end: float, text: str = "x") -> dict:
    return {"start": start, "end": end, "text": text}


def _item(text, start, end):
    return {"text": text, "start": start, "end": end}


def _vox(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "vocals"}


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
# even_words — LRC line text spread evenly (shares _spread_evenly with
# synthesize_words but floors the step so zero-width windows still yield words)
# --------------------------------------------------------------------------- #


def _lrc(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_even_words_spreads_tokens_uniformly_across_the_line():
    words = even_words([_lrc("one two three four", 0.0, 4.0)])
    assert [w["text"] for w in words] == ["one", "two", "three", "four"]
    assert [w["start"] for w in words] == [0.0, 1.0, 2.0, 3.0]
    assert [w["end"] for w in words] == [1.0, 2.0, 3.0, 4.0]


def test_even_words_covers_multiple_lines_back_to_back():
    words = even_words([_lrc("a b", 0.0, 2.0), _lrc("c d", 2.0, 4.0)])
    assert [w["text"] for w in words] == ["a", "b", "c", "d"]
    assert words[2]["start"] == 2.0 and words[3]["end"] == 4.0


def test_even_words_skips_empty_lines():
    words = even_words([
        _lrc("   ", 0.0, 1.0), _lrc("", 1.0, 2.0), _lrc("sung", 2.0, 3.0),
    ])
    assert [w["text"] for w in words] == ["sung"]


def test_even_words_keeps_end_after_start_for_a_zero_width_line():
    # A degenerate LRC window (start == end) must still yield end > start so the
    # words survive clean_words — this is where even_words diverges from
    # synthesize_words, which drops non-positive spans.
    words = even_words([_lrc("held", 5.0, 5.0)])
    assert len(words) == 1
    assert words[0]["end"] > words[0]["start"]


def test_even_words_empty_input_yields_no_words():
    assert even_words([]) == []


# --------------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------------- #


def test_manifest_has_exactly_the_spec_keys():
    m = build_manifest("asr", "tr", 240.0, [_item("a", 1.0, 2.0)], [_vox(0.0, 240.0)])
    assert set(m) == {"source", "language", "duration", "words", "vocal_activity"}
    assert m["source"] == "asr"
    assert m["language"] == "tr"
    assert m["duration"] == 240.0


def test_manifest_drops_degenerate_words():
    words = [
        _item("good", 1.0, 2.0),
        _item("", 2.0, 3.0),          # empty text
        _item("backwards", 5.0, 5.0),  # zero length
    ]
    m = build_manifest("lrclib", "en", 10.0, words, [])
    assert [w["text"] for w in m["words"]] == ["good"]


# --------------------------------------------------------------------------- #
# CLI preflight decisions
# --------------------------------------------------------------------------- #


def test_resolve_out_dir_default_is_video_stem():
    assert resolve_out_dir("some/dir/My Song (Live).mp4", None) == Path("My Song (Live)")


def test_resolve_out_dir_explicit():
    assert resolve_out_dir("video.mp4", "out") == Path("out")
