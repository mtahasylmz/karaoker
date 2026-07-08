"""Unit tests for repair.py — the qwen3 sanity policy: repair isolated,
reject systemic. Moved verbatim from core_test.py when repair.py was split
out of core.py."""

from __future__ import annotations

import pytest

from annemusic.repair import MAX_WORD_SPAN_S, SanityError, repair_words

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
    assert words[0]["end"] - words[0]["start"] == pytest.approx(MAX_WORD_SPAN_S)


def test_systemic_repairs_reject():
    items = [_item(f"w{i}", 0.5 + i * 0.1, 0.5 + i * 0.1) for i in range(20)]
    with pytest.raises(SanityError, match="systemic"):
        _repair(items)


def test_missing_fields_reject():
    with pytest.raises(SanityError, match="missing"):
        _repair([{"text": "hello", "start": 0.5, "end": None}])
