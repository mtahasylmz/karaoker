"""_align_qwen3's post-hoc sanity checker: repair vs reject.

Frame-quantized aligners legitimately emit zero-length word spans for short
tokens — the checker must nudge those (observed live: one tied pair used to
throw away an entire chunk's alignment), while genuinely decreasing spans
and out-of-window words still hard-fail to the whisperx fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from align import pipeline
from align.pipeline import _Qwen3SanityError, _align_qwen3


@dataclass
class _Item:
    text: str
    start_time: float
    end_time: float


class _FakeQwen:
    def __init__(self, items):
        self._items = items

    def align(self, audio, text, language):
        return [self._items]


def _run(monkeypatch, tmp_path: Path, items):
    monkeypatch.setattr(pipeline, "_load_qwen", lambda: _FakeQwen(items))
    audio = np.zeros(pipeline._AUDIO_SR * 12, dtype=np.float32)
    chunk = [{"text": "hello world again", "start": 2.0, "end": 10.0}]
    return _align_qwen3(tmp_path, audio, chunk, "en", idx=0)


def test_zero_length_span_is_nudged_not_rejected(monkeypatch, tmp_path):
    words = _run(monkeypatch, tmp_path, [
        _Item("hello", 0.50, 1.00),
        _Item("world", 3.93, 3.93),  # the live A40 case: tied pair
        _Item("again", 5.00, 5.40),
    ])
    assert [w["text"] for w in words] == ["hello", "world", "again"]
    nudged = words[1]
    assert nudged["end"] == pytest.approx(nudged["start"] + 0.02)


def test_decreasing_span_still_rejects(monkeypatch, tmp_path):
    with pytest.raises(_Qwen3SanityError, match="decreasing"):
        _run(monkeypatch, tmp_path, [
            _Item("hello", 0.50, 1.00),
            _Item("world", 4.00, 3.50),
        ])


def test_word_starting_outside_window_is_dropped(monkeypatch, tmp_path):
    # Observed live: a word stamped past the audio slice. Drop it, keep the rest.
    words = _run(monkeypatch, tmp_path, [
        _Item("hello", 0.50, 1.00),
        _Item("ghost", 9.60, 9.90),  # +chunk_start(2.0) → starts at 11.6 > hi(10.05)
        _Item("world", 5.00, 5.40),
    ])
    assert [w["text"] for w in words] == ["hello", "world"]


def test_word_ending_past_window_is_clamped(monkeypatch, tmp_path):
    words = _run(monkeypatch, tmp_path, [
        _Item("hello", 0.50, 1.00),
        _Item("tail", 7.50, 8.50),  # → [9.5, 10.5]; hi = 10.05
    ])
    assert words[1]["end"] == pytest.approx(10.05)


def test_all_words_outside_rejects(monkeypatch, tmp_path):
    with pytest.raises(_Qwen3SanityError, match="survived|systemic"):
        _run(monkeypatch, tmp_path, [
            _Item("ghost", 9.60, 9.90),
        ])


def test_held_note_span_accepted_verbatim(monkeypatch, tmp_path):
    # 7s 'what' — the live A40 case: a held sung note, not garbage.
    words = _run(monkeypatch, tmp_path, [
        _Item("oh", 0.10, 0.50),
        _Item("what", 0.60, 7.60),
    ])
    assert words[1]["end"] - words[1]["start"] == pytest.approx(7.0)


def test_absurd_span_clamped(monkeypatch, tmp_path):
    # chunk window is [2, 10] (8s) + 12s cap; use a tighter window via chunk
    # by checking the clamp against _MAX_WORD_SPAN_S directly on a long chunk.
    audio = np.zeros(pipeline._AUDIO_SR * 40, dtype=np.float32)
    chunk = [{"text": "loooong word", "start": 2.0, "end": 36.0}]
    monkeypatch.setattr(pipeline, "_load_qwen", lambda: _FakeQwen([
        _Item("loooong", 0.10, 30.0),
        _Item("word", 31.0, 31.5),
    ]))
    words = _align_qwen3(tmp_path, audio, chunk, "en", idx=0)
    assert words[0]["end"] - words[0]["start"] == pytest.approx(pipeline._MAX_WORD_SPAN_S)


def test_systemic_repairs_reject(monkeypatch, tmp_path):
    # Every word zero-length -> repairs exceed the 20% budget -> reject.
    items = [_Item(f"w{i}", 0.5 + i * 0.1, 0.5 + i * 0.1) for i in range(20)]
    with pytest.raises(_Qwen3SanityError, match="systemic"):
        _run(monkeypatch, tmp_path, items)
