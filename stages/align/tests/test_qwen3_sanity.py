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


def test_out_of_window_word_still_rejects(monkeypatch, tmp_path):
    with pytest.raises(_Qwen3SanityError, match="outside chunk"):
        _run(monkeypatch, tmp_path, [
            _Item("hello", 0.50, 9.50),  # +chunk_start(2.0) → end 11.5 > hi
        ])
