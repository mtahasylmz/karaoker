"""Unit tests for the little routing/fallback logic that lives in backends."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from annemusic import backends


class _FakeInfo:
    language = "tr"
    language_probability = 0.9


class _FakeWhisper:
    """Rejects explicit language codes it doesn't know, like faster-whisper."""

    VALID = {"tr", "en"}

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, **kwargs):
        self.calls.append(language)
        if language is not None and language not in self.VALID:
            raise ValueError(f"{language!r} is not a valid language code")
        seg = type("Seg", (), {"text": "la la", "start": 0.0, "end": 2.0})()
        return iter([seg]), _FakeInfo()


def test_whisper_invalid_language_hint_falls_back_to_autodetect(monkeypatch):
    fake = _FakeWhisper()
    monkeypatch.setattr(backends, "_load_whisper", lambda: fake)
    lang, segments = backends._transcribe_whisper(Path("v.wav"), "xx", None)
    assert fake.calls == ["xx", None]  # retried without the bad hint
    assert lang == "tr"
    assert segments and segments[0]["text"] == "la la"


def test_whisper_valid_language_hint_used_directly(monkeypatch):
    fake = _FakeWhisper()
    monkeypatch.setattr(backends, "_load_whisper", lambda: fake)
    lang, _ = backends._transcribe_whisper(Path("v.wav"), "en", None)
    assert fake.calls == ["en"]
    assert lang == "tr"


def test_whisper_autodetect_valueerror_is_not_swallowed(monkeypatch):
    class _Broken:
        def transcribe(self, audio, language=None, **kwargs):
            raise ValueError("decode failed")

    monkeypatch.setattr(backends, "_load_whisper", lambda: _Broken())
    with pytest.raises(ValueError, match="decode failed"):
        backends._transcribe_whisper(Path("v.wav"), None, None)


# --------------------------------------------------------------------------- #
# Whisper-fallback hallucination hardening (slice 3, unit-level, no scenario)
# --------------------------------------------------------------------------- #


def _seg(text, start, end, no_speech_prob=0.0):
    return type(
        "Seg", (),
        {"text": text, "start": start, "end": end, "no_speech_prob": no_speech_prob},
    )()


class _CapturingWhisper:
    """Records the transcribe kwargs and returns a fixed segment list."""

    def __init__(self, segments):
        self.kwargs = None
        self._segments = segments

    def transcribe(self, audio, language=None, **kwargs):
        self.kwargs = kwargs
        return iter(self._segments), _FakeInfo()


def _run_whisper(monkeypatch, segments, lyrics=None):
    fake = _CapturingWhisper(segments)
    monkeypatch.setattr(backends, "_load_whisper", lambda: fake)
    _, out = backends._transcribe_whisper(Path("v.wav"), "tr", lyrics)
    return fake, out


def test_whisper_disables_condition_on_previous_text(monkeypatch):
    # Stops hallucination cascades on repetitive sung lines.
    fake, _ = _run_whisper(monkeypatch, [_seg("la la", 0.0, 2.0)])
    assert fake.kwargs["condition_on_previous_text"] is False


def test_whisper_seeds_lyrics_prompt_without_bias(monkeypatch):
    fake, _ = _run_whisper(monkeypatch, [_seg("la la", 0.0, 2.0)])
    assert fake.kwargs["initial_prompt"] == "lyrics:"


def test_whisper_merges_known_lyrics_into_prompt(monkeypatch):
    fake, _ = _run_whisper(monkeypatch, [_seg("la", 0.0, 2.0)], lyrics="gerçek sözler")
    assert fake.kwargs["initial_prompt"] == "lyrics: gerçek sözler"


def test_whisper_clamps_merged_prompt_to_200_chars(monkeypatch):
    long = "x" * 500
    fake, _ = _run_whisper(monkeypatch, [_seg("la", 0.0, 2.0)], lyrics=long)
    assert fake.kwargs["initial_prompt"] == "lyrics: " + "x" * 200


def test_whisper_drops_high_no_speech_prob_segments(monkeypatch):
    segs = [
        _seg("real words", 0.0, 2.0, no_speech_prob=0.5),
        _seg("silence hallucination", 2.0, 4.0, no_speech_prob=0.95),
    ]
    _, out = _run_whisper(monkeypatch, segs)
    assert [s["text"] for s in out] == ["real words"]


def test_whisper_keeps_borderline_no_speech_prob(monkeypatch):
    # Exactly 0.9 is kept; only > 0.9 is dropped.
    segs = [_seg("edge segment here", 0.0, 2.0, no_speech_prob=0.9)]
    _, out = _run_whisper(monkeypatch, segs)
    assert [s["text"] for s in out] == ["edge segment here"]


def test_whisper_drops_dense_char_rate_segments(monkeypatch):
    # 100 chars over 1 s = 100 chars/s, far above the 37.5 cap: garbled dense
    # hallucination, dropped.
    segs = [
        _seg("normal line", 0.0, 2.0),          # ~5.5 chars/s, kept
        _seg("q" * 100, 5.0, 6.0),              # 100 chars/s, dropped
    ]
    _, out = _run_whisper(monkeypatch, segs)
    assert [s["text"] for s in out] == ["normal line"]


def test_whisper_keeps_normal_char_rate(monkeypatch):
    # 37 chars over 1 s = 37 chars/s, under the 37.5 cap: kept.
    segs = [_seg("a" * 37, 0.0, 1.0)]
    _, out = _run_whisper(monkeypatch, segs)
    assert [s["text"] for s in out] == ["a" * 37]


def test_pick_device_env_override(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DEVICE", "cpu")
    assert backends._pick_device() == "cpu"


def test_separate_unknown_model_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="unknown SEPARATE_MODEL"):
        backends.separate(tmp_path / "mix.wav", tmp_path, model="nope")


def test_run_cmd_failure_raises_with_stderr(tmp_path):
    with pytest.raises(RuntimeError, match="command failed"):
        backends._run_cmd(
            [sys.executable, "-c", "import sys; print('why', file=sys.stderr); sys.exit(3)"]
        )


def test_run_cmd_success_returns_process():
    proc = backends._run_cmd([sys.executable, "-c", "print('hi')"])
    assert proc.stdout.strip() == "hi"
