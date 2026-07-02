"""Unit tests for the little routing/fallback logic that lives in backends."""

from __future__ import annotations

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
