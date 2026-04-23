"""Dispatcher tests. Stubs out the two ASR backends and GCS download —
exercises only pipeline.py's routing + fallback logic, not the models."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcribe import pipeline, qwen3


# ---------- shared fixtures ----------


@pytest.fixture(autouse=True)
def _isolate_qwen3_singleton():
    """Some tests monkeypatch qwen3._load; make sure one test doesn't leak
    a cached handle into the next one."""
    qwen3.reset_for_tests()
    yield
    qwen3.reset_for_tests()


def _stub_download(monkeypatch):
    """Stub shared.download_file so the pipeline doesn't try real GCS. Creates
    an empty file at the requested path so anything that stats it is happy."""

    def fake(object_path: str, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).touch()
        return Path(local_path)

    monkeypatch.setattr("transcribe.pipeline.download_file", fake)


def _stub_vad(monkeypatch):
    """Deterministic VAD output so assertions don't care about audio content."""
    regions = [{"start": 0.0, "end": 10.0, "kind": "vocals"}]
    monkeypatch.setattr("transcribe.pipeline.vad.detect", lambda _p: regions)
    return regions


def _stub_whisper(monkeypatch, text: str = "whisper text", lang: str = "tr"):
    """Replace pipeline._load_model with a fake that returns a canned result."""

    class _FakeSeg:
        def __init__(self, text: str, start: float, end: float):
            self.text = text
            self.start = start
            self.end = end

    class _FakeInfo:
        language = lang
        language_probability = 0.99

    class _FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return iter([_FakeSeg(text, 0.0, 3.0)]), _FakeInfo()

    monkeypatch.setattr("transcribe.pipeline._load_model", lambda: _FakeModel())


# ---------- tests ----------


def test_qwen3_backend_when_flag_true(monkeypatch):
    """Language routes to qwen3 and the flag is on → response has source=qwen3."""
    _stub_download(monkeypatch)
    _stub_vad(monkeypatch)
    monkeypatch.setattr(pipeline, "_QWEN3_AVAILABLE", True)

    def fake_transcribe(_audio, _lang, _lyrics):
        return [{"text": "qwen sang", "start": 0.0, "end": 2.5}], "tr", "Qwen/Qwen3-ASR-Stub"

    monkeypatch.setattr("transcribe.qwen3.transcribe", fake_transcribe)

    out = pipeline.run(
        job_id="abcdef123456",
        vocals_uri="gs://bucket/stages/separate/abcdef123456/vocals.wav",
        source_uri="gs://bucket/uploads/abcdef123456.mp4",
        language="tr",
    )

    assert out["source"] == "qwen3"
    assert out["model_used"] == "Qwen/Qwen3-ASR-Stub"
    assert out["language"] == "tr"
    assert out["segments"] == [{"text": "qwen sang", "start": 0.0, "end": 2.5}]
    assert out["vocal_activity"] == [{"start": 0.0, "end": 10.0, "kind": "vocals"}]


def test_whisper_when_qwen3_flag_false(monkeypatch):
    """Flag off → dispatcher downgrades qwen3-preferred language to whisper."""
    _stub_download(monkeypatch)
    _stub_vad(monkeypatch)
    monkeypatch.setattr(pipeline, "_QWEN3_AVAILABLE", False)
    _stub_whisper(monkeypatch, text="small whisper", lang="tr")

    # Tripwire: if the dispatcher ever routes through qwen3 when the flag
    # is False, this stub makes the test fail loudly.
    def _boom(*_a, **_k):  # pragma: no cover
        raise AssertionError("qwen3.transcribe must not be called when flag is False")

    monkeypatch.setattr("transcribe.qwen3.transcribe", _boom)

    out = pipeline.run(
        job_id="abcdef123456",
        vocals_uri="gs://bucket/stages/separate/abcdef123456/vocals.wav",
        source_uri="gs://bucket/uploads/abcdef123456.mp4",
        language="tr",
    )

    assert out["source"] == "whisper"
    assert out["language"] == "tr"
    assert out["segments"] == [{"text": "small whisper", "start": 0.0, "end": 3.0}]


def test_qwen3_runtime_failure_falls_back_to_whisper(monkeypatch, caplog):
    """Qwen3 raises at inference time → pipeline logs warn and serves whisper."""
    _stub_download(monkeypatch)
    _stub_vad(monkeypatch)
    monkeypatch.setattr(pipeline, "_QWEN3_AVAILABLE", True)
    _stub_whisper(monkeypatch, text="fallback text", lang="en")

    def fake_transcribe(*_a, **_k):
        raise RuntimeError("simulated torch OOM")

    monkeypatch.setattr("transcribe.qwen3.transcribe", fake_transcribe)

    out = pipeline.run(
        job_id="abcdef123456",
        vocals_uri="gs://bucket/stages/separate/abcdef123456/vocals.wav",
        source_uri="gs://bucket/uploads/abcdef123456.mp4",
        language="en",
    )

    assert out["source"] == "whisper"
    assert out["language"] == "en"
    # Model reported is the whisper env var default, not the qwen3 id.
    assert out["model_used"] != "Qwen/Qwen3-ASR-Stub"
    assert out["segments"] == [{"text": "fallback text", "start": 0.0, "end": 3.0}]
