"""CLI behavior tests with all heavy backends monkeypatched out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annemusic import backends, cli, core, vad


@pytest.fixture()
def fake_pipeline(monkeypatch, tmp_path):
    """Stub every environment-touching seam; record calls for asserts."""
    calls: dict = {}

    def extract_audio(video, wav):
        Path(wav).write_bytes(b"RIFFfake")

    def separate(mix, work_dir):
        vocals = Path(work_dir) / "v.wav"
        instr = Path(work_dir) / "i.wav"
        vocals.write_bytes(b"RIFFvocals")
        instr.write_bytes(b"RIFFinstr")
        return vocals, instr

    def transcribe(mix, vocals, language, lyrics):
        calls["transcribe"] = {"language": language, "lyrics": lyrics}
        return "tr", [{"text": "la la la", "start": 0.0, "end": 3.0}]

    monkeypatch.setattr(backends, "extract_audio", extract_audio)
    monkeypatch.setattr(backends, "audio_duration", lambda p: 240.0)
    monkeypatch.setattr(backends, "separate", separate)
    monkeypatch.setattr(
        vad, "detect", lambda p: [{"start": 0.0, "end": 240.0, "kind": "vocals"}]
    )
    monkeypatch.setattr(backends, "detect_language", lambda p, va: "tr")
    monkeypatch.setattr(backends, "transcribe", transcribe)
    monkeypatch.setattr(backends, "qwen3_align_available", lambda: False)
    monkeypatch.setattr(
        backends,
        "align_whisperx",
        lambda vocals, segments, language: [
            {"text": "la", "start": 0.0, "end": 1.0},
            {"text": "la", "start": 1.0, "end": 2.0},
            {"text": "la", "start": 2.0, "end": 3.0},
        ],
    )

    video = tmp_path / "My Song.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.chdir(tmp_path)
    return {"video": video, "tmp": tmp_path, "calls": calls}


def test_happy_path_writes_all_four_artifacts(fake_pipeline):
    tmp = fake_pipeline["tmp"]
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(tmp / "out")])
    assert rc == 0
    out = tmp / "out"
    assert (out / "vocals.wav").exists()
    assert (out / "instrumental.wav").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest) == {"language", "duration", "words", "vocal_activity"}
    assert manifest["language"] == "tr"
    assert manifest["duration"] == 240.0
    assert len(manifest["words"]) == 3
    assert "\\kf" in (out / "lyrics.ass").read_text()


def test_language_hint_is_echoed_verbatim(fake_pipeline):
    tmp = fake_pipeline["tmp"]
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(tmp / "out"), "--language", "en"])
    assert rc == 0
    manifest = json.loads((tmp / "out" / "manifest.json").read_text())
    assert manifest["language"] == "en"
    # The hint short-circuits LID and reaches the transcriber.
    assert fake_pipeline["calls"]["transcribe"]["language"] == "en"


def test_default_out_dir_is_video_stem(fake_pipeline):
    rc = cli.main([str(fake_pipeline["video"])])
    assert rc == 0
    assert (fake_pipeline["tmp"] / "My Song" / "manifest.json").exists()


def test_lyrics_file_reaches_transcriber(fake_pipeline):
    tmp = fake_pipeline["tmp"]
    lyr = tmp / "lyrics.txt"
    lyr.write_text("gerçek sözler burada")
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(tmp / "out"), "--lyrics", str(lyr)])
    assert rc == 0
    assert fake_pipeline["calls"]["transcribe"]["lyrics"] == "gerçek sözler burada"


def test_missing_input_fails_fast_without_creating_out(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["does-not-exist.mp4", "-o", str(tmp_path / "out")])
    assert rc != 0
    assert "does-not-exist.mp4" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_refuses_to_clobber_without_force(fake_pipeline, capsys):
    tmp = fake_pipeline["tmp"]
    out = tmp / "out"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(out)])
    assert rc != 0
    assert "--force" in capsys.readouterr().err


def test_force_clobbers(fake_pipeline):
    tmp = fake_pipeline["tmp"]
    out = tmp / "out"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(out), "--force"])
    assert rc == 0
    assert json.loads((out / "manifest.json").read_text())["language"] == "tr"


def test_missing_lyrics_file_fails_clean(fake_pipeline, capsys):
    tmp = fake_pipeline["tmp"]
    rc = cli.main(
        [str(fake_pipeline["video"]), "-o", str(tmp / "out"), "--lyrics", "nope.txt"]
    )
    assert rc != 0
    assert "nope.txt" in capsys.readouterr().err


def test_align_falls_back_to_even_split(fake_pipeline, monkeypatch):
    def boom(vocals, segments, language):
        raise RuntimeError("worker dead")

    monkeypatch.setattr(backends, "align_whisperx", boom)
    tmp = fake_pipeline["tmp"]
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(tmp / "out")])
    assert rc == 0
    manifest = json.loads((tmp / "out" / "manifest.json").read_text())
    # even-split of "la la la" across [0, 3]
    assert len(manifest["words"]) == 3
    assert manifest["words"][0]["start"] == pytest.approx(0.0)
    assert manifest["words"][-1]["end"] == pytest.approx(3.0)


def test_pipeline_error_is_one_line_no_traceback(fake_pipeline, monkeypatch, capsys):
    def boom(mix, vocals, language, lyrics):
        raise RuntimeError("GPU on fire")

    monkeypatch.setattr(backends, "transcribe", boom)
    tmp = fake_pipeline["tmp"]
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(tmp / "out")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "RuntimeError: GPU on fire" in err
    assert "Traceback" not in err


def test_debug_env_reraises(fake_pipeline, monkeypatch):
    def boom(mix, vocals, language, lyrics):
        raise RuntimeError("boom")

    monkeypatch.setenv("ANNEMUSIC_DEBUG", "1")
    monkeypatch.setattr(backends, "transcribe", boom)
    with pytest.raises(RuntimeError, match="boom"):
        cli.main([str(fake_pipeline["video"]), "-o", str(fake_pipeline["tmp"] / "out")])


def _run_with_qwen3_aligner(fake_pipeline, monkeypatch, align_qwen3):
    """English run with the qwen3 aligner available; returns the manifest."""
    monkeypatch.setattr(backends, "qwen3_align_available", lambda: True)
    monkeypatch.setattr(backends, "align_qwen3", align_qwen3)
    monkeypatch.setattr(backends, "transcribe", lambda mix, vocals, language, lyrics: (
        "en", [{"text": "la la la", "start": 0.0, "end": 3.0}]))
    tmp = fake_pipeline["tmp"]
    rc = cli.main([str(fake_pipeline["video"]), "-o", str(tmp / "out"), "--language", "en"])
    assert rc == 0
    return json.loads((tmp / "out" / "manifest.json").read_text())


def test_qwen3_chunk_failure_falls_back_to_whisperx(fake_pipeline, monkeypatch):
    def align_qwen3(vocals, chunk, language):
        raise core.SanityError("systemic garbage")

    manifest = _run_with_qwen3_aligner(fake_pipeline, monkeypatch, align_qwen3)
    # The whisperx stub aligned the chunk the qwen3 aligner dropped.
    assert len(manifest["words"]) == 3


def test_qwen3_align_used_when_available(fake_pipeline, monkeypatch):
    seen = {}

    def align_qwen3(vocals, chunk, language):
        seen["chunk"] = chunk
        return [{"text": "la", "start": 0.0, "end": 3.0}]

    manifest = _run_with_qwen3_aligner(fake_pipeline, monkeypatch, align_qwen3)
    assert seen["chunk"]  # qwen3 aligner actually received the chunk
    assert len(manifest["words"]) == 1
