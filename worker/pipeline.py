"""End-to-end karaoke pipeline. CPU-only, idempotent per content hash."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import ass_builder
import gcs
import redis_client

log = logging.getLogger("pipeline")


def run(job_id: str, sha256: str, object_path: str) -> None:
    out_object = f"videos/{sha256}.mp4"

    # Idempotency: if a prior run already produced the output, short-circuit.
    if gcs.exists(out_object):
        url = gcs.url_for(out_object)
        redis_client.set_video_done(sha256, url)
        redis_client.update_job_status(job_id, "done")
        return

    with tempfile.TemporaryDirectory(prefix=f"annemusic-{sha256[:12]}-") as tmpdir:
        tmp = Path(tmpdir)

        redis_client.update_job_status(job_id, "downloading")
        video_path = _download_upload(object_path, tmp)

        redis_client.update_job_status(job_id, "separating")
        audio_path = _extract_audio(video_path, tmp)
        vocals_path, instrumental_path = _separate(audio_path, tmp)

        redis_client.update_job_status(job_id, "transcribing")
        words = _transcribe(vocals_path)

        redis_client.update_job_status(job_id, "rendering")
        ass_path = tmp / "lyrics.ass"
        ass_builder.write_ass(words, ass_path)
        out_path = _mux(video_path, instrumental_path, ass_path, tmp)

        redis_client.update_job_status(job_id, "uploading")
        video_url = gcs.upload(out_object, out_path)

    redis_client.set_video_done(sha256, video_url)
    redis_client.update_job_status(job_id, "done")


# ---------- steps ----------

def _download_upload(object_path: str, tmp: Path) -> Path:
    local = tmp / Path(object_path).name
    gcs.download(object_path, local)
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(f"download produced empty file for {object_path}")
    return local


def _extract_audio(video: Path, tmp: Path) -> Path:
    audio = tmp / "audio.wav"
    _run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ar", "44100", "-ac", "2",
        str(audio),
    ])
    return audio


def _separate(audio: Path, tmp: Path) -> tuple[Path, Path]:
    out_dir = tmp / "demucs"
    _run([
        "python", "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-o", str(out_dir),
        str(audio),
    ])
    stem_dir = next((out_dir / "htdemucs").iterdir())
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError(f"demucs output missing: {list(stem_dir.iterdir())}")
    return vocals, instrumental


def _transcribe(vocals: Path) -> list[dict]:
    """Return [{word, start, end}] via faster-whisper + whisperx forced alignment.

    faster-whisper handles transcription (its native Silero VAD works, unlike
    whisperx's dead S3-hosted pyannote VAD URL). Then we hand segments to
    whisperx.align() for wav2vec2 forced alignment — that's the step that
    gives crisp word-level timings, which matters for karaoke wipes.
    """
    from faster_whisper import WhisperModel
    import whisperx

    model_size = os.environ.get("WHISPER_MODEL", "small")
    device = "cpu"

    fw_model = WhisperModel(model_size, device=device, compute_type="int8")
    segments_iter, info = fw_model.transcribe(
        str(vocals), word_timestamps=False, vad_filter=True, beam_size=5,
    )
    log.info("transcribe language=%s (prob=%.2f)", info.language, info.language_probability)

    # whisperx.align expects a list of dicts with at least text/start/end.
    fw_segments = [
        {"text": seg.text, "start": float(seg.start), "end": float(seg.end)}
        for seg in segments_iter
    ]
    if not fw_segments:
        return []

    audio = whisperx.load_audio(str(vocals))

    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        aligned = whisperx.align(
            fw_segments, align_model, align_meta, audio, device,
            return_char_alignments=False,
        )
        out_segments = aligned["segments"]
    except Exception as e:
        log.warning("alignment failed (%s); falling back to segment-level timings", e)
        out_segments = fw_segments

    words: list[dict] = []
    for seg in out_segments:
        word_entries = seg.get("words") or _synthesize_words(seg)
        for w in word_entries:
            start = w.get("start")
            end = w.get("end")
            token = (w.get("word") or "").strip()
            if not token or start is None or end is None or end <= start:
                continue
            words.append({"word": token, "start": float(start), "end": float(end)})
    return words


def _synthesize_words(seg: dict) -> list[dict]:
    text = (seg.get("text") or "").strip()
    if not text:
        return []
    tokens = text.split()
    if not tokens:
        return []
    start = float(seg["start"])
    end = float(seg["end"])
    step = (end - start) / len(tokens)
    return [
        {"word": t, "start": start + i * step, "end": start + (i + 1) * step}
        for i, t in enumerate(tokens)
    ]


def _synthesize_words(seg: dict) -> list[dict]:
    text = (seg.get("text") or "").strip()
    if not text:
        return []
    tokens = text.split()
    if not tokens:
        return []
    start = float(seg["start"])
    end = float(seg["end"])
    step = (end - start) / len(tokens)
    return [
        {"word": t, "start": start + i * step, "end": start + (i + 1) * step}
        for i, t in enumerate(tokens)
    ]


def _mux(video: Path, instrumental: Path, ass: Path, tmp: Path) -> Path:
    out = tmp / "out.mp4"
    _run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(instrumental),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"ass={ass}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out),
    ])
    return out


def _run(cmd: list[str]) -> None:
    log.info("run: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd[:3])}...\n"
            f"stderr: {result.stderr[-2000:]}"
        )
