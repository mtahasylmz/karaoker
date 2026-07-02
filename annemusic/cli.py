"""annemusic CLI: music video in, karaoke artifacts out.

    annemusic video.mp4 [-o OUT] [--language LANG] [--lyrics FILE] [--force]

Artifacts written to OUT (default ./<video-stem>/): instrumental.wav,
vocals.wav, lyrics.ass (per-word \\kf karaoke), manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from annemusic import ass, backends, core, vad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="annemusic",
        description="Music video in -> instrumental + per-word karaoke subtitles out.",
    )
    p.add_argument("video", help="input music video (mp4/mov/webm/mkv)")
    p.add_argument("-o", "--out", help="output directory (default: ./<video-stem>/)")
    p.add_argument("--language", help="ISO language hint; skips detection")
    p.add_argument("--lyrics", help="known-lyrics text file to bias transcription")
    p.add_argument("--force", action="store_true", help="overwrite existing artifacts")
    args = p.parse_args(argv)

    video = Path(args.video)
    if not video.is_file():
        print(f"annemusic: input not found: {args.video}", file=sys.stderr)
        return 2

    lyrics = None
    if args.lyrics:
        lyrics_path = Path(args.lyrics)
        if not lyrics_path.is_file():
            print(f"annemusic: lyrics file not found: {args.lyrics}", file=sys.stderr)
            return 2
        lyrics = lyrics_path.read_text(encoding="utf-8")

    out_dir = core.resolve_out_dir(args.video, args.out)
    if (out_dir / "manifest.json").exists() and not args.force:
        print(
            f"annemusic: {out_dir / 'manifest.json'} already exists; "
            "pass --force to overwrite",
            file=sys.stderr,
        )
        return 3

    try:
        run(video, out_dir, args.language, lyrics)
    except Exception as e:  # spec: readable one-line error, no traceback
        if os.environ.get("ANNEMUSIC_DEBUG"):
            raise
        print(f"annemusic: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def run(
    video: Path, out_dir: Path, language_hint: str | None, lyrics: str | None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="annemusic-") as tmp_s:
        tmp = Path(tmp_s)
        mix = tmp / "mix.wav"
        _log("extracting audio")
        backends.extract_audio(video, mix)
        duration = backends.audio_duration(mix)

        _log("separating stems")
        vocals_src, instrumental_src = backends.separate(mix, tmp)
        vocals = out_dir / "vocals.wav"
        shutil.copyfile(vocals_src, vocals)
        shutil.copyfile(instrumental_src, out_dir / "instrumental.wav")

        vocal_activity = vad.detect(vocals)
        language = language_hint or backends.detect_language(vocals, vocal_activity)

        _log(f"transcribing (language={language or 'auto'})")
        asr_language, segments = backends.transcribe(
            mix=mix, vocals=vocals, language=language, lyrics=lyrics
        )

        _log("aligning words")
        words = _align(vocals, segments, vocal_activity, language or asr_language)

    # A --language hint is echoed verbatim; otherwise report what ASR saw.
    manifest = core.build_manifest(
        language_hint or asr_language, duration, words, vocal_activity
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "lyrics.ass").write_text(
        ass.build_ass(manifest["words"]), encoding="utf-8"
    )
    _log(f"done: {len(manifest['words'])} words -> {out_dir}")


def _align(
    vocals: Path,
    segments: list[dict],
    vocal_activity: list[dict],
    language: str,
) -> list[dict]:
    """Route alignment: qwen3 (CUDA, 11 languages) per ~120 s chunk with
    per-chunk whisperx fallback; whisperx for everything else; even-split as
    the last resort."""
    segments = core.split_at_vad_breaks(segments, vocal_activity)
    words: list[dict] = []
    if (
        language
        and language.lower() in core.QWEN_ALIGN_LANGS
        and backends.qwen3_align_available()
    ):
        remaining: list[dict] = []
        chunks = core.plan_chunks(
            segments, vocal_activity, target_seconds=core.QWEN_TARGET_SECONDS
        )
        for chunk in chunks:
            try:
                words += backends.align_qwen3(vocals, chunk, language)
            except Exception as e:
                _log(f"qwen3 align fallback for one chunk ({type(e).__name__}: {e})")
                remaining += chunk
        segments = remaining
        if not segments:
            return words

    try:
        words += backends.align_whisperx(vocals, segments, language)
    except Exception as e:
        _log(f"whisperx align failed ({type(e).__name__}: {e}); even-splitting")
        words += core.synthesize_words(segments)
    return words


def _log(msg: str) -> None:
    print(f"annemusic: {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
