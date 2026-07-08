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

from annemusic import ass, backends, core, lrclib, vad
from annemusic.backends import log


class _Preflight(Exception):
    """CLI refuses before any work starts; carries the exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="annemusic",
        description="Music video in -> instrumental + per-word karaoke subtitles out.",
    )
    p.add_argument("video", help="input music video (mp4/mov/webm/mkv)")
    p.add_argument("-o", "--out", help="output directory (default: ./<video-stem>/)")
    p.add_argument(
        "--language",
        type=str.lower,
        help="ISO language hint, case-insensitive; skips detection",
    )
    p.add_argument("--lyrics", help="known-lyrics text file to bias transcription")
    p.add_argument("--artist", help="song artist, for LRCLIB synced-lyrics lookup")
    p.add_argument("--title", help="song title, for LRCLIB synced-lyrics lookup")
    p.add_argument(
        "--no-lyrics-fetch",
        action="store_true",
        help="skip LRCLIB; always transcribe + align",
    )
    p.add_argument("--force", action="store_true", help="overwrite existing artifacts")
    return p.parse_args(argv)


def _preflight(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate inputs before any heavy work; raises _Preflight to refuse."""
    video = Path(args.video)
    if not video.is_file():
        raise _Preflight(2, f"input not found: {args.video}")
    if args.lyrics and not Path(args.lyrics).is_file():
        raise _Preflight(2, f"lyrics file not found: {args.lyrics}")
    out_dir = core.resolve_out_dir(args.video, args.out)
    if (out_dir / "manifest.json").exists() and not args.force:
        raise _Preflight(
            3,
            f"{out_dir / 'manifest.json'} already exists; pass --force to overwrite",
        )
    return video, out_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        video, out_dir = _preflight(args)
        run(video, out_dir, args)
    except _Preflight as e:
        print(f"annemusic: {e}", file=sys.stderr)
        return e.code
    except Exception as e:  # spec: readable one-line error, no traceback
        if os.environ.get("ANNEMUSIC_DEBUG"):
            raise
        print(f"annemusic: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def run(video: Path, out_dir: Path, args: argparse.Namespace) -> None:
    hint = args.language
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="annemusic-") as tmp_s:
        tmp = Path(tmp_s)
        mix = tmp / "mix.wav"
        log("extracting audio")
        backends.extract_audio(video, mix)
        duration = backends.audio_duration(mix)

        log("separating stems")
        vocals_src, instrumental_src = backends.separate(mix, tmp)
        vocals = out_dir / "vocals.wav"
        shutil.copyfile(vocals_src, vocals)
        shutil.copyfile(instrumental_src, out_dir / "instrumental.wav")

        vocal_activity = vad.detect(vocals)
        language = hint or backends.detect_language(vocals, vocal_activity)

        # Synced-lyrics fast path: human LRC line times drive timing (words
        # spread evenly within lines), skipping ASR + forced alignment.
        lrc_lines = None
        if args.artist and args.title and not args.no_lyrics_fetch:
            log("looking up synced lyrics (LRCLIB)")
            lrc_lines = lrclib.fetch(args.title, args.artist, duration)

        if lrc_lines:
            source, lines, words = "lrclib", lrc_lines, core.even_words(lrc_lines)
            log(f"synced lyrics: {len(lines)} lines")
        else:
            source = "asr"
            log(f"transcribing (language={language or 'auto'})")
            asr_language, segments = backends.transcribe(
                mix=mix, vocals=vocals, language=language, lyrics=_lyrics_text(args)
            )
            language = hint or asr_language
            log("aligning words")
            words = _align(vocals, segments, vocal_activity, language)
            lines = core.words_to_lines(words)

    manifest = core.build_manifest(source, language, duration, words, vocal_activity)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "lyrics.ass").write_text(ass.build_ass(lines), encoding="utf-8")
    log(f"done ({source}): {len(manifest['words'])} words -> {out_dir}")


def _lyrics_text(args: argparse.Namespace) -> str | None:
    return Path(args.lyrics).read_text(encoding="utf-8") if args.lyrics else None


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
    if _qwen3_alignable(language):
        words, segments = _align_qwen3_chunks(vocals, segments, vocal_activity, language)
        if not segments:
            return words

    try:
        words += backends.align_whisperx(vocals, segments, language)
    except Exception as e:
        log(f"whisperx align failed ({type(e).__name__}: {e}); even-splitting")
        words += core.synthesize_words(segments)
    return words


def _qwen3_alignable(language: str) -> bool:
    # core.flow_for owns the routing rules; don't re-derive them here.
    return core.flow_for(language).align == "qwen3" and backends.qwen3_align_available()


def _align_qwen3_chunks(
    vocals: Path,
    segments: list[dict],
    vocal_activity: list[dict],
    language: str,
) -> tuple[list[dict], list[dict]]:
    """Align per chunk; returns (words, segments left for the fallback)."""
    words: list[dict] = []
    remaining: list[dict] = []
    chunks = core.plan_chunks(
        segments, vocal_activity, target_seconds=core.QWEN_TARGET_SECONDS
    )
    for chunk in chunks:
        try:
            words += backends.align_qwen3(vocals, chunk, language)
        except Exception as e:
            log(f"qwen3 align fallback for one chunk ({type(e).__name__}: {e})")
            remaining += chunk
    return words, remaining


if __name__ == "__main__":
    sys.exit(main())
