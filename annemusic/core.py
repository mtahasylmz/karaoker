"""Pure pipeline logic: flow routing, word synthesis, manifest.

Zero GPU/IO — everything here is unit-testable on plain dicts and paths.
Harvested from the pre-fresh-start stages (shared/flows.py, align/pipeline.py)
and simplified.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from annemusic.repair import clean_words

# --------------------------------------------------------------------------- #
# Flow routing
# --------------------------------------------------------------------------- #

class Flow(NamedTuple):
    transcribe: str        # "qwen3" | "whisper"
    transcribe_input: str  # "mix" | "vocals"
    align: str             # "qwen3" | "whisperx"


# Qwen3-ForcedAligner-0.6B language coverage (the 11 langs Qwen ships an
# aligner for). Anything outside routes to whisperx/wav2vec2.
QWEN_ALIGN_LANGS = frozenset(
    {"en", "zh", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}
)

# Languages where Qwen3-ASR is preferred as the transcriber (30 languages,
# outperforms Whisper on singing per its published benchmarks).
QWEN_TRANSCRIBE_LANGS = frozenset(
    {
        "en", "zh", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko",
        "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi",
        "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
    }
)

DEFAULT_FLOW = Flow("whisper", "vocals", "whisperx")


def flow_for(language: str | None) -> Flow:
    if not language:
        return DEFAULT_FLOW
    lang = language.lower()
    transcribe = "qwen3" if lang in QWEN_TRANSCRIBE_LANGS else "whisper"
    return Flow(
        transcribe=transcribe,
        transcribe_input=input_for_backend(transcribe),
        align="qwen3" if lang in QWEN_ALIGN_LANGS else "whisperx",
    )


def input_for_backend(backend: str) -> str:
    """Audio the backend expects. Qwen3-ASR was trained on mixes with BGM;
    faster-whisper prefers the vocals stem."""
    return "mix" if backend == "qwen3" else "vocals"


# --------------------------------------------------------------------------- #
# Word synthesis: distribute a span's tokens when no aligner timing exists
# --------------------------------------------------------------------------- #

def _spread_evenly(
    span: dict, min_step: float = 0.0, drop_nonpositive: bool = True
) -> list[dict]:
    """One span's ``text`` tokens as words spread uniformly across
    [start, end]. Returns [] when there are no tokens. ``min_step`` floors the
    per-token duration so end > start even on a zero-width span (LRC lines);
    ``drop_nonpositive`` yields [] on a non-positive span (ASR segments)."""
    s, e = float(span["start"]), float(span["end"])
    toks = ((span.get("text") or "").strip()).split()
    if not toks or (drop_nonpositive and e <= s):
        return []
    step = max((e - s) / len(toks), min_step)
    return [
        {"text": tok, "start": s + i * step, "end": s + (i + 1) * step}
        for i, tok in enumerate(toks)
    ]


def synthesize_words(segments: list[dict]) -> list[dict]:
    """Last-resort fallback: evenly split each segment's tokens over its span."""
    out: list[dict] = []
    for seg in segments:
        out.extend(_spread_evenly(seg))
    return out


def even_words(lines: list[dict]) -> list[dict]:
    """Split each LRC line's text into words spread evenly across [start, end].
    Line windows are human LRC marks; within a line we don't know onsets, so
    distribute uniformly. A 1 ms step floor keeps end > start (clean_words)."""
    out: list[dict] = []
    for ln in lines:
        out.extend(_spread_evenly(ln, min_step=0.001, drop_nonpositive=False))
    return out


# --------------------------------------------------------------------------- #
# Manifest + CLI preflight decisions
# --------------------------------------------------------------------------- #

def build_manifest(
    source: str,
    language: str,
    duration: float,
    words: list[dict],
    vocal_activity: list[dict],
) -> dict:
    return {
        "source": source,
        "language": language,
        "duration": duration,
        "words": clean_words(words),
        "vocal_activity": vocal_activity,
    }


def resolve_out_dir(video: str, out: str | None) -> Path:
    return Path(out) if out else Path(Path(video).stem)
