"""Per-language backend routing for transcribe + align.

Python mirror of packages/contracts/src/flows.ts — SAME language sets, SAME
defaults. Keep in sync by hand; they're ~20 lines each and a shared contract
for the full pipeline. If you edit one, edit the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TranscribeSource = Literal["qwen3", "whisper"]
TranscribeInput = Literal["mix", "vocals"]
AlignSource = Literal["qwen3", "whisperx", "even-split"]


@dataclass(frozen=True)
class Flow:
    transcribe: TranscribeSource
    transcribe_input: TranscribeInput
    align: AlignSource


# Qwen3-ForcedAligner-0.6B language coverage (the 11 langs Qwen ships an
# aligner for). Anything outside this set routes to whisperx/wav2vec2.
_QWEN_ALIGN_LANGS: frozenset[str] = frozenset(
    {"en", "zh", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}
)

# Languages where we prefer Qwen3-ASR as the transcriber. Qwen3-ASR covers 30
# languages and outperforms Whisper on singing on its published benchmarks.
_QWEN_TRANSCRIBE_LANGS: frozenset[str] = frozenset(
    {
        "en", "zh", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko",
        "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi",
        "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
    }
)

DEFAULT_FLOW: Flow = Flow(
    transcribe="whisper", transcribe_input="vocals", align="whisperx"
)


def flow_for(language: str | None) -> Flow:
    if not language:
        return DEFAULT_FLOW
    lang = language.lower()
    transcribe: TranscribeSource = (
        "qwen3" if lang in _QWEN_TRANSCRIBE_LANGS else "whisper"
    )
    return Flow(
        transcribe=transcribe,
        transcribe_input="mix" if transcribe == "qwen3" else "vocals",
        align="qwen3" if lang in _QWEN_ALIGN_LANGS else "whisperx",
    )


def input_for_backend(backend: TranscribeSource) -> TranscribeInput:
    """Audio the *backend* expects, independent of flow/language.

    Use this when you've chosen a concrete backend (possibly via fallback) and
    need to decide which URI to feed it. Qwen3-ASR was trained on mixes with
    BGM; faster-whisper prefers the vocals stem.
    """
    return "mix" if backend == "qwen3" else "vocals"
