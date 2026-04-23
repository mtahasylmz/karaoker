// Per-language backend routing for transcribe + align.
//
// Not a Zod schema — shared code. Both stages import `flowFor(language)` so
// the routing truth lives in one place; adding a language is a one-line edit
// here, not a change in every stage.

import type { TranscribeSource } from "./transcribe.js";
import type { AlignSource } from "./align.js";

// Which audio the transcribe backend consumes. Qwen3-ASR was trained on full
// mixes with BGM and runs on "mix"; whisper paths consume the separated
// vocals stem.
export type TranscribeInput = "mix" | "vocals";

export type Flow = {
  transcribe: TranscribeSource;
  transcribe_input: TranscribeInput;
  align: AlignSource;
};

// Qwen3-ForcedAligner-0.6B language coverage (the 11 langs Qwen ships an
// aligner for). Anything outside this set routes to whisperx/wav2vec2.
const QWEN_ALIGN_LANGS = new Set([
  "en", "zh", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es",
]);

// Languages where we prefer Qwen3-ASR as the transcriber. Starting wide —
// Qwen3-ASR covers 30 languages and outperforms Whisper on singing on its
// published benchmarks. Carve out exceptions below when bench evidence
// suggests otherwise.
const QWEN_TRANSCRIBE_LANGS = new Set([
  "en", "zh", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru",
  "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs",
  "fil", "fa", "el", "hu", "mk", "ro",
]);

export const DEFAULT_FLOW: Flow = {
  transcribe: "whisper",
  transcribe_input: "vocals",
  align: "whisperx",
};

export function flowFor(language: string): Flow {
  const lang = language.toLowerCase();
  const transcribe = QWEN_TRANSCRIBE_LANGS.has(lang) ? "qwen3" : "whisper";
  return {
    transcribe,
    transcribe_input: transcribe === "qwen3" ? "mix" : "vocals",
    align: QWEN_ALIGN_LANGS.has(lang) ? "qwen3" : "whisperx",
  };
}
