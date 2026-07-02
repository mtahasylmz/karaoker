"""whisperx forced-alignment worker.

JSON request on stdin: {"audio_path": str, "segments": [...], "language": str,
"out": str}. Writes {"words": [{"text", "start", "end"}, ...]} to the ``out``
path (not stdout — library chatter would corrupt the stream).

Never calls whisperx.load_model — its VAD URL is dead. Only
load_align_model + align (wav2vec2).
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    req = json.load(sys.stdin)

    import whisperx

    audio = whisperx.load_audio(req["audio_path"])
    model, meta = whisperx.load_align_model(
        language_code=req["language"], device="cpu"
    )
    aligned = whisperx.align(
        req["segments"], model, meta, audio, "cpu", return_char_alignments=False
    )

    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if not text or start is None or end is None or end <= start:
                continue
            words.append({"text": text, "start": float(start), "end": float(end)})

    with open(req["out"], "w") as f:
        json.dump({"words": words}, f)


if __name__ == "__main__":
    main()
