"""Qwen3-ASR backend. Loads the model lazily on first call and caches it in
a module-level singleton.

Two load paths, tried in order:

1. `qwen3_asr_toolkit` — Alibaba's official wrapper. Handles long-form
   audio, context bias, and (depending on wrapper version) per-segment
   timestamps natively.
2. `transformers` — raw fallback. The Qwen3-ASR HF repo ships a processor
   + generation model; on recent releases the processor's apply_chat_template
   path yields a transcript string. Timestamps are not guaranteed — if the
   model returns a single blob we emit a single segment covering the whole
   clip and let `stages/align` produce word-level timings downstream.

Either load failure or inference failure raises. `pipeline.py` wraps the
call in a try/except and degrades to faster-whisper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from shared import create_logger

log = create_logger("transcribe.qwen3")

_DEFAULT_MODEL = "Qwen/Qwen3-ASR-Flash"
_PROMPT_MAX_CHARS = 200

_loaded: dict[str, Any] | None = None


def _resolve_device() -> str:
    requested = os.environ.get("QWEN_DEVICE", "auto").lower()
    if requested in {"cpu", "mps", "cuda"}:
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load() -> dict[str, Any]:
    """Load the model once and cache the handle. Returns a dict:
    {"kind": "toolkit"|"transformers", "model": ..., "processor": ..., "model_id": str, "device": str}.
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    model_id = os.environ.get("QWEN_MODEL", _DEFAULT_MODEL)
    device = _resolve_device()
    log.info(None, "loading qwen3", {"model_id": model_id, "device": device})

    # Path 1: toolkit
    try:
        import qwen3_asr_toolkit  # type: ignore

        loader = getattr(qwen3_asr_toolkit, "Qwen3ASR", None) or getattr(
            qwen3_asr_toolkit, "Qwen3ASRModel", None
        )
        if loader is None:
            raise ImportError("qwen3_asr_toolkit: no recognized model class")
        model = loader.from_pretrained(model_id, device=device)
        _loaded = {
            "kind": "toolkit",
            "model": model,
            "processor": None,
            "model_id": model_id,
            "device": device,
        }
        log.info(None, "qwen3 loaded", {"kind": "toolkit"})
        return _loaded
    except Exception as e:
        log.info(
            None,
            "qwen3 toolkit unavailable, trying transformers",
            {"error": f"{type(e).__name__}: {e}"},
        )

    # Path 2: transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    _loaded = {
        "kind": "transformers",
        "model": model,
        "processor": processor,
        "model_id": model_id,
        "device": device,
    }
    log.info(None, "qwen3 loaded", {"kind": "transformers"})
    return _loaded


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        return f.frames / float(f.samplerate)


def _load_audio_16k_mono(path: Path):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        # Simple linear-interp resample is fine here; the model will tolerate.
        # We keep the dep surface small — no librosa/torchaudio.
        ratio = 16000.0 / float(sr)
        n_new = int(round(data.shape[0] * ratio))
        if n_new > 0:
            xp = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False)
            fp = data
            x = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
            data = np.interp(x, xp, fp).astype("float32")
        sr = 16000
    return data, sr


def _run_toolkit(
    handle: dict[str, Any],
    audio_path: Path,
    language: str | None,
    known_lyrics: str | None,
) -> tuple[list[dict], str | None]:
    model = handle["model"]
    context = known_lyrics[:_PROMPT_MAX_CHARS] if known_lyrics else None
    # The toolkit's public API has varied release-to-release. We try the most
    # likely shapes and raise if none land.
    call_kwargs = {
        "audio": str(audio_path),
        "language": language,
        "context": context,
        "return_time_stamps": True,
    }
    for method_name in ("transcribe", "generate", "__call__"):
        fn = getattr(model, method_name, None)
        if not callable(fn):
            continue
        try:
            raw = fn(**{k: v for k, v in call_kwargs.items() if v is not None})
            return _normalize_toolkit_output(raw)
        except TypeError:
            # Method doesn't accept these kwargs; try the next shape.
            continue
    raise RuntimeError("qwen3 toolkit: no callable transcribe method accepted our kwargs")


def _normalize_toolkit_output(raw: Any) -> tuple[list[dict], str | None]:
    """Coerce the toolkit's response (which varies by version) into
    (segments, detected_language)."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        # Already a list of segments — trust it if start/end are present.
        segs = []
        for r in raw:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            start = float(r.get("start", 0.0))
            end = float(r.get("end", start))
            if end <= start:
                continue
            segs.append({"text": text, "start": start, "end": end})
        return segs, None

    if isinstance(raw, dict):
        lang = raw.get("language") or raw.get("detected_language")
        stamps = raw.get("time_stamps") or raw.get("timestamps") or raw.get("segments")
        if isinstance(stamps, list) and stamps and isinstance(stamps[0], dict):
            segs = []
            for r in stamps:
                text = (r.get("text") or r.get("word") or "").strip()
                if not text:
                    continue
                start = float(r.get("start", 0.0))
                end = float(r.get("end", start))
                if end <= start:
                    continue
                segs.append({"text": text, "start": start, "end": end})
            if segs:
                return segs, lang
        text = (raw.get("text") or "").strip()
        return ([{"text": text, "start": 0.0, "end": 0.0}] if text else []), lang

    if isinstance(raw, str):
        text = raw.strip()
        return ([{"text": text, "start": 0.0, "end": 0.0}] if text else []), None

    raise RuntimeError(f"qwen3 toolkit: unrecognized output shape: {type(raw).__name__}")


def _run_transformers(
    handle: dict[str, Any],
    audio_path: Path,
    language: str | None,
    known_lyrics: str | None,
) -> tuple[list[dict], str | None]:
    import torch

    model = handle["model"]
    processor = handle["processor"]
    device = handle["device"]

    audio, sr = _load_audio_16k_mono(audio_path)
    context = known_lyrics[:_PROMPT_MAX_CHARS] if known_lyrics else ""

    # Qwen3-ASR's chat-style prompt (per HF model card examples). Keep it
    # defensive: if the processor doesn't support this shape, raise and let
    # pipeline.py fall back to whisper.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {
                    "type": "text",
                    "text": (
                        (f"Language: {language}. " if language else "")
                        + (f"Context: {context}" if context else "Transcribe the audio.")
                    ),
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if device != "cpu":
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512")))

    # Trim the prompt tokens from the generated output before decoding.
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    output_ids = generated[:, input_len:] if input_len else generated
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Raw transformers path doesn't give us timestamps natively; emit one
    # segment covering [0, duration]. stages/align produces word-level timings.
    duration = _audio_duration(audio_path)
    segs = [{"text": text, "start": 0.0, "end": float(duration)}] if text else []
    return segs, None


def transcribe(
    audio_path: Path,
    language: str | None,
    known_lyrics: str | None,
) -> tuple[list[dict], str, str]:
    """Run Qwen3-ASR on `audio_path` (the full mix).

    Returns (segments, detected_language, model_id).
    Each segment: {"text": str, "start": float, "end": float}.
    """
    handle = _load()
    runner = _run_toolkit if handle["kind"] == "toolkit" else _run_transformers
    segments, detected = runner(handle, audio_path, language, known_lyrics)

    # If a segment has a zero-width timestamp (happens when the toolkit
    # returns only plain text), stretch it to cover the full clip so the
    # response contract (end > start) is satisfied.
    if segments and segments[0]["end"] <= segments[0]["start"]:
        duration = _audio_duration(audio_path)
        segments = [{"text": segments[0]["text"], "start": 0.0, "end": float(duration)}]

    detected_language = detected or language or "und"
    return segments, detected_language, handle["model_id"]


def reset_for_tests() -> None:
    """Drop the cached handle — used by tests that monkeypatch _load()."""
    global _loaded
    _loaded = None
