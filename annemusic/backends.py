"""Adapters around every environmentally unsuitable thing: ffmpeg, stem
separation, Qwen3-ASR, faster-whisper, the qwen3 CUDA aligner, and the
isolated whisperx worker. Logic-free wrappers — routing and repair live in
core.py so this file never needs a GPU to be unit-tested around.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from annemusic import core

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = REPO_ROOT / "workers" / "whisperx"

# audio-separator model filenames for the RoFormer candidates (bench winner
# first: mel_band_roformer_kim > bs_roformer > htdemucs, ~+2.6 dB SDR).
AS_MODEL_FILES = {
    "mel_band_roformer_kim": "vocals_mel_band_roformer.ckpt",
    "bs_roformer_ep317": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}
DEMUCS_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s"}

# Qwen3-ASR wants English-name language args, not ISO codes.
ISO_TO_QWEN = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}
QWEN_TO_ISO = {v: k for k, v in ISO_TO_QWEN.items()}

_LID_MIN_PROB = float(os.environ.get("LID_MIN_PROB", "0.5"))
_QWEN3_CONTEXT_MAX = int(os.environ.get("QWEN3_CONTEXT_MAX_CHARS", "4000"))

_whisper_model = None
_qwen3_model = None
_qwen3_device: str | None = None
_qwen_aligner = None


def _log(msg: str) -> None:
    print(f"annemusic: {msg}", file=sys.stderr, flush=True)


def _run_cmd(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd[:3])} ... "
            f"stderr: {result.stderr[-2000:]}"
        )
    return result


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #

def extract_audio(video: Path, wav: Path) -> None:
    """Demux + decode the video's audio to stereo 44.1 kHz WAV."""
    _run_cmd([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-vn", "-ar", "44100", "-ac", "2", str(wav),
    ])


def audio_duration(path: Path) -> float:
    import soundfile as sf  # lazy

    return float(sf.info(str(path)).duration)


# --------------------------------------------------------------------------- #
# Separation. Default: htdemucs subprocess (warm cache, minutes on this M4).
# Upgrade seam: SEPARATE_MODEL=mel_band_roformer_kim runs the bench winner
# (+2.6 dB SDR) via audio-separator in an isolated uvx env (it needs numpy>=2,
# this project pins <2) — measured 10+ min/song on M4 CPU, so opt-in only;
# any roformer failure still falls back to htdemucs.
# --------------------------------------------------------------------------- #

def separate(mix: Path, work_dir: Path, model: str | None = None) -> tuple[Path, Path]:
    """Return (vocals, instrumental) stem paths under work_dir."""
    active = (model or os.environ.get("SEPARATE_MODEL") or "htdemucs").strip()
    if active in AS_MODEL_FILES:
        try:
            return _separate_roformer(active, mix, Path(work_dir) / "stems")
        except Exception as e:
            _log(f"roformer separation failed ({type(e).__name__}: {e}); "
                 "falling back to htdemucs")
            active = "htdemucs"
    if active not in DEMUCS_MODELS:
        raise RuntimeError(
            f"unknown SEPARATE_MODEL={active!r}; expected one of "
            f"{sorted(AS_MODEL_FILES) + sorted(DEMUCS_MODELS)}"
        )
    return _separate_demucs(active, mix, Path(work_dir) / "demucs")


def _separate_roformer(model: str, mix: Path, out_dir: Path) -> tuple[Path, Path]:
    uvx = shutil.which("uvx")
    if uvx is None:
        raise RuntimeError("uvx not on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR") or "/tmp/audio-separator-models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    _run_cmd([
        uvx, "--from", "audio-separator[cpu]", "audio-separator", str(mix),
        "--model_filename", AS_MODEL_FILES[model],
        "--output_dir", str(out_dir),
        "--output_format", "WAV",
        "--model_file_dir", model_dir,
        "--custom_output_names",
        json.dumps({"Vocals": "vocals", "Other": "no_vocals",
                    "Instrumental": "no_vocals"}),
    ])
    vocals = out_dir / "vocals.wav"
    instrumental = out_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        # Older CLIs ignore custom names; fall back to pattern matching.
        wavs = sorted(out_dir.glob("*.wav"))
        vocals = next((p for p in wavs if "(Vocals)" in p.name), None)
        instrumental = next((p for p in wavs if "(Vocals)" not in p.name), None)
        if vocals is None or instrumental is None:
            raise RuntimeError(f"audio-separator outputs missing in {out_dir}")
    return vocals, instrumental


def _separate_demucs(model: str, mix: Path, out_dir: Path) -> tuple[Path, Path]:
    # sys.executable, never bare "python": demucs lives in THIS venv.
    _run_cmd([
        sys.executable, "-m", "demucs", "--two-stems=vocals",
        "-n", model, "-o", str(out_dir), str(mix),
    ])
    stem_dir = next((Path(out_dir) / model).iterdir())
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError(f"demucs output missing: {list(stem_dir.iterdir())}")
    return vocals, instrumental


# --------------------------------------------------------------------------- #
# Language identification (faster-whisper LID on a vocals-stem window)
# --------------------------------------------------------------------------- #

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # lazy

        size = os.environ.get("WHISPER_MODEL", "small")
        _log(f"loading whisper ({size})")
        _whisper_model = WhisperModel(
            size, device="cpu", compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        )
    return _whisper_model


def detect_language(vocals: Path, vocal_activity: list[dict]) -> str | None:
    """LID on a ~30 s vocals-stem window anchored at the first sung region —
    song intros poison LID at 0 s. Returns an ISO code, or None when
    confidence is below the routing threshold."""
    from faster_whisper.audio import decode_audio  # lazy

    start = next(
        (r["start"] for r in vocal_activity if r.get("kind") == "vocals"), 0.0
    )
    audio = decode_audio(str(vocals), sampling_rate=16000)
    lo = int(start * 16000)
    window = audio[lo: lo + 30 * 16000]
    if window.shape[0] < 16000:  # under ~1 s of usable audio — don't guess
        return None
    _, info = _load_whisper().transcribe(
        window, language=None, beam_size=1, vad_filter=False
    )
    prob = float(info.language_probability)
    _log(f"detected language {info.language} (p={prob:.2f})")
    return info.language if prob >= _LID_MIN_PROB else None


# --------------------------------------------------------------------------- #
# Transcription: Qwen3-ASR on the full mix, faster-whisper on the vocals stem
# --------------------------------------------------------------------------- #

def transcribe(
    mix: Path, vocals: Path, language: str | None, lyrics: str | None
) -> tuple[str, list[dict]]:
    """Returns (asr_language_iso, segments). Flow-routed with fallback."""
    flow = core.flow_for(language)
    if flow.transcribe == "qwen3" and os.environ.get("FORCE_WHISPER") != "1":
        try:
            return _transcribe_qwen3(mix, language, lyrics)
        except Exception as e:
            _log(f"qwen3 transcribe failed ({type(e).__name__}: {e}); "
                 "falling back to whisper")
    return _transcribe_whisper(vocals, language, lyrics)


def _pick_device() -> str:
    override = os.environ.get("TRANSCRIBE_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_qwen3(force_device: str | None = None):
    global _qwen3_model, _qwen3_device
    if _qwen3_model is not None and force_device is None:
        return _qwen3_model
    import torch  # lazy
    from qwen_asr import Qwen3ASRModel  # lazy

    repo = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-ASR-1.7B")
    device = force_device or _pick_device()
    if device.startswith("cuda"):
        dtype = torch.bfloat16
    elif device == "mps":
        # MPS can't load bf16 weights; fp16 is Apple's mixed-precision path.
        dtype = torch.float16
    else:
        dtype = torch.float32
    _log(f"loading qwen3 ASR on {device}")
    _qwen3_model = Qwen3ASRModel.from_pretrained(
        repo,
        dtype=dtype,
        device_map=device,
        max_new_tokens=int(os.environ.get("QWEN3_MAX_NEW_TOKENS", "512")),
    )
    _qwen3_device = device
    return _qwen3_model


def _transcribe_qwen3(
    mix: Path, language: str | None, lyrics: str | None
) -> tuple[str, list[dict]]:
    # known lyrics ride qwen-asr's `context` parameter (system-prompt biasing).
    context = (lyrics or "").strip()[:_QWEN3_CONTEXT_MAX]
    model = _load_qwen3()
    qwen_lang = ISO_TO_QWEN.get(language.lower()) if language else None
    kwargs = dict(
        audio=str(mix), context=context, language=qwen_lang,
        return_time_stamps=False,
    )
    try:
        results = model.transcribe(**kwargs)
    except (RuntimeError, NameError) as e:
        # MPS commonly hits op gaps at runtime; one-shot CPU retry.
        if _qwen3_device != "mps":
            raise
        _log(f"qwen3 mps failed ({e}); reloading on cpu")
        global _qwen3_model
        _qwen3_model = None
        results = _load_qwen3(force_device="cpu").transcribe(**kwargs)

    r = results[0]
    detected_iso = QWEN_TO_ISO.get(r.language, language or "und")
    duration = audio_duration(mix)
    text = (r.text or "").strip()
    # One coarse segment spanning the audio; word timing is the aligner's job.
    segments = [{"text": text, "start": 0.0, "end": duration}] if text else []
    return detected_iso, segments


def _transcribe_whisper(
    vocals: Path, language: str | None, lyrics: str | None
) -> tuple[str, list[dict]]:
    model = _load_whisper()
    segments_iter, info = model.transcribe(
        str(vocals),
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=False,
        initial_prompt=(lyrics[:200] if lyrics else None),
    )
    segments = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text or seg.end <= seg.start:
            continue
        segments.append({"text": text, "start": float(seg.start), "end": float(seg.end)})
    return info.language, segments


# --------------------------------------------------------------------------- #
# Alignment backends
# --------------------------------------------------------------------------- #

def qwen3_align_available() -> bool:
    """The qwen3 forced aligner is CUDA-only."""
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_audio_cache: tuple[str, object] | None = None


def _load_audio_16k(path: Path):
    """Decode to 16 kHz mono float32, memoized for the per-chunk align loop."""
    global _audio_cache
    key = str(path)
    if _audio_cache is None or _audio_cache[0] != key:
        from faster_whisper.audio import decode_audio  # lazy

        _audio_cache = (key, decode_audio(key, sampling_rate=16000))
    return _audio_cache[1]


def align_qwen3(vocals: Path, chunk: list[dict], language: str) -> list[dict]:
    """Forced-align one chunk via Qwen3-ForcedAligner (CUDA). Slices the
    audio to the chunk window, aligns, then repairs via core.repair_words
    (raises core.SanityError on systemic garbage — caller falls back)."""
    import soundfile as sf  # lazy

    global _qwen_aligner
    lang_name = ISO_TO_QWEN.get(language.lower())
    if lang_name is None or not chunk:
        raise core.SanityError(f"qwen3 aligner unavailable for language={language!r}")
    text = " ".join(
        t for t in ((seg.get("text") or "").strip() for seg in chunk) if t
    )
    if not text:
        return []

    audio = _load_audio_16k(vocals)
    chunk_start = float(chunk[0]["start"])
    chunk_end = float(chunk[-1]["end"])
    s = max(0, int(chunk_start * 16000))
    e = min(len(audio), int(chunk_end * 16000))
    if e <= s:
        raise core.SanityError(f"empty chunk window [{s}, {e})")

    if _qwen_aligner is None:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        _log("loading qwen3 forced aligner")
        _qwen_aligner = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        slice_path = f.name
    try:
        sf.write(slice_path, audio[s:e], 16000)
        results = _qwen_aligner.align(audio=slice_path, text=text, language=lang_name)
    finally:
        os.unlink(slice_path)
    if not results:
        raise core.SanityError("qwen3 aligner returned no results")
    items = [
        {
            "text": getattr(it, "text", None),
            "start": getattr(it, "start_time", None),
            "end": getattr(it, "end_time", None),
        }
        for it in results[0]
    ]
    return core.repair_words(items, chunk_start, chunk_end)


def align_whisperx(vocals: Path, segments: list[dict], language: str) -> list[dict]:
    """wav2vec2 alignment via the isolated whisperx worker (subprocess —
    whisperx's torch<2.9 pin conflicts with qwen-asr's torchaudio==2.9.1)."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv not on PATH; whisperx worker unavailable")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        payload = json.dumps({
            "audio_path": str(vocals),
            "segments": segments,
            "language": language,
            "out": out_path,
        })
        result = subprocess.run(
            [uv, "run", "--project", str(WORKER_DIR), "python",
             str(WORKER_DIR / "align_worker.py")],
            input=payload, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"whisperx worker failed: {result.stderr[-2000:]}")
        with open(out_path) as f:
            return json.load(f)["words"]
    finally:
        os.unlink(out_path)
