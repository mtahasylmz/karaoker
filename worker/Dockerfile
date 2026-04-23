FROM python:3.12-slim

# ffmpeg for audio/video; git for a few pip deps (pyannote).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU wheels first so subsequent installs don't pull CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchaudio==2.2.2 \
 && pip install --no-cache-dir -r requirements.txt

# Pre-warm model weights into the image so cold starts don't download ~2GB.
# These `|| true` guards keep the build green if the warmup API ever changes —
# worst case, the first request pays the download cost.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')" || true
RUN python -c "import whisperx; whisperx.load_model('small', 'cpu', compute_type='int8')" || true

COPY . .

ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_MODEL=small

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
