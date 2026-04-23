# CUDA 12.4 + cuDNN runtime — matches NVIDIA L4 on Cloud Run GPU.
# Keep the -runtime tag (not -devel); we don't compile CUDA kernels.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    # Qwen3 load target on prod; override per-deploy if needed.
    QWEN_DEVICE=cuda \
    QWEN_MODEL=Qwen/Qwen3-ASR-Flash

# System deps: Python 3.11 (via deadsnakes), ffmpeg for audio decode,
# libsndfile for soundfile, libgomp for numpy/torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common curl ca-certificates git \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
      ffmpeg libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# uv — same installer/version as local dev.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy only the workspace manifest first so dep resolution is cacheable.
COPY pyproject.toml uv.lock* ./
COPY packages/shared-py ./packages/shared-py
COPY stages/transcribe ./stages/transcribe

# Install with the qwen3 extra so torch + transformers land in the image.
RUN uv sync --package annemusic-stage-transcribe --extra qwen3 --no-dev

# Pre-fetching the Qwen3 weights at build time is tempting but bloats the
# image (~3.5 GB) and couples image rebuilds to model changes. We instead
# rely on Cloud Run's image-fs cache across container starts. First cold
# request downloads to $HF_HOME, warm requests reuse it.

EXPOSE 8102
CMD ["uv", "run", "--package", "annemusic-stage-transcribe", "--extra", "qwen3", \
     "uvicorn", "transcribe.main:app", "--host", "0.0.0.0", "--port", "8102"]
