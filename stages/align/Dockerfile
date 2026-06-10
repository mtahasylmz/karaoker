# CUDA 12.4 + cuDNN runtime for NVIDIA L4 on Cloud Run GPU.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common curl ca-certificates git \
      build-essential \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      ffmpeg libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/shared-py ./packages/shared-py
COPY stages/align ./stages/align
COPY stages/transcribe/pyproject.toml ./stages/transcribe/pyproject.toml
COPY stages/record-mix/pyproject.toml ./stages/record-mix/pyproject.toml
# json-schema/ is what shared.schemas.validate() loads at runtime.
COPY packages/contracts/json-schema ./packages/contracts/json-schema

RUN uv sync --package annemusic-stage-align --frozen --no-dev

# Qwen3-ForcedAligner (Apache-2.0) lives in qwen-asr[vllm]. Can't live in
# align/pyproject.toml because qwen-asr[vllm] pins torchaudio==2.9.1 which
# conflicts with the torch>=2.6,<2.9 / torchaudio>=2.6,<2.9 floor the
# whisperx+pyannote path needs. Install it here as a second pip step so the
# GPU image has both the Qwen3 and whisperx paths available at runtime.
# Use --no-deps so pip doesn't retry the torchaudio pin; align.pipeline
# already guards the import with _QWEN3_IMPORT_OK.
RUN uv pip install --no-deps 'qwen-asr[vllm]>=0.0.6' \
 || echo "qwen-asr[vllm] install failed; align will use whisperx-only path"

# wav2vec2 checkpoints (~360 MB English, ~1.2 GB Turkish) download on first
# request per revision and cache at $TORCH_HOME.

# Code-from-GCS support (see infra/stage-entrypoint.py). Production deploys
# leave CODE_GCS_PATH unset → entrypoint just exec's the baked code.
COPY infra/stage-entrypoint.py /entrypoint.py
ENV STAGE_NAME=align \
    PY_MODULE=align \
    VENV_PYTHON=/app/.venv/bin/python

EXPOSE 8103
CMD ["/app/.venv/bin/python", "/entrypoint.py"]
