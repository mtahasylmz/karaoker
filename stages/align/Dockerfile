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

# Qwen3-ForcedAligner (Apache-2.0) ships in qwen-asr. It can't live in
# align/pyproject.toml because qwen-asr pins torchaudio==2.9.1, conflicting
# with the torch>=2.6,<2.9 floor the whisperx path needs — so qwen-asr
# itself installs --no-deps, and its import-time deps that the whisperx
# tree doesn't already provide (librosa, nagisa, soynlp, qwen-omni-utils,
# sox, pytz, accelerate; transformers 4.57.6 is already locked) install
# normally. No [vllm] extra: the 0.6B aligner runs via transformers.
# The final import is the contract — if either backend can't import, the
# BUILD fails here instead of every prod job silently downgrading to
# CPU wav2vec2 on a GPU instance (pipeline._QWEN3_IMPORT_OK guard).
# hf_transfer rides along: HF_HUB_ENABLE_HF_TRANSFER=1 in the environment
# (GPU-host images commonly set it) makes huggingface_hub HARD-FAIL when the
# package is missing — observed on RunPod, where it silently downgraded the
# qwen3 aligner to whisperx. Having it also speeds cold weight downloads.
RUN uv pip install --no-deps 'qwen-asr>=0.0.6' \
 && uv pip install librosa 'nagisa==0.2.11' 'soynlp==0.0.493' \
      qwen-omni-utils sox pytz 'accelerate==1.12.0' hf_transfer \
 && /app/.venv/bin/python -c \
      "import whisperx; from qwen_asr import Qwen3ForcedAligner; print('align backends OK: whisperx + qwen3')"

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
