# CUDA 12.4 + cuDNN runtime for NVIDIA L4 on Cloud Run GPU.
# stages/separate has its OWN uv lockfile (stages/separate/uv.lock) because
# audio-separator needs numpy>=2 while align + transcribe are pinned numpy<2.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    AUDIO_SEPARATOR_MODEL_DIR=/app/.cache/audio-separator \
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

# separate's workspace is self-contained except for a path source to
# packages/shared-py — copy both.
COPY stages/separate/pyproject.toml stages/separate/uv.lock ./stages/separate/
COPY stages/separate/src ./stages/separate/src
COPY packages/shared-py ./packages/shared-py
# json-schema/ is what shared.schemas.validate() loads at runtime. Ship the
# generated artifacts instead of re-running the TS builder at image-build time.
COPY packages/contracts/json-schema ./packages/contracts/json-schema

WORKDIR /app/stages/separate

# Base sync uses audio-separator[cpu] per pyproject.toml. Swap to the GPU
# torch build so torch.cuda.is_available() returns True and audio-separator
# routes RoFormer inference to the GPU. We do this post-sync rather than
# editing the pyproject so local dev on Mac (no CUDA) still works with the
# [cpu] extra.
RUN uv sync --frozen --no-dev \
 && uv pip install --no-deps --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch torchaudio

# RoFormer weights (~913 MB) stay out of the image. First /process per
# revision downloads to $AUDIO_SEPARATOR_MODEL_DIR; warm requests reuse it.

# Code-from-GCS support (see infra/stage-entrypoint.py). Production deploys
# leave CODE_GCS_PATH unset → entrypoint just exec's the baked code.
# Separate has its own uv lockfile, so its venv lives at
# /app/stages/separate/.venv (not /app/.venv like the workspace stages).
WORKDIR /app
COPY infra/stage-entrypoint.py /entrypoint.py
ENV STAGE_NAME=separate \
    PY_MODULE=separate \
    VENV_PYTHON=/app/stages/separate/.venv/bin/python

EXPOSE 8101
CMD ["/app/stages/separate/.venv/bin/python", "/entrypoint.py"]
