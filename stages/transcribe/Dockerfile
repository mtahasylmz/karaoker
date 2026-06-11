# CUDA 12.4 + cuDNN runtime — matches NVIDIA L4 on Cloud Run GPU.
# -runtime (not -devel) because we don't compile CUDA kernels; -cudnn because
# torch + transformers use it for attention kernels.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    TRANSCRIBE_DEVICE=cuda:0 \
    QWEN3_MODEL=Qwen/Qwen3-ASR-1.7B

# Python 3.12 from deadsnakes (matches local dev), ffmpeg for audio decode,
# libsndfile for soundfile, libgomp for numpy/torch BLAS threads.
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

# Copy the workspace root first (pyproject + lock) so dep resolution is
# cacheable across code-only rebuilds (MVP scar #3).
COPY pyproject.toml uv.lock ./
COPY packages/shared-py ./packages/shared-py
COPY stages/transcribe ./stages/transcribe
# align + record-mix are workspace members too; uv refuses to resolve the
# lockfile without their manifests even when we only sync one package.
COPY stages/align/pyproject.toml ./stages/align/pyproject.toml
COPY stages/record-mix/pyproject.toml ./stages/record-mix/pyproject.toml
# json-schema/ is what shared.schemas.validate() loads at runtime.
COPY packages/contracts/json-schema ./packages/contracts/json-schema

RUN uv sync --package annemusic-stage-transcribe --frozen --no-dev

# Qwen3-ASR weights (~3.5 GB) stay out of the image — too big to push per
# rebuild. Cloud Run's writable filesystem is per-instance tmpfs (nothing
# survives an instance restart), so deploy-stage.sh mounts the GCS weight
# mirror (filled by infra/model-fetch) and stage-entrypoint.py copies it
# into HF_HOME at boot. Without the mirror, the first request per instance
# pays the full HuggingFace download into memory.

# Code-from-GCS support (see infra/stage-entrypoint.py). Production deploys
# leave CODE_GCS_PATH unset → entrypoint just exec's the baked code.
COPY infra/stage-entrypoint.py /entrypoint.py
ENV STAGE_NAME=transcribe \
    PY_MODULE=transcribe \
    VENV_PYTHON=/app/.venv/bin/python

EXPOSE 8102
CMD ["/app/.venv/bin/python", "/entrypoint.py"]
