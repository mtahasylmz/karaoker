# record-mix is CPU-only: ffmpeg subprocess for mix/loudnorm, python-demucs
# subprocess for bleed cleanup, numpy/scipy for GCC-PHAT alignment.
# rubberband-cli is reserved for the v2 autotune=smooth path (not yet wired
# in pipeline.py, but cheap to include).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git \
      ffmpeg libsndfile1 libgomp1 \
      rubberband-cli \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/shared-py ./packages/shared-py
COPY stages/record-mix ./stages/record-mix
COPY stages/align/pyproject.toml ./stages/align/pyproject.toml
COPY stages/transcribe/pyproject.toml ./stages/transcribe/pyproject.toml
# json-schema/ is what shared.schemas.validate() loads at runtime.
COPY packages/contracts/json-schema ./packages/contracts/json-schema

RUN uv sync --package annemusic-stage-record-mix --frozen --no-dev

# Bake the plate IR that pipeline.py references for the reverb filter.
# It's tiny (~350 KB), checked into the repo, and referenced via absolute
# path inside the package.

# demucs htdemucs weights (~320 MB) download on first request per revision
# and cache at $TORCH_HOME.

EXPOSE 8105
CMD ["uv", "run", "--package", "annemusic-stage-record-mix", \
     "python", "-m", "record_mix.main"]
