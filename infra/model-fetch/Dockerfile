# One-shot job container: HF → GCS model fetcher. Built once, run as a
# Cloud Run Job whenever new models need to be cached on GCS.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN pip install --no-cache-dir \
      huggingface_hub \
      hf_transfer \
      google-cloud-storage

WORKDIR /app
COPY fetch_models.py .

CMD ["python", "fetch_models.py"]
