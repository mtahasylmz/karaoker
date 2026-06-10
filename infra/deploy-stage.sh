#!/usr/bin/env bash
# Parametric Cloud Run deploy for a single stage.
#
# Usage:
#   bash infra/deploy-stage.sh <stage> [--gpu|--cpu] [--warm] [--shadow]
#                                      [--code-from-gcs] [--region <r>]
#
# Stages: separate | transcribe | align | compose | record-mix
#
# Defaults: separate / transcribe / align → GPU (L4); compose / record-mix → CPU.
# Use --cpu to override on an ML stage (e.g., when GPU quota is unavailable).
#
# Flags for active dev:
#   --warm           --min-instances 1 (no cold starts; ~$0.67/h per GPU svc)
#   --shadow         --no-traffic deploy + warm-up + traffic switch
#   --code-from-gcs  set CODE_GCS_PATH=code/<stage>; pair with `dev-sync.sh`
#
# Prereqs: `infra/setup.sh` already run, `gcloud` authenticated. Build runs
# on Cloud Build, so Apple Silicon / ARM laptops can deploy GPU images.

set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCS_BUCKET:?set GCS_BUCKET}"
: "${UPSTASH_REDIS_REST_URL:?set UPSTASH_REDIS_REST_URL (stages log to Redis Streams)}"
: "${UPSTASH_REDIS_REST_TOKEN:?set UPSTASH_REDIS_REST_TOKEN}"
: "${AR_REPO:=annemusic}"

STAGE="${1:?usage: deploy-stage.sh <stage> [--gpu|--cpu] [--warm] [--shadow] [--code-from-gcs] [--region <r>]}"
shift || true

case "$STAGE" in
  separate|transcribe|align|compose|record-mix) ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

# Default: GPU on for ML stages, CPU on for the others.
case "$STAGE" in
  separate|transcribe|align) GPU=1 ;;
  *)                         GPU=0 ;;
esac
WARM=0
SHADOW=0
CODE_FROM_GCS=0
# NVIDIA L4 on Cloud Run is GA in europe-west1, europe-west4, us-central1,
# asia-southeast1 (as of 2026-04). karaoke-494118's bucket lives in
# us-central1; matching here so the GCS FUSE mount has zero-cost reads.
REGION="${GCP_REGION:-us-central1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=1; shift ;;
    --cpu) GPU=0; shift ;;
    --warm) WARM=1; shift ;;
    --shadow) SHADOW=1; shift ;;
    --code-from-gcs) CODE_FROM_GCS=1; shift ;;
    --region) REGION="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  separate|transcribe|align|compose|record-mix) ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

SVC="annemusic-${STAGE}"
WORKER_SA="annemusic-worker@${GCP_PROJECT}.iam.gserviceaccount.com"
API_SA="annemusic-api@${GCP_PROJECT}.iam.gserviceaccount.com"
SHA="$(git rev-parse --short HEAD)"
IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/${STAGE}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "==> Building ${IMAGE}:${SHA} on Cloud Build"
# Pull :latest (if it exists) to seed the build cache — Cloud Build otherwise
# caches nothing layer-wise across submissions. On a code-only rebuild this
# takes builds from ~10 min to ~2 min (MVP scar #3).
cd "$REPO_ROOT"
cat > /tmp/annemusic-${STAGE}-cloudbuild.yaml <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    entrypoint: bash
    args:
      - -c
      - |
        docker pull ${IMAGE}:latest 2>/dev/null || true
        docker build \
          --cache-from ${IMAGE}:latest \
          -t ${IMAGE}:${SHA} \
          -t ${IMAGE}:latest \
          -f stages/${STAGE}/Dockerfile \
          .
  - name: gcr.io/cloud-builders/docker
    args: ['push', '${IMAGE}:${SHA}']
  - name: gcr.io/cloud-builders/docker
    args: ['push', '${IMAGE}:latest']
options:
  machineType: N1_HIGHCPU_8
timeout: 1800s
YAML

gcloud builds submit \
  --project "$GCP_PROJECT" \
  --config=/tmp/annemusic-${STAGE}-cloudbuild.yaml \
  .

echo "==> Deploying ${SVC} to Cloud Run (${REGION})"

ENV_PAIRS="GCS_BUCKET=${GCS_BUCKET},UPSTASH_REDIS_REST_URL=${UPSTASH_REDIS_REST_URL},UPSTASH_REDIS_REST_TOKEN=${UPSTASH_REDIS_REST_TOKEN}"
if [[ $CODE_FROM_GCS == 1 ]]; then
  ENV_PAIRS="${ENV_PAIRS},CODE_GCS_PATH=code/${STAGE}"
fi

DEPLOY_FLAGS=(
  --image "${IMAGE}:${SHA}"
  --region "$REGION"
  --service-account "$WORKER_SA"
  --no-allow-unauthenticated
  --set-env-vars "$ENV_PAIRS"
  --project "$GCP_PROJECT"
)

if [[ $GPU == 1 ]]; then
  DEPLOY_FLAGS+=(
    --gpu 1
    --gpu-type nvidia-l4
    --no-gpu-zonal-redundancy
    --cpu 4
    --memory 16Gi
    --no-cpu-throttling
    --max-instances 1
    --timeout 3600
  )
else
  # ML-class stages (separate/transcribe/align/record-mix) import torch +
  # model deps at startup and need ≥4 GiB even on CPU. compose is pure TS.
  case "$STAGE" in
    compose) MEM=2Gi; CPU=1 ;;
    record-mix) MEM=4Gi; CPU=2 ;;
    *) MEM=8Gi; CPU=2 ;;
  esac
  DEPLOY_FLAGS+=(
    --cpu "$CPU"
    --memory "$MEM"
    --max-instances 1
    --timeout 3600
  )
fi

if [[ $WARM == 1 ]]; then
  DEPLOY_FLAGS+=( --min-instances 1 )
fi

if [[ $SHADOW == 1 ]]; then
  # Deploy as a non-traffic-bearing revision tagged "next" so the cold-start
  # + model load happens out of the user's way; promote with traffic switch.
  DEPLOY_FLAGS+=( --no-traffic --tag next )
fi

gcloud run deploy "$SVC" "${DEPLOY_FLAGS[@]}"

if [[ $SHADOW == 1 ]]; then
  TAG_URL=$(gcloud run services describe "$SVC" --region="$REGION" --project="$GCP_PROJECT" \
    --format='value(status.traffic[?(@.tag=="next")].url)' | head -n1)
  echo "==> Shadow revision URL: $TAG_URL"
  if [[ -n "$TAG_URL" ]]; then
    TOKEN=$(gcloud auth print-identity-token 2>/dev/null || true)
    echo "==> Pinging /ping until 200 (warm-up)…"
    for i in $(seq 1 60); do
      code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 \
        ${TOKEN:+-H "Authorization: Bearer $TOKEN"} "$TAG_URL/ping" || echo 0)
      if [[ "$code" == "200" ]]; then echo "  ready after $i tries"; break; fi
      sleep 2
    done
  fi
  echo "==> Switching traffic to latest revision"
  gcloud run services update-traffic "$SVC" --region="$REGION" --project="$GCP_PROJECT" \
    --to-latest --quiet
fi

echo "==> Granting api SA invoker on ${SVC}"
gcloud run services add-iam-policy-binding "$SVC" \
  --member="serviceAccount:${API_SA}" \
  --role=roles/run.invoker \
  --region="$REGION" \
  --project "$GCP_PROJECT" >/dev/null

URL=$(gcloud run services describe "$SVC" \
  --region="$REGION" --project "$GCP_PROJECT" \
  --format='value(status.url)')

# env var naming matches the existing .env convention (RECORD_MIX_URL, etc).
case "$STAGE" in
  record-mix) ENV_VAR="RECORD_MIX_URL" ;;
  *)          ENV_VAR="$(echo "$STAGE" | tr '[:lower:]' '[:upper:]')_URL" ;;
esac

cat <<EOF

Deployed ${SVC}: ${URL}

Paste into apps/api .env (and wherever the proxy reads URLs from):
  ${ENV_VAR}=${URL}
EOF
