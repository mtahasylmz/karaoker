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
# Stages run with open Cloud Run ingress (QStash, which executes the
# orchestrator's context.call, cannot mint GCP OIDC tokens) and verify this
# shared bearer token on /process instead. Generate once:
#   openssl rand -hex 32
# and set the same value on the orchestrator and apps/api.
: "${STAGE_AUTH_TOKEN:?set STAGE_AUTH_TOKEN (openssl rand -hex 32; shared with orchestrator + api)}"
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

ENV_PAIRS="GCS_BUCKET=${GCS_BUCKET},UPSTASH_REDIS_REST_URL=${UPSTASH_REDIS_REST_URL},UPSTASH_REDIS_REST_TOKEN=${UPSTASH_REDIS_REST_TOKEN},STAGE_AUTH_TOKEN=${STAGE_AUTH_TOKEN}"
if [[ $CODE_FROM_GCS == 1 ]]; then
  ENV_PAIRS="${ENV_PAIRS},CODE_GCS_PATH=code/${STAGE}"
fi
if [[ $GPU == 1 ]]; then
  # HF weight mirror, populated by infra/model-fetch (Cloud Run Job).
  # The bucket mounts read-only at /gcs (volume flags below) and
  # stage-entrypoint.py copies models/hf into local HF_HOME at boot —
  # same-region FUSE reads instead of multi-GB HuggingFace downloads on
  # every cold start.
  ENV_PAIRS="${ENV_PAIRS},MODEL_CACHE_SRC=/gcs/models/hf"
fi

DEPLOY_FLAGS=(
  --image "${IMAGE}:${SHA}"
  --region "$REGION"
  --service-account "$WORKER_SA"
  # Open ingress by design: the orchestrator's context.call runs from
  # Upstash's infra and can't carry a GCP OIDC token. /process enforces
  # STAGE_AUTH_TOKEN at the app layer instead (shared-py auth.py).
  --allow-unauthenticated
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
    # One request at a time: the stages run a single ML job per instance
    # (in-process job lock mirrors this); Cloud Run's default of 80 would
    # queue requests against a busy worker and time them out together.
    --concurrency 1
    --timeout 3600
    --add-volume "name=models,type=cloud-storage,bucket=${GCS_BUCKET},readonly=true"
    --add-volume-mount "volume=models,mount-path=/gcs"
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
  # compose is light pure-TS and handles parallel requests fine; every other
  # stage runs one heavy ML/DSP job at a time (see the in-process job lock).
  if [[ "$STAGE" != "compose" ]]; then
    DEPLOY_FLAGS+=( --concurrency 1 )
  fi
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
  # gcloud's projection language has no JSONPath filters; pull the traffic
  # block as JSON and pick the "next"-tagged URL in python. (The previous
  # [?(@.tag=="next")] projection silently returned empty, skipping warm-up.)
  TAG_URL=$(gcloud run services describe "$SVC" --region="$REGION" --project="$GCP_PROJECT" \
    --format='json(status.traffic)' | python3 -c '
import json, sys
traffic = json.load(sys.stdin).get("status", {}).get("traffic", [])
urls = [t.get("url", "") for t in traffic if t.get("tag") == "next" and t.get("url")]
print(urls[0] if urls else "")')
  echo "==> Shadow revision URL: ${TAG_URL:-<not found>}"
  if [[ -z "$TAG_URL" ]]; then
    echo "WARNING: could not resolve the 'next' tag URL; traffic switch will hit a cold revision" >&2
  else
    echo "==> Pinging /ping until 200 (warm-up)…"
    for i in $(seq 1 60); do
      code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "$TAG_URL/ping" || echo 0)
      if [[ "$code" == "200" ]]; then echo "  ready after $i tries"; break; fi
      sleep 2
    done
  fi
  echo "==> Switching traffic to latest revision"
  gcloud run services update-traffic "$SVC" --region="$REGION" --project="$GCP_PROJECT" \
    --to-latest --quiet
fi

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
