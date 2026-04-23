#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:?set GCP_REGION}"
: "${GCS_BUCKET:?set GCS_BUCKET}"
: "${UPSTASH_REDIS_REST_URL:?set UPSTASH_REDIS_REST_URL}"
: "${UPSTASH_REDIS_REST_TOKEN:?set UPSTASH_REDIS_REST_TOKEN}"
: "${AR_REPO:=annemusic}"

IMAGE_BASE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/worker"
IMAGE="${IMAGE_BASE}:$(date +%s)"
IMAGE_LATEST="${IMAGE_BASE}:latest"
WORKER_SA="annemusic-worker@${GCP_PROJECT}.iam.gserviceaccount.com"
API_SA="annemusic-api@${GCP_PROJECT}.iam.gserviceaccount.com"

cd "$(dirname "$0")/../worker"

echo "==> Building $IMAGE with layer cache from $IMAGE_LATEST (fast on code-only changes)…"
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions="_IMAGE=${IMAGE},_IMAGE_LATEST=${IMAGE_LATEST}" \
  --project "$GCP_PROJECT" --timeout=30m

echo "==> Deploying worker to Cloud Run…"
gcloud run deploy worker \
  --image "$IMAGE" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --service-account "$WORKER_SA" \
  --no-allow-unauthenticated \
  --cpu 4 --memory 8Gi \
  --timeout 1800 --concurrency 1 \
  --min-instances 0 --max-instances 3 \
  --set-env-vars "UPSTASH_REDIS_REST_URL=${UPSTASH_REDIS_REST_URL},UPSTASH_REDIS_REST_TOKEN=${UPSTASH_REDIS_REST_TOKEN},GCS_BUCKET=${GCS_BUCKET},GCS_URL_MODE=${GCS_URL_MODE:-public},WHISPER_MODEL=${WHISPER_MODEL:-small},OIDC_AUDIENCE=${OIDC_AUDIENCE}" \
  --clear-secrets

WORKER_URL=$(gcloud run services describe worker --region "$GCP_REGION" --project "$GCP_PROJECT" --format='value(status.url)')
echo "==> Worker URL: $WORKER_URL"

if [ "$OIDC_AUDIENCE" != "$WORKER_URL" ]; then
  echo "WARN: OIDC_AUDIENCE in env ($OIDC_AUDIENCE) != actual worker URL ($WORKER_URL)."
  echo "      Update OIDC_AUDIENCE in .env and redeploy if Cloud Tasks OIDC fails."
fi

# Allow API SA to invoke the worker via Cloud Tasks OIDC.
gcloud run services add-iam-policy-binding worker \
  --region "$GCP_REGION" --project "$GCP_PROJECT" \
  --member="serviceAccount:$API_SA" --role=roles/run.invoker >/dev/null

echo "Export WORKER_URL=$WORKER_URL before running deploy-api.sh"
