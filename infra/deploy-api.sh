#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:?set GCP_REGION}"
: "${UPSTASH_REDIS_REST_URL:?set UPSTASH_REDIS_REST_URL}"
: "${UPSTASH_REDIS_REST_TOKEN:?set UPSTASH_REDIS_REST_TOKEN}"
: "${WORKER_URL:?set WORKER_URL (printed by deploy-worker.sh)}"
: "${TASKS_QUEUE:=karaoke-jobs}"
: "${AR_REPO:=annemusic}"

IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/api:$(date +%s)"
API_SA="annemusic-api@${GCP_PROJECT}.iam.gserviceaccount.com"

cd "$(dirname "$0")/../api"

echo "==> Building ${IMAGE}"
gcloud builds submit --tag "$IMAGE" --project "$GCP_PROJECT"

echo "==> Deploying api to Cloud Run"
gcloud run deploy api \
  --image "$IMAGE" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --service-account "$API_SA" \
  --allow-unauthenticated \
  --cpu 1 --memory 512Mi \
  --min-instances 0 --max-instances 5 \
  --set-env-vars "UPSTASH_REDIS_REST_URL=${UPSTASH_REDIS_REST_URL},UPSTASH_REDIS_REST_TOKEN=${UPSTASH_REDIS_REST_TOKEN},WORKER_URL=${WORKER_URL},TASKS_QUEUE=${TASKS_QUEUE},GCP_PROJECT=${GCP_PROJECT},GCP_REGION=${GCP_REGION},TASKS_INVOKER_SA=${API_SA},CORS_ORIGINS=${CORS_ORIGINS:-*},GCS_BUCKET=${GCS_BUCKET}"

API_URL=$(gcloud run services describe api --region "$GCP_REGION" --project "$GCP_PROJECT" --format='value(status.url)')
echo "==> API URL: $API_URL"
echo "Update web/index.html: set localStorage.annemusic.api_url to $API_URL, or hard-code API_URL."
