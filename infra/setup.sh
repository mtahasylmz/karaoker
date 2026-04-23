#!/usr/bin/env bash
# One-time GCP bootstrap for annemusic.
# Prereq: `gcloud auth login`, `gcloud config set project <project>` done.
# Upstash Redis is created separately in the Upstash dashboard.

set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:?set GCP_REGION (e.g. us-central1)}"
: "${GCS_BUCKET:?set GCS_BUCKET (e.g. annemusic-videos)}"
: "${TASKS_QUEUE:=karaoke-jobs}"
: "${AR_REPO:=annemusic}"

API_SA="annemusic-api@${GCP_PROJECT}.iam.gserviceaccount.com"
WORKER_SA="annemusic-worker@${GCP_PROJECT}.iam.gserviceaccount.com"

echo "==> Enabling APIs…"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudtasks.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  --project "$GCP_PROJECT"

echo "==> Creating Artifact Registry repo (idempotent)…"
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker --location="$GCP_REGION" \
  --project "$GCP_PROJECT" 2>/dev/null || true

echo "==> Creating GCS bucket (idempotent)…"
gcloud storage buckets create "gs://$GCS_BUCKET" \
  --location="$GCP_REGION" --uniform-bucket-level-access \
  --project "$GCP_PROJECT" 2>/dev/null || true

# Make videos publicly readable (family project; swap for signed URLs if privacy matters).
gcloud storage buckets add-iam-policy-binding "gs://$GCS_BUCKET" \
  --member=allUsers --role=roles/storage.objectViewer \
  --project "$GCP_PROJECT" >/dev/null

echo "==> Creating Cloud Tasks queue (idempotent)…"
gcloud tasks queues create "$TASKS_QUEUE" \
  --location="$GCP_REGION" \
  --max-dispatches-per-second=1 \
  --max-concurrent-dispatches=5 \
  --max-attempts=3 \
  --project "$GCP_PROJECT" 2>/dev/null || true

echo "==> Creating service accounts (idempotent)…"
gcloud iam service-accounts create annemusic-api \
  --display-name="annemusic API" --project "$GCP_PROJECT" 2>/dev/null || true
gcloud iam service-accounts create annemusic-worker \
  --display-name="annemusic worker" --project "$GCP_PROJECT" 2>/dev/null || true

echo "==> Granting IAM roles…"
PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT" --format='value(projectNumber)')
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# In GCP projects created after ~2024-04, Cloud Build uses the Compute Engine
# default service account and it doesn't inherit the old builder permissions.
gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
  --member="serviceAccount:$COMPUTE_SA" --role=roles/cloudbuild.builds.builder >/dev/null

# API: may enqueue Cloud Tasks and mint OIDC tokens for its own SA.
gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
  --member="serviceAccount:$API_SA" --role=roles/cloudtasks.enqueuer >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$API_SA" \
  --member="serviceAccount:$API_SA" \
  --role=roles/iam.serviceAccountTokenCreator --project "$GCP_PROJECT" >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$API_SA" \
  --member="serviceAccount:$API_SA" \
  --role=roles/iam.serviceAccountUser --project "$GCP_PROJECT" >/dev/null

# Worker: read/write bucket.
gcloud storage buckets add-iam-policy-binding "gs://$GCS_BUCKET" \
  --member="serviceAccount:$WORKER_SA" --role=roles/storage.objectAdmin \
  --project "$GCP_PROJECT" >/dev/null

# API: needs object-create on the bucket — signed PUT URLs inherit the signer's
# GCS permissions, and the API SA signs URLs for browser uploads.
gcloud storage buckets add-iam-policy-binding "gs://$GCS_BUCKET" \
  --member="serviceAccount:$API_SA" --role=roles/storage.objectAdmin \
  --project "$GCP_PROJECT" >/dev/null

# API SA may invoke worker Cloud Run via Cloud Tasks.
# (Bound after worker is deployed — see deploy-worker.sh.)

cat <<EOF

Setup complete.

Next:
  1. Create an Upstash Redis database at https://console.upstash.com and copy
     UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN into your .env.
  2. Deploy worker first:   bash infra/deploy-worker.sh
  3. Then API:              bash infra/deploy-api.sh
  4. Then frontend:         cd web && firebase deploy  (or serve statically)
EOF
