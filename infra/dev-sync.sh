#!/usr/bin/env bash
# Fast iteration helper for the code-from-GCS pattern.
#
# Usage:
#   bash infra/dev-sync.sh <stage>
#
# Stages: separate | transcribe | align
#
# What it does (target: <60 s end-to-end):
#   1. rsync stages/<stage>/src/ → gs://${GCS_BUCKET}/code/<stage>/
#   2. force a Cloud Run revision restart by bumping a DEV_NONCE env var
#      (same image; layers stay cached; the entrypoint resyncs at startup)
#   3. poll /ping (with OIDC) until 200, then print elapsed time
#
# Prereq: the service was deployed with `deploy-stage.sh <stage> --code-from-gcs`
# at least once (so the entrypoint reads CODE_GCS_PATH at startup). Run with
# --warm to keep the GPU hot between iterations.

set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCS_BUCKET:?set GCS_BUCKET}"

STAGE="${1:?usage: dev-sync.sh <stage>}"
case "$STAGE" in
  separate|transcribe|align) ;;
  *) echo "dev-sync only supports the 3 ML stages; got: $STAGE" >&2; exit 2 ;;
esac

REGION="${GCP_REGION:-us-central1}"
SVC="annemusic-${STAGE}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="${REPO_ROOT}/stages/${STAGE}/src"
DST="gs://${GCS_BUCKET}/code/${STAGE}"

T0=$(date +%s)

echo "==> rsync ${SRC} → ${DST}"
gcloud storage rsync --recursive --delete-unmatched-destination-objects \
  "$SRC" "$DST" --project="$GCP_PROJECT" 2>&1 | tail -5
T_RSYNC=$(( $(date +%s) - T0 ))

echo "==> force revision restart on $SVC"
gcloud run services update "$SVC" \
  --region="$REGION" --project="$GCP_PROJECT" \
  --update-env-vars="DEV_NONCE=$(date +%s)" \
  --quiet 2>&1 | grep -E "Service|revision" | tail -3
T_DEPLOY=$(( $(date +%s) - T0 ))

echo "==> waiting for /ping 200"
URL=$(gcloud run services describe "$SVC" --region="$REGION" --project="$GCP_PROJECT" \
  --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
for i in $(seq 1 60); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 \
    -H "Authorization: Bearer $TOKEN" "$URL/ping" || echo 0)
  if [[ "$code" == "200" ]]; then
    T_TOTAL=$(( $(date +%s) - T0 ))
    cat <<EOF
==> ready
   rsync:   ${T_RSYNC}s
   deploy:  $((T_DEPLOY - T_RSYNC))s
   warm:    $((T_TOTAL - T_DEPLOY))s
   total:   ${T_TOTAL}s   ($URL)
EOF
    exit 0
  fi
  sleep 2
done

echo "==> /ping never returned 200 within 120s" >&2
exit 1
