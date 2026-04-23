#!/usr/bin/env bash
# Wipe all annemusic data from Upstash Redis + GCS bucket.
# Dry-run by default; pass --yes to actually delete.
#
# Usage:
#   bash infra/wipe.sh         # show what would be deleted
#   bash infra/wipe.sh --yes   # delete everything

set -euo pipefail

# Load .env from project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi

: "${UPSTASH_REDIS_REST_URL:?set UPSTASH_REDIS_REST_URL (in .env)}"
: "${UPSTASH_REDIS_REST_TOKEN:?set UPSTASH_REDIS_REST_TOKEN (in .env)}"
: "${GCS_BUCKET:?set GCS_BUCKET (in .env)}"

YES="${1:-}"

# Upstash REST POST helper. Body is a JSON array of command args.
rpost() {
  curl -sS -X POST "${UPSTASH_REDIS_REST_URL}/$1" \
    -H "Authorization: Bearer ${UPSTASH_REDIS_REST_TOKEN}"
}

rpost_body() {
  curl -sS -X POST "${UPSTASH_REDIS_REST_URL}/pipeline" \
    -H "Authorization: Bearer ${UPSTASH_REDIS_REST_TOKEN}" \
    -d "$1"
}

collect_keys() {
  local pattern="$1"
  # MATCH returns {"result": ["key1", "key2", ...]}
  rpost "keys/${pattern}" \
    | python3 -c 'import sys, json; print("\n".join(json.load(sys.stdin).get("result", [])))'
}

echo "=== annemusic state ==="

all_keys=()
for pattern in 'user:*' 'job:*' 'upload:*' 'video:*'; do
  keys=$(collect_keys "$pattern")
  count=$(printf '%s\n' "$keys" | grep -c . || true)
  printf "  %-20s %d\n" "$pattern" "$count"
  if [ "$count" -gt 0 ]; then
    while IFS= read -r k; do [ -n "$k" ] && all_keys+=("$k"); done <<< "$keys"
  fi
done

# dedupe (user:*:jobs matches user:*); macOS bash 3.2 has no readarray.
unique_keys=()
while IFS= read -r k; do
  [ -n "$k" ] && unique_keys+=("$k")
done < <(printf '%s\n' "${all_keys[@]:-}" | sort -u | grep -v '^$' || true)

echo "  TOTAL redis keys: ${#unique_keys[@]}"

echo
echo "GCS bucket ${GCS_BUCKET}:"
for prefix in 'uploads/' 'videos/'; do
  count=$(gcloud storage ls "gs://${GCS_BUCKET}/${prefix}" 2>/dev/null | grep -c . || true)
  printf "  %-20s %d\n" "$prefix" "$count"
done
total_bytes=$(gcloud storage du "gs://${GCS_BUCKET}" --summarize 2>/dev/null | awk '{print $1}')
total_mb=$(echo "$total_bytes" | awk '{printf "%.2f", $1 / 1e6}')
echo "  total bucket size: ${total_mb} MB"

if [ "$YES" != "--yes" ]; then
  echo
  echo "DRY RUN. Re-run with --yes to delete everything above."
  exit 0
fi

echo
echo "==> DELETING"

# Delete Redis keys in chunks of 100 via the /pipeline endpoint.
if [ ${#unique_keys[@]} -gt 0 ]; then
  for ((i = 0; i < ${#unique_keys[@]}; i += 100)); do
    chunk=("${unique_keys[@]:i:100}")
    body='['
    first=1
    for k in "${chunk[@]}"; do
      [ $first -eq 1 ] && first=0 || body+=','
      body+='["DEL",'
      body+=$(printf '%s' "$k" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')
      body+=']'
    done
    body+=']'
    rpost_body "$body" > /dev/null
  done
  echo "deleted ${#unique_keys[@]} redis keys"
fi

# Delete GCS objects.
for prefix in 'uploads/' 'videos/'; do
  if gcloud storage ls "gs://${GCS_BUCKET}/${prefix}" >/dev/null 2>&1; then
    gcloud storage rm -r "gs://${GCS_BUCKET}/${prefix}" 2>&1 | tail -1
  fi
done

echo
echo "Clean."
