#!/bin/bash
# deploy_arms.sh
# Builds the app image once and deploys 6 Cloud Run services:
#   3 arms (bb/e/h) × 2 groups (g0=test / g1=official).
# Each service runs the SAME image, differentiated only by env vars:
#   ARM                 → pins the agent, skips the selector screen
#   SESSIONS_COLLECTION → isolates Firestore data (sessions_g0 vs sessions)
#   GROUP               → stored on each session doc for convenience
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Secrets in Secret Manager already used by the current single service.
#     Adjust --set-secrets / --set-env-vars below to MATCH that service.
#
# Change log: 2026-06-22 — initial build (arm × group split)

set -e

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ID="case-interview-coach"
REGION="europe-west1"
IMAGE="europe-west1-docker.pkg.dev/case-interview-coach/ba-agent/ba-agent:latest"

SHARED_SECRETS="/secrets/firebase_key.json=firebase-key:latest,GEMINI_API_KEY=gemini-api-key:latest,NEO4J_URI=neo4j-uri:latest,NEO4J_PASSWORD=neo4j-password:latest"
SHARED_ENV="NEO4J_USERNAME=aa8af79d,FIREBASE_KEY_PATH=/secrets/firebase_key.json"

echo "=== Step 1: Build and push app image ==="
docker buildx build --platform linux/amd64 \
  -f Dockerfile \
  -t "$IMAGE" \
  --push .

# ── Deploy matrix ───────────────────────────────────────────────────────────
# arm key | service-name slug
# slug: bb=black_box, e=explainable, h=hitl
# group: g0=test, g1=official
ARMS=("black_box:bb" "explainable:e" "hitl:h")

deploy_one() {
  local arm_key="$1" slug="$2" group="$3" collection="$4"
  local service="ba-agent-${slug}-${group}"
  echo "=== Deploying ${service} (ARM=${arm_key}, GROUP=${group}, COLLECTION=${collection}) ==="
  gcloud run deploy "$service" \
    --image="$IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=2Gi \
    --timeout=300 \
    --set-secrets="$SHARED_SECRETS" \
    --set-env-vars="${SHARED_ENV},ARM=${arm_key},GROUP=${group},SESSIONS_COLLECTION=${collection}" \
    --project="$PROJECT_ID"
}

for entry in "${ARMS[@]}"; do
  arm_key="${entry%%:*}"
  slug="${entry##*:}"
  deploy_one "$arm_key" "$slug" "g0" "sessions_g0"
  deploy_one "$arm_key" "$slug" "g1" "sessions_g1"
done

echo ""
echo "✅ Done! 6 services deployed. Get their URLs with:"
echo "  gcloud run services list --region=$REGION --project=$PROJECT_ID --format='table(metadata.name, status.url)'"
