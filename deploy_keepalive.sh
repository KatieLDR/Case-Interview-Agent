#!/bin/bash
# deploy_keepalive.sh
# Sets up the AuraDB keep-alive Cloud Run Job + Cloud Scheduler trigger.
# Run this once from Google Cloud Shell.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Secrets already in Secret Manager: neo4j-uri, neo4j-password
#   - Existing service account: 129190537087-compute@developer.gserviceaccount.com
#
# Change log: 2026-04-16 — initial build

set -e  # Exit on any error

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ID="case-interview-coach"
REGION="europe-west1"
IMAGE="europe-west1-docker.pkg.dev/case-interview-coach/ba-agent/ba-agent-keepalive:latest"
SERVICE_ACCOUNT="129190537087-compute@developer.gserviceaccount.com"
JOB_NAME="auradb-keepalive"
SCHEDULER_JOB_NAME="auradb-keepalive-trigger"

echo "=== Step 1: Build and push keep-alive image ==="
docker buildx build --platform linux/amd64 \
  -f Dockerfile.keepalive \
  -t $IMAGE \
  --push .

echo "=== Step 2: Create Cloud Run Job ==="
gcloud run jobs create $JOB_NAME \
  --image=$IMAGE \
  --region=$REGION \
  --service-account=$SERVICE_ACCOUNT \
  --set-secrets="NEO4J_URI=neo4j-uri:latest,NEO4J_PASSWORD=neo4j-password:latest" \
  --set-env-vars="NEO4J_USERNAME=aa8af79d" \
  --max-retries=2 \
  --task-timeout=60s \
  --project=$PROJECT_ID

echo "=== Step 3: Grant Scheduler permission to invoke the job ==="
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/run.invoker"

echo "=== Step 4: Create Cloud Scheduler job (every 24h at 08:00 UTC) ==="
gcloud scheduler jobs create http $SCHEDULER_JOB_NAME \
  --location=$REGION \
  --schedule="0 8 * * *" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
  --message-body="{}" \
  --oauth-service-account-email=$SERVICE_ACCOUNT \
  --project=$PROJECT_ID

echo ""
echo "✅ Done! Keep-alive job scheduled daily at 08:00 UTC."
echo ""
echo "To test manually right now:"
echo "  gcloud run jobs execute $JOB_NAME --region=$REGION"
echo ""
echo "To view logs:"
echo "  gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME' --limit=20 --project=$PROJECT_ID"
