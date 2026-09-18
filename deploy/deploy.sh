#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# deploy.sh - Build and deploy GBench UI dashboard to Google Cloud Run
#
# Usage:
#   ./deploy.sh
# ==============================================================================

set -e

# Optional local overrides. deploy/.env is matched by the `.env` rule in
# .gitignore, so a deployment's own values stay out of the repository. Write it
# with the `:=` form shown in deploy/.env.example, which leaves any value
# already in the environment untouched.
ENV_FILE="$(dirname "$0")/.env"
if [ -f "${ENV_FILE}" ]; then
  echo "Loading local overrides from ${ENV_FILE}"
  set -a
  . "${ENV_FILE}"
  set +a
fi

# Deployment-specific configuration. GCP_PROJECT and GBENCH_TF_STATE_BUCKET are
# required rather than defaulted, so that no single deployment's project is
# baked into this repository.
: "${GCP_PROJECT:?Set GCP_PROJECT to the GCP project to deploy into}"
: "${GBENCH_TF_STATE_BUCKET:?Set GBENCH_TF_STATE_BUCKET to the GCS bucket holding terraform state}"

PROJECT_ID=${GCP_PROJECT}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="${SERVICE_NAME:-gbench-dashboard}"
REPO_ID="${REPO_ID:-gbench}"
IMAGE_NAME="gbench-dashboard"
TAG="latest"

IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_ID}/${IMAGE_NAME}:${TAG}"

echo "========================================="
echo "Starting deployment of GBench UI/API to prod"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_URL}"
echo "========================================="

# 1. Configure docker authentication for Artifact Registry
echo "Configuring docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# 2. Build docker image
echo "Building GBench docker image..."
BUILDKIT_PROGRESS=plain docker build -t ${IMAGE_URL} .

# 3. Push docker image
echo "Pushing image to GCP Artifact Registry..."
docker push ${IMAGE_URL}

# 4. Deploy to Cloud Run via Terraform
echo "Deploying via Terraform..."
cd terraform

# Locate the local terraform binary or fall back to system command
TF_CMD="terraform"
if [ -f "$HOME/bin/terraform" ]; then
  TF_CMD="$HOME/bin/terraform"
fi

$TF_CMD init -backend-config="bucket=${GBENCH_TF_STATE_BUCKET}"
$TF_CMD apply \
  -var="project_id=${PROJECT_ID}" \
  -var="region=${REGION}" \
  -var="deploy_timestamp=$(date +%s)" \
  -auto-approve

echo "Deployment complete!"

