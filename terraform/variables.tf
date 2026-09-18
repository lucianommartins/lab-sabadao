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

variable "project_id" {
  type        = string
  description = "The GCP project to deploy into. Required, so that no single deployment's project name is baked into the repository."
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "service_account" {
  type        = string
  default     = null
  description = "The service account email to run the Cloud Run service as. If not provided, defaults to gemma-gke-sa@<project_id>.iam.gserviceaccount.com"
}

variable "dashboard_service_name" {
  type        = string
  default     = "gbench-dashboard"
  description = "The name of the GBench Web Dashboard Cloud Run service."
}

variable "iap_members" {
  type        = list(string)
  default     = []
  description = "The list of users, groups, or domains permitted to access the IAP-secured dashboard. See access.auto.tfvars.example."
}

variable "eval_service_invokers" {
  type        = list(string)
  default     = []
  description = "The list of principals granted roles/run.invoker on the serving service. Empty by default, so a deployment grants access explicitly rather than inheriting somebody else's allowlist. See access.auto.tfvars.example."
}

variable "models_bucket" {
  type        = string
  default     = null
  description = "GCS bucket holding staged model artifacts. Defaults to <project_id>-model-artifacts."
}

variable "results_bucket" {
  type        = string
  default     = null
  description = "GCS bucket holding benchmark results. Defaults to <project_id>-results."
}

variable "deploy_timestamp" {
  type        = string
  default     = ""
  description = "Timestamp to force redeployment of Cloud Run services when image tag is static."
}

variable "serving_image_tag" {
  type        = string
  default     = "latest"
  description = "Tag of the pre-built vLLM serving image in the Artifact Registry repo, referenced as gemma-serving:<tag>. NOTE: deploy.sh / cloudbuild.yaml build only the dashboard image; build and push the serving image to <repo>/gemma-serving:<tag> yourself before deploying."
}


