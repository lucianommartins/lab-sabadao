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

terraform {
  required_providers {
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 6.31.0"
    }
  }
  # Partial backend configuration. The state bucket is deployment-specific, so
  # supply it at init time rather than committing it:
  #   terraform init -backend-config="bucket=YOUR_TF_STATE_BUCKET"
  backend "gcs" {
    prefix = "state"
  }
}


provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Bucket names default to a project-derived convention so a fresh deployment
# needs one input rather than three. Override either variable to point at
# buckets you already own.
locals {
  models_bucket  = coalesce(var.models_bucket, "${var.project_id}-model-artifacts")
  results_bucket = coalesce(var.results_bucket, "${var.project_id}-results")
}

# Create a dedicated Artifact Registry for the gbench custom images
resource "google_artifact_registry_repository" "gbench_repo" {
  provider      = google-beta
  location      = var.region
  repository_id = "gbench"
  description   = "Docker repository for gbench Gemma 4 evaluations"
  format        = "DOCKER"
}

resource "google_cloud_run_service" "gemma_eval_service" {
  provider = google-beta
  name     = "gbench-eval-service"
  location = var.region
  project  = var.project_id


  metadata {
    annotations = {
      "run.googleapis.com/launch-stage" = "BETA"
    }
  }

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"                 = "1"
        "autoscaling.knative.dev/maxScale"                 = "1"
        "run.googleapis.com/execution-environment"         = "gen2"
        "run.googleapis.com/gpu-zonal-redundancy-disabled" = "true"
        "run.googleapis.com/accelerator"                   = "nvidia-l4"
        "run.googleapis.com/cpu-throttling"                = "false"
      }
    }
    spec {
      service_account_name = coalesce(var.service_account, "gemma-gke-sa@${var.project_id}.iam.gserviceaccount.com")
      timeout_seconds      = 600
      container_concurrency = 64
      containers {
        image = "${google_artifact_registry_repository.gbench_repo.location}-docker.pkg.dev/${google_artifact_registry_repository.gbench_repo.project}/${google_artifact_registry_repository.gbench_repo.repository_id}/gemma-serving:${var.serving_image_tag}"
        
        args = [
          "vllm",
          "serve",
          "--model=gemma-4-E2B-it",
          "--enable-chunked-prefill",
          "--enable-prefix-caching",
          "--generation-config=auto",
          "--enable-auto-tool-choice",
          "--tool-call-parser=gemma4",
          "--reasoning-parser=gemma4",
          "--dtype=bfloat16",
          "--max-num-seqs=64",
          "--gpu-memory-utilization=0.85",
          "--max-model-len=131072",
          "--tensor-parallel-size=1",
          "--port=8080",
          "--host=0.0.0.0"
        ]

        env {
          name  = "MODEL_ID"
          value = "gemma-4-E2B-it"
        }
        env {
          name  = "AIP_STORAGE_URI"
          value = "gs://${local.models_bucket}/gemma-4-E2B-it/"
        }

        resources {
          limits = {
            cpu    = "8"
            memory = "32Gi"
            "nvidia.com/gpu" = "1"
          }
        }

        startup_probe {
          initial_delay_seconds = 10
          period_seconds        = 10
          failure_threshold     = 60  # 60 * 10s = 600s max startup window (10 mins for HF download)
          timeout_seconds       = 5
          tcp_socket {
            port = 8080
          }
        }
      }
    }
  }
}

resource "google_cloud_run_service_iam_member" "binding" {
  provider = google-beta
  for_each = toset(var.eval_service_invokers)
  project  = google_cloud_run_service.gemma_eval_service.project
  location = google_cloud_run_service.gemma_eval_service.location
  service  = google_cloud_run_service.gemma_eval_service.name
  role     = "roles/run.invoker"
  member   = each.key
}

# Fetch the project metadata (number) dynamically to configure the IAP service agent
data "google_project" "project" {
  provider   = google-beta
  project_id = var.project_id
}

# GBench Web Dashboard Cloud Run service (v2) with direct IAP integration enabled
resource "google_cloud_run_v2_service" "gbench_dashboard" {
  provider = google-beta
  name     = var.dashboard_service_name
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${google_artifact_registry_repository.gbench_repo.location}-docker.pkg.dev/${google_artifact_registry_repository.gbench_repo.project}/${google_artifact_registry_repository.gbench_repo.repository_id}/gbench-dashboard:latest"

      ports {
        container_port = 8080
      }


      env {
        name  = "GBENCH_REMOTE_ENDPOINT"
        value = "${google_cloud_run_service.gemma_eval_service.status[0].url}/v1"
      }

      env {
        name  = "GBENCH_RESULTS_BUCKET"
        value = local.results_bucket
      }

      # Passed explicitly so the service code can default these to empty
      # rather than carrying a deployment's project name as a literal.
      env {
        name  = "GBENCH_MODELS_BUCKET"
        value = local.models_bucket
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }

      env {
        name  = "DEPLOY_TIMESTAMP"
        value = var.deploy_timestamp
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "4096Mi"
        }
        cpu_idle = false
      }
    }
    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }
  }

  iap_enabled = true
}


# Authorize permitted users to access the dashboard through IAP
resource "google_iap_web_cloud_run_service_iam_member" "gbench_dashboard_iap_member" {
  provider               = google-beta
  for_each               = toset(var.iap_members)
  project                = google_cloud_run_v2_service.gbench_dashboard.project
  location               = google_cloud_run_v2_service.gbench_dashboard.location
  cloud_run_service_name = google_cloud_run_v2_service.gbench_dashboard.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.key
}

# Authorize the GCP Identity-Aware Proxy service agent to invoke the dashboard container
resource "google_cloud_run_v2_service_iam_member" "gbench_dashboard_iap_invoker" {
  provider = google-beta
  project  = google_cloud_run_v2_service.gbench_dashboard.project
  location = google_cloud_run_v2_service.gbench_dashboard.location
  name     = google_cloud_run_v2_service.gbench_dashboard.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-iap.iam.gserviceaccount.com"
}

