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

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:gemma4

# This custom Dockerfile wraps the Vertex AI vLLM image. Build and push it to your own
# Artifact Registry so the container is served from a registry you control (set the
# base image below to whatever serving image your deployment uses).

# We can optionally bake the HuggingFace token in at build-time if needed,
# though passing it at runtime via Secret Manager is preferred.
# ENV HF_TOKEN=""

# We could also add a custom entrypoint script here if we needed to 
# download weights from GCS before starting vLLM, e.g.:
# COPY entrypoint.sh /usr/local/bin/
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# The Vertex AI base image already contains the highly optimized vLLM entrypoint.
