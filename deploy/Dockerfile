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

# Build Vite React UI
FROM node:22-slim AS ui-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
RUN npm run build

# Final Python backend container
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (git, curl, node 22, pnpm)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

# Pre-clone and pre-install gemmaclaw to speed up runtime and avoid OOM
RUN mkdir -p /root/.cache/gbench && \
    git clone --depth 1 https://github.com/gemmaclaw/gemmaclaw.git /root/.cache/gbench/gemmaclaw && \
    cd /root/.cache/gbench/gemmaclaw && \
    pnpm install && \
    node scripts/run-node.mjs --version && \
    git status && \
    commit_hash=$(git rev-parse HEAD) && \
    mkdir -p dist && \
    echo "{\"head\": \"${commit_hash}\", \"builtAt\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"}" > dist/.buildstamp


# Install python dependencies for UI and service (bypassing heavy vLLM / PyTorch dependencies)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    google-cloud-storage \
    google-api-python-client \
    pandas \
    pyarrow \
    numpy \
    huggingface-hub \
    transformers

# Copy source code and install GBench package without dependencies
COPY . .
RUN pip install -e . --no-deps

# Copy Vite UI static assets from builder stage
COPY --from=ui-builder /app/ui/dist ./ui/dist

# Expose port and start server
EXPOSE 8080
ENV GBENCH_ENV=prod
ENV PORT=8080

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
