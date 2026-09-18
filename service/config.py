# -*- coding: utf-8 -*-
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

import os
from pathlib import Path

class Settings:
    # Environment control: 'local' or 'prod'
    GBENCH_ENV: str = os.getenv("GBENCH_ENV", "local").lower()

    # Paths
    # Resolve relative to this config file: gbench/service/config.py -> gbench/
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()

    # Where local results are written
    LOCAL_RESULTS_DIR: Path = BASE_DIR / "results"

    # Path to gemmaclaw cache (used for scenario scanning)
    DEFAULT_GEMMACLAW_CACHE: Path = Path("~/.cache/gbench/gemmaclaw").expanduser()
    GEMMACLAW_CACHE_PATH: Path = Path(os.getenv("GEMMACLAW_CACHE_PATH", DEFAULT_GEMMACLAW_CACHE))

    # GCS Configurations (for production mode).
    #
    # MODELS_BUCKET and GCP_PROJECT default to empty rather than to any one
    # deployment's names. The terraform stack sets both on the Cloud Run
    # container, so a deployment provisioned from terraform/ still gets them.
    RESULTS_BUCKET: str = os.getenv("GBENCH_RESULTS_BUCKET", "")
    MODELS_BUCKET: str = os.getenv("GBENCH_MODELS_BUCKET", "")

    # Cloud Run configurations
    VLLM_SERVICE_NAME: str = os.getenv("VLLM_SERVICE_NAME", "")
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "")
    GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")

    @property
    def is_local(self) -> bool:
        return self.GBENCH_ENV == "local"

settings = Settings()
