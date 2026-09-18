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

"""GBench UI Service entrypoint.

Provides a FastAPI backend service for managing benchmark runs, models,
scenarios, and analytics in the web UI.
"""

from gbench import __version__
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.routes import models, runs, scenarios, analytics
from service.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GBench UI API", version=__version__)

# Allow CORS for local development of frontend
if settings.is_local:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])

@app.get("/api/health")
def health_check():
    return {"status": "ok", "environment": settings.GBENCH_ENV}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

ui_dist_path = Path(__file__).parent.parent / "ui" / "dist"
if ui_dist_path.exists():
    app.mount("/assets", StaticFiles(directory=str(ui_dist_path / "assets")), name="assets")

    @app.get("/{catchall:path}")
    def serve_ui(catchall: str):
        # Try to serve direct static file
        file_path = ui_dist_path / catchall
        if file_path.is_file():
            return FileResponse(file_path)
        # Default to index.html for SPA client routing
        return FileResponse(ui_dist_path / "index.html")
else:
    logger.warning("ui/dist folder not found. UI static files will not be served.")

