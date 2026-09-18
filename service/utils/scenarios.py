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

import logging
import subprocess
from pathlib import Path
from service.config import settings
from service.models.schemas import ScenarioNode

logger = logging.getLogger(__name__)

def _ensure_gemmaclaw_repo():
    repo_path = settings.GEMMACLAW_CACHE_PATH
    if not repo_path.exists():
        logger.info(f"Cloning gemmaclaw to {repo_path} to read scenarios...")
        try:
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/gemmaclaw/gemmaclaw.git", str(repo_path)],
                check=True
            )

        except Exception as e:
            logger.error(f"Failed to clone gemmaclaw repo: {e}")

def build_scenarios_tree() -> list[ScenarioNode]:
    """Build a tree representing the hierarchy of gemmaclaw QA scenarios."""
    _ensure_gemmaclaw_repo()
    scenarios_dir = settings.GEMMACLAW_CACHE_PATH / "qa" / "scenarios"
    if not scenarios_dir.exists():
        logger.warning(f"Scenarios directory does not exist at {scenarios_dir}")
        return []
    return _traverse_directory(scenarios_dir, scenarios_dir)

def _traverse_directory(dir_path: Path, base_dir: Path) -> list[ScenarioNode]:
    nodes = []
    try:
        # Sort items: directories first, then files
        items = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        for item in items:
            # Ignore hidden files/dirs, config folders, media, and general index
            if item.name.startswith(".") or item.name in ["index.md", "media", "config"]:
                continue
                
            relative_path = str(item.relative_to(base_dir))
            
            if item.is_dir():
                children = _traverse_directory(item, base_dir)
                if children:  # Only add directory if it contains at least one scenario
                    nodes.append(ScenarioNode(
                        name=item.name.replace("_", " ").title(),
                        path=relative_path,
                        type="directory",
                        children=children
                    ))
            elif item.suffix == ".md":
                # Convert kabob-case or snake_case filename to display name
                display_name = item.stem.replace("-", " ").replace("_", " ").title()
                nodes.append(ScenarioNode(
                    name=display_name,
                    path=relative_path,
                    type="file"
                ))
    except Exception as e:
        logger.error(f"Error traversing scenarios directory {dir_path}: {e}")
    return nodes
