#!/usr/bin/env python3
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

"""Helper script to generate model definitions for models.py from download_gemma.sh.

This ensures consistency between the download script and the model registry.
Run this whenever download_gemma.sh is updated.
"""

import re
from pathlib import Path

# Read download script
script_path = Path(__file__).parent / "download_gemma.sh"
script_content = script_path.read_text()

# Extract model definitions using regex
p0_pattern = r'\["([^"]+)"\]="([^:]+):([^"]+)"'
p0_models = re.findall(r'declare -A P0_MODELS=\(\s*((?:\s*\[.*\n)+)\)', script_content)
p1_models = re.findall(r'declare -A P1_MODELS=\(\s*((?:\s*\[.*\n)+)\)', script_content)
p2_models = re.findall(r'declare -A P2_MODELS=\(\s*((?:\s*\[.*\n)+)\)', script_content)

def parse_models(models_block):
    """Parse model definitions from bash associative array."""
    models = {}
    for match in re.finditer(p0_pattern, models_block):
        name, repo, file = match.groups()
        models[name] = {"repo": repo, "file": file}
    return models

if p0_models:
    print("P0 Models:")
    for model, info in parse_models(p0_models[0]).items():
        print(f"  {model}: {info['file']}")

if p1_models:
    print("\nP1 Models:")
    for model, info in parse_models(p1_models[0]).items():
        print(f"  {model}: {info['file']}")

if p2_models:
    print("\nP2 Models:")
    for model, info in parse_models(p2_models[0]).items():
        print(f"  {model}: {info['file']}")

print("\n✓ Review these filenames and update models.py accordingly")
