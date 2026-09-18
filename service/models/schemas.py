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

from pydantic import BaseModel
from typing import Any, Optional

class ScenarioNode(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
    children: Optional[list["ScenarioNode"]] = None

# Resolve self-reference for children in ScenarioNode
ScenarioNode.model_rebuild()

class RunTriggerRequest(BaseModel):
    model_id: str
    format: str = "hf"  # "hf", "gguf", "both"
    run_type: str = "quality"  # "performance", "quality", "all"
    preset: str = "default"  # "quick", "default"
    num_iterations: Optional[int] = None
    batch_sizes: Optional[list[int]] = None
    input_lengths: Optional[list[int]] = None
    output_lengths: Optional[list[int]] = None
    selected_scenarios: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    num_gpus: int = 1

class RunStatusResponse(BaseModel):
    run_id: str
    model_name: Optional[str] = None
    model_short: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: dict
