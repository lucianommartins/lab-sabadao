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

import sys
from unittest import mock

# Mock vllm and its submodules before anything else imports them
try:
    import vllm
except ImportError:
    vllm_mock = mock.MagicMock()
    class MockTaskType:
        GENERATION = "GENERATION"
    vllm_mock.benchmarks = mock.MagicMock()
    vllm_mock.benchmarks.serve = mock.MagicMock()
    vllm_mock.benchmarks.serve.TaskType = MockTaskType
    vllm_mock.benchmarks.datasets = mock.MagicMock()
    vllm_mock.tokenizers = mock.MagicMock()
    
    sys.modules["vllm"] = vllm_mock
    sys.modules["vllm.benchmarks"] = vllm_mock.benchmarks
    sys.modules["vllm.benchmarks.serve"] = vllm_mock.benchmarks.serve
    sys.modules["vllm.benchmarks.datasets"] = vllm_mock.benchmarks.datasets
    sys.modules["vllm.tokenizers"] = vllm_mock.tokenizers

import json
import os
import tempfile
from unittest import mock
import pytest

# Create a temporary config.json to return from mock hf_hub_download
_temp_dir = tempfile.TemporaryDirectory()
_dummy_config_path = os.path.join(_temp_dir.name, "config.json")

# A realistic dummy config that supports MoE and has typical params
_dummy_config = {
    "hidden_size": 2048,
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 256000,
    "max_position_embeddings": 131072,
    "architectures": ["Gemma3ForCausalLM"]
}

with open(_dummy_config_path, "w") as f:
    json.dump(_dummy_config, f)


# Start patches at module load time so they apply during import of core.models
_mock_download = mock.patch("huggingface_hub.hf_hub_download", return_value=_dummy_config_path)
_mock_download.start()

# Mock model_info to return something that doesn't have safetensors info
# so it falls back to arch estimation (which tests our math)
_mock_info = mock.patch("huggingface_hub.model_info")
_mock_info_func = _mock_info.start()
_mock_info_func.return_value = mock.MagicMock(safetensors=None)


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dir():
    yield
    _temp_dir.cleanup()
