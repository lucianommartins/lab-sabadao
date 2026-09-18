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

"""Unit tests for gbench custom evaluation plugin loader and JSONL runner."""

from pathlib import Path
import pytest
from gbench.runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
from gbench.runners.eval_suites.custom_jsonl import _load_custom_jsonl_samples, _eval_custom_jsonl
from gbench.runners.eval_suites import SUITES


def test_plugin_discovery_and_registration():
    examples_dir = Path(__file__).parent.parent / "examples" / "custom_evals"
    assert examples_dir.exists()

    discovered = discover_and_register_plugins([str(examples_dir)])
    assert "custom_qa" in discovered
    assert "custom_qa" in SUITES
    assert "custom_qa" in CUSTOM_PILLARS
    assert "CUSTOM DOMAIN" in CUSTOM_PILLARS["custom_qa"]


def test_custom_jsonl_loader():
    jsonl_file = Path(__file__).parent.parent / "examples" / "custom_evals" / "sample_benchmark.jsonl"
    assert jsonl_file.exists()

    samples = _load_custom_jsonl_samples(jsonl_file, limit=3)
    assert len(samples) == 3
    messages, gold, extra = samples[0]
    assert len(messages) >= 1
    assert "speed of light" in messages[0]["content"].lower()
    # The gold now carries the row's `eval_type` so the scorer can honour it; without
    # this the declared type stayed in the sample metadata, which eval_fn never sees.
    assert gold == {"__gbench_eval_type": "contains", "gold": "299792458"}
    assert extra["category"] == "Physics"
    assert extra["eval_type"] == "contains"


def test_custom_jsonl_eval_matching():
    # Exact / Substring
    assert _eval_custom_jsonl("The capital of France is Paris.", "Paris") is True
    assert _eval_custom_jsonl("London", "Paris") is False

    # Numeric
    assert _eval_custom_jsonl("The calculated value is 210 exactly.", "210") is True
    assert _eval_custom_jsonl("The answer is 210.00", "210") is True
    assert _eval_custom_jsonl("The answer is 195", "210") is False

    # Choice
    assert _eval_custom_jsonl("The correct answer is (B).", "B") is True
    assert _eval_custom_jsonl("Option C", "B") is False


def test_plugin_dynamic_sample_loading(tmp_path):
    plugin_file = tmp_path / "dynamic_sample_plugin.py"
    plugin_code = """
from typing import List, Tuple, Dict, Any

PILLAR = "Dynamic Domain"

def _load_samples(enable_thinking=False, limit=None):
    return [([{"role": "user", "content": "What is 2+2?"}], "4", {"category": "math"})]

def run_dynamic_sample_plugin(model_name, base_url, **kwargs):
    return {"accuracy": 100.0}
"""
    plugin_file.write_text(plugin_code)
    discovered = discover_and_register_plugins([str(tmp_path)])
    assert "dynamic_sample_plugin" in discovered
    assert CUSTOM_PILLARS.get("dynamic_sample_plugin") == "Dynamic Domain"


def test_recursive_subdirectory_discovery(tmp_path):
    sub = tmp_path / "domain" / "specialized"
    sub.mkdir(parents=True)
    plugin_code = """
PILLAR = "Domain Specialized"
def run_deep_specialized_test(model_name, base_url, **kwargs):
    return {"score": 1.0}
"""
    (sub / "deep_specialized_test.py").write_text(plugin_code)

    discovered = discover_and_register_plugins([str(tmp_path)])
    assert "deep_specialized_test" in discovered
    assert CUSTOM_PILLARS.get("deep_specialized_test") == "Domain Specialized"





def test_custom_jsonl_honours_the_declared_eval_type():
    from gbench.runners.eval_suites.custom_jsonl import _ET_KEY

    def graded(resp, gold, eval_type):
        return _eval_custom_jsonl(resp, {_ET_KEY: eval_type, "gold": gold})

    # `exact` must not pass on a substring, which every row previously did
    assert graded("The capital of France is Paris.", "Paris", "contains") is True
    assert graded("The capital of France is Paris.", "Paris", "exact") is False
    assert graded("Paris", "Paris", "exact") is True

    # `numeric` must not pass on incidental text, and `exact` must not fall through to it
    assert graded("The answer is 210.00", "210", "numeric") is True
    assert graded("Section 210 discusses this", "210", "exact") is False

    # `multiple_choice` takes the model's stated choice, not the first letter it mentions
    assert graded("A looks tempting, but the answer is (C).", "C", "multiple_choice") is True
    assert graded("A looks tempting, but the answer is (C).", "A", "multiple_choice") is False

    # an unknown type falls back to the documented default rather than failing the row
    assert graded("The capital of France is Paris.", "Paris", "something_else") is True
