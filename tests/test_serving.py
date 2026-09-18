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

"""Serving runner regression tests."""

from gbench.core.config import BenchmarkConfig
from gbench.runners.serving import ServingBenchmarkRunner


def test_no_think_fields_disables_reasoning_by_default():
    """The perf path (serving/stress) must run no-think by default so a reasoning
    prefix doesn't pollute the controlled fixed-length workload. It sends BOTH
    switches because different backends honor different ones (verified live):
      - reasoning_effort:'none'          -> honored by Ollama's /v1
      - chat_template_kwargs.enable_thinking:False -> honored by vLLM
    Backends that don't support a field ignore it (HTTP 200, no error)."""
    r = ServingBenchmarkRunner(BenchmarkConfig())
    assert r.no_think_fields() == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }


def test_no_think_fields_empty_under_thinking():
    """--thinking must NOT suppress reasoning (the model reasons on purpose)."""
    cfg = BenchmarkConfig()
    cfg.thinking = True
    assert ServingBenchmarkRunner(cfg).no_think_fields() == {}
