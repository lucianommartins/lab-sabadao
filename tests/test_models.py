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

"""Unit tests for model registry and param-based config functions."""

from gbench.core.models import (
    ModelCategory,
    ModelConfig,
    ModelFormat,
    ModelRegistry,
    Priority,
    registry,
    MODELS,
    PROJECT_ROOT,
)
from gbench.core.config import (
    get_batch_sizes,
    get_num_gpus,
    get_server_timeout,
    get_tensor_parallel,
    get_gpu_memory_utilization,
    get_max_model_len,
)


# ── Registry tests ──────────────────────────────────────────

def test_registry_starts_empty():
    """No built-in model table - the registry is populated on demand from
    --models ids via register_hf_model()."""
    assert len(MODELS) == 0


# NOTE: the previous nine registry tests iterated the built-in MODELS list / registry.filter(...),
# which are now permanently empty (models register on demand via register_hf_model). Those loop
# bodies never executed, so the tests passed vacuously with zero coverage; they were removed. The
# same invariants (params>0, local_path under models/gbench/, category<->supports_multimodal, and
# filter() correctness) are exercised on the LIVE dynamic-registration path by
# test_dynamic_registration_enriches below. test_gemma4_paths_not_under_gbench was also removed: it
# guarded an unreachable premise (register_hf_model always sets local_path under models/gbench/).


def test_dynamic_registration_enriches():
    """A --models HF id auto-registers and enriches from config.json (mocked in
    conftest): it resolves, has a local_path under models/gbench/, params are
    derived, category agrees with supports_multimodal, and it is filterable by category."""
    m = registry.get("google/gemma-4-enrich-test-it")  # '/' -> register_hf_model
    try:
        assert m is not None and m.hf_model_id
        assert registry.get(m.short_name) is m  # bare short-name now resolves
        assert m.local_path and "gbench" in m.local_path
        assert m.total_params_b > 0, "enrichment did not populate total_params_b"
        if m.category == ModelCategory.MULTIMODAL:
            assert m.supports_multimodal
        else:
            assert not m.supports_multimodal
        # filter() works over dynamically-registered models (replaces the old vacuous
        # MODELS-iteration filter tests, which looped an always-empty registry).
        assert m in registry.filter(category=m.category)
    finally:
        registry._models.pop(m.short_name.lower(), None)
        # Also purge the _by_category index that register_hf_model appends to, else a stale
        # ModelConfig lingers in the registry after the test.
        cat_list = getattr(registry, "_by_category", {}).get(m.category)
        if cat_list and m in cat_list:
            cat_list.remove(m)
        if m in MODELS:
            MODELS.remove(m)


# ── Param-based config function tests ───────────────────────

def test_gpu_allocation_fairness():
    """Models of similar size get the same GPU count."""
    # All ≤20B → 1 GPU
    assert get_num_gpus(1.0) == 1
    assert get_num_gpus(14.0) == 1
    assert get_num_gpus(20.0) == 1

    # All 20-80B → 2 GPUs
    assert get_num_gpus(25.0) == 2
    assert get_num_gpus(36.0) == 2
    assert get_num_gpus(80.0) == 2

    # All >80B → 8 GPUs
    assert get_num_gpus(100.0) == 8
    assert get_num_gpus(235.0) == 8


def test_tensor_parallel_matches_gpus():
    """TP size always equals GPU count."""
    for params in [1.0, 14.0, 30.0, 120.0, 235.0]:
        assert get_tensor_parallel(params) == get_num_gpus(params)


def test_batch_sizes_default_single_stream():
    """Default serving is single-stream ([1]) for every model size and preset.

    Concurrency sweeping is opt-in via --batch-sizes; sustained load is measured by
    the stress pillar (open-loop QPS), so the default serving pillar isolates
    single-stream latency. See BenchmarkConfig.get_batch_sizes."""
    assert get_batch_sizes(4.0, "quick") == [1]
    assert get_batch_sizes(4.0, "default") == [1]
    assert get_batch_sizes(30.0, "default") == [1]
    assert get_batch_sizes(120.0, "default") == [1]


def test_timeout_increases_with_params():
    """Larger models get longer timeouts."""
    t_small = get_server_timeout(4.0, False)
    t_large = get_server_timeout(30.0, False)
    t_xlarge = get_server_timeout(120.0, False)

    assert t_small < t_large < t_xlarge


def test_timeout_moe_multiplier():
    """MoE models get longer timeouts than dense at same size."""
    t_dense = get_server_timeout(30.0, False)
    t_moe = get_server_timeout(30.0, True)

    assert t_moe > t_dense


def test_timeout_bounds():
    """Timeouts are within floor/cap bounds."""
    assert get_server_timeout(1.0, False) >= 600
    assert get_server_timeout(500.0, True) <= 3600


def test_max_model_len_uniform():
    """max_model_len is always 4096."""
    assert get_max_model_len() == 4096


def test_gpu_memory_utilization():
    """GPU memory utilization is uniform 0.90."""
    assert get_gpu_memory_utilization(4.0) == 0.90
    assert get_gpu_memory_utilization(120.0) == 0.90
