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

"""Benchmark configuration management.

This module derives batch sizes and timeouts directly from model parameters
(total_params_b and is_moe), rather than hand-tuned lookup tables.

GPU allocation is NOT auto-applied to the served run: the user must pass
``--num-gpus`` for local performance runs (see cli.py's local-run guard).
``get_num_gpus`` below encodes the fair-tier RECOMMENDATION and is used by
``validate_gpu_config`` to warn/abort when the requested count is off-tier or
won't fit - it does not itself set the tensor-parallel degree. The fair tiers:
    ≤20B total params  → 1 GPU
    ≤80B total params  → 2 GPUs
    >80B total params  → 8 GPUs
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Param-based benchmark functions ─────────────────────────


def get_num_gpus(total_params_b: float) -> int:
    """Derive GPU count from total model parameters.

    Uses discrete tiers for fair benchmarking - all models in the same
    size range get identical GPU allocation regardless of vendor.

    Boundaries chosen so competing models always share the same tier:
      ≤20B:  1 GPU  - small/mid dense and edge MoE
      ≤80B:  2 GPUs - large dense (23-36B), mid MoE
      >80B:  8 GPUs - all large MoE (100B+) on equal footing
    """
    if total_params_b <= 20:
        return 1
    elif total_params_b <= 80:
        return 2
    else:
        return 8


def get_tensor_parallel(total_params_b: float) -> int:
    """Tensor parallel size = GPU count (1:1 mapping)."""
    return get_num_gpus(total_params_b)


def get_batch_sizes(total_params_b: float, preset: str = "default") -> list[int]:
    """Default concurrency is a single stream (batch size 1) for every model and
    preset. Concurrency sweeping is opt-in via --batch-sizes; the load dimension
    is otherwise covered by the stress test's arrival-rate (QPS) sweep.

    Args:
        total_params_b: Total model parameters in billions (unused; kept for API).
        preset: Benchmark preset (unused; kept for API).
    """
    return [1]


# ── Per-campaign SLOs (capacity-under-SLO) ─────────────────────────────────────
# TTFT (first-byte) budget scales with prefill/input size - an 8k-token prompt
# legitimately tolerates a longer first token than a quick chat. ITL/TBT (decode
# cadence) is prompt-independent (~human reading speed, ~10 tok/s). These are
# DEFAULTS: overridable globally (--stress-threshold / --itl-slo-ms) or per
# campaign, and frozen into the run fingerprint so a run's SLOs are reproducible
# and applied IDENTICALLY to every model being compared (tunable across runs,
# fixed within a run).
CAMPAIGN_SLOS: dict[str, dict[str, int]] = {
    "chat-like":     {"ttft_ms": 1000, "itl_ms": 100},
    "decode-heavy":  {"ttft_ms": 1000, "itl_ms": 100},
    "mixed":         {"ttft_ms": 2500, "itl_ms": 100},
    "agentic":       {"ttft_ms": 5000, "itl_ms": 100},
    "prefill-heavy": {"ttft_ms": 5000, "itl_ms": 100},
    "long-decode":   {"ttft_ms": 5000, "itl_ms": 100},
}
_DEFAULT_SLO = {"ttft_ms": 5000, "itl_ms": 200}   # no-campaign fallback


def resolve_campaign_slo(campaign, ttft_override=None, itl_override=None) -> tuple[int, int]:
    """Resolve (ttft_ms, itl_ms) for a campaign.

    Precedence: explicit override (global, applies to all campaigns) > per-campaign
    default > no-campaign fallback. A None override means 'use the per-campaign
    default' - which is why the CLI SLO flags default to None, not a number.
    """
    base = CAMPAIGN_SLOS.get(campaign, _DEFAULT_SLO)
    ttft = ttft_override if ttft_override is not None else base["ttft_ms"]
    itl = itl_override if itl_override is not None else base["itl_ms"]
    return int(ttft), int(itl)


def campaign_ctx(campaign, input_lengths, output_lengths) -> int:
    """Auto-derive the served max_model_len for a campaign.

    Clears the RandomDataset 1.5x length tail (stress samples range_ratio=0.5) with
    slack, rounded up to 256. sharegpt/chat-like keeps the 4096 default (the loader
    filters prompts to <=2048). This removes the 4-of-6-campaign truncation/abort
    risk for direct gbench users who don't pass --max-model-len.
    """
    # sharegpt/chat-like: loader filters prompts to <=2048, so 4096 always fits.
    if campaign == "chat-like" or not input_lengths or not output_lengths:
        return 4096
    in_len = int(input_lengths[0])
    out_len = int(output_lengths[0])
    need = int(1.5 * (in_len + out_len)) + 256
    return max(4096, ((need + 255) // 256) * 256)


def get_server_timeout(total_params_b: float, is_moe: bool) -> int:
    """Derive vLLM server startup timeout from model parameters.

    Generous timeouts for parallel execution - under parallel GPU load,
    weight loading competes for PCIe/NVLink bandwidth and CUDA graph
    compilation is slower. Uses 3x safety margin over solo estimates.

    Args:
        total_params_b: Total model parameters in billions.
        is_moe: Whether the model uses Mixture of Experts.

    Returns:
        Timeout in seconds. Floor 600s, cap 3600s.
    """
    solo = 300 + total_params_b * 10
    parallel = solo * (1.3 if is_moe else 1.0) * 3
    return int(max(600, min(3600, parallel)))


def get_gpu_memory_utilization(total_params_b: float) -> float:
    """GPU memory utilization - uniform 0.90 for fair comparison.

    Leaves ~8GB headroom per 80GB GPU for CUDA graphs + sampler.
    """
    return 0.90


def get_max_model_len(override: Optional[int] = None) -> int:
    """Context length to serve at.

    Returns ``override`` when a positive value is given (the
    ``--max-model-len`` / ``config.max_model_len`` escape hatch for
    long-context campaigns), otherwise the uniform 4096 default that
    matches the MLPerf v4.0 standard context for chat workloads. The
    default prevents OOM from models with large native contexts (10M+,
    128K+) and avoids penalizing small models with excessive KV cache
    reservation; overriding it trades that uniformity for reach.
    """
    if override and override > 0:
        return override
    return 4096


def estimate_required_vram_gb(
    total_params_b: float, bytes_per_param: float = 2.0
) -> float:
    """Estimate the VRAM needed to serve a model, in GB.

    Weights dominate: ``total_params_b * bytes_per_param``. The default of
    2.0 bytes/param is the bf16 weight footprint gbench benchmarks. For
    MoE models this uses TOTAL params, not active - every expert stays
    resident in VRAM regardless of per-token routing, so a "26B-A4B" costs
    ~52GB, not ~8GB.

    Adds a fixed ~10GB for CUDA graphs, activations, and a minimal KV cache
    at get_max_model_len() (4096). This is a floor for a "does it physically
    fit" check, not a capacity-planning estimate.
    """
    weights_gb = total_params_b * bytes_per_param
    runtime_overhead_gb = 10.0
    return weights_gb + runtime_overhead_gb


# The GemmaClaw commit the quality pillar scores against unless
# ``--gemmaclaw-commit`` overrides it.
#
# A sha and not ``main``, because this is the value almost every run uses
# and it is hashed into the quality ``scaffold_id``. A default that tracks
# a branch makes two unflagged runs a week apart two different experiments,
# and the id correctly moves to say so, which means the series breaks for a
# reason nobody chose. Pinning it here is what makes the default
# reproducible: the same bare ``gbench --quality-only`` gives the same
# scaffold_id in six months.
#
# Promoted by hand, which is the point rather than a shortcoming. Pick the
# newest ``gemmaclaw-v*`` tag, put its sha here, update the tag name and
# date below, and say so in the PR. Changing the scorer is then a
# deliberate reviewable commit instead of something that happens to you.
# ``scripts/`` has no automation for this on purpose.
#
# Currently gemmaclaw-v2026.8.3, tagged 2026-06-23.
DEFAULT_GEMMACLAW_COMMIT = "d8bc6989133e0e93535b5ad84e6acda019642839"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Includes optimized vLLM settings for maximum throughput:
    - gpu_memory_utilization: 0.90 (uniform headroom for CUDA graphs)
    - enable_chunked_prefill: True (~20-30% throughput improvement)
    - max_num_batched_tokens: 16384 (optimal scheduler throughput)
    """

    # Iteration control for statistical reliability
    num_iterations: int = 3  # Run each config N times
    warmup_iterations: int = 1  # Warmup runs (not included in stats)
    min_acceptable_cv_percent: float = 5.0  # Max CV% for variance check

    # --personal (device-peak) mode. serving_num_prompts_override forces a fixed
    # small serving sample count (the real lever - _serving_num_prompts, not
    # num_prompts); personal_mode switches the report to the device-peak block.
    serving_num_prompts_override: Optional[int] = None
    personal_mode: bool = False
    # Images per MM request. None -> the runner default (4). --personal sets 1:
    # lighter (280 vs 1120 img tokens) AND universally supported - a single image
    # needs no server-side --limit-mm-per-prompt>1, which a remote endpoint
    # (e.g. Ollama) may not have configured.
    mm_images_per_request: Optional[int] = None
    # Seconds to wait for the remote multimodal capability probe. None -> 10.
    # --personal defaults it higher (cold CPU vision is slow to first token, so
    # 10s falsely reads as "unsupported"). Set via --mm-probe-timeout.
    mm_probe_timeout: Optional[int] = None
    # Per-request serving timeout in SECONDS (set via --timeout). Applied to both
    # remote paths, but the semantics differ by transport and are documented here:
    #   * text (urllib): a per-read STALL timeout - fires when the server sends no
    #     bytes for this long (e.g. a hung prefill), NOT a cap on total decode.
    #   * MM (aiohttp): a TOTAL wall-clock cap on the whole request.
    # Default 600 is generous for personal single-stream CPU runs; raise it for
    # large models / long outputs on slow devices, lower it to fail hangs faster.
    timeout: int = 600
    # Thinking / reasoning mode (set via --thinking). Default False: gbench sends a
    # no-think system prompt on chat requests so a reasoning model (e.g. gemma-4 on
    # Ollama) measures pure answer generation, not an accidental chain-of-thought
    # workload it inherits from the backend's default system prompt. --thinking
    # skips the override and lets the model reason (reasoning tokens are counted).
    thinking: bool = False

    # Batch sizes to test
    batch_sizes: list[int] = field(
        default_factory=lambda: [1]   # single-stream default; override with --batch-sizes
    )

    # Input lengths for throughput benchmarks (tokens)
    input_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 2048]
    )

    # Output lengths for throughput benchmarks (tokens)
    output_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 1024]
    )

    # Number of images per request (multimodal only)
    images_per_request: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )

    # Base configuration
    num_prompts: int = 200
    num_prompts_throughput: int = 200
    request_rate: str = "inf"
    dataset: str = "random"
    dataset_path: Optional[str] = None
    dataset_multimodal: str = "random-mm"
    remote_endpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    # Output configuration
    results_dir: Path = field(
        default_factory=lambda: Path("./results")
    )
    enable_logging: bool = True
    log_samples: bool = False

    # Resource limits and vLLM optimizations
    gpu_memory_utilization: float = 0.90
    max_num_seqs: Optional[int] = 256
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 16384
    # Override for the served context window. None keeps the uniform 4096
    # default (get_max_model_len); raise it to run long-context campaigns
    # (agentic/prefill-heavy/mixed/long-decode) whose input+output exceed
    # 4096. Recorded into the scaffold fingerprint, so a non-default value
    # is a distinct experiment.
    max_model_len: Optional[int] = None
    # True when the user pinned a workload geometry (a --campaign or explicit
    # --input/--output-lengths). The stress test reads this to decide whether
    # to probe a real campaign shape/dataset or fall back to its neutral
    # 128/128 laptop default.
    workload_shape_explicit: bool = False
    # Active campaign name (set by apply_campaign_to_config). Drives per-campaign
    # SLO resolution and auto max_model_len; recorded in the run fingerprint.
    campaign: Optional[str] = None

    # Stress-test knobs. All four resolve flag > GBENCH_STRESS_* env > runner
    # default. stress_reps: full sweeps per point (None -> runner default 3; the
    # knee is reported as mean+CI over reps since a single sweep varies run-to-run).
    # stress_client_procs: OS processes for the multi-process load generator
    # (None -> runner default 8, capped to cpu_count-1) - keeps the client from
    # becoming the knee.
    stress_reps: Optional[int] = None
    stress_client_procs: Optional[int] = None
    # stress_max_qps: safety-cap on the open-loop arrival sweep (None -> runner
    # default 512). Raise it when a fast model saturates the 512 cap (the knee is
    # then censored at 512, not measured). stress_max_prompts: cap on requests per
    # rate point (None -> runner default 1200); raise it so high-QPS points still
    # span a real steady-state window instead of a sub-second burst. Both also read
    # GBENCH_STRESS_MAX_QPS / GBENCH_STRESS_MAX_PROMPTS (flag wins over env).
    stress_max_qps: Optional[float] = None
    stress_max_prompts: Optional[int] = None
    # stress_min_samples: min steady-state completions to trust a rate point (None ->
    # runner default 15). Lower it for a slow box that meets the SLO but cannot
    # complete 15 per point (else every point is TOO-FEW and the knee is 0); a smaller
    # floor gives a faster, wider-CI, lower-confidence knee.
    stress_min_samples: Optional[int] = None

    # Multi-GPU configuration
    num_gpus: int = 1
    tensor_parallel_size: Optional[int] = None

    # Execution control
    dry_run: bool = False
    skip_existing: bool = False

    # Quality benchmark (gemmaclaw) configuration
    gemmaclaw_commit: str = DEFAULT_GEMMACLAW_COMMIT
    gemmaclaw_path: Optional[str] = None
    remote_endpoint: Optional[str] = None
    selected_scenarios: Optional[list[str]] = None

    # Golden Set benchmark configuration
    golden: bool = False
    golden_only: bool = False
    selected_golden_tasks: Optional[list[str]] = None
    # Model name to put in the golden request payload. Leave unset to let
    # the runner resolve it from the endpoint's /models listing.
    golden_model_id: Optional[str] = None

    # Evaluation suites configuration
    evals: Optional[list[str]] = None
    eval_thinking: bool = False
    eval_max_output_tokens: Optional[int] = None
    eval_max_soft_tokens: int = 1120
    eval_n_shot: Optional[int] = None  # None -> suite canonical default (mmlu/mmlu_pro: 5-shot); explicit 0 = 0-shot
    #: Per-suite wall-clock budget (s). None -> per-suite defaults in evals.py.
    suite_timeout: Optional[int] = None
    eval_categories: Optional[str] = None
    eval_limit: Optional[int] = None
    sandboxes: Optional[int] = None
    temperature: Optional[float] = None
    attempt_count: int = 1
    eval_plugins_dir: Optional[list[str]] = None
    eval_custom_jsonl: Optional[str] = None

    # Run metadata tracking
    tags: Optional[list[str]] = None

    def __post_init__(self):
        """Validate configuration. LogManager is deferred to initialize()."""
        self.log_manager = None

    def initialize(self):
        """Create LogManager after all config fields (incl. --results-dir) are set.

        Must be called after CLI argument overrides are applied to this config.
        """
        from gbench.utils import LogManager

        self.log_manager = LogManager(self.results_dir)
        self.results_dir = self.log_manager.results_dir

    def _serving_num_prompts(self, batch: int) -> tuple[int, bool]:
        """Sample count for one serving config, sized to a wall-clock target.

        Problem-B fix: a flat count is the wrong knob - request cost spans 3
        orders of magnitude across campaigns, and single-stream (batch 1) is
        slow regardless of count. So target a fixed wall-clock per config:
        wall_clock ~= n * out_len / (batch * tok/s), solved for n. Samples thus
        scale UP with concurrency and DOWN with output length, keeping each
        campaign's real shape while bounding runtime. Fast/high-concurrency
        configs get many samples (tight P99); slow/low-concurrency ones get few
        (bounded) and are flagged low_confidence so a thinly-sampled P99 is never
        passed off as solid. (Very-long-output campaigns still hit the MIN floor
        and stay expensive at low batch - those are better skipped at the sweep
        level; stress TPOT + offline throughput already cover them.)
        """
        RELIABLE_P99 = 150               # below this, P99 is low-confidence
        # --personal: a fixed small count (device tok/s converges fast; the wall-
        # clock auto-sizer's TPS_PER_STREAM=30 is GPU-calibrated and wrong on CPU).
        if self.serving_num_prompts_override is not None:
            n = max(1, int(self.serving_num_prompts_override))
            return n, (n < RELIABLE_P99)
        TARGET_CONFIG_S = 300            # ~5 min wall-clock target per config
        TPS_PER_STREAM = 30              # rough single-stream decode tok/s
        MIN_PROMPTS, MAX_PROMPTS = 30, 500
        out_len = int(self.output_lengths[0]) if self.output_lengths else 512
        n = int(TARGET_CONFIG_S * max(int(batch), 1) * TPS_PER_STREAM / max(out_len, 64))
        n = max(MIN_PROMPTS, min(n, MAX_PROMPTS))
        return n, (n < RELIABLE_P99)

    def get_serving_configs(self) -> list[dict]:
        """Generate all serving benchmark configurations."""
        configs = []
        for batch in self.batch_sizes:
            n, low_conf = self._serving_num_prompts(batch)
            configs.append({
                "batch_size": batch,
                "num_prompts": n,
                "request_rate": self.request_rate,
                "dataset": self.dataset,
                "low_confidence": low_conf,
            })
        return configs

    def get_throughput_configs(self) -> list[dict]:
        """Generate all throughput benchmark configurations."""
        configs = []
        for input_len in self.input_lengths:
            for output_len in self.output_lengths:
                for batch in self.batch_sizes:
                    configs.append({
                        "input_length": input_len,
                        "output_length": output_len,
                        "batch_size": batch,
                        "num_prompts": self.num_prompts_throughput,
                    })
        return configs

    def get_multimodal_configs(self) -> list[dict]:
        """Generate multimodal-specific configurations."""
        configs = []
        for num_images in self.images_per_request:
            for batch in self.batch_sizes:
                configs.append({
                    "num_images": num_images,
                    "batch_size": batch,
                    "num_prompts": self.num_prompts,
                    "dataset": self.dataset_multimodal,
                })
        return configs


@dataclass
class PerformanceTargets:
    """Target performance metrics for validation."""

    ttft_p50_target_ms: float = 200.0
    ttft_p95_target_ms: float = 500.0
    itl_p50_target_ms: float = 50.0
    itl_p95_target_ms: float = 100.0
    tpot_target_ms: float = 100.0
    throughput_target_tps: float = 1000.0
    memory_target_gb: float = 45.0
    gguf_quality_delta_max: float = 2.0
    repeatability_variance_max: float = 5.0

    def validate_serving_results(
        self, results: dict
    ) -> dict[str, bool]:
        """Validate serving benchmark results against targets."""
        checks = {}
        checks["ttft_p50"] = (
            results.get("ttft_p50", float("inf"))
            <= self.ttft_p50_target_ms
        )
        checks["ttft_p95"] = (
            results.get("ttft_p95", float("inf"))
            <= self.ttft_p95_target_ms
        )
        checks["itl_p50"] = (
            results.get("itl_p50", float("inf"))
            <= self.itl_p50_target_ms
        )
        checks["itl_p95"] = (
            results.get("itl_p95", float("inf"))
            <= self.itl_p95_target_ms
        )
        return checks


# Default configurations
DEFAULT_CONFIG = BenchmarkConfig()
DEFAULT_TARGETS = PerformanceTargets()

# Quick test configuration (smoke test)
QUICK_CONFIG = BenchmarkConfig(
    num_iterations=1,
    warmup_iterations=0,
    batch_sizes=[1],
    input_lengths=[128],
    output_lengths=[512],
    num_prompts=10,
    num_prompts_throughput=10,
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
    max_num_batched_tokens=16384,
)

# Default configuration (production benchmark)
DEFAULT_CONFIG = BenchmarkConfig(
    num_iterations=3,
    warmup_iterations=1,
    batch_sizes=[1],   # single-stream default; override with --batch-sizes
    input_lengths=[128],
    output_lengths=[512],
    num_prompts=1000,
    num_prompts_throughput=1000,
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
    max_num_batched_tokens=16384,
)


def get_available_gpus() -> int:
    """Get the number of available NVIDIA GPUs.

    Respects CUDA_VISIBLE_DEVICES if set.
    """
    import subprocess

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return len([x.strip() for x in cuda_visible.split(",") if x.strip()])

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.splitlines() if "GPU" in l])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _gpu_total_vram_gb() -> float:
    """Total VRAM of the smallest visible GPU, in GB (0.0 if unknown).

    Queries nvidia-smi ``memory.total``, respecting CUDA_VISIBLE_DEVICES so
    it reflects the GPUs a run will actually use. Returns 0.0 when
    nvidia-smi is absent or unreadable, so callers can skip the VRAM check
    (fail open) rather than block a run on a missing tool.
    """
    import subprocess

    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_ids:
        cmd.append("--id=" + gpu_ids)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            totals = [
                float(x) / 1024
                for x in result.stdout.split()
                if x.strip()
            ]
            if totals:
                return min(totals)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0.0


def _gpu_min_free_vram_gb() -> float:
    """Free VRAM of the smallest visible GPU, in GB (0.0 if unknown).

    Mirrors ``_gpu_total_vram_gb`` but queries ``memory.free``; respects
    CUDA_VISIBLE_DEVICES. 0.0 (fail-open) when nvidia-smi is absent/unreadable.
    """
    import subprocess

    cmd = ["nvidia-smi", "--query-gpu=memory.free",
           "--format=csv,noheader,nounits"]
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_ids:
        cmd.append("--id=" + gpu_ids)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            frees = [float(x) / 1024 for x in result.stdout.split() if x.strip()]
            if frees:
                return min(frees)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0.0


# Per-rank overhead (CUDA graph capture, activations, NCCL buffers). Incurred on
# EVERY GPU, so it is NOT shard-divided by TP.
_PER_RANK_OVERHEAD_GB = 3.0


def weights_fit_advisory(
    total_params_b: float,
    tensor_parallel: int,
    gpu_memory_utilization: float,
    min_free_override: Optional[float] = None,
) -> tuple[bool, str]:
    """Best-effort, NON-BLOCKING check that a model's WEIGHTS fit the visible GPUs.

    Gates ``weights_per_gpu = total_params_b*2/TP + per-rank overhead`` against
    the usable budget ``min(free, util*total)`` per GPU. Advisory only - vLLM's
    load-time profiler is the hard authority for the full weights+KV fit; this
    gives an early heads-up so a doomed run isn't discovered only after a long
    load. KV is intentionally NOT modeled (per-layer-type: sliding-window / MLA
    make a flat estimate ~5-40x wrong - vLLM sizes it correctly). Returns
    ``(fits, message)``; ``fits=True`` (skip) when the model size or VRAM can't
    be read. ``min_free_override`` (from --min-free-gb) REPLACES the computed
    requirement.
    """
    tp = max(1, int(tensor_parallel or 1))
    if not (isinstance(total_params_b, (int, float)) and total_params_b > 0):
        return True, "weights-fit check skipped (unknown model size - deferring to vLLM)."
    total = _gpu_total_vram_gb()
    free = _gpu_min_free_vram_gb()
    caps = [c for c in (free, total * gpu_memory_utilization) if c > 0]
    if not caps:
        return True, "weights-fit check skipped (could not read VRAM - deferring to vLLM)."
    usable = min(caps)   # bounded by BOTH free memory now AND util*total
    weights_per_gpu = total_params_b * 2.0 / tp + _PER_RANK_OVERHEAD_GB
    if min_free_override is not None:
        # Explicit user floor on FREE memory - the SAME meaning as --min-free-gb in
        # check_gpu_ready (not clamped by util*total), so the flag is consistent.
        free_basis = free if free > 0 else usable
        if free_basis < min_free_override:
            return False, (
                f"weights-fit advisory: free VRAM ~{free_basis:.0f}GB/GPU is below "
                f"the --min-free-gb={min_free_override:.0f}GB floor. Model may not "
                f"fit - vLLM's profiler will decide.")
        return True, (f"weights-fit OK: free ~{free_basis:.0f}GB/GPU >= "
                      f"--min-free-gb {min_free_override:.0f}GB.")
    if usable < weights_per_gpu:
        return False, (
            f"weights-fit advisory: ~{weights_per_gpu:.0f}GB/GPU needed "
            f"(={total_params_b:.0f}B x2 / TP{tp} + {_PER_RANK_OVERHEAD_GB:.0f}GB "
            f"per-rank), but only ~{usable:.0f}GB usable/GPU "
            f"(min of free & {gpu_memory_utilization:.0%}xtotal). Model may not fit "
            f"- vLLM's profiler will decide. Lower --gpu-memory-utilization, raise "
            f"--num-gpus, or set --min-free-gb to override.")
    return True, (f"weights-fit OK: ~{weights_per_gpu:.0f}GB/GPU needed, "
                  f"~{usable:.0f}GB usable/GPU.")


def validate_gpu_config(
    num_gpus: int, total_params_b: float,
    gpu_memory_utilization: Optional[float] = None,
) -> tuple[bool, str]:
    """Validate GPU configuration for a model.

    Args:
        num_gpus: Number of GPUs requested.
        total_params_b: Total model parameters in billions.
        gpu_memory_utilization: Fraction of VRAM vLLM will use (the value the
            server is actually launched with). Defaults to the uniform 0.90 when
            not supplied, so this stays consistent with the launched engine and
            the run fingerprint rather than a separate hardcoded value.

    Returns:
        Tuple of (is_valid, message).
    """
    available = get_available_gpus()
    recommended = get_num_gpus(total_params_b)

    if available == 0:
        return False, "No NVIDIA GPUs detected"

    if num_gpus > available:
        return False, f"Requested {num_gpus} GPUs but only {available} available"

    VALID_TP_SIZES = {1, 2, 4, 8, 16}
    if num_gpus not in VALID_TP_SIZES:
        return False, (
            f"GPU count {num_gpus} not supported for tensor parallel. "
            f"Valid sizes: {sorted(VALID_TP_SIZES)}"
        )

    if num_gpus < recommended:
        # `recommended` (see get_num_gpus) is a FAIRNESS tier: it assumes
        # 80GB GPUs and exists so competing models get identical resources
        # for comparable leaderboard numbers. It is NOT a hard physical
        # limit. Only block a below-tier request when the model genuinely
        # will not fit in the requested GPUs' actual VRAM; otherwise allow
        # it with a warning that the run is off the fair tier.
        util = (gpu_memory_utilization if gpu_memory_utilization is not None
                else get_gpu_memory_utilization(total_params_b))
        per_gpu_gb = _gpu_total_vram_gb()
        required_gb = estimate_required_vram_gb(total_params_b)
        usable_gb = per_gpu_gb * num_gpus * util
        if per_gpu_gb <= 0:
            # VRAM couldn't be read (nvidia-smi absent/unreadable). Do NOT claim
            # the model "fits" - we never checked. Defer the physical-fit decision
            # to vLLM's load-time profiler, which is authoritative.
            return True, (
                f"Warning: using {num_gpus} GPU(s) for a {total_params_b:.0f}B "
                f"model is below the {recommended}-GPU fairness tier. Could not "
                f"read VRAM (nvidia-smi unreadable) - deferring the physical-fit "
                f"check to vLLM's load-time profiler."
            )
        if required_gb > usable_gb:
            return False, (
                f"Model (~{required_gb:.0f}GB bf16) does not fit in "
                f"{num_gpus}x{per_gpu_gb:.0f}GB GPU(s) at {util:.0%} "
                f"utilization. Recommended tier: {recommended} GPU(s)."
            )
        return True, (
            f"Warning: using {num_gpus} GPU(s) for a {total_params_b:.0f}B "
            f"model is below the {recommended}-GPU fairness tier - "
            f"leaderboard numbers may not be comparable across models. "
            f"Model fits (~{required_gb:.0f}GB bf16)."
        )

    if num_gpus > recommended * 2:
        return True, (
            f"Warning: Using {num_gpus} GPUs for {total_params_b:.0f}B model "
            f"may not provide linear scaling. Recommended: {recommended}"
        )

    return True, (
        f"Using {num_gpus} GPU(s) for {total_params_b:.0f}B model"
    )


def check_gpu_ready(min_free_memory_gb: Optional[float] = None) -> tuple[bool, str]:
    """Check if GPU is ready for benchmarking (no conflicting processes).

    Uses nvidia-smi to detect other processes and (optionally) a free-memory
    floor. Respects CUDA_VISIBLE_DEVICES to only check assigned GPUs.

    ``min_free_memory_gb`` defaults to None = NO hard free-memory floor (the old
    fixed 70GB floor blocked every <70GB GPU even for tiny models - the actual
    physical-fit decision is model-aware, done by the weights-fit advisory +
    validate_gpu_config + vLLM's own profiler). Pass a number to re-enable a hard
    floor (e.g. via --min-free-gb).
    """
    import subprocess

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    try:
        proc_cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            proc_cmd, capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            return True, "nvidia-smi unavailable, skipping GPU check"

        if not gpu_ids:
            processes = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            if processes:
                process_info = "; ".join(processes[:3])
                return False, (
                    f"GPU is currently in use by other processes.\n"
                    f"Active GPU processes: {process_info}\n"
                    f"Please terminate these processes before running benchmarks.\n"
                    f"Tip: Use 'nvidia-smi' to view processes and 'kill <PID>' to stop them."
                )

        mem_cmd = [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        if gpu_ids:
            mem_cmd.extend(["--id=" + gpu_ids])

        mem_result = subprocess.run(
            mem_cmd, capture_output=True, text=True, timeout=10,
        )

        if mem_result.returncode == 0:
            lines = mem_result.stdout.strip().splitlines()
            if lines:
                min_free_gb = float("inf")
                total_gb_first = 0
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        free_mb = float(parts[0].strip())
                        total_mb = float(parts[1].strip())
                        free_gb = free_mb / 1024
                        if total_gb_first == 0:
                            total_gb_first = total_mb / 1024
                        min_free_gb = min(min_free_gb, free_gb)

                if min_free_memory_gb is not None and min_free_gb < min_free_memory_gb:
                    return False, (
                        f"Insufficient GPU memory.\n"
                        f"Free: {min_free_gb:.1f}GB / {total_gb_first:.1f}GB "
                        f"total (min across {len(lines)} GPU(s))\n"
                        f"Required: {min_free_memory_gb:.1f}GB minimum (--min-free-gb)\n"
                        f"Try waiting 30 seconds or restart the terminal."
                    )

                # Soft busy-card signal when there is no hard floor. Under
                # CUDA_VISIBLE_DEVICES the process-conflict check above is skipped,
                # so this is the only pre-launch hint that an assigned card is
                # already occupied. WARN (never block - a genuinely small GPU can
                # be low in absolute GB yet fine); vLLM's profiler is authoritative.
                if (min_free_memory_gb is None and total_gb_first > 0
                        and min_free_gb < 0.15 * total_gb_first):
                    return True, (
                        f"⚠ assigned GPU appears busy: only {min_free_gb:.1f}GB / "
                        f"{total_gb_first:.1f}GB free - proceeding (vLLM will confirm; "
                        f"if this is unexpected, free the GPU or wait)."
                    )

                return True, (
                    f"GPU ready: {min_free_gb:.1f}GB / {total_gb_first:.1f}GB "
                    f"free ({len(lines)} GPU(s))"
                )

        return True, "GPU check passed"

    except FileNotFoundError:
        return True, "nvidia-smi not found, skipping GPU check"
    except subprocess.TimeoutExpired:
        return True, "nvidia-smi timed out, skipping GPU check"
    except Exception as e:
        return True, f"GPU check failed: {e}, continuing anyway"
