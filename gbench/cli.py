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

"""Command-line interface for Gemma benchmark suite.

This module provides a rich CLI with comprehensive options for running
benchmarks on Gemma models.
"""

import argparse
import logging
import os
import re
import sys
import time
from types import ModuleType
dummy_check = ModuleType("transformers.dependency_versions_check")
def dep_version_check(*args, **kwargs):
    pass
dummy_check.dep_version_check = dep_version_check
# Universal Gemini API key multi-key handling and third-party library sanitization
raw_gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
if raw_gemini_key and "," in raw_gemini_key:
    gemini_keys_list = [k.strip() for k in raw_gemini_key.split(",") if k.strip()]
    if gemini_keys_list:
        os.environ["GEMINI_API_KEYS"] = ",".join(gemini_keys_list)
        os.environ["GEMINI_API_KEY"] = gemini_keys_list[0]

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .core import (
    BenchmarkConfig,
    DEFAULT_CONFIG,
    DEFAULT_GEMMACLAW_COMMIT,
    QUICK_CONFIG,
    ModelCategory,
    ModelFormat,
    Priority,
    registry,
    check_gpu_ready,
    validate_gpu_config,
    weights_fit_advisory,
    get_available_gpus,
    get_batch_sizes,
    resolve_campaign_slo,
    campaign_ctx,
)

logger = logging.getLogger(__name__)

# Exit codes. A caller pointing gbench at its own endpoint needs to tell
# "the model got something wrong" apart from "the run never happened",
# because only the first is evidence about the model.
EXIT_OK = 0
EXIT_MODEL_FAILURE = 1
EXIT_HARNESS_ERROR = 2

GOLDEN_VERDICT = {"passed": "PASS", "failed": "FAIL", "error": "ERROR",
                  "not_applicable": "N/A"}


def golden_exit_code(results: list[dict]) -> int:
    """Reduce Golden Set results to a process exit code.

    Harness errors outrank model failures: a run that could not reach the
    endpoint has no verdict on the model at all, so it must not be
    reported with the same code as a real regression.
    """
    golden = [r for r in results if r.get("benchmark_type") == "golden"]
    if any(r.get("status") == "error" for r in golden):
        return EXIT_HARNESS_ERROR
    if any(r.get("status") == "failed" for r in golden):
        return EXIT_MODEL_FAILURE
    return EXIT_OK


def golden_category_breakdown(task_results: list[dict]) -> list[dict]:
    """Aggregate per-case Golden Set results into one row per category.

    A bare "9/12" says something regressed but not where, which is the
    first thing you need in order to judge whether it matters. Losing
    both tool_use cases is a different morning from losing one
    translation.

    Args:
        task_results: Per-case result dicts from the Golden runner.

    Returns:
        One row per category, worst first, so problem areas sort to the
        top instead of landing wherever the alphabet puts them. The
        offending task ids are deliberately not repeated here: the
        FAIL and ERROR lines printed below the table already name each
        one, along with the detail you need to reproduce it.
    """
    buckets: dict[str, dict] = {}
    for task in task_results:
        category = task.get("category") or "uncategorized"
        bucket = buckets.setdefault(category, {
            "category": category,
            "total": 0,
            "passed": 0,
            "na": 0,
            "statuses": set(),
        })
        bucket["total"] += 1
        status = task.get("status")
        bucket["statuses"].add(status)
        if status == "passed":
            bucket["passed"] += 1
        elif status == "not_applicable":
            bucket["na"] += 1

    rows = []
    for bucket in buckets.values():
        # Same precedence as golden_exit_code: a category that could not
        # be measured is not a category that passed. A category that is
        # entirely not-applicable (the model lacks that modality) is N/A,
        # not a pass.
        if "error" in bucket["statuses"]:
            bucket["status"] = "error"
        elif "failed" in bucket["statuses"]:
            bucket["status"] = "failed"
        elif bucket["na"] == bucket["total"] and bucket["passed"] == 0:
            bucket["status"] = "not_applicable"
        else:
            bucket["status"] = "passed"
        del bucket["statuses"]
        rows.append(bucket)

    rank = {"error": 0, "failed": 1, "passed": 2, "not_applicable": 3}
    return sorted(rows, key=lambda b: (rank.get(b["status"], 4),
                                       b["category"]))


def _save_eval_summary_csv(
    results_dir: Path,
    eval_results: list,
    eval_failures: list,
    eval_pillars: list,
    custom_pillars: dict,
) -> Optional[Path]:
    """Save clean, structured CSV report for all evaluation results to eval_summary.csv."""
    import csv

    csv_path = results_dir / "eval_summary.csv"
    suite_to_pillar = {}
    for p_title, keys in eval_pillars:
        clean_title = re.sub(r"^\d+\.\s*", "", p_title).strip()
        for k in keys:
            suite_to_pillar[k.lower()] = clean_title

    fieldnames = [
        "Pillar",
        "Model",
        "Format",
        "Eval Suite",
        "Subcategory",
        "Thinking",
        "Questions",
        "Effective N",
        "Correct",
        "Accuracy (%)",
        "Status",
        "Duration (s)",
    ]

    rows = []
    all_runs = [(r, True) for r in eval_results] + [(r, False) for r in eval_failures]

    for r, is_success in all_runs:
        eval_name = r.get("eval_name", "").lower()
        model = r.get("model_short", r.get("model_name", "N/A"))
        fmt = r.get("format", "N/A")
        thinking = "Yes" if r.get("thinking", False) else "No"
        total_q = r.get("total_questions", 0) if is_success else 0
        effective_n = r.get("effective_n", total_q) if is_success else "-"
        correct = r.get("correct_answers", 0) if is_success else 0
        # accuracy can be present-but-None (e.g. omnidocbench's composite is undefined when the
        # sampled pages have no tables/formulas); .get(...,0.0) does NOT substitute the default for
        # an existing None, so `f"{None:.2f}"` used to crash the whole model summary (outside the
        # per-eval guard) and abort the run. Treat a non-numeric accuracy as "-".
        _acc = r.get("accuracy")
        acc = f"{_acc:.2f}" if (is_success and isinstance(_acc, (int, float))) else "-"
        status = r.get("status", "success" if is_success else "failed")
        duration = r.get("duration_s", "")

        if eval_name in suite_to_pillar:
            pillar = suite_to_pillar[eval_name]
        elif eval_name in custom_pillars:
            clean_cp = re.sub(r"^\d+\.\s*", "", custom_pillars[eval_name]).strip()
            pillar = f"[CUSTOM] {clean_cp}"
        else:
            pillar = "[CUSTOM] OTHER"

        # Extract subcategory information
        cat_acc = r.get("category_accuracy", {})
        traces = r.get("sample_traces", [])
        trace_cat = traces[0].get("category") if traces and isinstance(traces[0], dict) else None

        if len(cat_acc) == 1:
            single_cat = list(cat_acc.keys())[0]
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": single_cat,
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Status": status,
                "Duration (s)": duration,
            })
        elif len(cat_acc) > 1:
            # Main overall suite row
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": "OVERALL",
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Status": status,
                "Duration (s)": duration,
            })
            # Individual subcategory rows
            for cat, cstats in sorted(cat_acc.items()):
                ctot = cstats.get("total", 0)
                ccorr = cstats.get("correct", 0)
                _cacc = cstats.get("accuracy")   # can be None (undefined subcategory metric)
                cacc = f"{float(_cacc):.2f}" if isinstance(_cacc, (int, float)) else "-"
                rows.append({
                    "Pillar": pillar,
                    "Model": model,
                    "Format": fmt,
                    "Eval Suite": eval_name.upper(),
                    "Subcategory": cat,
                    "Thinking": thinking,
                    "Questions": ctot,
                    "Effective N": ctot,
                    "Correct": ccorr,
                    "Accuracy (%)": cacc,
                    "Status": status,
                    "Duration (s)": "",
                })
        else:
            # Fallback to trace category if available
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": trace_cat or "-",
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Status": status,
                "Duration (s)": duration,
            })

    if not rows:
        return None

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved evaluation CSV report to: {csv_path}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to write CSV summary to {csv_path}: {e}")
        return None


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ["httpx", "httpcore", "datasets", "huggingface_hub", "urllib3", "fsspec", "filelock", "google_genai", "google", "grpc"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Open Model Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test on single model
  gbench --preset quick --models gemma-4-E4B-it

  # Run full baseline (HF only)
  gbench --format hf --models gemma-4-E4B-it gemma-4-31B-it

  # Benchmark any HuggingFace model (auto-registers)
  gbench --models google/gemma-4-31B-it

  # Multimodal models with vision benchmarks
  gbench --category multimodal --multimodal-only

  # Dry run to see what would be executed
  gbench --dry-run --models gemma-4-E4B-it

Environment knobs: every per-suite knob is GBENCH_<SUITE>_<KNOB> (older bare names like TAU2_* still
work as deprecated aliases). Full list incl. decoding/judge/search knobs: docs/evals/env-knobs.md.
Per-suite temperature: GBENCH_<SUITE>_TEMPERATURE (overrides --temperature).
        """,
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"gbench {__version__}",
    )

    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark - use registry names (e.g., gemma-4-E4B-it) or HuggingFace IDs (e.g., google/gemma-4-31B-it)",
    )
    model_group.add_argument(
        "--category",
        choices=["text", "embedding", "multimodal"],
        help="Filter by model category",
    )
    model_group.add_argument(
        "--priority",
        choices=["P0", "P1", "P2"],
        help="Filter by priority level",
    )
    model_group.add_argument(
        "--format",
        choices=["hf", "gguf", "both"],
        default="both",
        help="Model format to test (default: both)",
    )

    # Benchmark type selection
    bench_group = parser.add_argument_group("Benchmark Types")
    bench_group.add_argument(
        "--personal",
        action="store_true",
        help="Personal-device mode: 'what will this model do on my machine'. "
             "Single-stream, text-only, serving-only, few samples, full decode; "
             "reports peak decode tok/s + TTFT. Use with --remote-endpoint (e.g. "
             "Ollama). Incompatible with --campaign.",
    )
    bench_group.add_argument(
        "--serving-only",
        action="store_true",
        help="Run only serving benchmarks",
    )
    bench_group.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmarks",
    )
    bench_group.add_argument(
        "--text-only",
        action="store_true",
        help="Run only text benchmarks, skip multimodal even for capable models",
    )
    bench_group.add_argument(
        "--multimodal-only",
        action="store_true",
        help="Run only multimodal benchmarks (for models that support it)",
    )
    bench_group.add_argument(
        "--stress-test",
        action="store_true",
        help="Run only stress test (ramp-up to find max sustainable throughput)",
    )
    bench_group.add_argument(
        "--no-stress-test",
        action="store_true",
        help="Skip stress test (included by default in all presets)",
    )
    bench_group.add_argument(
        "--stress-threshold",
        type=int,
        default=None,
        help="P99 TTFT SLO in ms for the stress QPS sweep. GLOBAL override "
             "(applies to every campaign); default None => use the per-campaign "
             "SLO map (config.CAMPAIGN_SLOS: 1000ms chat/decode, 2500ms mixed, "
             "5000ms big-prefill). Tunable for laptop/H100/GB200.",
    )
    bench_group.add_argument(
        "--itl-slo-ms",
        "--stress-tpot-threshold",
        dest="itl_slo_ms",
        type=int,
        default=None,
        help="P99 ITL/TBT (inter-token, decode-cadence) SLO in ms for the stress "
             "QPS sweep. GLOBAL override; default None => per-campaign map "
             "(100ms ~= 10 tok/s). A rate is sustainable only if BOTH the TTFT "
             "and the ITL/TBT SLO hold. (--stress-tpot-threshold is a back-compat alias.)",
    )
    bench_group.add_argument(
        "--stress-reps",
        type=int,
        default=None,
        help="Number of full QPS sweeps per stress point; the knee is reported as "
             "mean + bootstrap CI over reps (a single sweep varies run-to-run). "
             "Env GBENCH_STRESS_REPS (flag wins). Default None => 3.",
    )
    bench_group.add_argument(
        "--stress-client-procs",
        type=int,
        default=None,
        help="OS processes for the multi-process stress load generator (removes "
             "the single-asyncio-client CPU-contention confound at the knee). "
             "Auto-capped to cpu_count-1, so a large value is safe on small hosts. "
             "Env GBENCH_STRESS_CLIENT_PROCS (flag wins). Default None => min(32, cpu_count-1).",
    )
    bench_group.add_argument(
        "--stress-max-qps",
        type=float,
        default=None,
        help="Safety cap on the open-loop arrival-rate sweep (env GBENCH_STRESS_MAX_QPS; "
             "flag wins). Default None => 8192. Raise it further only if a host passes the SLO "
             "at 512 (the knee is then censored at 512, not measured); pair with a higher "
             "--stress-client-procs so the client can offer the extra load without becoming "
             "the bottleneck (weak clients hit client_bound first and are excluded).",
    )
    bench_group.add_argument(
        "--stress-max-prompts",
        type=int,
        default=None,
        help="Cap on requests per stress rate point (env GBENCH_STRESS_MAX_PROMPTS; flag "
             "wins). Default None => 8000. Raise it so very-high-QPS points still span a real "
             "steady-state window (at 1000 req/s, 1200 prompts is ~1.2s; 8000 is ~8s).",
    )
    bench_group.add_argument(
        "--stress-min-samples",
        type=int,
        default=None,
        help="Min steady-state completions to trust a stress rate point (env "
             "GBENCH_STRESS_MIN_SAMPLES; flag wins). Default None => 15. LOWER it for a slow "
             "box that meets the SLO but is too slow to complete 15 per point (else every point "
             "is TOO-FEW and the reported knee is 0). A smaller floor gives a faster, "
             "wider-CI, lower-confidence knee - use it for modest-hardware smoke runs.",
    )
    bench_group.add_argument(
        "--quality",
        action="store_true",
        help="Run quality (gemmaclaw agentic) benchmarks",
    )
    bench_group.add_argument(
        "--quality-only",
        action="store_true",
        help="Run only quality (gemmaclaw agentic) benchmarks",
    )
    bench_group.add_argument(
        "--scenarios",
        nargs="+",
        help="List of specific scenario file paths to run (relative to qa/scenarios/)",
    )
    bench_group.add_argument(
        "--golden",
        action="store_true",
        help="Run Golden Set deterministic smoke-test tasks",
    )
    bench_group.add_argument(
        "--golden-only",
        action="store_true",
        help="Run only Golden Set deterministic smoke-test tasks",
    )
    bench_group.add_argument(
        "--golden-tasks",
        nargs="+",
        help="List of specific Golden task IDs or JSON files to run",
    )
    bench_group.add_argument(
        "--golden-model-id",
        help=(
            "Model name to send in the Golden Set request payload. "
            "Defaults to whatever the endpoint's /models listing reports, "
            "falling back to the model's HF id. Required when an endpoint "
            "serves several models"
        ),
    )

    # Evaluation benchmarks
    eval_group = parser.add_argument_group("Evaluation Benchmarks")
    eval_group.add_argument(
        "--evals",
        nargs="+",
        help="Select evaluation benchmark suites to run (space or comma-separated, or 'all', 'plugins')",
    )
    eval_group.add_argument(
        "--eval-thinking",
        "--enable-thinking",
        "--thinking",
        dest="eval_thinking",
        action="store_true",
        help="Let the model think/reason (default OFF everywhere). For SERVING, "
             "gbench otherwise sends a no-think system prompt so a reasoning model "
             "(e.g. gemma-4 on Ollama) measures pure answer generation, not an "
             "accidental chain-of-thought workload (which inflates TTFT and causes "
             "0-content 'empty' replies). For EVALS, enables reasoning mode where "
             "supported (aime, bfcl, bundled_detection, causalbench, gpqa_diamond, "
             "infographicvqa, loft_x_arxiv, mmlu_pro, mmmu_pro, amc_aime, "
             "putnam, semantic_keypoint).",
    )
    eval_group.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generation tokens per eval response. REQUIRED for eval runs: it bounds "
             "generation and materially affects scores (truncation reads as a failure), so it "
             "must be explicit for runs to be comparable. e.g. --max-output-tokens 65536.",
    )
    eval_group.add_argument(
        "--eval-max-soft-tokens",
        type=int,
        choices=[70, 140, 280, 560, 1120],
        default=None,
        help="Image soft token budget for vision evals (70, 140, 280, 560, 1120). Default: 1120",
    )
    eval_group.add_argument(
        "--suite-timeout",
        type=int,
        default=None,
        help="Per-suite wall-clock budget in seconds (default: OFF). When set, a suite "
             "that exceeds it is reported as 'timeout' and the sweep continues, so one "
             "wedged harness cannot stall the run. Destructive: the suite's partial "
             "results are lost and its child processes/containers are not reaped, so "
             "leave it unset unless running unattended.",
    )
    eval_group.add_argument(
        "--eval-n-shot",
        type=int,
        default=None,
        help="Few-shot exemplar count for MMLU (dev split), MMLU-Pro (validation split), and GSM8K (train CoT). Default: canonical 5-shot for mmlu/mmlu_pro and 8-shot for gsm8k; pass 0 for 0-shot.",
    )
    eval_group.add_argument(
        "--eval-categories",
        type=str,
        default=None,
        help="Comma-separated category override for BFCL (e.g. 'simple_python,multi_turn')",
    )
    eval_group.add_argument(
        "--eval-limit",
        "--limit",
        dest="eval_limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples per benchmark suite (e.g. --eval-limit 10)",
    )
    eval_group.add_argument(
        "--shard",
        dest="shard",
        type=str,
        default=None,
        help="Run shard I of N (e.g. --shard 3/8) to split each eval across machines: a "
             "reproducible, non-overlapping subset of every suite that unions back to the full set. "
             "Partitions the full ordered sample set (before --eval-limit). Sets GBENCH_SHARD.",
    )
    eval_group.add_argument(
        "--sandboxes",
        type=int,
        default=None,
        help="Concurrency level for containerized/sandboxed evaluations (e.g. terminal_bench, copilot_bench_swe, the swe_bench family, multipl_e, bigcodebench). Defaults to --batch-sizes. Suites whose harness runs strictly one task at a time (e.g. ui_control_osworld, a single-VM loop) ignore it.",
    )
    eval_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for ALL benchmarks. Default is THINK-AWARE: 0.0 (greedy) "
             "for a no-think run, 1.0 (the model's shipped generation_config) with --thinking. "
             "This flag overrides both, and whatever a suite encodes. Measured 2026-08-17: at "
             "no-think, 1.0 halves degenerate repetition versus greedy (78%%->38%% on looping "
             "prompts) - so 0.0 no-think loops more, which the looping metrics surface. A "
             "single eval can be overridden without touching the rest of the run via "
             "GBENCH_<EVAL>_TEMPERATURE, which takes precedence over this flag.",
    )
    eval_group.add_argument(
        "--attempt-count",
        "--attempts",
        dest="attempt_count",
        type=int,
        default=1,
        help="Generate this many independent attempts per sample and report avg@k, pass@k "
             "and pass^k (default: 1). Most published numbers for small benchmarks are "
             "not single-sample - AIME is avg@4-64, GPQA-Diamond avg@10, ARC-AGI pass@2 "
             "by rule. At --temperature 0 the attempts differ only through server "
             "nondeterminism, which measures reproducibility rather than the benchmark. "
             "Multiplies generation cost by k. Batch-scored suites "
             "that have not opted in run 1 attempt and say so on the result.",
    )
    eval_group.add_argument(
        "--eval-plugins-dir",
        "--eval-plugins-path",
        dest="eval_plugins_dir",
        nargs="+",
        default=None,
        help="Directory or Python file path(s) containing custom/private eval suite plugins (*.py)",
    )
    eval_group.add_argument(
        "--eval-custom-jsonl",
        type=str,
        default=None,
        help="Path to custom JSONL evaluation dataset file for zero-code adhoc evaluation",
    )
    eval_group.add_argument(
        "--evals-only",
        action="store_true",
        help="Run only specific evaluation benchmarks (skip serving, throughput, stress, quality, golden)",
    )
    eval_group.add_argument(
        "--list-plugins",
        nargs="?",
        const=".",
        default=None,
        metavar="PATH",
        help="List all evaluation plugins discovered at PATH (defaults to current directory) and exit",
    )
    eval_group.add_argument(
        "--eval-provenance",
        action="store_true",
        help="Print the canonical-sync provenance table (when each eval was last reconciled against "
             "its upstream, and against what version) and exit. Suites shown as 'unverified' have no "
             "reconcile record yet.",
    )
    eval_group.add_argument(
        "--calibrate",
        action="store_true",
        help="Inspect local hardware (GPUs/VRAM, CPU, RAM, Docker) and print recommended "
             "--num-gpus / --sandboxes / --batch-sizes, then exit. A requested --sandboxes that "
             "would over-subscribe the machine is refused before a real run (override with "
             "GBENCH_SKIP_CALIBRATION_GUARD=1).",
    )
    eval_group.add_argument(
        "--list",
        dest="list_what",
        nargs="?",
        const="evals",
        default=None,
        choices=["evals", "pillars", "presets", "golden", "models", "plugins", "all"],
        help="List what gbench can run (evals, pillars, presets, golden, models, plugins, all) and exit",
    )

    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--preset",
        choices=["quick", "default"],
        default="default",
        help="Configuration preset (default: default)",
    )
    config_group.add_argument(
        "--campaign",
        nargs="+",
        choices=["chat-like", "agentic", "prefill-heavy", "decode-heavy", "mixed", "long-decode"],
        help="Specific performance campaign scenario(s) to run",
    )
    config_group.add_argument(
        "--dataset",
        choices=["random", "custom", "hf"],
        help="Override the serving/throughput data source (random, custom, or hf). ShareGPT is not "
             "selectable here - run it via `--campaign chat-like`, which pairs it with the correct "
             "SLOs and context.",
    )
    config_group.add_argument(
        "--dataset-path",
        type=str,
        help="Path or HF ID for custom dataset",
    )
    config_group.add_argument(
        "--num-iterations",
        type=int,
        help="Number of iterations per config (default: 3)",
    )
    config_group.add_argument(
        "--warmup-iterations",
        type=int,
        help="Number of warmup iterations (default: 1)",
    )
    config_group.add_argument(
        "--max-cv-percent",
        type=float,
        help="Max acceptable CV%% for validation (default: 5.0)",
    )
    config_group.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Custom batch sizes to test",
    )
    config_group.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        help="Custom input lengths for throughput tests",
    )
    config_group.add_argument(
        "--output-lengths",
        nargs="+",
        type=int,
        help="Custom output lengths for throughput tests",
    )
    config_group.add_argument(
        "--num-prompts",
        type=int,
        help="(advanced) Fallback prompt count. The SERVING pillar auto-sizes its own sample count "
             "per config, so this does not control it (and is rejected with --serving-only); tune "
             "serving via --batch-sizes / --output-lengths or --personal. Throughput uses its own "
             "fixed count.",
    )
    config_group.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (validates against available GPUs). Required "
             "for any local run that serves a model (performance, evals, quality) so "
             "tensor-parallel matches the model size; omit only for --remote-endpoint / "
             "--golden-only / --dry-run.",
    )
    config_group.add_argument(
        "--tensor-parallel",
        type=int,
        help="Tensor parallel size (default: same as num-gpus)",
    )
    config_group.add_argument(
        "--max-model-len",
        type=int,
        help="Override the served context window for performance runs "
             "(default: 4096). Raise it to run long-context campaigns "
             "(agentic/prefill-heavy/mixed/long-decode) whose input+output "
             "exceed 4096; note the larger KV cache lowers max concurrency.",
    )
    config_group.add_argument(
        "--gemmaclaw-commit",
        type=str,
        default=DEFAULT_GEMMACLAW_COMMIT,
        help=(
            "Target gemmaclaw git commit, branch or tag (default: the pinned "
            f"release {DEFAULT_GEMMACLAW_COMMIT[:7]}). A branch or tag is "
            "resolved to a commit sha before the run, so the quality "
            "scaffold_id moves when the scorer does. Pass 'main' to score "
            "against the current development tip instead"
        ),
    )
    config_group.add_argument(
        "--gemmaclaw-path",
        type=str,
        help="Path to local gemmaclaw repository checkout (optional)",
    )
    config_group.add_argument(
        "--remote-endpoint",
        type=str,
        help="Remote API endpoint URL to benchmark instead of starting local vLLM (e.g. https://.../v1)",
    )
    config_group.add_argument(
        "--tokenizer",
        type=str,
        help="HuggingFace model ID or local directory path for tokenization when benchmarking custom remote endpoints or Ollama tags",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for benchmark results (default: ./results)",
    )
    output_group.add_argument(
        "--tags",
        nargs="+",
        help="List of arbitrary key:value tags to assign to this run (e.g., family:gemma-1b stage:prod)",
    )
    output_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Reuse a previous result for a suite instead of running it, if one is found "
             "anywhere under --results-dir (newest match wins). OFF by default: reusing a "
             "result silently mixes it with whatever the code measures now, so opt in only "
             "when resuming an interrupted run against unchanged code.",
    )
    output_group.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Explicitly re-run every benchmark (this is the default).",
    )

    # Execution options
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    exec_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    exec_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    exec_group.add_argument(
        "--mm-probe-timeout",
        type=int,
        default=None,
        help="Seconds to wait for the remote multimodal capability probe "
             "(default 10; --personal defaults to 120 for slow CPU vision). Raise "
             "it if a capable-but-slow endpoint is wrongly detected as text-only.",
    )
    exec_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-request serving timeout in seconds (default 600). Applied to "
             "both remote paths: text (urllib) uses it as a per-read STALL timeout "
             "(hung prefill), MM (aiohttp) as a TOTAL wall-clock cap. Raise it for "
             "large models / long outputs on slow devices.",
    )
    exec_group.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Explicit minimum free VRAM per GPU (GB). Overrides the model-aware "
             "weights-fit advisory and re-enables a hard free-memory pre-check. "
             "Use to unblock/limit on non-datacenter GPUs (default: model-aware, "
             "advisory-only).",
    )

    # Staging utilities
    staging_group = parser.add_argument_group("Staging Utilities")
    staging_group.add_argument(
        "--stage-to-gcs",
        type=str,
        help="Stage models locally (downloading from HF if needed) and upload to the specified GCS destination (e.g. gs://my-bucket/path/). Benchmarks will not be run.",
    )

    return parser


def apply_campaign_to_config(config, campaign: str, args=None) -> None:
    """Apply specific campaign scenario parameters to config."""
    config.campaign = campaign
    if campaign == "chat-like":
        config.dataset = "sharegpt"
        # (serving sample count is now auto-sized by output length in
        # get_serving_configs; throughput uses num_prompts_throughput.)
    elif campaign == "agentic":
        config.dataset = "random"
        config.input_lengths = [8000]
        config.output_lengths = [400]
    elif campaign == "prefill-heavy":
        config.dataset = "random"
        config.input_lengths = [8192]
        config.output_lengths = [128]
    elif campaign == "decode-heavy":
        config.dataset = "random"
        config.input_lengths = [128]
        config.output_lengths = [2048]
    elif campaign == "mixed":
        config.dataset = "random"
        config.input_lengths = [4096]
        config.output_lengths = [1024]
    elif campaign == "long-decode":
        config.dataset = "random"
        config.input_lengths = [8192]
        config.output_lengths = [8192]
    if args and getattr(args, "dataset", None):
        config.dataset = args.dataset
    if args and getattr(args, "dataset_path", None):
        config.dataset_path = args.dataset_path
    # Auto-size the served context to this campaign's shape unless the user
    # pinned --max-model-len. RandomDataset (range_ratio=0.5) samples up to
    # 1.5x(in+out), so the fixed 4096 default would truncate/abort the four
    # long-context campaigns; campaign_ctx() clears that tail with slack.
    # Gated on the explicit override so it stays correct at the per-campaign
    # loop call sites (where args.max_model_len was already applied) too.
    if not (args and getattr(args, "max_model_len", None)):
        config.max_model_len = campaign_ctx(
            campaign, config.input_lengths, config.output_lengths
        )


def _split_golden_tasks(raw: Optional[list[str]]) -> Optional[list[str]]:
    """Normalise --golden-tasks into a flat list of task ids.

    argparse nargs="+" only splits on spaces, so a comma separated list
    arrives as a single token that matches nothing. Accept either form,
    and both mixed together.

    Args:
        raw: Values as argparse produced them, or None if the flag was
            not passed.

    Returns:
        The flattened task ids, or None when the flag was not passed. An
        explicitly empty selection stays empty rather than becoming None,
        so it is reported as "matched nothing" instead of silently
        running the whole dataset.
    """
    if raw is None:
        return None
    return [part.strip() for item in raw for part in item.split(",")
            if part.strip()]


def get_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Create BenchmarkConfig from CLI arguments."""
    # Start with preset
    if args.preset == "quick":
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Override with custom iteration values
    if args.num_iterations:
        config.num_iterations = args.num_iterations
    if args.warmup_iterations is not None:
        config.warmup_iterations = args.warmup_iterations
    if args.max_cv_percent:
        config.min_acceptable_cv_percent = args.max_cv_percent

    # Override with custom configuration values
    if hasattr(args, "campaign") and args.campaign:
        # Concurrency defaults to a single stream (batch size 1); pass
        # --batch-sizes to sweep concurrency. Stress ignores batch_sizes anyway
        # (it runs its own arrival-rate sweep).
        campaigns = args.campaign if isinstance(args.campaign, list) else [args.campaign]
        apply_campaign_to_config(config, campaigns[0], args)

    # Record whether the user pinned a workload geometry (campaign or explicit
    # lengths); the stress test uses this to pick a real shape vs its 128/128
    # neutral default.
    config.workload_shape_explicit = bool(
        getattr(args, "campaign", None)
        or args.input_lengths
        or args.output_lengths
    )

    if args.batch_sizes:
        config.batch_sizes = args.batch_sizes
    if args.input_lengths:
        config.input_lengths = args.input_lengths
    if args.output_lengths:
        config.output_lengths = args.output_lengths
    if args.num_prompts:
        config.num_prompts = args.num_prompts
    if getattr(args, "dataset", None):
        config.dataset = args.dataset
    if getattr(args, "dataset_path", None):
        config.dataset_path = args.dataset_path
    # Set GPU configuration. None (unset) leaves the dataclass default (1) so
    # exempt modes (remote/quality/golden/dry-run) keep a valid TP + scaffold
    # fingerprint; local performance runs are required to pass it (guarded below).
    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    if args.tensor_parallel:
        config.tensor_parallel_size = args.tensor_parallel
    else:
        config.tensor_parallel_size = config.num_gpus  # Default: TP = num GPUs
    if getattr(args, "max_model_len", None):
        config.max_model_len = args.max_model_len

    # Stress-test knobs (None => keep config defaults).
    if getattr(args, "stress_reps", None) is not None:
        config.stress_reps = args.stress_reps
    if getattr(args, "stress_client_procs", None) is not None:
        config.stress_client_procs = args.stress_client_procs
    if getattr(args, "stress_max_qps", None) is not None:
        config.stress_max_qps = args.stress_max_qps
    if getattr(args, "stress_max_prompts", None) is not None:
        config.stress_max_prompts = args.stress_max_prompts
    if getattr(args, "stress_min_samples", None) is not None:
        config.stress_min_samples = args.stress_min_samples

    # Set output options
    config.results_dir = args.results_dir
    config.skip_existing = args.skip_existing
    config.dry_run = args.dry_run
    config.gpu_memory_utilization = args.gpu_memory_utilization
    if abs(float(args.gpu_memory_utilization) - 0.90) > 1e-9:
        logger.warning(
            f"--gpu-memory-utilization={args.gpu_memory_utilization} differs from the "
            f"0.90 default; KV-cache capacity (and thus stress/throughput numbers) is "
            f"NOT comparable to runs at 0.90. It is recorded in the run fingerprint.")

    # Quality benchmark parameters
    config.gemmaclaw_commit = args.gemmaclaw_commit
    config.gemmaclaw_path = args.gemmaclaw_path
    config.remote_endpoint = args.remote_endpoint
    config.tokenizer = getattr(args, "tokenizer", None)
    config.selected_scenarios = args.scenarios
    config.tags = args.tags

    # Evaluation parameters
    config.evals = getattr(args, "evals", None)
    if not config.evals:
        if getattr(args, "eval_custom_jsonl", None):
            # A bare --eval-custom-jsonl (no explicit --evals) dispatches the custom_jsonl
            # suite; do NOT let --evals-only expand it to the full 'all' set.
            config.evals = ["custom_jsonl"]
        elif getattr(args, "evals_only", False):
            config.evals = ["plugins"] if getattr(args, "eval_plugins_dir", None) else ["all"]
    config.eval_thinking = getattr(args, "eval_thinking", False)
    config.eval_max_output_tokens = getattr(args, "max_output_tokens", None)
    config.eval_max_soft_tokens = getattr(args, "eval_max_soft_tokens", 1120)
    config.eval_n_shot = getattr(args, "eval_n_shot", 0)
    config.suite_timeout = getattr(args, "suite_timeout", None)
    config.eval_categories = getattr(args, "eval_categories", None)
    config.eval_limit = getattr(args, "eval_limit", None)
    config.shard = getattr(args, "shard", None)
    if config.shard:
        from .runners.eval_suites.sampling import parse_shard
        try:
            parse_shard(config.shard)          # validate now: fail fast on a bad I/N spec
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        # Transport via env so it reaches both the native path (run_eval_suite reads GBENCH_SHARD)
        # and container evals (forwarded into the orchestrator). An operator can also set
        # GBENCH_SHARD directly per swarm node without the flag.
        os.environ["GBENCH_SHARD"] = config.shard
    config.sandboxes = getattr(args, "sandboxes", None)
    config.temperature = getattr(args, "temperature", None)
    config.attempt_count = getattr(args, "attempt_count", 1)
    config.eval_plugins_dir = getattr(args, "eval_plugins_dir", None)
    config.eval_custom_jsonl = getattr(args, "eval_custom_jsonl", None)

    # Golden Set parameters
    config.golden = args.golden or args.golden_only
    config.golden_only = args.golden_only
    # nargs="+" splits on spaces, but a comma separated list is the more
    # natural thing to type and silently matches no tasks. Accept both.
    config.selected_golden_tasks = _split_golden_tasks(args.golden_tasks)
    config.golden_model_id = args.golden_model_id

    # Initialize LogManager NOW - after all config fields are set
    # (must be after results_dir is assigned from CLI args)
    config.initialize()

    return config


def get_models_from_args(args: argparse.Namespace) -> list:
    """Get list of models to benchmark based on CLI arguments."""
    models = []

    # If specific models requested (takes precedence over auto-detection)
    if args.models:
        for model_name in args.models:
            model = registry.get(model_name)
            if not model:
                model = registry.register_hf_model(model_name)
            models.append(model)
        return models

    # If remote endpoint specified without --models, auto-detect served model ID from /v1/models
    if getattr(args, "remote_endpoint", None):
        base_url = args.remote_endpoint
        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
            base_url = f"{base_url.rstrip('/')}/v1"
        import urllib.request
        import json

        served_id = None
        try:
            url = f"{base_url.rstrip('/')}/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models_data = data.get("data", [])
                if models_data:
                    served_id = models_data[0]["id"]
        except Exception as e:
            logger.warning(
                f"Could not query /v1/models from {base_url} to auto-detect model: {e}"
            )
        if served_id:
            model = registry.get(served_id)
            if not model:
                model = registry.register_hf_model(served_id)
            logger.info(
                f"Auto-detected model '{served_id}' served at {args.remote_endpoint}"
            )
            # The served id is often an un-introspectable tag (e.g. Ollama
            # 'gemma4-qat:4b') -> 0B/dense/text defaults. If a --tokenizer HF id
            # was given, borrow real metadata (params, MoE, modality) from it so
            # the report is accurate; identity (name) stays the served id.
            if getattr(args, "tokenizer", None) and (not model.total_params_b):
                from .core.models import enrich_metadata_from
                if enrich_metadata_from(model, args.tokenizer):
                    logger.info(
                        f"Enriched metadata from --tokenizer '{args.tokenizer}': "
                        f"{model.total_params_b:.1f}B, "
                        f"{'MoE' if model.is_moe else 'dense'}, "
                        f"{'multimodal' if model.supports_multimodal else 'text'}")
            return [model]

    # If specific models requested
    if args.models:
        for model_name in args.models:
            model = registry.get(model_name)
            if not model:
                logger.warning(
                    f"Unknown model: {model_name}. "
                    f"Use a HuggingFace model ID (e.g. org/model-name) "
                    f"to auto-register any vLLM-compatible model."
                )
                continue
            models.append(model)
    else:
        # Filter by category and priority
        category = (
            ModelCategory[args.category.upper()]
            if args.category
            else None
        )
        priority = Priority[args.priority] if args.priority else None

        models = registry.filter(
            category=category,
            priority=priority,
            supports_gguf=(args.format == "gguf"),
        )

    return models


def get_formats_from_args(args: argparse.Namespace) -> list:
    """Get list of formats to test."""
    if getattr(args, "remote_endpoint", None):
        return [ModelFormat.REMOTE]
    elif args.format == "hf":
        return [ModelFormat.HF]
    elif args.format == "gguf":
        return [ModelFormat.GGUF]
    else:  # both
        return [ModelFormat.HF, ModelFormat.GGUF]



def _handle_list(args) -> int:
    """`--list <what>`: show what gbench can run, then exit.

    The suites, pillars and presets were previously only discoverable by reading the
    source or scraping --help, which made it hard to know what a sweep would actually
    execute (or which suites would skip).
    """
    what = args.list_what
    show = (lambda k: what in (k, "all"))

    if show("pillars") or show("evals"):
        from .runners.evals import BUILTIN_PILLARS
        from .runners.eval_suites import SUITES
        in_pillars = {e for _, evs in BUILTIN_PILLARS for e in evs}
        if show("pillars"):
            print(f"\nBuilt-in pillars ({len(BUILTIN_PILLARS)}):\n")
            for pillar, evs in BUILTIN_PILLARS:
                print(f"  {pillar}  ({len(evs)})")
                print("      " + "  ".join(sorted(evs)))
        if show("evals"):
            print(f"\nBuilt-in eval suites ({len(SUITES)}):\n")
            pillar_of = {e: p for p, evs in BUILTIN_PILLARS for e in evs}
            width = max((len(n) for n in SUITES), default=10) + 2
            for name in sorted(SUITES):
                tag = pillar_of.get(name, "(not in --evals all)")
                print(f"  {name:<{width}} {tag}")
            orphan = sorted(set(SUITES) - in_pillars)
            if orphan:
                print(f"\n  NB {len(orphan)} suite(s) are registered but NOT part of "
                      f"`--evals all`: {', '.join(orphan)}")

    if show("presets"):
        print("\nPresets (--preset):\n  quick    reduced scenario set\n  default  full scenario set")

    if show("golden"):
        # Golden tasks are JSON files under gbench/golden_dataset/, each with an "id";
        # there is no module-level constant to import.
        import json as _json
        from pathlib import Path as _Path
        dataset_dir = _Path(__file__).parent / "golden_dataset"
        if not dataset_dir.is_dir():
            print(f"\nGolden Set tasks: dataset directory not found at {dataset_dir}")
        else:
            rows = []
            for fp in sorted(dataset_dir.glob("*.json")):
                try:
                    data = _json.loads(fp.read_text(encoding="utf-8"))
                except Exception as e:
                    rows.append((fp.stem, f"<unreadable: {type(e).__name__}>", ""))
                    continue
                rows.append((str(data.get("id", fp.stem)),
                             str(data.get("category", "uncategorized")),
                             str(data.get("description", data.get("name", "")))[:52]))
            print(f"\nGolden Set tasks ({len(rows)}) - use with --golden / --golden-tasks:\n")
            width = max((len(r[0]) for r in rows), default=10) + 2
            cat_w = max((len(r[1]) for r in rows), default=8) + 2
            for tid, cat, desc in rows:
                print(f"  {tid:<{width}} {cat:<{cat_w}} {desc}")

    if show("models"):
        try:
            from .core.models import MODELS
            entries = MODELS.values() if isinstance(MODELS, dict) else MODELS
            entries = list(entries)
            print(f"\nModels ({len(entries)}) - use with --models:\n")
            width = max((len(str(getattr(m, "name", m))) for m in entries), default=10) + 2
            for m in sorted(entries, key=lambda x: str(getattr(x, "name", x))):
                name = str(getattr(m, "name", m))
                cat = getattr(getattr(m, "category", None), "value", "") or ""
                prio = getattr(getattr(m, "priority", None), "value", "") or ""
                hf = getattr(m, "hf_model_id", "") or ""
                print(f"  {name:<{width}} {str(cat):<12} {str(prio):<4} {hf}")
        except Exception as e:
            print(f"\nModels: unavailable ({type(e).__name__}: {e}); see --models")

    if show("plugins"):
        from .runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
        paths = getattr(args, "eval_plugins_dir", None)
        if not paths:
            env_paths = os.environ.get("GBENCH_EVAL_PLUGINS_PATH", "").strip()
            if env_paths:
                paths = [p.strip() for p in env_paths.split(os.pathsep) if p.strip()]
            elif Path("plugins").is_dir():
                paths = ["plugins"]
            elif Path("gbench/plugins").is_dir():
                paths = ["gbench/plugins"]
        if not paths:
            print("\nPlugins (--list plugins): no plugin directory specified.\n  Pass --eval-plugins-dir <path> or set GBENCH_EVAL_PLUGINS_PATH to discover custom suites.")
        else:
            discovered = discover_and_register_plugins(paths if isinstance(paths, list) else [paths])
            print(f"\nPlugins discovered in {paths} ({len(discovered)}):")
            for name in sorted(discovered):
                print(f"  {name:<52} [{CUSTOM_PILLARS.get(name, 'Custom Plugin')}]")

    print()
    return 0


def main(argv: Optional[list[str]] = None):
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Show help if no arguments provided
    import sys
    if (argv is None and len(sys.argv) == 1) or (argv is not None and len(argv) == 0):
        parser.print_help()
        return 0

    # List plugins mode (independent of any benchmark execution)
    if getattr(args, "list_what", None):
        return _handle_list(args)

    if getattr(args, "calibrate", False):
        from .utils.calibrate import calibration_report
        print("\n" + calibration_report() + "\n")
        return 0

    if getattr(args, "eval_provenance", False):
        from .runners.eval_suites import SUITES
        from .runners.eval_suites.canonical_sync import canonical_sync_for
        rows = []
        for name in sorted(SUITES):
            cs = canonical_sync_for(name)
            if cs.get("status") == "reconciled":
                ver = cs.get("checksum") or cs.get("revision") or cs.get("dataset") or cs.get("upstream") or ""
                pinned = "pinned" if (cs.get("revision") or cs.get("checksum")) else "date-only"
                rows.append((name, cs.get("synced", ""), pinned, str(ver)[:44]))
            else:
                rows.append((name, "unverified", "", ""))
        w = max((len(r[0]) for r in rows), default=8)
        recon = sum(1 for r in rows if r[1] != "unverified")
        pinned_n = sum(1 for r in rows if r[2] == "pinned")
        print(f"\nCanonical-sync provenance ({recon}/{len(rows)} reconciled, "
              f"{pinned_n} version-pinned, {recon - pinned_n} date-only):\n")
        print(f"  {'suite':<{w}}  {'synced':<11}  {'status':<9}  upstream/checksum ref")
        print(f"  {'-'*w}  {'-'*11}  {'-'*9}  {'-'*24}")
        for n, s, st, v in rows:
            print(f"  {n:<{w}}  {s:<11}  {st:<9}  {v}")
        print()
        return 0

    if getattr(args, "list_plugins", None) is not None:
        target_path = args.list_plugins
        from .runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
        discovered = discover_and_register_plugins([target_path])
        if not discovered:
            print(f"No plugins found in '{target_path}'.")
            return 0
        print(f"\nDiscovered {len(discovered)} evaluation plugin(s) in '{target_path}':\n")
        max_name_len = max(len(name) for name in discovered.keys())
        for name, fn in sorted(discovered.items()):
            pillar = CUSTOM_PILLARS.get(name, "Custom Plugin")
            print(f"  • {name:<{max_name_len + 2}} [{pillar}]")
        print()
        return 0

    # Validation constraints
    if getattr(args, "remote_endpoint", None) and not getattr(args, "tokenizer", None):
        parser.error(
            "--tokenizer <hf_repo_or_local_path> is required when using --remote-endpoint (e.g. --tokenizer google/gemma-4-E4B-it)."
        )
    if getattr(args, "remote_endpoint", None) and getattr(args, "format", None) and args.format != "both":
        parser.error("--remote-endpoint and --format are mutually exclusive.")
    if getattr(args, "campaign", None) and (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None)):
        parser.error("--campaign and --input-lengths/--output-lengths are mutually exclusive.")
    if getattr(args, "num_prompts", None) and getattr(args, "serving_only", False):
        parser.error(
            "--num-prompts cannot be used with --serving-only: the serving pillar auto-sizes its own "
            "sample count per config (a ~5-minute wall-clock target that scales up with --batch-sizes "
            "and down with --output-lengths). Change the count via --batch-sizes / --output-lengths, "
            "or use --personal for a fixed small count.")
    if not getattr(args, "campaign", None) and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None)) and not getattr(args, "preset", None):
        # --stage-to-gcs only copies model artifacts to a bucket. It never
        # runs a benchmark, so it has no workload geometry to describe and
        # belongs with the other modes exempted here.
        if not getattr(args, "evals", None) and not getattr(args, "evals_only", False) and not getattr(args, "golden_only", False) and not getattr(args, "quality_only", False) and not getattr(args, "stress_test", False) and not getattr(args, "stage_to_gcs", None):
            parser.error("Either --preset, --campaign, or --input-lengths/--output-lengths must be provided.")

    # Strict Eval Argument Compatibility Validation
    from .runners.eval_suites import discover_and_register_plugins, SUITES
    discover_and_register_plugins(getattr(args, "eval_plugins_dir", None))

    raw_evals = getattr(args, "evals", None) or []
    flat_evals = []
    for item in raw_evals:
        for sub in str(item).split(","):
            sub_clean = sub.strip()
            if sub_clean:
                flat_evals.append(sub_clean)
    args.evals = flat_evals
    evals_selected = set(flat_evals)
    if evals_selected:
        for e in evals_selected:
            if e not in {"all", "plugins"} and e.lower() not in SUITES:
                parser.error(f"Unknown eval suite '{e}'. Available: {', '.join(sorted(SUITES.keys()))}")

    if getattr(args, "eval_categories", None):
        if not evals_selected or any(e not in {"bfcl", "all"} for e in evals_selected):
            parser.error("--eval-categories is only supported when running 'bfcl'. Cannot combine with unsupported evals.")
    if getattr(args, "eval_max_soft_tokens", None) is not None:
        if getattr(args, "remote_endpoint", None):
            parser.error("--eval-max-soft-tokens cannot be used with --remote-endpoint (soft tokens are configured server-side).")
        if not evals_selected or any(e not in {"mmmu_pro", "screenspot", "semantic_keypoint", "textvqa", "infographicvqa", "bundled_detection", "all"} for e in evals_selected):
            parser.error("--eval-max-soft-tokens is only supported by vision evals ('mmmu_pro', 'screenspot', 'semantic_keypoint', 'textvqa', 'infographicvqa', 'bundled_detection').")
    if getattr(args, "eval_n_shot", None) is not None:
        if not evals_selected or any(e not in {"mmlu", "mmlu_pro", "gsm8k", "all"} for e in evals_selected):
            parser.error("--eval-n-shot is only supported by 'mmlu', 'mmlu_pro' and 'gsm8k'.")

    # --max-output-tokens is required for eval runs: it bounds generation and materially
    # affects scores (truncation reads as a failure), so it must be explicit rather than
    # silently taking a per-suite default - otherwise two runs are quietly incomparable.
    if (evals_selected or getattr(args, "evals_only", False)) \
            and getattr(args, "max_output_tokens", None) is None:
        parser.error(
            "--max-output-tokens is required for eval runs: it bounds generation and "
            "materially affects scores (truncation reads as a failure), so it must be "
            "explicit for runs to be comparable. Set it, e.g. --max-output-tokens 65536.")




    # Setup logging
    setup_logging(args.verbose)

    # --personal (device-peak) mode: a single flag for "what will this model do
    # on my machine". Implemented as a thin convenience over existing gates -
    # serving-only + text-only (so stress/throughput/MM are skipped by the usual
    # guards) + few full-decode samples + the device-peak report. Incompatible
    # with --campaign (it uses a fixed single-stream shape).
    if getattr(args, "personal", False):
        # Reject flags that would silently produce a zero-benchmark no-op or fight
        # the single-stream shape. (--personal forces serving-only; combining it
        # with --stress-test/--throughput-only cancels every pillar.)
        _conflict = [name for name, flag in (
            ("--campaign", getattr(args, "campaign", None)),
            ("--stress-test", getattr(args, "stress_test", False)),
            ("--throughput-only", getattr(args, "throughput_only", False)),
        ) if flag]
        if _conflict:
            logger.error(f"--personal is incompatible with {'/'.join(_conflict)} "
                         f"(it runs single-stream text serving only). Drop one.")
            return 1
        if not getattr(args, "remote_endpoint", None):
            logger.warning("--personal is intended for --remote-endpoint (e.g. Ollama at "
                           "http://localhost:11434/v1); without it a local vLLM + GPU is "
                           "required and the personal-device framing doesn't apply.")
        # Serving-only (single-stream) is the personal-device metric. MM is NOT
        # forced off: it is probe-gated (runs only if the endpoint accepts images),
        # so a vision model on this device still gets measured. Pass --text-only to
        # skip MM (it's slow single-stream on CPU).
        args.serving_only = True
        if getattr(args, "remote_endpoint", None) and not args.text_only:
            logger.info("--personal: multimodal will run IF the endpoint accepts images "
                        "(probe-gated); note MM single-stream is slow on CPU. Add "
                        "--text-only to skip it.")

    # Get configuration
    config = get_config_from_args(args)
    # MM capability-probe timeout: honor the flag for any run; --personal defaults
    # it higher (cold CPU vision is slow to first token, so 10s falsely reads as
    # "unsupported" and skips MM).
    config.mm_probe_timeout = getattr(args, "mm_probe_timeout", None)
    # Per-request serving timeout: override the config default (600) only when the
    # flag is given, so unset runs keep the documented default.
    if getattr(args, "timeout", None) is not None:
        config.timeout = args.timeout
    # Thinking/reasoning mode: default OFF (no-think system prompt on chat
    # serving); the existing --thinking/--enable-thinking flag (dest eval_thinking)
    # enables it for serving too, so one flag governs "let the model think".
    config.thinking = bool(getattr(args, "eval_thinking", False))
    if getattr(args, "personal", False):
        config.personal_mode = True
        config.num_iterations = 1
        config.warmup_iterations = 1
        config.serving_num_prompts_override = 12
        config.batch_sizes = [1]
        # 1 image/request (280 tokens) - lighter than 4 (1120) AND needs no
        # server-side --limit-mm-per-prompt>1, which a remote endpoint (Ollama)
        # may not have. MM still probe-gated; skipped if the endpoint refuses images.
        config.mm_images_per_request = 1
        if config.mm_probe_timeout is None:
            config.mm_probe_timeout = 120   # slow CPU vision needs a longer probe
        if not args.output_lengths:
            config.output_lengths = [512]
        if not args.input_lengths:
            config.input_lengths = [512]
    evals_only_mode = getattr(args, "evals_only", False) or (
        bool(getattr(args, "evals", None))
        and not getattr(args, "campaign", None)
        and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None))
    )
    logger.info(f"Using configuration: {args.preset}")
    if evals_only_mode:
        logger.info(f"  Mode: Evaluation Benchmarks Only")
        logger.info(f"  Evals: {config.evals}")
        logger.info(f"  Concurrency: {config.batch_sizes[0] if config.batch_sizes else 8} (from --batch-sizes)")
        logger.info(f"  Thinking enabled: {config.eval_thinking}")
        eff_max = config.eval_max_output_tokens or (16384 if config.eval_thinking else 8192)
        logger.info(f"  Max output tokens: {eff_max}")
    else:
        logger.info(f"  Iterations: {config.num_iterations} (+{config.warmup_iterations} warmup)")
        logger.info(f"  Batch sizes: {config.batch_sizes}")
        logger.info(f"  Input/Output lengths: {config.input_lengths}/{config.output_lengths}")
        # The serving sample count is NOT config.num_prompts (dead for serving) -
        # it's computed per config by _serving_num_prompts (honors the --personal
        # override). Show the effective value so the summary isn't misleading.
        try:
            _serv_n = config._serving_num_prompts(
                config.batch_sizes[0] if config.batch_sizes else 1)[0]
        except Exception:
            _serv_n = config.num_prompts
        logger.info(f"  Num prompts: serving={_serv_n} (auto-sized per config; ~5-min wall-clock "
                    f"target, scales with batch size and output length - NOT --num-prompts), "
                    f"throughput={config.num_prompts_throughput}")
        if getattr(args, "num_prompts", None):
            logger.warning(
                "  --num-prompts=%s does NOT control the serving pillar (it auto-sizes per config, "
                "shown above). Use --batch-sizes / --output-lengths to change the serving count, or "
                "--personal for a fixed count. Throughput uses its own count (%s).",
                args.num_prompts, config.num_prompts_throughput)
        stress_disabled = args.no_stress_test or args.serving_only or args.throughput_only
        logger.info(f"  Stress test: {'disabled' if stress_disabled else 'enabled (ramp-up)'}")
    logger.info(f"Results directory: {config.results_dir}")

    # Pre-flight over-subscription guard: a requested containerized-eval concurrency beyond this
    # host's CPU/RAM would exhaust it (OOM / Docker address-pool) mid-sweep. Refuse gracefully with
    # a recommendation (bypass with GBENCH_SKIP_CALIBRATION_GUARD=1). The effective concurrency is
    # --sandboxes when set, else --batch-sizes[0] (evals.py:_run_eval_suite derives it that way), so
    # the guard covers the DEFAULT knob (--batch-sizes) too, not only --sandboxes. Eval runs only.
    _running_evals = bool(getattr(config, "evals", None))
    _eff_conc = (config.sandboxes if config.sandboxes
                 else (config.batch_sizes[0] if (config.batch_sizes and _running_evals) else None))
    if _eff_conc and not config.dry_run:
        from .utils.calibrate import detect_hardware, check_oversubscription
        _ok, _msg = check_oversubscription(_eff_conc, detect_hardware())
        if not _ok:
            logger.error(_msg)
            return 1

    # Pre-flight GPU check: ensure vLLM engine and GPU are ready for local benchmarking
    if not config.dry_run and not args.stage_to_gcs and not config.remote_endpoint:
        from gbench.utils import require_vllm_engine
        require_vllm_engine("Local GPU benchmarking")
        # Process/idle check (no hard free-memory floor by default - the old fixed
        # 70GB blocked every <70GB GPU; physical fit is model-aware below + vLLM's
        # profiler). --min-free-gb re-enables an explicit hard floor.
        gpu_ready, gpu_msg = check_gpu_ready(min_free_memory_gb=getattr(args, "min_free_gb", None))
        if not gpu_ready:
            logger.error(
                f"GPU pre-flight check failed:\n{gpu_msg}\n"
                "No local CUDA GPU? Run against a remote vLLM server with "
                "--remote-endpoint <url> (serving + stress; offline throughput "
                "needs a local GPU). Add --text-only unless that server was "
                "started with --limit-mm-per-prompt for images.")
            return 1
        elif gpu_msg.startswith("⚠"):
            logger.warning(gpu_msg)
        else:
            logger.info(gpu_msg)

    # Get models to benchmark
    models = get_models_from_args(args)
    if not models:
        logger.error("No models selected")
        return 1

    if args.stage_to_gcs:
        stage_models_to_gcs(models, args.stage_to_gcs)
        return 0

    logger.info(f"Selected {len(models)} model(s) for benchmarking")
    
    # Save run metadata
    if config.log_manager and not args.stage_to_gcs and not config.dry_run:
        metadata = {
            "timestamp": config.log_manager.timestamp,
            "models": [m.name for m in models],
            "tags": config.tags or [],
            "preset": args.preset,
            "benchmark_mode": args.quality_only and "quality" or "performance",
        }
        config.log_manager.save_metadata(metadata)
    for model in models:
        moe_tag = f", MoE {model.num_experts}x{model.num_active_experts}" if model.is_moe else ""
        logger.info(f"  - {model.name} ({model.total_params_b:.0f}B{moe_tag})")

    # Validate GPU configuration against first model (skip if remote endpoint is used)
    first_model = models[0]
    if config.remote_endpoint:
        logger.info(f"Remote endpoint specified ({config.remote_endpoint}); checking if endpoint is functional...")
        from .utils import verify_endpoint_functional
        is_up, err_msg, max_model_len = verify_endpoint_functional(config.remote_endpoint)
        if not is_up:
            logger.error(f"❌ Remote endpoint '{config.remote_endpoint}' is unreachable or not functional: {err_msg}")
            logger.error("Cancelling benchmark run.")
            return 1
        logger.info(f"✅ Remote endpoint '{config.remote_endpoint}' is functional and answering requests.")
        if max_model_len:
            logger.info(f"   Server reported max_model_len: {max_model_len} tokens")
        logger.info(f"Remote endpoint specified ({config.remote_endpoint}); skipping local GPU count check.")
    else:
        # We are in the LOCAL branch (no --remote-endpoint). A local vLLM server is launched whenever
        # a performance run, evals, or quality runs; its TP = config.num_gpus, so omitting --num-gpus
        # silently serves a large model at TP=1 and OOMs. Exempt only runs that serve NO local model:
        # --dry-run, and --golden-only WHEN it is not ALSO running evals/quality (golden-only can be
        # combined with --evals / --quality, which do serve a local model). Remote is handled above.
        # Post-dispatch guard (NOT arg required=True) so --list/--help/--remote/--dry-run are never
        # blocked at parse time.
        _also_evals_or_quality = bool(config.evals) or args.quality or args.quality_only
        _exempt_gpu = config.dry_run or (args.golden_only and not _also_evals_or_quality)
        if args.num_gpus is None and not _exempt_gpu:
            logger.error(
                "--num-gpus is required for a local run that serves a model (performance, evals, or "
                "quality) so tensor-parallel matches the model size (e.g. --num-gpus 1). Omit it "
                "only with --remote-endpoint, a pure --golden-only run, or --dry-run.")
            return 1
        is_valid, gpu_msg = validate_gpu_config(
            config.num_gpus, first_model.total_params_b,
            gpu_memory_utilization=config.gpu_memory_utilization)
        # Model-aware weights-fit ADVISORY (WARN only; vLLM's profiler is the hard
        # authority). Sized on the largest selected model, at the resolved TP.
        _max_params = max((m.total_params_b for m in models), default=first_model.total_params_b)
        _fits, _fit_msg = weights_fit_advisory(
            _max_params, config.tensor_parallel_size or config.num_gpus or 1,
            config.gpu_memory_utilization, min_free_override=getattr(args, "min_free_gb", None))
        (logger.warning if not _fits else logger.info)(_fit_msg)
        if not is_valid:
            if _exempt_gpu:
                logger.warning(f"GPU configuration error (ignored: this run serves no local model): {gpu_msg}")
            else:
                logger.error(f"GPU configuration error: {gpu_msg}")
                return 1
        elif "Warning" in gpu_msg:
            logger.warning(gpu_msg)
        else:
            logger.info(gpu_msg)

    # Get formats
    formats = get_formats_from_args(args)
    if config.remote_endpoint and len(formats) > 1:
        logger.info("Remote endpoint specified; testing single active server format (hf).")
        formats = [ModelFormat.HF]
    logger.info(f"Testing formats: {[f.value for f in formats]}")

    # Determine benchmark modes
    # Default: run text + multimodal for multimodal-capable models
    # --text-only: skip multimodal benchmarks entirely
    # --multimodal-only: skip text benchmarks, only run multimodal
    # --serving-only / --throughput-only: control benchmark type, not text vs multimodal
    evals_only = getattr(args, "evals_only", False) or (
        bool(getattr(args, "evals", None))
        and not getattr(args, "campaign", None)
        and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None))
    )
    run_golden = args.golden or args.golden_only
    run_quality = args.quality or args.quality_only
    run_text = not args.multimodal_only and not args.quality_only and not args.golden_only and not evals_only
    run_multimodal = not args.text_only and not args.quality_only and not args.golden_only and not evals_only
    
    if args.multimodal_only:
        run_multimodal = True
        run_text = False
    
    # Display benchmark mode
    if evals_only:
        benchmark_mode = "evals only"
    elif args.golden_only:
        benchmark_mode = "golden set only"
    elif args.quality_only:
        benchmark_mode = "quality only"
    elif args.multimodal_only:
        benchmark_mode = "multimodal only"
    elif run_text and run_multimodal:
        benchmark_mode = "text and multimodal (for capable models)"
    elif run_text:
        benchmark_mode = "text only"
    else:
        benchmark_mode = "multimodal only"
    logger.info(f"Benchmark mode: {benchmark_mode}")

    # Create runners
    from .runners import ServingBenchmarkRunner, ThroughputBenchmarkRunner, StressTestRunner, QualityBenchmarkRunner, GoldenBenchmarkRunner, EvalsBenchmarkRunner
    serving_runner = ServingBenchmarkRunner(config)
    throughput_runner = ThroughputBenchmarkRunner(config)
    # Resolve the SLO for the active campaign: CLI global override (if given) else
    # the per-campaign default map. Frozen here for this run (all models compared
    # in one run share it) and recorded in the fingerprint downstream.
    _ttft_slo, _itl_slo = resolve_campaign_slo(
        config.campaign, args.stress_threshold, args.itl_slo_ms
    )
    stress_runner = StressTestRunner(
        config,
        ttft_threshold_ms=_ttft_slo,
        tpot_threshold_ms=_itl_slo,
    )
    quality_runner = QualityBenchmarkRunner(config)
    golden_runner = GoldenBenchmarkRunner(config)
    evals_runner = EvalsBenchmarkRunner(config)

    # Determine if stress test should run
    # --stress-test means run stress test only (stress_test_only)
    # --no-stress-test means skip stress test entirely
    # --serving-only or --throughput-only also skip stress test
    stress_test_only = args.stress_test  # --stress-test means ONLY run stress test
    run_stress_test = (
        (args.stress_test or not args.no_stress_test)
        and not args.serving_only
        and not args.throughput_only
        and not args.quality_only
        and not args.golden_only
        and not evals_only
    )

    # Scaffold contract: record every (pillar, model, format) condition this
    # run plans to produce. See docs/scaffold-versioning.md.
    #
    # `conditions` is the planned matrix, not the outcome. Multimodal pillars
    # are listed from the model's declared capability, and a remote endpoint
    # probe can still skip one at run time. The results say what completed.
    from .core.scaffold import (
        Pillar,
        build_condition,
        gbench_commit,
        render_conditions,
        resolve_gemmaclaw_sha,
    )

    # A branch name is not a pin. The default is already a sha, so this is
    # a no-op that touches nothing for an unflagged run, but anyone passing
    # `--gemmaclaw-commit main` or a tag needs it resolved. Resolve once,
    # here, so the scaffold built below and the checkout the quality runner
    # performs later are the same commit by construction. Resolving in both
    # places instead would let `main` advance in between and pin a scorer
    # the run never used.
    #
    # Gated on the pillar actually being planned, because a non-sha ref
    # reaches the network and a serving-only run has no business doing that.
    if run_quality:
        resolved = resolve_gemmaclaw_sha(config.gemmaclaw_commit, config.gemmaclaw_path)
        if resolved:
            if resolved != config.gemmaclaw_commit:
                logger.info(
                    f"Resolved gemmaclaw ref '{config.gemmaclaw_commit}' to {resolved}"
                )
            config.gemmaclaw_commit = resolved
        else:
            logger.warning(
                f"Could not resolve gemmaclaw ref '{config.gemmaclaw_commit}' to a "
                "commit. The quality scaffold_id will pin the ref itself, so it "
                "will not move when that ref does."
            )

    conditions = []
    is_remote = bool(getattr(args, "remote_endpoint", None))
    for model in models:
        mm = run_multimodal and model.supports_multimodal
        pillars = [
            (Pillar.SERVING, run_text and not args.throughput_only and not stress_test_only),
            (Pillar.THROUGHPUT, run_text and not args.serving_only and not stress_test_only and not is_remote),
            (Pillar.SERVING_MULTIMODAL,
             mm and not args.throughput_only and not stress_test_only),
            (Pillar.THROUGHPUT_MULTIMODAL,
             mm and not args.serving_only and not stress_test_only and not is_remote),
            (Pillar.STRESS_TEST, run_stress_test),
            (Pillar.STRESS_TEST_MULTIMODAL, mm and run_stress_test),
            (Pillar.QUALITY, run_quality),
            (Pillar.GOLDEN, run_golden),
            (Pillar.EVALS, bool(config.evals)),
        ]
        for pillar, planned in pillars:
            if not planned:
                continue
            for fmt in formats:
                is_golden = pillar is Pillar.GOLDEN
                conditions.append(build_condition(
                    pillar,
                    model,
                    fmt,
                    dataset_dir=golden_runner.dataset_dir if is_golden else None,
                    selected_tasks=getattr(args, "golden_tasks", None) if is_golden else None,
                    config=config,
                ))

    # Printed before the persist, and outside its condition, because the
    # console and metadata.json are two views of the same objects rather
    # than one depending on the other. A run with no log manager, a dry
    # run, and a GCS staging run are all still entitled to know what
    # scaffold they are about to measure against.
    if conditions:
        print("\n" + render_conditions(conditions))

    # A SECOND save_metadata rather than a move of the first one. The early
    # write happens before the plan exists, and removing it would leave an
    # aborted run as a directory with no metadata.json at all, which
    # service/utils/storage.py renders as a run with unknown models.
    if config.log_manager and not args.stage_to_gcs and not config.dry_run:
        metadata["conditions"] = [c.to_dict() for c in conditions]
        # Beside the conditions rather than inside any id. Results are
        # never uploaded anywhere central by a CLI run, so a run
        # directory has to be self-describing: there is no index to
        # reconstruct later which harness produced it.
        metadata["gbench"] = gbench_commit()
        config.log_manager.save_metadata(metadata)

    if config.dry_run:
        logger.info("Dry-run complete. Exiting without executing benchmarks.")
        return 0

    # Track all results for final summary
    all_results = []

    # Execute benchmarks
    total_runs = 0
    failed_models = []
    for model in models:
      try:
        # Check multimodal compatibility (dynamic HTTP probe for remote endpoints)
        if getattr(args, "remote_endpoint", None) and run_multimodal:
            from gbench.utils import probe_multimodal_support
            _probe_to = int(getattr(config, "mm_probe_timeout", None) or 10)
            logger.info(f"Probing remote endpoint multimodal capability for {model.name} "
                        f"(timeout {_probe_to}s)...")
            model_supports_multimodal = probe_multimodal_support(
                args.remote_endpoint, model.hf_model_id or model.name, timeout=_probe_to)
            if model_supports_multimodal:
                logger.info(f"✅ Remote model {model.name} supports multimodal requests.")
            else:
                logger.info(f"ℹ️ Remote model {model.name}: endpoint rejected the image "
                            f"probe (text-only model, or endpoint unreachable) - skipping "
                            f"MM. A real multimodal model accepts the payload even when a "
                            f"cold vision load makes it slow to respond.")
        else:
            # Image-based MM benchmarks require VISION, so key off
            # supports_multimodal (has_vision) - NOT category==MULTIMODAL, which
            # is vision OR audio; an audio-only model would otherwise be handed
            # image benchmarks it cannot serve.
            model_supports_multimodal = model.supports_multimodal

        if run_multimodal and not model_supports_multimodal:
            logger.warning(
                f"{model.name} does not support multimodal, skipping multimodal benchmarks"
            )
        
        for format in formats:
            # Skip GGUF if not available
            if format == ModelFormat.GGUF and not model.gguf_model_id:
                logger.warning(
                    f"GGUF not available for {model.name}, skipping"
                )
                continue

            logger.info(
                f"\n{'='*60}\n"
                f"Benchmarking: {model.name} ({format.value})\n"
                f"{'='*60}"
            )

            # Apply param-based batch sizes (unless user specified --batch-sizes)
            if not args.batch_sizes:
                model_batch_sizes = get_batch_sizes(model.total_params_b, args.preset)
                config.batch_sizes = model_batch_sizes
                logger.info(f"  Batch sizes for {model.total_params_b:.0f}B: {model_batch_sizes}")

            # Run text-only serving benchmarks
            if run_text and not args.throughput_only and not stress_test_only:
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running serving benchmarks (text) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running serving benchmarks (text)...")
                    results = serving_runner.run_all(model, format)
                    for result in results:
                        result['benchmark_type'] = 'serving'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run text-only throughput benchmarks
            if run_text and not args.serving_only and not stress_test_only and not getattr(args, "remote_endpoint", None):
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running throughput benchmarks (text) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running throughput benchmarks (text)...")
                    results = throughput_runner.run_all(model, format)
                    for result in results:
                        result['benchmark_type'] = 'throughput'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run multimodal serving benchmarks - sends real image+text requests
            if run_multimodal and model_supports_multimodal and not args.throughput_only and not stress_test_only:
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running serving benchmarks (multimodal) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running serving benchmarks (multimodal)...")
                    results = serving_runner.run_all(model, format, multimodal=True)
                    for result in results:
                        result['benchmark_type'] = 'serving_multimodal'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run multimodal benchmarks (throughput) - uses custom offline inference
            if run_multimodal and model_supports_multimodal and not args.serving_only and not stress_test_only and not getattr(args, "remote_endpoint", None):
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running throughput benchmarks (multimodal) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running throughput benchmarks (multimodal)...")
                    results = throughput_runner.run_all(model, format, multimodal=True)
                    for result in results:
                        result['benchmark_type'] = 'throughput_multimodal'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run stress test (ramp-up to find max sustainable throughput)
            if run_stress_test:
                logger.info("Running stress test (finding max sustainable throughput)...")
                # Honor --text-only / --multimodal-only for stress too: the text
                # pass is gated by run_text, the multimodal pass by run_multimodal
                # (and MM-capability, inside run_all). Each result is stamped by
                # its own mode below.
                stress_results = stress_runner.run_all(
                    model, format,
                    include_text=run_text,
                    # Gate MM stress on model capability (matches MM serving at the
                    # top of this loop). Without this, a text-only model - or a MM
                    # model against a text-only remote endpoint - would build images
                    # and trip the MM preflight, tracebacking on work that can't run.
                    include_multimodal=run_multimodal and model_supports_multimodal,
                )
                for result in stress_results:
                    result['benchmark_type'] = (
                        'stress_test_multimodal' if result.get('multimodal')
                        else 'stress_test'
                    )
                    result['model'] = model.short_name
                    result['model_name'] = model.name
                    result['model_short'] = model.short_name
                    result['format'] = format.value
                    all_results.append(result)
                total_runs += 1

            # Run quality benchmarks (gemmaclaw agentic)
            if run_quality:
                logger.info("Running quality benchmarks (gemmaclaw)...")
                result = quality_runner.run(model, format)
                result['benchmark_type'] = 'quality'
                result['model'] = model.short_name
                result['model_name'] = model.name
                result['model_short'] = model.short_name
                result['format'] = format.value
                all_results.append(result)
                total_runs += 1

            # Run Golden Set exact-match benchmarks
            if run_golden:
                logger.info("Running Golden Set benchmarks...")
                result = golden_runner.run(model)
                result['benchmark_type'] = 'golden'
                result['model'] = model.short_name
                result['model_name'] = model.name
                result['model_short'] = model.short_name
                result['format'] = format.value
                all_results.append(result)
                total_runs += 1

            # Run Evaluation benchmarks (bfcl, gpqa, gsm8k, mmlu, mmmu_pro, mrcr, screenspot)
            if config.evals:
                logger.info("Running Evaluation benchmarks...")
                eval_results = evals_runner.run(model, format)
                for eval_res in eval_results:
                    all_results.append(eval_res)
                    total_runs += 1


      except Exception as e:
        logger.error(f"❌ Model {model.name} failed: {e}")
        logger.error("Continuing with next model...")
        failed_models.append(model.name)

    # Print comprehensive results summary
    print_results_summary(all_results, config)

    if failed_models:
        logger.error(f"❌ {len(failed_models)} model(s) failed: {', '.join(failed_models)}")
        return EXIT_MODEL_FAILURE

    # A perf pillar (stress/serving/throughput) that breaks mid-campaign returns
    # a {'failed': True} dict instead of raising, so it never reaches
    # failed_models above and gbench would exit 0 - masking a broken run from
    # eval_performance.sh's `set -e`. Surface genuine TEXT-pillar harness
    # failures as EXIT_MODEL_FAILURE. Match on `is True` (harness dicts set a
    # bool; vLLM's raw "failed" is an int request-count) AND benchmark_type
    # (text pillars only). Multimodal passes are logged but TOLERATED: their
    # auto-launched server may lack the gemma4 MM flags, a known non-fatal error
    # - otherwise a benign MM error would abort the model and skip models 2-4.
    TEXT_PILLARS = ("serving", "throughput", "stress_test")
    MM_PILLARS = ("serving_multimodal", "throughput_multimodal", "stress_test_multimodal")
    text_harness_failures = [
        r for r in all_results
        if r.get("failed") is True and r.get("benchmark_type") in TEXT_PILLARS
    ]
    mm_harness_failures = [
        r for r in all_results
        if r.get("failed") is True and r.get("benchmark_type") in MM_PILLARS
    ]
    if mm_harness_failures:
        logger.warning(
            "⚠️  %d multimodal pass(es) failed (tolerated: auto-server may lack "
            "gemma4 MM flags).", len(mm_harness_failures)
        )
    if text_harness_failures:
        camps = ", ".join(sorted({str(r.get("campaign", "?")) for r in text_harness_failures}))
        logger.error(
            "❌ %d text benchmark configuration(s) failed (campaigns: %s). "
            "Exiting %d (model failure).",
            len(text_harness_failures), camps, EXIT_MODEL_FAILURE,
        )
        return EXIT_MODEL_FAILURE

    golden_code = golden_exit_code(all_results)
    if golden_code == EXIT_HARNESS_ERROR:
        logger.error(
            "❌ Golden Set could not complete. Exiting %d (harness error). "
            "This is not a verdict on the model.", EXIT_HARNESS_ERROR
        )
    elif golden_code == EXIT_MODEL_FAILURE:
        logger.error(
            "❌ Golden Set has failing cases. Exiting %d (model failure).",
            EXIT_MODEL_FAILURE,
        )
    return golden_code


def print_results_summary(results: list[dict], config: BenchmarkConfig):
    """Print comprehensive summary of all benchmark results."""
    if not results:
        logger.warning("No benchmark results to summarize")
        return

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY".center(80))
    print("="*80 + "\n")

    # Separate successful and failed results
    successful_results = [r for r in results if not r.get('failed', False)]
    failed_results = [r for r in results if r.get('failed', False)]

    # Group results by benchmark type (text and multimodal separately)
    serving_results = [r for r in successful_results if r.get('benchmark_type') == 'serving']
    serving_mm_results = [r for r in successful_results if r.get('benchmark_type') == 'serving_multimodal']
    throughput_results = [r for r in successful_results if r.get('benchmark_type') == 'throughput']
    throughput_mm_results = [r for r in successful_results if r.get('benchmark_type') == 'throughput_multimodal']
    # Include BOTH text and multimodal stress (the table prints the mode per row);
    # keying only on 'stress_test' silently dropped every MM stress result.
    stress_results = [r for r in successful_results if r.get('benchmark_type') in ('stress_test', 'stress_test_multimodal')]
    quality_results = [r for r in successful_results if r.get('benchmark_type') == 'quality']
    golden_results = [r for r in successful_results
                      if r.get('benchmark_type') == 'golden']
    serving_failures = [r for r in failed_results if r.get('benchmark_type') == 'serving']
    serving_mm_failures = [r for r in failed_results if r.get('benchmark_type') == 'serving_multimodal']
    throughput_failures = [r for r in failed_results if r.get('benchmark_type') in ('throughput', 'throughput_multimodal')]
    stress_failures = [r for r in failed_results if r.get('benchmark_type') in ('stress_test', 'stress_test_multimodal')]
    quality_failures = [r for r in failed_results if r.get('benchmark_type') == 'quality']

    # ── PERSONAL-DEVICE PEAK (--personal): single-stream tok/s + TTFT headline ──
    if getattr(config, "personal_mode", False) and (serving_results or serving_mm_results):
        def _md(r, m):
            # Cover the pooled keys AND the legacy/remote aggregate's _mean-suffixed
            # keys (remote goes through aggregate_benchmark_results -> mean_{m}_ms_mean).
            for k in (f'{m}_p50_ms', f'median_{m}_ms', f'mean_{m}_ms',
                      f'median_{m}_ms_mean', f'mean_{m}_ms_mean'):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    return v
            return None
        print("=" * 80)
        print("PERSONAL-DEVICE PERFORMANCE".center(80))
        print("=" * 80)
        for r, mode in ([(x, 'text') for x in serving_results]
                        + [(x, 'image+text') for x in serving_mm_results]):
            model = r.get('model_short', r.get('model', 'N/A'))
            served = r.get('format', 'N/A')
            tpot = _md(r, 'tpot'); ttft = _md(r, 'ttft')
            toks = (1000.0 / tpot) if isinstance(tpot, (int, float)) and tpot > 0 else None
            # Support count = requests that actually BACK the percentiles, not the
            # offered prompt count (they differ when requests are empty/error).
            n = (r.get('completed_requests') or r.get('pooled_sample_count')
                 or r.get('num_prompts'))
            emptyc = int(r.get('empty_requests') or 0)
            errc = int(r.get('failed_requests') or 0)
            offered = int(r.get('offered_requests') or 0) or ((n or 0) + emptyc + errc)
            realized = r.get('output_length_realized')
            cv = next((r.get(k) for k in ('mean_tpot_ms_cv_percent',
                       'median_tpot_ms_cv_percent', 'tpot_cv_percent')
                       if isinstance(r.get(k), (int, float))), None)
            print(f"  Model: {model}   ·   served via: {served}   ·   {mode}, single stream (batch=1)")
            print("  " + "-" * 66)
            print(f"    Decode speed         {(f'~{toks:.1f} tok/s' if toks else 'n/a'):<16} how fast the reply streams")
            print(f"    Time to first token  {(f'~{ttft/1000:.1f} s' if isinstance(ttft,(int,float)) else 'n/a'):<16} wait before text appears")
            print(f"    Per-token latency    {(f'~{tpot:.0f} ms' if isinstance(tpot,(int,float)) else 'n/a'):<16} (median TPOT)")
            # Reliability: how many offered requests produced a usable reply, vs
            # empty (model returned 0 tokens) vs error (transport/HTTP failure).
            rel = (f"{n}/{offered} ok" if offered else f"{n} ok")
            if emptyc:
                rel += f" · {emptyc} empty"
            if errc:
                rel += f" · {errc} error"
            _rl = (f"median reply {int(realized)} tok"
                   if isinstance(realized, (int, float)) else "")
            print(f"    Reliability          {rel:<16} {_rl}")
            warn = []
            if emptyc:
                warn.append(f"{emptyc} empty (model returned nothing)")
            if errc:
                reasons = r.get('error_reasons') or {}
                warn.append(f"{errc} errored" + (f" {reasons}" if reasons else ""))
            if r.get('low_confidence') or (isinstance(n, (int, float)) and n < 30):
                warn.append(f"n={n}")
            if isinstance(cv, (int, float)) and cv > 20:
                warn.append(f"TPOT CV {cv:.0f}%")
            if warn:
                print(f"    ⚠ indicative only ({', '.join(warn)})")
        print("  " + "-" * 66)
        print("  Median single-stream (batch=1) on THIS device - a rough LOCAL figure,")
        print("  NOT comparable to datacenter/GPU runs. If the backend ignores ignore_eos")
        print("  (e.g. Ollama), replies stop at natural EOS, so tok/s reflects natural-")
        print("  length outputs, not a fixed decode length.")
        print("=" * 80 + "\n")

    # ── SERVING - single-stream latency (batch=1), text + multimodal together ──
    def _p50(r, m):
        # Prefer the true median across every path's key spelling; include the
        # legacy/remote `_mean`-suffixed median BEFORE the mean, so a remote/
        # legacy row never shows its MEAN under a "p50" header (mirror `_md`).
        for k in (f'{m}_p50_ms', f'median_{m}_ms', f'median_{m}_ms_mean',
                  f'mean_{m}_ms', f'mean_{m}_ms_mean'):
            v = r.get(k)
            if isinstance(v, (int, float)):
                return v
        return 'N/A'
    serv_all = ([(x, 'text') for x in serving_results]
                + [(x, 'MM') for x in serving_mm_results])
    serv_fail = ([(x, 'text') for x in serving_failures]
                 + [(x, 'MM') for x in serving_mm_failures])
    if serv_all or serv_fail:
        print("SERVING - single-stream latency, batch=1 (one request at a time; latency IS the metric)")
        print("-" * 92)
        print(f"{'Model':<20} {'Campaign':<14} {'Mode':<5} {'Input':<8} {'TTFT p50':<10} {'TPOT p50':<10} {'ITL p50':<10}")
        print("-" * 92)
        def _fmt1(v): return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
        has_mm_serv = False
        for r, mode in serv_all:
            model = r.get('model_short', 'N/A')[:20]
            camp = str(r.get('campaign', '-'))[:14]
            inp = r.get('input_length')
            inp_s = f"{int(inp)}" if isinstance(inp, (int, float)) else "-"
            if mode == 'MM':
                has_mm_serv = True
            ok = r.get('request_throughput_repeatability_valid', True)
            print(f"{model:<20} {camp:<14} {mode:<5} {inp_s:<8} {_fmt1(_p50(r,'ttft')):<10} "
                  f"{_fmt1(_p50(r,'tpot')):<10} {_fmt1(_p50(r,'itl')):<10} {'✓' if ok else '✗'}")
        for r, mode in serv_fail:
            model = r.get('model', 'N/A')[:20]; camp = str(r.get('campaign', '-'))[:14]
            print(f"{model:<20} {camp:<14} {mode:<5} {'-':<8} {'FAILED':<10} {'-':<10} {'-':<10} ✗")
        print("  note: pooled across iterations, shown as P50 (median). batch=1 = single-stream,")
        print("        so P99≈P50 (no queueing) - use --batch-sizes for tail-under-load latency.")
        if has_mm_serv:
            _mm = next((r for r, m in serv_all if m == 'MM'), {})
            _ni = int(_mm.get('mm_images') or 1)
            _act = _mm.get('input_length_actual')
            _actnote = (f"; server actually prefilled ~{int(_act)} tok"
                        if isinstance(_act, (int, float)) and _act > 0 else "")
            print(f"  note: MM 'Input' = full campaign text + {_ni} image(s) "
                  f"(nominal ~280 tok/img budget){_actnote} ON TOP,")
            print("        so MM Input >= text Input and MM TTFT >= text at the same campaign")
            print("        (identical text prefill + extra vision-tower work) - directly comparable.")
        _rel_bits = []
        for r, mode in serv_all:
            e = int(r.get('empty_requests') or 0); f = int(r.get('failed_requests') or 0)
            if e or f:
                seg = f"{mode} {r.get('completed_requests')}/{r.get('offered_requests')} ok"
                if e: seg += f", {e} empty"
                if f: seg += f", {f} error"
                _rel_bits.append(seg)
        if _rel_bits:
            print("  reliability: " + "; ".join(_rel_bits)
                  + "  (empty = model returned 0 tokens on a nonsense prompt, NOT an error)")
        print()

    # ── THROUGHPUT - offline peak (max batch, latency ignored), text + MM ──────
    thr_all = ([(x, 'text') for x in throughput_results]
               + [(x, 'MM') for x in throughput_mm_results])
    thr_fail = [(x, 'text') for x in throughput_failures]
    if thr_all or thr_fail:
        print("THROUGHPUT - offline peak: max batching (<= max_num_seqs, KV-bound), LATENCY IGNORED")
        print("-" * 92)
        print(f"{'Model':<20} {'Campaign':<14} {'Mode':<5} {'In/Out':<12} {'out tok/s':<11} {'CV%':<7}")
        print("-" * 92)
        has_mm = False
        for r, mode in thr_all:
            model = r.get('model_short', 'N/A')[:20]
            camp = str(r.get('campaign', '-'))[:14]
            in_out = f"{r.get('input_length','?')}/{r.get('output_length','?')}"
            tp = r.get('output_tokens_per_second_mean', r.get('output_tokens_per_second', 'N/A'))
            cv = r.get('output_tokens_per_second_cv_percent', 'N/A')
            tp_s = f"{tp:.0f}" if isinstance(tp, (int, float)) else str(tp)
            cv_s = f"{cv:.1f}%" if isinstance(cv, (int, float)) else str(cv)
            if mode == 'MM':
                has_mm = True
            print(f"{model:<20} {camp:<14} {mode:<5} {in_out:<12} {tp_s:<11} {cv_s:<7} {'✓'}")
        for r, mode in thr_fail:
            model = r.get('model', 'N/A')[:20]; camp = str(r.get('campaign', '-'))[:14]
            in_out = f"{r.get('input_length','?')}/{r.get('output_length','?')}"
            print(f"{model:<20} {camp:<14} {mode:<5} {in_out:<12} {'FAILED':<11} {'-':<7} ✗")
        if has_mm:
            _mmt = next((r for r, m in thr_all if m == 'MM'), {})
            _nit = int(_mmt.get('mm_images') or 4)
            print(f"  note: MM rows = full campaign text + {_nit} image(s) ON TOP (~280 tok/img")
            print("        budget each). 'In' = text + image tokens, so it exceeds the text row's In.")
            print("        MM does the SAME text prefill as text plus the images, so MM out tok/s")
            print("        <= text - directly comparable (cost of adding images).")
        print()

    # (Multimodal serving + throughput are now merged into the unified SERVING
    # and THROUGHPUT tables above via the Mode column.)

    # Print stress test results - open-loop capacity-under-SLO (goodput) format.
    if stress_results or stress_failures:
        print("STRESS TEST BENCHMARKS (open-loop capacity under SLO)")
        print("=" * 84)

        for r in stress_results:
            model = r.get('model', r.get('model_short', 'N/A'))
            fmt = r.get('format', 'N/A')
            mode = 'Multimodal' if r.get('multimodal', False) else 'Text'
            camp = r.get('campaign')
            ttft_thr = r.get('ttft_threshold_ms', 5000)
            itl_thr = r.get('itl_threshold_ms', r.get('tpot_threshold_ms', 200))
            reps = r.get('reps', 1)

            hdr = f"Model: {model} ({fmt}) - {mode}"
            if camp:
                hdr += f" - campaign: {camp}"
            print(hdr)
            print(f"SLO: P99 TTFT <= {ttft_thr}ms  AND  P99 ITL <= {itl_thr}ms "
                  f"| {reps} reps | {r.get('num_client_procs', '?')} client procs")
            print("-" * 84)

            # Per-rate sweep frontier (from rep 0): arrival -> achieved, tail latencies.
            sweep = r.get('sweep_points', [])
            if sweep:
                print(f"{'Arrival':>9} {'Achieved':>9} {'P99 TTFT':>10} {'P99 ITL':>9} "
                      f"{'%SLO':>5} {'Status':>16}")
                used_server_itl = False
                for pt in sweep:
                    arr = pt.get('arrival_qps', pt.get('request_rate_qps', 0))
                    ach = pt.get('achieved_qps', 0)
                    pttft = pt.get('p99_ttft_ms', 0)
                    # Show the ITL the knee actually GATED on: the server's own P99
                    # ITL (vllm /metrics) when available, else the client-measured
                    # ITL, else per-request TPOT. Showing the client ITL next to a
                    # server-ITL-gated PASS reads as self-contradictory (e.g. 1181ms
                    # ITL with a PASS under a 200ms SLO).
                    if pt.get('p99_itl_server_ms'):
                        pitl = pt.get('p99_itl_server_ms')
                        used_server_itl = True
                    else:
                        pitl = pt.get('p99_itl_ms', 0) or pt.get('p99_tpot_ms', 0)
                    slo = pt.get('slo_attainment', 0) * 100
                    st = "PASS" if pt.get('passed') else "fail"
                    if pt.get('client_bound'):
                        st += " CLIENT-BOUND"
                    print(f"{arr:>9.2f} {ach:>9.2f} {pttft:>9.0f}ms {pitl:>8.0f}ms "
                          f"{slo:>4.0f}% {st:>16}")
                print("-" * 84)
                if used_server_itl:
                    print("P99 ITL = server-side decode latency (vllm /metrics), the metric the knee "
                          "gated on;\nclient-measured ITL is confounded by multiproc-async read "
                          "scheduling under bursty load.")

            # Headline capacity: median knee over reps + mean/CI/CV.
            max_qps = r.get('max_sustainable_qps', 0)
            kstats = r.get('max_sustainable_qps_stats', {}) or {}
            n_users = r.get('little_law_n_users', 0)
            per_rep = r.get('per_rep_knees', [])
            if max_qps and max_qps > 0:
                mn = kstats.get('min', max_qps)
                mx = kstats.get('max', max_qps)
                cv = kstats.get('cv_percent', 0.0)
                # Lead with median + observed [min-max] + the actual per-rep
                # values - with few reps that is more honest than a noisy CV%,
                # which is shown last as a rough dispersion hint only.
                print(f"✅ Sustainable load: {max_qps:.2f} req/s "
                      f"(median of {len(per_rep)} reps {per_rep}, range [{mn:.2f}-{mx:.2f}], CV {cv:.1f}%)")
                print(f"   = max arrival rate holding P99 TTFT <= {ttft_thr}ms AND P99 ITL <= {itl_thr}ms")
                if n_users and n_users > 0:
                    e2e_s = r.get('mean_e2e_ms', 0) / 1000.0
                    print(f"   ≈ {n_users:.0f} concurrent in-flight requests "
                          f"(Little's Law: {max_qps:.1f} req/s × {e2e_s:.1f}s mean latency at this load)")
            elif r.get('preflight_no_stream'):
                print(f"⚠ Stress skipped - the endpoint produced no output tokens for any "
                      f"single-stream probe, so the sweep cannot measure capacity.")
                print(f"   Check that the endpoint streams chat/completions for this model, "
                      f"or use --no-stress-test.")
            elif r.get('below_stress_floor'):
                ttft = float(r.get('preflight_single_stream_ttft_ms', 0) or 0)
                print(f"⚠ Below the stress floor - a single request already takes "
                      f"{ttft:.0f}ms to first token (> {ttft_thr}ms SLO), so sustainable "
                      f"open-loop QPS is ~0 on this hardware.")
                print(f"   Use --no-stress-test, or raise --stress-threshold to measure "
                      f"capacity at this latency.")
            else:
                print(f"⚠ No sustainable rate found - even the lowest probed arrival "
                      f"rate missed the SLO.")
                # If some rates MET the SLO but still weren't counted, the blocker is the
                # steady-sample floor (a slow model can't complete enough per point), not
                # latency - point at --stress-min-samples rather than the SLO knobs.
                pts = r.get('sweep_points', []) or []
                if any(float(p.get('slo_attainment', 0) or 0) >= 0.99 for p in pts):
                    print(f"   NOTE: some rates met the SLO but were TOO-FEW (a slow model "
                          f"can't complete enough steady requests per point). Lower "
                          f"--stress-min-samples to register a lower-confidence knee here.")
                print(f"   Otherwise relax --stress-threshold/--stress-tpot-threshold, or "
                      f"the model is capacity-bound for this campaign.")
            print("=" * 84)
            print()
        
        # Print failures
        for r in stress_failures:
            model = r.get('model', 'N/A')
            fmt = r.get('format', 'N/A')
            mode = 'Multimodal' if r.get('multimodal', False) else 'Text'
            error = r.get('error', 'Unknown error')
            print(f"Model: {model} ({fmt}) - {mode} Mode")
            print("-" * 80)
            print(f"Status: ✗ FAILED")
            print(f"Error:  {error}")
            print("=" * 80)
            print()

    # Print quality benchmarks results table
    if quality_results or quality_failures:
        print("QUALITY BENCHMARKS (AGENTIC)")
        print("-" * 80)
        print(f"{'Model':<20} {'Format':<8} {'Commit':<12} {'Scenarios':<12} {'Pass Rate':<10}")
        print("-" * 80)
        
        for r in quality_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            commit = r.get('gemmaclaw_commit', 'N/A')[:7]
            passed = r.get('passed_scenarios', 0)
            total = r.get('total_scenarios', 0)
            scenarios_str = f"{passed}/{total}"
            pass_rate = r.get('pass_rate', 0.0)
            
            print(f"{model:<20} {fmt:<8} {commit:<12} {scenarios_str:<12} {pass_rate:.1f}%")
        
        # Print failures
        for r in quality_failures:
            model = r.get('model', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            print(f"{model:<20} {fmt:<8} {'-':<12} {'FAILED':<12} -")
        
        print()

    # Print evaluation benchmarks results table
    eval_results = [r for r in successful_results if r.get('benchmark_type') == 'eval']
    eval_failures = [r for r in failed_results if r.get('benchmark_type') == 'eval']
    if eval_results or eval_failures:
        from .runners.evals import BUILTIN_PILLARS
        eval_pillars = BUILTIN_PILLARS

        builtin_suite_keys = set()
        for _, keys in eval_pillars:
            builtin_suite_keys.update(keys)

        builtin_results = [r for r in eval_results if r.get('eval_name', '').lower() in builtin_suite_keys]
        builtin_fails = [r for r in eval_failures if r.get('eval_name', '').lower() in builtin_suite_keys]

        custom_results = [r for r in eval_results if r.get('eval_name', '').lower() not in builtin_suite_keys]
        custom_fails = [r for r in eval_failures if r.get('eval_name', '').lower() not in builtin_suite_keys]

        from .runners.eval_suites import CUSTOM_PILLARS

        table_width = 113
        label_width = 83
        grand_total_q = 0
        grand_correct_q = 0

        # -------------------------------------------------------------------------
        # SECTION 1: BUILT-IN EVALUATION BENCHMARKS (GBENCH STANDARD)
        # -------------------------------------------------------------------------
        if builtin_results or builtin_fails:
            print("BUILT-IN EVALUATION BENCHMARKS (GBENCH STANDARD)")
            print("=" * table_width)
            print(f"{'Model':<18} {'Format':<16} {'Eval Suite':<36} {'Thinking':<10} {'Questions':>9} {'Correct':>9} {'Accuracy':>10}")
            print("=" * table_width)

            builtin_tot = 0
            builtin_corr = 0

            for pillar_title, suite_keys in eval_pillars:
                p_res = [r for r in builtin_results if r.get('eval_name', '').lower() in suite_keys]
                p_fails = [r for r in builtin_fails if r.get('eval_name', '').lower() in suite_keys]
                if not p_res and not p_fails:
                    continue

                print(f"\n▶ {pillar_title}")
                print("-" * table_width)
                p_tot = 0
                p_corr = 0

                for r in p_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0)
                    correct = r.get('correct_answers', 0)
                    acc = r.get('accuracy', 0.0)
                    p_tot += total_q
                    p_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {total_q:>9} {correct:>9} {acc:>9.2f}%")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  └ WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                    cat_acc = r.get("category_accuracy", {})
                    if 1 < len(cat_acc) <= 25:
                        for cat, cstats in sorted(cat_acc.items()):
                            cat_label = f"  └ {cat}"[:label_width]
                            ctot = cstats.get("total", 0)
                            ccorr = cstats.get("correct", 0)
                            cacc = cstats.get("accuracy", 0.0)
                            print(f"{cat_label:<{label_width}} {ctot:>9} {ccorr:>9} {cacc:>9.2f}%")

                for r in p_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {'FAILED':>9} {'-':>9} {'-':>10}")

                if p_tot > 0:
                    p_acc = (p_corr / p_tot * 100.0)
                    print(f"{'  └ Pillar Subtotal':<{label_width}} {p_tot:>9} {p_corr:>9} {p_acc:>9.2f}%")
                    builtin_tot += p_tot
                    builtin_corr += p_corr

            if builtin_tot > 0:
                b_acc = (builtin_corr / builtin_tot * 100.0)
                print("-" * table_width)
                print(f"{'BUILT-IN EVALS SUBTOTAL':<{label_width}} {builtin_tot:>9} {builtin_corr:>9} {b_acc:>9.2f}%")
                print("  ⚠ Subtotal % is a micro-average of raw correct/total across suites with")
                print("    DIFFERENT metrics (multiple-choice, code pass@k, ANLS, judge-mean, sub-step)")
                print("    - it is NOT a comparable score. Use the per-suite Accuracy column. Suites")
                print("    whose headline is not correct/total (HealthBench mean-rubric, aider pass@2,")
                print("    scicode sub-step) contribute only their raw counts here, not that headline.")
                print("=" * table_width)
                grand_total_q += builtin_tot
                grand_correct_q += builtin_corr
            print()

        # -------------------------------------------------------------------------
        # SECTION 2: CUSTOM / PLUGIN EVALUATION BENCHMARKS
        # -------------------------------------------------------------------------
        if custom_results or custom_fails:
            print("CUSTOM / PLUGIN EVALUATION BENCHMARKS")
            print("=" * table_width)
            print(f"{'Model':<18} {'Format':<16} {'Eval Suite':<36} {'Thinking':<10} {'Questions':>9} {'Correct':>9} {'Accuracy':>10}")
            print("=" * table_width)

            custom_pillar_map: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            truly_uncat_res = []
            truly_uncat_fails = []

            for r in custom_results:
                ename = r.get('eval_name', '').lower()
                if ename in CUSTOM_PILLARS:
                    p_name = CUSTOM_PILLARS[ename]
                    custom_pillar_map.setdefault(p_name, ([], []))[0].append(r)
                else:
                    truly_uncat_res.append(r)

            for r in custom_fails:
                ename = r.get('eval_name', '').lower()
                if ename in CUSTOM_PILLARS:
                    p_name = CUSTOM_PILLARS[ename]
                    custom_pillar_map.setdefault(p_name, ([], []))[1].append(r)
                else:
                    truly_uncat_fails.append(r)

            custom_tot = 0
            custom_corr = 0

            for cp_title, (cp_res, cp_fails) in sorted(custom_pillar_map.items()):
                clean_cp_title = re.sub(r"^\d+\.\s*", "", cp_title).strip()
                print(f"\n▶ [CUSTOM] {clean_cp_title.upper()}")
                print("-" * table_width)
                cp_tot = 0
                cp_corr = 0
                for r in cp_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0)
                    correct = r.get('correct_answers', 0)
                    acc = r.get('accuracy', 0.0)
                    cp_tot += total_q
                    cp_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {total_q:>9} {correct:>9} {acc:>9.2f}%")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  └ WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                    cat_acc = r.get("category_accuracy", {})
                    if 1 < len(cat_acc) <= 25:
                        for cat, cstats in sorted(cat_acc.items()):
                            cat_label = f"  └ {cat}"[:label_width]
                            ctot = cstats.get("total", 0)
                            ccorr = cstats.get("correct", 0)
                            cacc = cstats.get("accuracy", 0.0)
                            print(f"{cat_label:<{label_width}} {ctot:>9} {ccorr:>9} {cacc:>9.2f}%")

                for r in cp_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {'FAILED':>9} {'-':>9} {'-':>10}")

                if cp_tot > 0:
                    cp_acc = (cp_corr / cp_tot * 100.0)
                    print(f"{'  └ Pillar Subtotal':<{label_width}} {cp_tot:>9} {cp_corr:>9} {cp_acc:>9.2f}%")
                    custom_tot += cp_tot
                    custom_corr += cp_corr

            if truly_uncat_res or truly_uncat_fails:
                print(f"\n▶ [CUSTOM] OTHER EVALUATIONS")
                print("-" * table_width)
                uncat_tot = 0
                uncat_corr = 0
                for r in truly_uncat_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0)
                    correct = r.get('correct_answers', 0)
                    acc = r.get('accuracy', 0.0)
                    uncat_tot += total_q
                    uncat_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {total_q:>9} {correct:>9} {acc:>9.2f}%")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  └ WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                for r in truly_uncat_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {'FAILED':>9} {'-':>9} {'-':>10}")

                if uncat_tot > 0:
                    u_acc = (uncat_corr / uncat_tot * 100.0)
                    print(f"{'  └ Pillar Subtotal':<{label_width}} {uncat_tot:>9} {uncat_corr:>9} {u_acc:>9.2f}%")
                    custom_tot += uncat_tot
                    custom_corr += uncat_corr

            if custom_tot > 0:
                c_acc = (custom_corr / custom_tot * 100.0)
                print("-" * table_width)
                print(f"{'CUSTOM PLUGINS SUBTOTAL':<{label_width}} {custom_tot:>9} {custom_corr:>9} {c_acc:>9.2f}%")
                print("=" * table_width)
                grand_total_q += custom_tot
                grand_correct_q += custom_corr
            print()

        # Print overall total if both built-in and custom were executed
        if (builtin_results or builtin_fails) and (custom_results or custom_fails) and grand_total_q > 0:
            grand_acc = (grand_correct_q / grand_total_q * 100.0)
            print("=" * table_width)
            print(f"{'OVERALL EVALS TOTAL':<{label_width}} {grand_total_q:>9} {grand_correct_q:>9} {grand_acc:>9.2f}%")
            print("  ⚠ micro-average of correct/total across EVERY suite (built-in + custom) and")
            print("    every metric - the least comparable number in this report. It is a coverage")
            print("    tally, NOT a benchmark score; compare models per-suite, never on this %.")
            print("=" * table_width)
            print()

        print()

        # Generate CSV summary report
        target_results_dir = config.log_manager.results_dir if getattr(config, "log_manager", None) else Path(getattr(config, "results_dir", "results"))
        csv_file = _save_eval_summary_csv(
            target_results_dir,
            eval_results,
            eval_failures,
            eval_pillars,
            CUSTOM_PILLARS,
        )

    # Print Golden Set results table
    if golden_results:
        print("GOLDEN SET (DETERMINISTIC SMOKE TEST)")
        print("-" * 80)
        print(f"{'Model':<20} {'Requested':<24} {'Cases':<10} {'Verdict':<10}")
        print("-" * 80)

        for r in golden_results:
            model = r.get('model_short', r.get('model', 'N/A'))[:20]
            requested = (r.get('requested_model') or '(endpoint default)')[:24]
            passed = r.get('passed_cases', 0)
            # Denominator is the cases APPLICABLE to this model's modalities, so
            # a case the model architecturally cannot run (e.g. audio on a
            # no-audio-tower model) does not count against it.
            applicable = r.get('applicable_tasks', r.get('total_tasks', 0))
            na = r.get('not_applicable_cases', 0)
            cases = f"{passed}/{applicable}"
            verdict = GOLDEN_VERDICT.get(r.get('status'), 'UNKNOWN')
            print(f"{model:<20} {requested:<24} {cases:<10} {verdict:<10}")

            rows = golden_category_breakdown(r.get('task_results', []))
            if len(rows) > 1:
                for row in rows:
                    cat_label = f"  └ {row['category']}"[:45]
                    cat_cases = f"{row['passed']}/{row['total']}"
                    cat_verdict = GOLDEN_VERDICT.get(row['status'], 'UNKNOWN')
                    print(f"{cat_label:<45} {cat_cases:<10} "
                          f"{cat_verdict:<10}".rstrip())

            if na:
                na_ids = [t.get('task_id') for t in r.get('task_results', [])
                          if t.get('status') == 'not_applicable']
                print(f"  └ {na} case(s) N/A (model lacks the required "
                      f"modality): {', '.join(na_ids)}")

        print()

        for r in golden_results:
            for err in r.get('harness_errors', []):
                print(f"  ERROR  {err}")
            for t in r.get('task_results', []):
                if t.get('status') == 'failed':
                    print(f"  FAIL   {t.get('task_id')}: {t.get('details')}")
        print()

    # Print file locations using LogManager
    print("="*80)
    print("RESULTS & LOGS")
    print("="*80)
    
    lm = getattr(config, "log_manager", None)
    if lm:
        # Save top-level aggregated summary.json for programmatic consumption
        summary_payload = {
            "timestamp": lm.timestamp,
            "results_dir": str(lm.results_dir.absolute()),
            "total_configurations": len(results),
            "successful_configurations": len(successful_results),
            "failed_configurations": len(failed_results),
            "models": successful_results,
            "failures": failed_results,
        }
        lm.save_summary(summary_payload)

        summary = lm.get_summary()
        print(f"\nResults Directory: {summary['results_dir']}")
        
        # CSV summary file
        csv_path = lm.results_dir / "eval_summary.csv"
        if csv_path.exists():
            print(f"\nSummary Report (CSV):")
            print(f"  - {csv_path.name} (ready for Google Sheets / Excel)")

        # List result files
        result_files = summary['result_files']
        if result_files:
            print(f"\nResult Files ({len(result_files)} total):")
            for f in result_files[:10]:
                print(f"  - {f}")
            if len(result_files) > 10:
                print(f"  ... and {len(result_files) - 10} more files")
        
        # List log files
        log_files = summary['log_files']
        if log_files:
            print(f"\nLog Files ({len(log_files)} total):")
            for f in log_files[:5]:
                print(f"  - {f}")
            if len(log_files) > 5:
                print(f"  ... and {len(log_files) - 5} more files")
    else:
        results_dir = getattr(config, "results_dir", "results")
        print(f"\nResults Directory: {results_dir}")

    # Statistical validation summary
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION")
    print("="*80)
    
    total_configs = len(results)
    num_failed = len(failed_results)
    num_passed = len(successful_results)
    
    print(f"\nTotal Configurations: {total_configs}")
    print(f"Successful: {num_passed}")
    if num_failed > 0:
        print(f"Failed: {num_failed}")
    
    print("\n" + "="*80)
    if num_failed > 0:
        print(f"Benchmark suite complete! {num_passed} passed, {num_failed} failed.")
    else:
        print(f"Benchmark suite complete! All {total_configs} configurations passed.")

    # A configuration "passed" here only means its runner returned. A
    # Golden Set run that returned a FAIL or ERROR verdict is one of
    # those, so without this line the footer reads "All 1 configurations
    # passed" directly above a non-zero exit code.
    golden_bad = [r for r in golden_results
                  if r.get('status') in ('failed', 'error')]
    if golden_bad:
        verdicts = ", ".join(
            f"{r.get('model_short', r.get('model', 'N/A'))} "
            f"{GOLDEN_VERDICT.get(r.get('status'), 'UNKNOWN')}"
            for r in golden_bad
        )
        print(f"Golden Set did not pass: {verdicts}. See the table above.")
    print("="*80 + "\n")


def stage_models_to_gcs(models: list, gcs_destination: str) -> None:
    """Stage model weights from local/HF to GCS for remote deployment."""
    import subprocess
    import os
    from gbench.core.models import ModelFormat

    logger.info(f"Starting GCS staging to: {gcs_destination}")

    for model in models:
        logger.info(f"Processing model: {model.name}")
        try:
            local_path = model.get_model_path(ModelFormat.HF)
        except Exception as e:
            logger.error(f"Failed to resolve local path for {model.name}: {e}")
            continue

        dest_url = gcs_destination.rstrip("/") + "/" + model.short_name + "/"
        logger.info(f"Uploading {local_path} to {dest_url} ...")

        src_items = [os.path.join(local_path, item) for item in os.listdir(local_path)]
        if not src_items:
            logger.warning(f"No files found in {local_path} to upload.")
            continue

        cmd = ["gcloud", "storage", "cp", "-r"] + src_items + [dest_url]
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully uploaded {model.name} to GCS.")
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to upload {model.name} to GCS: {e}")


if __name__ == "__main__":
    sys.exit(main())

