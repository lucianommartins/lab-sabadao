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

"""Throughput benchmark runner for vLLM bench throughput command."""

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..core.config import BenchmarkConfig, get_max_model_len
from ..core.models import ModelConfig, ModelFormat
from ..analysis.statistics import (
    aggregate_benchmark_results,
    format_statistics_summary,
    validate_repeatability,
)

logger = logging.getLogger(__name__)


class ThroughputBenchmarkRunner:
    """Runner for vLLM throughput benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the throughput benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._first_run = True  # Track if this is first run (no wait needed)
        self._mm_tok = None  # Lazily loaded; sizes MM padded-text prompt

    def _run_fingerprint(self, model, num_prompts, max_model_len, multimodal: bool) -> dict:
        """The exact knobs that determine a throughput number.

        A peak tokens/s figure is only comparable/reproducible if the config
        that produced it is recorded alongside it. Stamped onto every result so
        each datapoint is self-describing (and a mismatch between two runs is a
        distinct experiment, not noise). ignore_eos is True because `vllm bench
        throughput` forces it in SamplingParams, so output_length is honored
        exactly; range_ratio 0.0 means every request is exactly (in, out).
        """
        return {
            "engine": "vllm-bench-throughput-offline",
            "shape_deterministic": True,     # range_range_ratio 0.0
            "random_range_ratio": 0.0,
            "ignore_eos": True,              # forced by vllm bench throughput
            "num_prompts": int(num_prompts),
            "num_iterations": int(self.config.num_iterations),
            "warmup_iterations": 0,          # dropped: fresh process per run
            "max_model_len": int(max_model_len) if max_model_len else None,
            "max_num_seqs": self.config.max_num_seqs,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "tensor_parallel_size": self.config.tensor_parallel_size or 1,
            # No --dtype is forced (see _build_command), so vLLM resolves it from the model config.
            # Record "auto" rather than asserting bf16, which would mislabel a non-bf16 checkpoint
            # (the project mandate is bf16 and never quantized, but the fingerprint must be truthful).
            "dtype": "auto",
            "seed": 83,
            "multimodal": bool(multimodal),
        }

    def _stamp_run_metadata(self, result: dict, fingerprint: dict) -> None:
        """Attach the fingerprint + a derived duration/validity flag in place.

        measured_duration_s is reconstructed from total tokens and the reported
        tokens/s (vllm bench prints no elapsed time we parse). short_run flags a
        peak measurement so brief it may be ramp-dominated rather than steady -
        surfaced, never silently trusted, and never "fixed" by shrinking the
        dataset (num_prompts is fixed by mandate).
        """
        result["run_fingerprint"] = fingerprint
        # Works for BOTH a raw single-iteration result (output_tokens_per_second,
        # total_output_tokens) and a 3-iter AGGREGATE (only *_mean present, and
        # aggregate_benchmark_results drops total_output_tokens). Fall back to the
        # _mean rate and reconstruct total tokens from num_prompts * output_length
        # (deterministic shape + ignore_eos => exact) so duration/short_run are
        # populated on the aggregate too, not just the raw result.
        tps = result.get("output_tokens_per_second")
        if not tps:
            tps = result.get("output_tokens_per_second_mean")
        tot_tok = result.get("total_output_tokens")
        if not tot_tok:
            npr = fingerprint.get("num_prompts")
            ol = result.get("output_length")
            try:
                if npr and ol is not None and int(ol) > 0:
                    tot_tok = int(npr) * int(ol)
            except (TypeError, ValueError):
                tot_tok = None
        try:
            if tot_tok and tps and float(tps) > 0:
                dur = float(tot_tok) / float(tps)
                result["measured_duration_s"] = round(dur, 1)
                result["short_run"] = dur < 600.0
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    def _offline_timeout_s(self, num_prompts: int, output_length) -> int:
        """Generous wall-clock backstop for one offline `vllm bench` run.

        A stalled offline engine otherwise hangs the whole unattended sweep
        forever (text path had NO timeout; MM's flat 1800s falsely kills
        long-decode). This is a RUNAWAY-HANG guard, not a tight SLA: scale with
        the total tokens to generate so a legitimate long-decode run (1000 x
        8192 out) is never killed, with a 4h floor. Err high.
        """
        total_out = int(num_prompts) * max(int(output_length or 128), 1)
        return int(max(14400, total_out // 300))

    def _run_offline_subprocess(self, cmd: list, timeout: int):
        """Run an offline `vllm bench` subprocess, orphan-safe on timeout.

        subprocess.run(timeout=...) SIGKILLs only the direct child, orphaning the
        vLLM EngineCore grandchild (which holds ~all GPU VRAM) so the next
        Phase-2 run OOMs. Launch in a new session and group-kill on timeout so
        nothing survives to poison the next campaign. Returns (rc, stdout,
        stderr); raises RuntimeError on timeout.
        """
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise RuntimeError(
                f"Offline throughput subprocess exceeded {timeout}s and was "
                f"group-killed (stalled engine?)."
            )
        return proc.returncode, stdout, stderr

    def _build_pad_text(self, n_tokens: int, tokenizer) -> str:
        """Text of ~n_tokens tokens to pad an MM throughput prompt to in_len."""
        base = "Analyze the attached image and the following context in detail. "
        if not tokenizer or n_tokens <= 0:
            return base * max(1, n_tokens // 12)
        filler = base * (max(1, n_tokens // 8) + 8)
        ids = tokenizer(filler).input_ids[:n_tokens]
        return tokenizer.decode(ids)

    def _wait_for_gpu_memory(self, min_free_gb: Optional[float] = None, max_wait_seconds: int = 120):
        """Wait for GPU memory to be reclaimed by CUDA driver.

        Uses nvidia-smi to actively poll until sufficient memory is free.
        Respects CUDA_VISIBLE_DEVICES to only check the assigned GPUs.

        Args:
            min_free_gb: Minimum free memory required in GB. None => 80% of the
                GPU's total VRAM (scales to any card; the old fixed 70GB made this
                wait the full timeout on every <70GB GPU).
            max_wait_seconds: Maximum time to wait before giving up
        """
        if self._first_run:
            self._first_run = False
            return  # No wait needed for first run

        import subprocess

        if min_free_gb is None:
            from ..core.config import _gpu_total_vram_gb
            _total = _gpu_total_vram_gb()
            min_free_gb = 0.8 * _total if _total > 0 else 8.0

        logger.info(f"Waiting for GPU memory to be reclaimed (need {min_free_gb:.0f}GB free)...")
        
        # Build nvidia-smi command targeting only our assigned GPUs
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        smi_cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        if gpu_ids:
            smi_cmd.extend(["--id=" + gpu_ids])
        
        for elapsed in range(0, max_wait_seconds, 5):
            try:
                result = subprocess.run(
                    smi_cmd,
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Check ALL assigned GPUs have enough free memory
                    free_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                    if free_values:
                        min_free_mb = min(free_values)
                        free_gb = min_free_mb / 1024
                        if free_gb >= min_free_gb:
                            logger.info(f"GPU memory ready: {free_gb:.1f}GB free (min across {len(free_values)} GPU(s))")
                            return
                        logger.info(f"  GPU memory: {free_gb:.1f}GB free (min), waiting... ({elapsed}s)")
            except Exception:
                pass  # nvidia-smi failed, fall back to fixed wait
            time.sleep(5)
        
        logger.warning(f"GPU memory wait timed out after {max_wait_seconds}s, proceeding anyway")

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: str = "random",
    ) -> dict:
        """Run throughput benchmark for a specific configuration.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            input_length: Input prompt length in tokens
            output_length: Output generation length in tokens
            batch_size: Batch size
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset name (random or random-mm)

        Returns:
            Dictionary with benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        if getattr(self.config, "remote_endpoint", None):
            logger.info("Skipping throughput benchmark for remote endpoint (throughput benchmarks require local offline vLLM GPU engine).")
            return {
                "skipped": True, 
                "reason": "remote_endpoint", 
                "model": model.short_name, 
                "format": format.value,
                "batch_size": batch_size
            }

        num_prompts = num_prompts or self.config.num_prompts_throughput
        lm = self.config.log_manager

        # Generate output filename
        output_file = lm.get_throughput_result_path(
            model.short_name, format.value, input_length, output_length, batch_size
        )

        # Build command
        cmd = self._build_command(
            model,
            format,
            input_length,
            output_length,
            batch_size,
            num_prompts,
            output_file,
            dataset,
        )

        # Log clear benchmark configuration banner
        # Note: batch_size is not shown because vllm bench throughput doesn't use it
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Throughput\n"
            f"  Mode:           Text\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Input length:   {input_length}\n"
            f"  Output length:  {output_length}\n"
            f"  Num prompts:    {num_prompts}\n"
            f"{'='*60}"
        )

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return {"dry_run": True, "command": cmd}

        from gbench.utils import require_vllm_engine
        require_vllm_engine("Offline throughput benchmarks")

        try:
            rc, out, err = self._run_offline_subprocess(
                cmd, timeout=self._offline_timeout_s(num_prompts, output_length)
            )
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd, out, err)
            result = subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)

            # Parse plain text output from vllm bench throughput
            # Output format: "Throughput: X requests/s, Y total tokens/s, Z output tokens/s"
            benchmark_result = self._parse_throughput_output(result.stdout)

            # Save results to file
            lm.save_result(output_file, benchmark_result)

            # Log key metrics. Values may be the "N/A" string on a partial/failed
            # parse, so format defensively - applying :.2f to a str raised ValueError
            # and crashed the whole throughput run at the log line.
            def _fmt(v):
                return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
            req_tput = benchmark_result.get("request_throughput", "N/A")
            total_tps = benchmark_result.get("total_tokens_per_second", "N/A")
            out_tps = benchmark_result.get("output_tokens_per_second", "N/A")
            logger.info(
                f"Results: {_fmt(req_tput)} req/s, "
                f"{_fmt(total_tps)} total tok/s, {_fmt(out_tps)} output tok/s"
            )

            # Log output if enabled
            if self.config.enable_logging:
                log_file = lm.get_throughput_log_path(
                    model.short_name, format.value,
                    input_length, output_length, batch_size
                )
                lm.save_log(log_file, result.stdout, result.stderr)

            return benchmark_result

        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Throughput benchmark failed: {e}")


    def run_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: str = "random",
    ) -> dict:
        """Run benchmark with multiple iterations and statistical analysis.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            input_length: Input prompt length in tokens
            output_length: Output generation length in tokens
            batch_size: Batch size
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset name (random or random-mm)

        Returns:
            Dictionary with aggregated results and statistics
        """
        if self.config.num_iterations == 1:
            result = self.run(
                model,
                format,
                input_length,
                output_length,
                batch_size,
                num_prompts,
                dataset,
            )
            # Add configuration metadata for single-iteration case
            result["model"] = model.short_name
            result["format"] = format.value
            result["input_length"] = input_length
            result["output_length"] = output_length
            result["batch_size"] = batch_size
            return result

        logger.info(
            f"Running {self.config.num_iterations} throughput iterations "
            f"(no warmup: each offline run is a fresh vLLM engine process - its "
            f"in-process graph capture/profiling already runs at startup, so a "
            f"separate warmup iteration carries nothing forward and is pure cost)"
        )

        # Actual benchmark iterations
        results = []
        for i in range(self.config.num_iterations):
            logger.info(
                f"Iteration {i+1}/{self.config.num_iterations}"
            )
            try:
                result = self.run(
                    model,
                    format,
                    input_length,
                    output_length,
                    batch_size,
                    num_prompts,
                    dataset,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Iteration {i+1} failed: {e}")
                continue

        if not results:
            raise RuntimeError("All iterations failed")

        # Aggregate statistics using field names from _parse_throughput_output
        metrics = [
            "request_throughput",
            "total_tokens_per_second",
            "output_tokens_per_second",
        ]

        aggregated = aggregate_benchmark_results(results, metrics)

        # Add configuration metadata
        aggregated["model"] = model.short_name
        aggregated["format"] = format.value
        aggregated["input_length"] = input_length
        aggregated["output_length"] = output_length
        aggregated["batch_size"] = batch_size

        # Validate repeatability
        for metric in ["output_tokens_per_second"]:
            is_valid, msg = validate_repeatability(
                aggregated,
                metric,
                self.config.min_acceptable_cv_percent,
            )
            logger.info(f"{metric}: {msg}")
            aggregated[f"{metric}_repeatability_valid"] = is_valid

            summary = format_statistics_summary(aggregated, metric)
            logger.info(summary)

        return aggregated

    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> list[dict]:
        """Run all throughput benchmark configurations for a model.

        Args:
            model: Model configuration
            format: Model format
            multimodal: If True, run multimodal throughput using offline inference

        Returns:
            List of aggregated result dictionaries
        """
        if getattr(self.config, "remote_endpoint", None):
            mode_str = "multimodal " if multimodal else ""
            logger.info(f"Skipping {mode_str}throughput benchmark for remote endpoint (throughput benchmarks require local offline vLLM GPU engine).")
            return [{
                "skipped": True,
                "reason": "remote_endpoint",
                "model": model.short_name,
                "format": format.value,
                "batch_size": "N/A"
            }]

        results = []
        configs = self.config.get_throughput_configs()

        # Deduplicate configs because batch_size is NOT used by vllm bench throughput.
        # The batch_size parameter was removed from the CLI in newer vLLM versions.
        # Without deduplication, we'd run the exact same benchmark multiple times.
        if multimodal:
            # Multimodal: dedupe by (output_length, num_prompts)
            seen_configs = set()
            deduplicated_configs = []
            for cfg in configs:
                key = (cfg["output_length"], cfg["num_prompts"])
                if key not in seen_configs:
                    seen_configs.add(key)
                    deduplicated_configs.append(cfg)
            configs = deduplicated_configs
        else:
            # Text: dedupe by (input_length, output_length, num_prompts)
            seen_configs = set()
            deduplicated_configs = []
            for cfg in configs:
                key = (cfg["input_length"], cfg["output_length"], cfg["num_prompts"])
                if key not in seen_configs:
                    seen_configs.add(key)
                    deduplicated_configs.append(cfg)
            configs = deduplicated_configs

        for cfg in configs:
            try:
                # Wait for GPU memory from previous run to be reclaimed
                self._wait_for_gpu_memory()
                
                if multimodal:
                    # Multi-iteration MM throughput (same as text path; no warmup:
                    # each offline run is a fresh vLLM engine process, so a warmup
                    # iteration carries nothing forward and is pure cost).
                    logger.info(
                        f"Running {self.config.num_iterations} MM iterations "
                        f"(no warmup: fresh engine process per run)"
                    )
                    # Actual iterations
                    mm_iter_results = []
                    for mi in range(self.config.num_iterations):
                        logger.info(f"MM iteration {mi+1}/{self.config.num_iterations}")
                        try:
                            mm_r = self._run_multimodal_throughput(
                                model, format,
                                num_prompts=cfg["num_prompts"],
                                output_length=cfg["output_length"],
                                input_length=cfg["input_length"],
                            )
                            mm_iter_results.append(mm_r)
                        except Exception as ie:
                            logger.error(f"MM iteration {mi+1} failed: {ie}")
                            continue
                    if not mm_iter_results:
                        raise RuntimeError("All MM iterations failed")
                    # Aggregate with same metrics as text throughput
                    mm_metrics = [
                        "request_throughput",
                        "total_tokens_per_second",
                        "output_tokens_per_second",
                    ]
                    result = aggregate_benchmark_results(mm_iter_results, mm_metrics)
                    # Validate repeatability
                    for mm_m in ["output_tokens_per_second"]:
                        is_valid, msg = validate_repeatability(
                            result, mm_m, self.config.min_acceptable_cv_percent,
                        )
                        logger.info(f"{mm_m}: {msg}")
                        result[f"{mm_m}_repeatability_valid"] = is_valid
                    # Add config values for display. Report the ACTUAL MM prefill
                    # length (clamped text + image placeholder tokens) from the last
                    # iteration, not a bare "img" - small campaigns collapse text to
                    # the floor so the images dominate the real prefill.
                    _last_mm = mm_iter_results[-1]
                    result["batch_size"] = "N/A"  # Not applicable for offline inference
                    result["input_length"] = _last_mm.get("input_length", "img")
                    result["input_length_nominal"] = _last_mm.get(
                        "input_length_nominal", cfg["input_length"])
                    result["mm_images"] = _last_mm.get("mm_images", 4)
                    result["output_length"] = cfg["output_length"]
                else:
                    # Use standard CLI-based throughput
                    result = self.run_with_iterations(
                        model,
                        format,
                        input_length=cfg["input_length"],
                        output_length=cfg["output_length"],
                        batch_size=cfg["batch_size"],  # Kept for display, but not used by vLLM
                        num_prompts=cfg["num_prompts"],
                        dataset="random",
                    )
                    # Mark batch_size as N/A in results since it's not actually used
                    result["batch_size"] = "N/A"
                # Stamp the reproducibility fingerprint + derived duration so the
                # peak-throughput number is self-describing and comparable.
                _mml = get_max_model_len(self.config.max_model_len)
                self._stamp_run_metadata(
                    result,
                    self._run_fingerprint(model, cfg["num_prompts"], _mml, multimodal),
                )
                result["multimodal"] = multimodal
                # Persist the rich AGGREGATE (fingerprint + CV + duration) to the
                # per-config artifact, matching what serving does. Otherwise only
                # the raw LAST-iteration result that run() wrote is on disk, and
                # the aggregate lives only in summary.json.
                try:
                    lm = self.config.log_manager
                    _in = "img" if multimodal else cfg["input_length"]
                    out_file = lm.get_throughput_result_path(
                        model.short_name, format.value, _in,
                        cfg["output_length"], cfg["batch_size"])
                    lm.save_result(out_file, result)
                except Exception as _se:
                    logger.warning(f"Could not save throughput aggregate: {_se}")
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed benchmark for {model.short_name}: {e}"
                )
                # Track failure in results
                results.append({
                    "failed": True,
                    "error": str(e),
                    "model": model.short_name,
                    "format": format.value,
                    "input_length": cfg["input_length"],
                    "output_length": cfg["output_length"],
                    "batch_size": cfg["batch_size"],
                    "multimodal": multimodal,
                })

        return results

    def _run_multimodal_throughput(
        self,
        model: ModelConfig,
        format: ModelFormat,
        num_prompts: int,
        output_length: int,
        input_length: int = 0,
    ) -> dict:
        """Run multimodal throughput benchmark using the random-mm dataset.

        Uses `vllm bench throughput --dataset-name random-mm`, which generates
        synthetic images + text in-memory at the campaign's input/output length
        with N images per request - no dataset file, no local media dir, and no
        sharegpt length filter (which discarded every one of our campaign
        shapes).

        Args:
            model: Model configuration
            format: Model format
            num_prompts: Number of prompts to process
            output_length: Output tokens per prompt
            input_length: Target text input tokens (images added on top)

        Returns:
            Dictionary with throughput metrics
        """
        # Log clear benchmark configuration banner
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Throughput\n"
            f"  Mode:           Multimodal\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Output length:  {output_length}\n"
            f"  Num prompts:    {num_prompts}\n"
            f"{'='*60}"
        )
        
        # Check for dry run
        if self.config.dry_run:
            logger.info("[DRY RUN] Would execute multimodal throughput benchmark")
            return {"dry_run": True}
            
        if getattr(self.config, "remote_endpoint", None):
            logger.info("Skipping multimodal throughput benchmark for remote endpoint (throughput benchmarks require local offline vLLM GPU engine).")
            return {
                "skipped": True, 
                "reason": "remote_endpoint", 
                "model": model.short_name, 
                "format": format.value,
                "batch_size": "N/A"
            }
        
        # random-mm generates synthetic images + text IN-MEMORY at arbitrary
        # shapes, so it honors the campaign input/output length AND multiple
        # images without the offline sharegpt loader's hardcoded 1024-prompt /
        # 2048-total filter (which discards every one of our campaign shapes and
        # made the old sharegpt stub fail for essentially every campaign).
        import os
        # Images per request: honor the config override (--personal forces 1) so
        # the workload AND the reported note agree; default to 4 otherwise.
        n_img = int(getattr(self.config, "mm_images_per_request", None) or 4)
        # Per-image token count is model-specific (gemma-4 = 280); read it from the
        # model metadata, falling back to 280 conservatively (never 0).
        per_img = getattr(model, "vision_tokens_per_image", None) or 280
        img_tokens = n_img * per_img
        max_model_len = get_max_model_len(self.config.max_model_len)
        # FULL campaign text with images ON TOP (not replacing text): MM does the
        # same text prefill as the text row PLUS the 4 images, so MM is a strict
        # superset (directly comparable: MM out-tok/s <= text, MM total input >
        # text input). Clamp only if text + images + output would overflow context.
        text_len = min(int(input_length),
                       max_model_len - img_tokens - int(output_length or 128) - 128)
        text_len = max(16, text_len)

        try:
            # random-mm uses the default vllm backend and builds the multimodal
            # requests itself; no dataset file or local media dir is needed.
            cmd = [
                "vllm", "bench", "throughput",
                "--model", model.get_model_path(format),
                # REQUIRED for multimodal: only the vllm-chat backend attaches the
                # images. With the default backend, get_samples is called with
                # multimodal_backends=() and the images are silently dropped.
                # vllm-chat sets enable_multimodal_chat=True, so random-mm emits a
                # chat-format prompt (list: [{text},{image_url}xN]) that
                # run_vllm_chat feeds to llm.chat, and the vision tower encodes them.
                # VERIFIED by controlled A/B (24 prompts, 64 text tok, 256x256 imgs):
                # 0 img = 57.1 req/s, 2 img -> 4 img falls monotonically to ~39.5
                # req/s (~31% slower = 4x280 soft-tokens/req of real vision work).
                # NOTE: "Total num prompt tokens" counts TEXT ONLY, so it is
                # identical with/without images and is NOT a valid image-presence
                # check; use throughput (req/s) instead.
                "--backend", "vllm-chat",
                "--dataset-name", "random-mm",
                "--num-prompts", str(num_prompts),
                # random-mm reads --random-input-len/--random-output-len (NOT the
                # generic --input-len/--output-len, which it silently ignores).
                "--random-input-len", str(text_len),
                "--random-output-len", str(int(output_length or 128)),
                "--random-range-ratio", "0.0",  # deterministic in/out length
                "--random-mm-base-items-per-request", str(n_img),
                "--random-mm-num-mm-items-range-ratio", "0.0",  # fixed image count
                "--random-mm-limit-mm-per-prompt", json.dumps({"image": n_img}),
                "--random-mm-bucket-config", str({(256, 256, 1): 1.0}),
                "--limit-mm-per-prompt", json.dumps({"image": n_img}),
                "--mm-processor-cache-gb", "0",
                # Parity with serving/stress + the text throughput path: no APC, so
                # repeated images/prefixes are never served from cache (real prefill).
                "--no-enable-prefix-caching",
                "--output-json", "/tmp/mm_throughput_result.json",
            ]
            
            # Add GPU memory utilization
            mm_gpu_mem = self.config.gpu_memory_utilization
            if mm_gpu_mem:
                cmd.extend([
                    "--gpu-memory-utilization",
                    str(mm_gpu_mem),
                ])
            
            # GGUF models need explicit tokenizer
            if format == ModelFormat.GGUF:
                cmd.extend(["--tokenizer", model.hf_model_id])
            
            # Multi-GPU: tensor parallel (critical for models requiring TP>1)
            num_gpus = self.config.tensor_parallel_size or 1
            if num_gpus > 1:
                cmd.extend(["--tensor-parallel-size", str(num_gpus)])
            
            # Uniform context length for fair comparison (overridable via
            # --max-model-len for long-context campaigns)
            max_model_len = get_max_model_len(self.config.max_model_len)
            if max_model_len:
                cmd.extend(["--max-model-len", str(max_model_len)])

            # Limit max concurrent sequences for uniform memory usage
            if self.config.max_num_seqs:
                cmd.extend(["--max-num-seqs", str(self.config.max_num_seqs)])
            
            # Performance optimization flags (parity with text throughput path)
            if self.config.enable_chunked_prefill:
                cmd.append("--enable-chunked-prefill")
            
            if self.config.max_num_batched_tokens:
                cmd.extend([
                    "--max-num-batched-tokens",
                    str(self.config.max_num_batched_tokens),
                ])
            
            # (output length is set via --random-output-len above; random-mm
            # ignores the generic --output-len.)

            # Reproducibility seed (matches text throughput path)
            cmd.extend(["--seed", "83"])

            # Preflight guard so MM throughput can never silently regress to
            # text-only (the original bug). (1) Command integrity: the image-
            # bearing flags must be present. (2) Dataset integrity: the random-mm
            # generator must actually emit n_img images with these params. Engine-
            # level expansion of those image_url parts is separately proven
            # (run_vllm_chat -> prompt_token_ids grow ~256/img); the runner always
            # uses --backend vllm-chat, asserted here.
            if "vllm-chat" not in cmd or "--limit-mm-per-prompt" not in cmd:
                raise RuntimeError(
                    "MM throughput preflight FAILED: command is missing the "
                    "multimodal flags (--backend vllm-chat / --limit-mm-per-prompt) "
                    "- images would be dropped. Aborting.")
            try:
                from vllm.benchmarks.datasets.datasets import RandomMultiModalDataset
                from gbench.utils import safe_get_tokenizer
                _tok = safe_get_tokenizer(model.get_model_path(format))
                _probe = RandomMultiModalDataset(random_seed=83).sample(
                    tokenizer=_tok, num_requests=1, input_len=max(16, int(text_len)),
                    output_len=int(output_length or 128), range_ratio=0.0,
                    base_items_per_request=n_img, num_mm_items_range_ratio=0.0,
                    limit_mm_per_prompt={"image": n_img},
                    bucket_config={(256, 256, 1): 1.0}, enable_multimodal_chat=True)
                _p = _probe[0].prompt
                _content = _p[0].get("content", []) if isinstance(_p, list) else []
                _nimg = sum(1 for c in _content
                            if isinstance(c, dict) and c.get("type") == "image_url")
                if _nimg != n_img:
                    raise RuntimeError(
                        f"MM throughput preflight FAILED: random-mm produced {_nimg} "
                        f"images (expected {n_img}); images would be dropped. Aborting.")
                logger.info(f"MM throughput preflight OK: dataset emits {_nimg} "
                            f"images/request; MM flags present.")
            except RuntimeError:
                raise
            except Exception as _pe:
                logger.warning(f"MM throughput dataset preflight skipped (non-fatal): {_pe}")

            logger.info(f"Command: {' '.join(cmd)}")

            rc, out, err = self._run_offline_subprocess(
                cmd, timeout=self._offline_timeout_s(num_prompts, output_length)
            )
            result = subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)

            if result.returncode != 0:
                logger.error(f"Multimodal throughput STDOUT: {result.stdout}")
                logger.error(f"Multimodal throughput STDERR: {result.stderr}")
                raise RuntimeError(
                    f"Multimodal throughput failed: {result.stderr}"
                )
            
            # Log stdout for debugging
            logger.debug(f"Throughput STDOUT: {result.stdout[-1000:]}")
            
            # Parse output JSON if it exists
            import os
            json_path = "/tmp/mm_throughput_result.json"
            if os.path.exists(json_path):
                logger.info(f"Reading results from {json_path}")
                with open(json_path) as f:
                    metrics = json.load(f)
                logger.info(f"Throughput metrics: {metrics}")
            else:
                # Parse from stdout as fallback
                logger.warning(f"No JSON output file, parsing stdout")
                logger.info(f"STDOUT (last 500 chars): {result.stdout[-500:]}")
                metrics = self._parse_throughput_output(result.stdout)
            
            # Map field names for CLI compatibility
            metrics["model_short"] = model.short_name
            metrics["model"] = model.short_name
            metrics["format"] = format.value
            # vLLM's 'tokens_per_second' is TOTAL (prompt-text + output, image
            # tokens excluded) - NOT output. Mapping it to output_tokens_per_second
            # made prefill-heavy MM report ~44k "output" tok/s (mostly the 7k-token
            # prompt). Report total and output SEPARATELY: output = generated
            # tokens / elapsed (num_prompts * output_len, since ignore_eos forces
            # the full length), which is the real decode throughput.
            if "tokens_per_second" in metrics:
                metrics["total_tokens_per_second"] = metrics["tokens_per_second"]
            elapsed = metrics.get("elapsed_time") or metrics.get("duration") or 0
            out_toks = int(num_prompts) * int(output_length or 128)
            metrics["total_output_tokens"] = out_toks
            if elapsed and elapsed > 0:
                metrics["output_tokens_per_second"] = out_toks / float(elapsed)
            # Actual MM prefill = clamped text + image placeholder tokens (what the
            # engine prefills), plus the nominal campaign input, so the row is not
            # labeled with a bare "img" or the collapsed text length alone.
            metrics["input_length"] = int(text_len + img_tokens)
            metrics["input_length_nominal"] = int(input_length)
            metrics["mm_text_tokens"] = int(text_len)
            metrics["mm_images"] = int(n_img)
            return metrics
            
        finally:
            if os.path.exists("/tmp/mm_throughput_result.json"):
                os.unlink("/tmp/mm_throughput_result.json")

    def _build_command(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: int,
        output_file: Path,
        dataset: str = "random",
    ) -> list[str]:
        """Build vllm bench throughput command."""
        # For multimodal, use HF dataset with VQA
        # random-mm is not available for throughput in current vLLM
        is_multimodal = dataset == "random-mm"
        
        cmd = [
            "vllm",
            "bench",
            "throughput",
            "--model",
            model.get_model_path(format),
        ]
        
        if is_multimodal:
            # Use HuggingFace VQA dataset for multimodal throughput
            cmd.extend([
                "--dataset-name", "hf",
                "--dataset", "lmms-lab/textvqa",
                "--hf-split", "validation",
            ])
        else:
            cmd.extend([
                "--dataset-name", dataset,
                # MUST use --random-input-len/--random-output-len for the random
                # dataset. The generic --input-len/--output-len are SILENTLY
                # IGNORED here (verified: asking 128/512 produced vLLM's defaults
                # 1024/128), which truncated every campaign to a ~128-token
                # generation and made all throughput numbers a meaningless uniform
                # ~2670 tok/s. random-* are the entry points get_samples actually
                # reads (the MM path already uses them, which is why MM honored
                # its output length while text did not).
                "--random-input-len", str(input_length),
                "--random-output-len", str(output_length),
                # Deterministic shape: every request is EXACTLY (input_length,
                # output_length), so total tokens = num_prompts*(in+out) and the
                # reported peak tokens/s is reproducible and comparable across
                # models/campaigns. (vllm bench throughput forces ignore_eos=True,
                # so with the length actually applied, full output is generated.)
                "--random-range-ratio", "0.0",
                "--seed", "83",  # Reproducible randomness
            ])
        
        cmd.extend([
            "--num-prompts", str(num_prompts),
        ])

        # Add GPU memory utilization
        gpu_mem = self.config.gpu_memory_utilization
        if gpu_mem:
            cmd.extend(
                [
                    "--gpu-memory-utilization",
                    str(gpu_mem),
                ]
            )

        # Add performance optimization flags
        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")

        # Disable prefix caching for parity with serving/stress. The benchmark
        # prompts are fixed-seed; with APC on (vLLM's V1 default) a shared prefix or
        # any repeat would be served from cache and inflate throughput above real
        # prefill. random prompts rarely share a prefix, but this makes it correct
        # by construction (and matches the serving/stress servers).
        cmd.append("--no-enable-prefix-caching")

        if self.config.max_num_batched_tokens:
            cmd.extend([
                "--max-num-batched-tokens",
                str(self.config.max_num_batched_tokens),
            ])

        # GGUF models need explicit tokenizer from HF model ID
        if format == ModelFormat.GGUF:
            cmd.extend(["--tokenizer", model.hf_model_id])
        
        # Multi-GPU: tensor parallel
        num_gpus = self.config.tensor_parallel_size or 1
        if num_gpus > 1:
            cmd.extend(["--tensor-parallel-size", str(num_gpus)])
        
        # Uniform context length for fair comparison (overridable via
        # --max-model-len for long-context campaigns)
        max_model_len = get_max_model_len(self.config.max_model_len)
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Limit max concurrent sequences for uniform memory usage
        if self.config.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.config.max_num_seqs)])

        return cmd

    def _parse_throughput_output(self, stdout: str) -> dict:
        """Parse plain text output from vllm bench throughput.
        
        Expected format:
        Throughput: X requests/s, Y total tokens/s, Z output tokens/s
        Total num prompt tokens: NNN
        Total num output tokens: MMM
        
        Args:
            stdout: Raw stdout from vllm bench throughput
            
        Returns:
            Parsed metrics as dictionary
        """
        import re
        
        result = {}
        
        # Parse throughput line
        throughput_match = re.search(
            r'Throughput:\s+([\d.]+)\s+requests/s,\s+([\d.]+)\s+total tokens/s,\s+([\d.]+)\s+output tokens/s',
            stdout
        )
        if throughput_match:
            result["request_throughput"] = float(throughput_match.group(1))
            result["total_tokens_per_second"] = float(throughput_match.group(2))
            result["output_tokens_per_second"] = float(throughput_match.group(3))
        
        # Parse prompt tokens
        prompt_match = re.search(r'Total num prompt tokens:\s+(\d+)', stdout)
        if prompt_match:
            result["total_prompt_tokens"] = int(prompt_match.group(1))
        
        # Parse output tokens
        output_match = re.search(r'Total num output tokens:\s+(\d+)', stdout)
        if output_match:
            result["total_output_tokens"] = int(output_match.group(1))
        
        if not result:
            raise RuntimeError(f"Could not parse throughput output: {stdout[-500:]}")
        
        return result
