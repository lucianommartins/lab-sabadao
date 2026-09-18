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

"""Stress test runner - capacity under a latency SLO.

SOTA serving-capacity methodology: drive the endpoint OPEN-LOOP at a target
arrival rate (QPS) with Poisson arrivals and no concurrency cap, sweep the
rate to find the maximum sustainable rate ("goodput") that keeps BOTH P99
TTFT and P99 TPOT within their SLOs. This is what academic/industry serving
benchmarks (MLPerf server scenario, vLLM/Orca/Sarathi/DistServe) report -
not a closed-loop concurrency ceiling.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import re
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import BenchmarkConfig, get_max_model_len
from ..core.models import ModelConfig, ModelFormat
from .serving import ServingBenchmarkRunner

# vLLM benchmark imports will be loaded dynamically in active methods

logger = logging.getLogger(__name__)


def _delta_piece(choice: dict, content_field: str) -> str:
    """Extract the streamed text from one SSE `choices[i]` object.

    Counts content OR reasoning/thinking tokens: a reasoning model (e.g. gemma via
    Ollama) streams its reply as `reasoning`/`reasoning_content` deltas, which is real
    decode - not an empty completion. Mirrors the serving pillar's parser; ignoring
    them made every stress request look like a 0-token reply -> 'no usable stream'."""
    if content_field == "delta":
        d = choice.get("delta") or {}
        return d.get("content") or d.get("reasoning") or d.get("reasoning_content") or ""
    return choice.get("text") or ""


def _stress_client_worker(task: dict) -> dict:
    """Run ONE client process's share of an open-loop rate point.

    Module-level (picklable for the 'spawn' start method) so K copies run in
    separate OS processes - this is what removes the single-asyncio-client
    CPU-contention confound at the knee: SSE parsing for many concurrent streams
    no longer starves one event loop and inflates client-side TTFT/TPOT. Uses
    epoch ``time.time()`` (NOT perf_counter, whose origin is per-process) so the
    parent can merge records across workers on a common clock. Fires its payload
    chunk at Poisson(qps) and records per-request send/ttft/tpot/itls/done/ok,
    plus the worst cumulative schedule slippage (the client-bound signal).

    Args:
        task: dict with payloads, qps, seed, api_url, content_field,
            request_timeout, max_point_s, itl_cap.

    Returns:
        {"records": [...], "sched_lag_s": float}.
    """
    import aiohttp  # local import: worker-only dependency

    payloads = task["payloads"]
    qps = float(task["qps"])
    seed = int(task["seed"])
    api_url = task["api_url"]
    content_field = task["content_field"]
    request_timeout = float(task["request_timeout"])
    max_point_s = float(task["max_point_s"])
    itl_cap = int(task["itl_cap"])

    rng = random.Random(seed)
    records: list = []
    state = {"sched_lag_s": 0.0}

    async def _run():
        connector = aiohttp.TCPConnector(limit=0)  # open-loop: no client cap
        timeout = aiohttp.ClientTimeout(total=request_timeout)

        async def _send(session, payload):
            rec = {"send": time.time(), "ttft": None, "tpot": None,
                   "itls": None, "done": None, "ok": False}
            records.append(rec)
            ct = []  # arrival epoch of each real content token
            try:
                async with session.post(api_url, json=payload) as resp:
                    if resp.status != 200:
                        rec["done"] = time.time()
                        return
                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:") or "[DONE]" in line:
                            continue
                        try:
                            ch = json.loads(line[5:].strip()).get("choices", [{}])[0]
                            piece = _delta_piece(ch, content_field)
                        except Exception:
                            piece = ""
                        if piece:
                            ct.append(time.time())
                rec["done"] = time.time()
                if ct:
                    rec["ttft"] = (ct[0] - rec["send"]) * 1000.0
                    if len(ct) >= 2:
                        gaps = [(ct[j] - ct[j - 1]) * 1000.0
                                for j in range(1, len(ct))]
                        rec["tpot"] = sum(gaps) / len(gaps)
                        # Subsample gaps (uniform, preserves the distribution) so
                        # the parent's pooled P99-ITL tail gate stays cheap to
                        # pickle back on long-decode (8192 gaps/req).
                        if len(gaps) > itl_cap:
                            step = len(gaps) / itl_cap
                            rec["itls"] = [gaps[int(k * step)] for k in range(itl_cap)]
                        else:
                            rec["itls"] = gaps
                    rec["ok"] = True
            except Exception:
                rec["done"] = time.time()

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            t0 = time.time()
            planned = 0.0
            for k, payload in enumerate(payloads):
                tasks.append(asyncio.create_task(_send(session, payload)))
                if k < len(payloads) - 1 and qps > 0:
                    dt = rng.expovariate(qps)
                    planned += dt
                    state["sched_lag_s"] = max(
                        state["sched_lag_s"], (time.time() - t0) - planned)
                    await asyncio.sleep(dt)
            pending = set(tasks)
            deadline = time.time() + max_point_s
            while pending:
                rem = deadline - time.time()
                if rem <= 0:
                    for t in pending:
                        t.cancel()
                    break
                _d, pending = await asyncio.wait(pending, timeout=min(30.0, rem))

    asyncio.run(_run())
    return {"records": records, "sched_lag_s": state["sched_lag_s"]}


def _interpolate_knee(low_pt: dict, high_pt: dict, ttft_thr: float,
                      itl_thr: float, keepup_frac: float) -> float:
    """Linear-interpolate the sustainable arrival rate between a passing bracket
    (low_pt) and the failing bracket just above it (high_pt).

    The binary search brackets the knee to ±QPS_PRECISION; this pins it inside
    that bracket by finding where the binding constraint's normalized margin
    crosses zero. Margin = min over constraints of (headroom / limit); positive
    at low_pt (passes), negative at high_pt (fails). Whichever of TTFT-P99,
    ITL-P99 or keep-up binds first sets the crossing.

    Returns:
        Interpolated arrival QPS in [low_rate, high_rate]; falls back to the low
        rate if the margins don't straddle zero.
    """
    def margin(pt):
        ttft = pt.get("p99_ttft_ms") or 0.0
        # Match _test_rate's decode gate EXACTLY: prefer the SERVER's P99 ITL
        # (from /metrics), fall back to the client P99 ITL, then P99 TPOT (no ITL
        # captured). Using the client ITL here while the gate used the server ITL
        # would interpolate the crossing on a different constraint than the one
        # that set pass/fail. Stored values are 0.0 (not None) when absent.
        itl = pt.get("p99_itl_server_ms") or 0.0
        if itl <= 0:
            itl = pt.get("p99_itl_ms") or 0.0
        if itl <= 0:
            itl = pt.get("p99_tpot_ms") or 0.0
        rate = pt.get("arrival_qps") or pt.get("request_rate_qps") or 0.0
        achieved = pt.get("achieved_qps") or 0.0
        m_ttft = (ttft_thr - ttft) / ttft_thr if ttft_thr else 1.0
        m_itl = (itl_thr - itl) / itl_thr if itl_thr else 1.0
        m_keep = (achieved / rate - keepup_frac) if rate > 0 else -1.0
        return min(m_ttft, m_itl, m_keep)

    r_low = low_pt.get("arrival_qps") or low_pt.get("request_rate_qps") or 0.0
    r_high = high_pt.get("arrival_qps") or high_pt.get("request_rate_qps") or 0.0
    m_low, m_high = margin(low_pt), margin(high_pt)
    # Require a STRICT sign change (m_high < 0): if the "failing" point sits
    # exactly at threshold (m_high == 0) the crossing is at r_high itself, which
    # just failed - return the last passing rate instead of the failing one.
    if r_high <= r_low or not (m_low > 0 > m_high):
        return r_low
    frac = m_low / (m_low - m_high)   # in (0,1) given m_low>0>m_high
    frac = min(1.0, max(0.0, frac))
    return r_low + frac * (r_high - r_low)


def _knee_stats(knees: list) -> dict:
    """Summarize the knee across sweep reps: mean/std/CV/min/max + a percentile
    bootstrap CI. With only a few reps the CI is wide by construction - that
    honesty is the point (the notes record FA4 knees varying up to 16%
    run-to-run, so a single sweep's knee is not trustworthy on its own)."""
    vals = np.asarray([k for k in knees if k is not None and np.isfinite(k)], dtype=float)
    out = {"reps": int(vals.size)}
    if vals.size == 0:
        return out
    mean = float(np.mean(vals))
    out["mean"] = mean
    # Median is the reported headline: with few reps it rejects a single spurious
    # rep (e.g. a transient 0-knee from one bad sweep) that would otherwise drag
    # the mean down by ~1/reps. mean/std/CI are still reported for transparency.
    out["median"] = float(np.median(vals))
    out["std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    out["cv_percent"] = (out["std"] / mean * 100.0) if mean > 0 else 0.0
    out["min"] = float(np.min(vals))
    out["max"] = float(np.max(vals))
    if vals.size >= 2:
        rng = np.random.default_rng(83)
        boot = np.mean(rng.choice(vals, size=(2000, vals.size), replace=True), axis=1)
        out["ci_low"] = float(np.percentile(boot, 2.5))
        out["ci_high"] = float(np.percentile(boot, 97.5))
    else:
        out["ci_low"] = out["ci_high"] = mean
    return out


class StressTestRunner:
    """Runner for stress testing to find max sustainable throughput under SLO.

    Sweeps the OPEN-LOOP arrival rate (QPS, Poisson) to find the maximum
    sustainable rate that keeps P99 TTFT and P99 TPOT within their SLOs.
    Supports both text-only and multimodal stress tests.
    """

    # ── Open-loop request-rate (QPS) sweep parameters ─────────
    # Stress finds the max sustainable arrival rate (goodput) that keeps both
    # latency SLOs - the SOTA capacity-under-SLO measurement, not a concurrency
    # ceiling. Load is Poisson (exponential inter-arrival), concurrency uncapped.
    INITIAL_QPS = 4.0            # starting arrival rate (req/s)
    MIN_QPS = 0.5               # lower bound for a downward search
    MIN_DOWN_QPS = 0.05         # floor for the ascending sweep's rare downward fallback
    MAX_QPS = 8192.0           # safety cap: high enough that a fast multi-GPU server finds its
                               # real SLO-break knee out of the box (512 censored fast servers -
                               # they passed the SLO at every point up to the cap). Override with
                               # --stress-max-qps for an even more powerful host.
    QPS_PRECISION = 1.15        # stop binary search when high/low < this (±~15%)

    # ── Steady-state measurement (capacity, not a transient latency knee) ──────
    # A rate is "sustainable" only if, AT STEADY STATE, the server KEEPS UP
    # (achieved ≈ arrival - the queue isn't growing) AND meets both latency SLOs
    # for ≥SLO_ATTAINMENT of requests. We size each point to ~MEASURE_ARRIVAL_S
    # of Poisson arrivals, then compute achieved throughput / SLO attainment over
    # the STEADY middle of the run (dropping the pipeline fill and final drain),
    # so an over-capacity rate shows achieved << arrival instead of a transiently
    # low P99. Each point is hard-capped at MAX_POINT_S so long-decode can't run
    # forever. (The old flat 20s window << request latency reported arrival-rate
    # knees far above real capacity - e.g. 72 "req/s" when the server did ~21.)
    MEASURE_ARRIVAL_S = 60      # target seconds of arrivals per rate point
    MIN_PROMPTS = 40            # floor so the steady window has enough samples
    MAX_PROMPTS = 8000          # cap so high-QPS points don't send unboundedly, but large enough
                               # that a point near the raised MAX_QPS still spans a real steady
                               # window (1200 was ~0.15s at 8k req/s; 8000 is ~1s + ~8k samples).
    MIN_STEADY = 15             # min steady-state completions to trust a point

    # ── Multi-process load generator ──────────────────────────────────────────
    # The arrival generation + SSE parsing is spread across NUM_CLIENT_PROCS OS
    # processes so a single Python event loop's CPU cost never becomes the knee
    # (the classic single-asyncio-client confound). Each worker drives qps/K with
    # its own seeded Poisson stream (deterministic, isolated by construction) and
    # returns epoch-timestamped records the parent merges. CLIENT_KEEPUP_FRAC is
    # the rate-sustainability guard: if the aggregate ACHIEVED SEND rate falls
    # below this fraction of the target, the CLIENT (not the server) is the
    # bottleneck and the point is flagged client_bound (its arrival label is not
    # a server capacity result).
    NUM_CLIENT_PROCS = 32       # max client processes (capped to cpu_count-1). Raised from 8 so the
                               # load generator can actually OFFER the higher MAX_QPS on a big host
                               # (8 asyncio procs hit client_bound at ~1-2k req/s, re-censoring the
                               # knee); small hosts still auto-cap to cpu_count-1.
    CLIENT_KEEPUP_FRAC = 0.90   # send-rate floor before a point is client_bound
    ITL_SAMPLE_CAP = 512        # max inter-token gaps kept per request (P99 pool)
    # Minimum requests PER worker. The worker count is scaled so each worker
    # drives a stable Poisson stream; splitting a small low-QPS point across all
    # NUM_CLIENT_PROCS gives ~5 requests/worker, and the aggregate arrival window
    # (max over workers of each worker's last-send time) then stretches far past
    # the target - deterministically dropping the realized rate to ~37% at
    # 0.44 req/s (simulation byte-matches the observed 0.44->0.164). So use FEWER
    # workers at low QPS: k = min(NUM_CLIENT_PROCS, num_prompts // this). Multi-proc
    # still kicks in at high QPS, where the single-client CPU cost is the real risk
    # and each worker has plenty of requests.
    MIN_PROMPTS_PER_WORKER = 30

    # ── Reps for a trustworthy knee ───────────────────────────────────────────
    # A single sweep's knee is not reliable (run-to-run variation up to ~16% in
    # the notes), so the whole sweep runs STRESS_REPS times with varied seeds and
    # the knee is reported as mean + bootstrap CI over reps.
    STRESS_REPS = 3
    FILL_FRAC = 0.30            # drop the first 30% (pipeline fill)
    DRAIN_FRAC = 0.15           # drop the last 15% (final drain)
    # Loose saturation backstop only. At low QPS with long requests the steady
    # `achieved` undercounts ~15-20% even when the server keeps up (it responds
    # in ~40ms), so a tight 0.85 gate spuriously fired SATURATED there and
    # collapsed the knee non-reproducibly (decode-heavy 0.3 vs 2.5). The P99-SLO
    # gate now catches real saturation on its own (over-capacity -> queue -> P99
    # TTFT blows past the SLO within the steady window). 0.5 rejects only gross
    # saturation (e.g. achieved 0.29x arrival) while passing the low-QPS undercount.
    KEEP_UP_FRAC = 0.50         # sustainable iff achieved >= this * arrival
    SLO_ATTAINMENT = 0.99       # and >= this fraction of requests meet both SLOs
    MAX_FAILURE_FRAC = 0.02     # a point is INVALID if > this fraction of OFFERED
                                # requests error/drop (non-200, timeout, truncated).
                                # Failures leave achieved & slo_attainment (both
                                # successful-only denominators) untouched, so an
                                # ungated point with drops silently inflates the knee.
    REQUEST_TIMEOUT_S = 900     # per-request ceiling (full-length long decodes)
    MAX_POINT_S = 1200          # hard wall-clock cap per rate point (bounds cost)
    WARMUP_PROMPTS = 20         # (legacy) count; the gentle warmup below uses fewer, sequential
    WARMUP_PROMPTS_GENTLE = 3   # sequential (concurrency-1) warmup requests - never floods the queue
    PREFLIGHT_PROBES = 3        # sequential single-stream requests to establish the latency floor

    # Multimodal stress request shape. MM input is images + padded text sized to
    # the campaign's input length; output = campaign output (ignore_eos). Needs
    # the server started with --limit-mm-per-prompt image=IMAGES_PER_REQUEST.
    IMAGES_PER_REQUEST = 4
    PER_IMAGE_SOFT_TOKENS = 280   # gemma-4 vision_soft_tokens_per_image
    MIN_MM_TEXT_TOKENS = 16

    # ── Dual latency SLO (P99) ────────────────────────────────
    # TTFT = prefill responsiveness; TPOT = decode smoothness. A rate is
    # "sustainable" only if BOTH percentiles hold.
    DEFAULT_TTFT_THRESHOLD_MS = 5000   # laptop-friendly default (CLI overridable)
    DEFAULT_TPOT_THRESHOLD_MS = 200    # ~5 tok/s per stream floor (CLI overridable)

    def __init__(
        self,
        config: BenchmarkConfig,
        ttft_threshold_ms: int = None,
        tpot_threshold_ms: int = None,
    ):
        """Initialize stress test runner.

        Args:
            config: Benchmark configuration
            ttft_threshold_ms: P99 TTFT SLO in ms (overrides default)
            tpot_threshold_ms: P99 TPOT SLO in ms (overrides default)
        """
        self.config = config
        self.ttft_threshold_ms = ttft_threshold_ms or self.DEFAULT_TTFT_THRESHOLD_MS
        self.tpot_threshold_ms = tpot_threshold_ms or self.DEFAULT_TPOT_THRESHOLD_MS
        self.initial_qps = 1.0 if config.remote_endpoint else self.INITIAL_QPS
        self._serving_runner: Optional[ServingBenchmarkRunner] = None
        self._tested_rates = {}  # Cache tested QPS points (current rep)
        self._tokenizer = None  # Cached tokenizer
        self._api_model_id = None  # Cached served model id
        self._multimodal = False  # Whether current run is multimodal
        self._mm_image_dir = None  # Temp dir for generated images
        self._mm_image_paths = []  # Generated image file paths
        # Client processes: cap to cpu_count-1 (leave one core for the parent +
        # OS), overridable via config.stress_client_procs. 'spawn' context avoids
        # inheriting any parent state into the workers.
        # Flag (config) > GBENCH_STRESS_* env > class default, uniformly for every
        # stress knob (client procs / reps / max qps / max prompts).
        want = (getattr(config, "stress_client_procs", None)
                or (os.environ.get("GBENCH_STRESS_CLIENT_PROCS") or None)
                or self.NUM_CLIENT_PROCS)
        self._num_client_procs = max(1, min(int(want), max(1, (os.cpu_count() or 2) - 1)))
        self._reps = int(getattr(config, "stress_reps", None)
                         or (os.environ.get("GBENCH_STRESS_REPS") or None)
                         or self.STRESS_REPS)
        # MAX_QPS safety cap + MAX_PROMPTS steady-window size are overridable so a
        # fast model on strong hardware is not censored at the 512 default. Precedence:
        # --stress-max-qps / --stress-max-prompts (config) > GBENCH_STRESS_* env > default.
        _mq = getattr(config, "stress_max_qps", None)
        if _mq is None:
            _mq = os.environ.get("GBENCH_STRESS_MAX_QPS") or None
        self.max_qps = float(_mq) if _mq is not None else self.MAX_QPS
        _mp = getattr(config, "stress_max_prompts", None)
        if _mp is None:
            _mp = os.environ.get("GBENCH_STRESS_MAX_PROMPTS") or None
        self.max_prompts = int(_mp) if _mp is not None else self.MAX_PROMPTS
        # Min steady-state completions to trust a point. Lower it for a slow box that
        # meets the SLO but cannot complete the default 15 per point (else every point
        # is TOO-FEW and the knee is 0). Precedence: --stress-min-samples (config) >
        # GBENCH_STRESS_MIN_SAMPLES env > class default.
        _msamp = getattr(config, "stress_min_samples", None)
        if _msamp is None:
            _msamp = os.environ.get("GBENCH_STRESS_MIN_SAMPLES") or None
        self._min_steady = max(1, int(_msamp)) if _msamp is not None else self.MIN_STEADY
        # Prompts-per-point floor scales with the sample floor: enough to yield
        # ~_min_steady completions after dropping FILL/DRAIN, capped at the class
        # default (so the DEFAULT min-samples keeps the original 40-prompt floor and
        # byte-for-byte behavior; a lower floor makes each point smaller and faster).
        _steady_frac = max(0.1, 1.0 - self.FILL_FRAC - self.DRAIN_FRAC)
        self._min_prompts = (self.MIN_PROMPTS if self._min_steady >= self.MIN_STEADY
                             else min(self.MIN_PROMPTS,
                                      max(self._min_steady + 3,
                                          int(self._min_steady / _steady_frac) + 3)))
        self._rep_seed = 83     # base seed for the current rep (set per rep)
        # Single-stream latency measured by the preflight, used to seed the ascending
        # sweep's start rate near half the implied capacity (set in _preflight_floor_check).
        self._probe_ttft_ms = None
        self._probe_e2e_ms = None
        self._client_pool = None  # reusable multi-process client pool (per sweep)
        # Multiprocessing context for the client pool. 'spawn' avoids inheriting any
        # parent state (asyncio loop, sockets) into the workers. This used to sit after a
        # `return` in _images_per_request() -> unreachable, so `_mp_ctx` was never set and
        # every _get_client_pool() raised AttributeError, silently forcing the
        # single-process fallback while the artifact still claimed multi-process. Init here.
        try:
            self._mp_ctx = mp.get_context("spawn")
        except ValueError:
            self._mp_ctx = mp.get_context()

    def _images_per_request(self) -> int:
        """Images per MM request: config override (--personal sets 1) else default 4."""
        return int(getattr(self.config, "mm_images_per_request", None) or self.IMAGES_PER_REQUEST)

    def _generate_requests(self, num_prompts: int, model=None, format=None) -> list:
        """Generate sample requests for the stress sweep.

        Neutral default is a fixed 128-in / 128-out random shape - the laptop
        smoke-test workload. When the user pins a geometry (a --campaign or
        explicit --input/--output-lengths, i.e. config.workload_shape_explicit),
        stress the model at THAT shape and dataset instead, reusing the serving
        runner's dataset generation (random / sharegpt / custom / hf).
        """
        if (getattr(self.config, "workload_shape_explicit", False)
                and self._serving_runner is not None and model is not None):
            in_len = self.config.input_lengths[0] if self.config.input_lengths else 128
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            requests, tok = self._serving_runner._generate_sample_requests(
                model=model,
                format=format,
                num_prompts=num_prompts,
                input_len=in_len,
                output_len=out_len,
            )
            if tok is not None:
                self._tokenizer = tok
            return requests

        try:
            from vllm.benchmarks.datasets import RandomDataset
            dataset = RandomDataset()
            return dataset.sample(
                tokenizer=self._tokenizer,
                num_requests=num_prompts,
                prefix_len=0,
                input_len=128,
                output_len=128,
                range_ratio=0.5,
            )
        except ImportError:
            from dataclasses import dataclass
            @dataclass
            class SampleRequest:
                prompt: str
                expected_output_len: int = 128

            prompt = "Explain quantum computing in simple terms. " * 5
            return [SampleRequest(prompt=prompt, expected_output_len=128) for _ in range(num_prompts)]

    def _workload_shape(self) -> tuple:
        """(dataset, input_len, output_len) the stress sweep will use."""
        if getattr(self.config, "workload_shape_explicit", False):
            in_len = self.config.input_lengths[0] if self.config.input_lengths else 128
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            return (self.config.dataset, in_len, out_len)
        return ("random", 128, 128)

    def _mm_text_tokens(self, in_len: int, out_len: int) -> int:
        """Text-token budget for a MM request: the FULL campaign input text, with
        the images added ON TOP (not replacing text), so MM is a strict superset
        of the text workload (same text + 4 images) and is directly comparable to
        it. Only clamped down if text + images + output would overflow the context.
        Shared by the request builder AND the reported workload length so the two
        never drift."""
        img_tokens = self._images_per_request() * getattr(
            self, "_vision_tokens_per_image", self.PER_IMAGE_SOFT_TOKENS)
        max_tokens = max(int(out_len), 16)
        max_len = get_max_model_len(getattr(self.config, "max_model_len", None))
        return max(self.MIN_MM_TEXT_TOKENS,
                   min(int(in_len),
                       max_len - img_tokens - max_tokens - 128))

    def _mm_effective_input_len(self, in_len: int, out_len: int) -> int:
        """Actual MM prefill length = clamped text tokens + image placeholder
        tokens. This is what the server prefills (and what drives TTFT), NOT the
        campaign's nominal text input - for small-input campaigns the images
        dominate, so labeling the row with the nominal input understates it ~9x."""
        return int(self._mm_text_tokens(in_len, out_len)
                   + self._images_per_request() * getattr(
                       self, "_vision_tokens_per_image", self.PER_IMAGE_SOFT_TOKENS))

    def _workload_desc(self) -> str:
        """Human-readable workload description for the config banner."""
        _ds, in_len, out_len = self._workload_shape()
        if getattr(self, "_multimodal", False):
            return f"multimodal (256x256 image + text, {min(max(out_len, 16), 512)} out)"
        if _ds == "sharegpt":
            return "sharegpt (dynamic real prompt lengths)"
        return f"{_ds} in={in_len} out={out_len}"

    def _setup_mm_images(self, num_images: int = 50) -> str:
        """Generate synthetic images for multimodal stress test.

        Creates a temp directory with random JPEG images.

        Args:
            num_images: Number of images to generate

        Returns:
            Path to the temp directory containing images
        """
        import numpy as np
        from PIL import Image

        img_dir = tempfile.mkdtemp(prefix="stress_mm_images_")
        self._mm_image_paths = []

        for i in range(num_images):
            # 256x256 random images - small enough for fast I/O,
            # large enough to exercise the vision encoder
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(img_dir, f"stress_img_{i:03d}.jpg")
            img.save(img_path, format="JPEG", quality=85)
            self._mm_image_paths.append(img_path)

        logger.info(f"Generated {num_images} synthetic images in {img_dir}")
        self._mm_image_dir = img_dir
        return img_dir

    def _cleanup_mm_images(self):
        """Clean up temp image directory."""
        if self._mm_image_dir and os.path.exists(self._mm_image_dir):
            import shutil
            shutil.rmtree(self._mm_image_dir, ignore_errors=True)
            logger.info(f"Cleaned up MM image directory: {self._mm_image_dir}")
            self._mm_image_dir = None
            self._mm_image_paths = []

    def _build_text(self, n_tokens: int) -> str:
        """A text prompt of approximately n_tokens tokens (MM input padding)."""
        base = "Analyze the attached images and the following context in detail. "
        if not self._tokenizer or n_tokens <= 0:
            return base * max(1, n_tokens // 12)  # rough char-based fallback
        filler = base * (max(1, n_tokens // 8) + 8)
        ids = self._tokenizer(filler).input_ids[:n_tokens]
        return self._tokenizer.decode(ids)

    def _build_payloads(self, model, format, num_prompts: int) -> tuple:
        """Build (api_url, content_field, [payload, ...]) for one rate point.

        Built once in the PARENT and split across client workers, so payload
        construction (tokenization, image-URL rotation) is not repeated in every
        process. Text uses /v1/completions; multimodal uses /v1/chat/completions
        with N images + campaign-sized padded text. Every request forces
        ignore_eos so it generates the campaign's full output length.
        """
        model_path = self._api_model_id or self._serving_runner._resolve_model_id(model, format)
        # Normalize like serving.py: a remote endpoint already ending in /v1 must
        # not get a second /v1 appended (that 404s every request). Works for the
        # local path too (base_url has no /v1 -> the /v1 branch is taken).
        base_url = (self.config.remote_endpoint or f"http://127.0.0.1:{self._serving_runner.server_port}").rstrip("/")
        _ds, in_len, out_len = self._workload_shape()
        max_tokens = max(int(out_len), 16)
        # Portable no-think for the perf workload (mirrors serving); {} under --thinking.
        _snt = getattr(self._serving_runner, "no_think_fields", None)
        nt = _snt() if callable(_snt) else {}

        if self._multimodal:
            api_url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
            n_img = self._images_per_request()
            target_text = self._mm_text_tokens(in_len, out_len)
            prompt_text = self._build_text(target_text)

            def _img_url(path):
                if self.config.remote_endpoint:
                    import base64
                    with open(path, "rb") as f:
                        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                return f"file://{path}"

            payloads = []
            for i in range(num_prompts):
                urls = [
                    _img_url(self._mm_image_paths[(i * n_img + k) % len(self._mm_image_paths)])
                    for k in range(n_img)
                ]
                content = [{"type": "text", "text": prompt_text}]
                content += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
                payloads.append({
                    "model": model_path,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens, "ignore_eos": True, "stream": True, **nt,
                })
            return api_url, "delta", payloads

        # REMOTE endpoints (e.g. Ollama) commonly IGNORE ignore_eos, so a random-token
        # /v1/completions prompt draws an immediate-EOS (empty) reply - every rate point
        # then records 0 completions and an infinite TTFT. Mirror the serving pillar's
        # proven remote path: a COHERENT chat prompt padded to in_len + a long-output
        # directive over /v1/chat/completions, which reliably sustains output. LOCAL vLLM
        # keeps the controlled-shape RandomDataset + /v1/completions + ignore_eos below.
        if (self.config.remote_endpoint and self._serving_runner is not None
                and self._tokenizer is not None):
            chat_url = (f"{base_url}/chat/completions" if base_url.endswith("/v1")
                        else f"{base_url}/v1/chat/completions")
            payloads = [{
                "model": model_path,
                "messages": [
                    {"role": "system", "content": self._serving_runner.NOTHINK_SYSTEM},
                    {"role": "user", "content": self._serving_runner._build_text_prompt(
                        in_len, out_len, self._tokenizer, seed=i)},
                ],
                "max_tokens": max_tokens, "ignore_eos": True, "stream": True, **nt,
            } for i in range(num_prompts)]
            return chat_url, "delta", payloads

        api_url = f"{base_url}/completions" if base_url.endswith("/v1") else f"{base_url}/v1/completions"
        reqs = self._generate_requests(num_prompts, model, format)
        prompts = [getattr(r, "prompt", "") for r in reqs] or [""]
        payloads = [{
            "model": model_path, "prompt": prompts[i % len(prompts)],
            "max_tokens": max_tokens, "ignore_eos": True, "stream": True, **nt,
        } for i in range(num_prompts)]
        return api_url, "text", payloads

    # Server-side ITL SLO: seconds to wait for the /metrics scrape. The knee gates
    # on the SERVER's own inter_token_latency_seconds (true decode smoothness) when
    # the endpoint exposes /metrics, because the client-measured P99 ITL is
    # confounded by the multiproc-async client's SSE read scheduling under bursty
    # load (verified: at 3 req/s MM the client measured P99 ITL 195-1193ms while the
    # server's own P99 ITL was ~25ms). Falls back to client ITL when /metrics is
    # absent (non-vLLM endpoints).
    METRICS_TIMEOUT_S = 5

    def _metrics_url(self) -> Optional[str]:
        """The served endpoint's Prometheus /metrics URL (server root + /metrics),
        or None if we don't have an endpoint. /metrics lives at the SERVER ROOT,
        not under /v1, so strip a trailing /v1."""
        if getattr(self.config, "remote_endpoint", None):
            base = self.config.remote_endpoint
        elif self._serving_runner is not None and getattr(
                self._serving_runner, "server_port", None):
            base = f"http://127.0.0.1:{self._serving_runner.server_port}"
        else:
            return None
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        return base + "/metrics"

    def _server_itl_snapshot(self) -> Optional[tuple]:
        """Cumulative vllm:inter_token_latency_seconds histogram, summed across DP
        engines: ({le: count}, total_count). None if /metrics is unreachable or the
        metric is absent (so the caller falls back to client-measured ITL). This
        build names the per-token metric inter_token_latency_seconds (NOT
        time_per_output_token_seconds)."""
        url = self._metrics_url()
        if not url:
            return None
        try:
            raw = urllib.request.urlopen(url, timeout=self.METRICS_TIMEOUT_S).read().decode()
        except Exception:
            return None
        buckets: dict = {}
        count = 0.0
        for ln in raw.splitlines():
            if ln.startswith("vllm:inter_token_latency_seconds_bucket"):
                m = re.search(r'le="([^"]+)".*?\}\s+([\d.eE+]+)', ln)
                if m:
                    le = float("inf") if m.group(1) == "+Inf" else float(m.group(1))
                    buckets[le] = buckets.get(le, 0.0) + float(m.group(2))
            elif ln.startswith("vllm:inter_token_latency_seconds_count"):
                count += float(ln.split()[-1])
        if not buckets or count <= 0:
            return None
        return buckets, count

    @staticmethod
    def _hist_delta_p99_ms(before: Optional[tuple], after: Optional[tuple],
                           pct: float = 99.0) -> Optional[float]:
        """Approx P99 (ms) of the inter-token latency of the requests that ran
        BETWEEN two cumulative-histogram snapshots (the current rate point).
        None if either snapshot is missing or no tokens were emitted in the window
        (so the caller keeps the client-side ITL)."""
        if not before or not after:
            return None
        b_buckets, b_cnt = before
        a_buckets, a_cnt = after
        d_cnt = a_cnt - b_cnt
        if d_cnt <= 0:
            return None
        les = sorted(a_buckets)
        target = pct / 100.0 * d_cnt
        cum = 0.0
        for le in les:
            cum += (a_buckets.get(le, 0.0) - b_buckets.get(le, 0.0))
            if cum >= target:
                return float("inf") if le == float("inf") else le * 1000.0
        return float("inf")

    def _measure_rate_point(
        self,
        model: ModelConfig,
        format: ModelFormat,
        qps: float,
        num_prompts: int,
    ) -> dict:
        """Measure ONE open-loop Poisson arrival rate at STEADY STATE, using the
        MULTI-PROCESS client so the load generator can't become the bottleneck.

        Splits num_prompts across K worker processes (each driving qps/K with its
        own seeded Poisson stream), merges their epoch-timestamped records, and
        derives achieved throughput + dual-SLO attainment over the steady middle
        of the run via _steady_state_metrics - so a rate the server cannot
        service shows achieved << arrival (not a transiently-passing knee), and a
        rate the CLIENT cannot emit is flagged client_bound.
        """
        api_url, content_field, payloads = self._build_payloads(model, format, num_prompts)
        if not payloads:
            return self._steady_state_metrics([], qps)

        # Snapshot the server's own inter-token-latency histogram before the load so
        # the knee can gate on TRUE decode smoothness (the delta over this point),
        # immune to the client-side multiproc-async ITL confound.
        srv_itl_before = self._server_itl_snapshot()

        # Scale workers with request count so each worker has a stable Poisson
        # stream (>= MIN_PROMPTS_PER_WORKER). At low QPS this collapses to 1
        # worker, which emits the target rate faithfully; at high QPS it grows to
        # NUM_CLIENT_PROCS where the single-client CPU cost is the real risk.
        k = max(1, min(self._num_client_procs,
                       len(payloads) // self.MIN_PROMPTS_PER_WORKER or 1))
        # Round-robin split so each worker's chunk spans the whole prompt set
        # (same shape distribution); per-worker rate is qps/k.
        chunks = [payloads[w::k] for w in range(k)]
        qps_per = qps / k
        base = self._rep_seed * 1_000_003 + int(round(qps * 1000))
        tasks = [{
            "payloads": chunks[w],
            "qps": qps_per,
            "seed": (base + w * 2_654_435_761) & 0x7FFFFFFF,
            "api_url": api_url,
            "content_field": content_field,
            "request_timeout": self.REQUEST_TIMEOUT_S,
            "max_point_s": self.MAX_POINT_S,
            "itl_cap": self.ITL_SAMPLE_CAP,
        } for w in range(k) if chunks[w]]

        records: list = []
        max_lag = 0.0
        n_procs = len(tasks)
        try:
            ex = self._get_client_pool()
            for out in ex.map(_stress_client_worker, tasks):
                records.extend(out.get("records", []))
                max_lag = max(max_lag, float(out.get("sched_lag_s", 0.0)))
        except Exception as e:
            # Pool unusable (e.g. spawn blocked in a restricted sandbox, or a
            # worker died). Fall back to a SINGLE in-process asyncio loop driving
            # ALL payloads at the FULL qps - this preserves the aggregate arrival
            # rate. A per-worker sequential fallback would instead run each
            # chunk's qps/k in back-to-back windows, collapsing the effective rate
            # to qps/k and flagging every point client_bound. The single-process
            # client may be CPU-bound at high QPS, but client_bound flags that.
            logger.warning(f"Multi-process client unavailable ({e}); falling back "
                           f"to single-process in-loop client (contention may apply).")
            self._shutdown_client_pool()
            n_procs = 1
            try:
                out = _stress_client_worker({
                    "payloads": payloads, "qps": qps, "seed": base,
                    "api_url": api_url, "content_field": content_field,
                    "request_timeout": self.REQUEST_TIMEOUT_S,
                    "max_point_s": self.MAX_POINT_S, "itl_cap": self.ITL_SAMPLE_CAP,
                })
                records = out.get("records", [])
                max_lag = float(out.get("sched_lag_s", 0.0))
            except Exception as e2:
                logger.error(f"In-process fallback also failed: {e2}")
                records = []

        srv_itl_after = self._server_itl_snapshot()
        metrics = self._steady_state_metrics(records, qps)
        metrics["client_sched_lag_s"] = max_lag
        metrics["num_client_procs"] = n_procs
        # True server-side P99 ITL over THIS rate point (None when /metrics is
        # absent -> caller keeps client ITL). Immune to the client read-scheduling
        # confound that otherwise under-reports the MM knee ~7x.
        metrics["p99_itl_server_ms"] = self._hist_delta_p99_ms(
            srv_itl_before, srv_itl_after)
        # Decide client_bound from the DIRECT schedule-slippage signal, NOT from
        # actual_send_qps. The empirical send span (max over K workers of each
        # worker's last-send time) is biased HIGH, so actual_send_qps reads ~10-15%
        # low even when the client is nowhere near saturated - in GPU validation it
        # flagged EVERY point, including ones where achieved > arrival. sched_lag
        # (cumulative asyncio-schedule overshoot: ~0 for a healthy client, grows
        # only when the event loop is CPU-starved) is the unbiased signal. A worker
        # is client-bound only if it fell >10% of the arrival window behind (1s floor).
        window_s = num_prompts / max(qps, 1e-6)
        metrics["client_bound"] = bool(max_lag > max(1.0, 0.10 * window_s))
        return metrics

    def _get_client_pool(self) -> ProcessPoolExecutor:
        """Lazily create ONE reusable client process pool for the whole sweep.

        Reused across every rate point and rep so the K workers stay warm - no
        per-point re-spawn cost and, more importantly, no repeated worker
        start-up stagger eating into the ~MEASURE_ARRIVAL_S window (which would
        depress actual_send_qps and spuriously trip client_bound).
        """
        if self._client_pool is None:
            self._client_pool = ProcessPoolExecutor(
                max_workers=self._num_client_procs, mp_context=self._mp_ctx)
        return self._client_pool

    def _shutdown_client_pool(self):
        """Tear down the reusable client pool (end of sweep, or after a fault)."""
        if self._client_pool is not None:
            try:
                self._client_pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._client_pool = None

    def _steady_state_metrics(self, records: list, qps: float) -> dict:
        """Achieved throughput + dual-SLO attainment over the steady middle.

        Drops the pipeline-fill (first FILL_FRAC) and final-drain (last
        DRAIN_FRAC) of the run by completion time, so achieved throughput
        reflects equilibrium: a keep-up rate yields achieved ≈ arrival, an
        over-capacity rate yields achieved ≈ server-max << arrival.
        """
        recs = [r for r in records if r.get("send") is not None]

        # Rate-sustainability guard: did the CLIENT actually emit arrivals near
        # the target? If the aggregate send rate is well below qps, the client
        # (not the server) is the bottleneck and the point's arrival label is not
        # a server-capacity result. Computed on ALL sends (not just completed).
        actual_send_qps = 0.0
        if len(recs) >= 2:
            sends = sorted(r["send"] for r in recs)
            actual_send_qps = (len(sends) - 1) / max(1e-6, sends[-1] - sends[0])
        client_bound = bool(qps > 0 and len(recs) >= 2
                            and actual_send_qps < self.CLIENT_KEEPUP_FRAC * qps)

        def _empty(failed, completed=0):
            return {"achieved_qps": 0.0, "arrival_qps": qps, "slo_attainment": 0.0,
                    "p99_ttft_ms": float("inf"), "p99_tpot_ms": float("inf"),
                    "p99_itl_ms": float("inf"), "actual_send_qps": actual_send_qps,
                    "client_bound": client_bound,
                    "completed": completed, "failed": failed, "steady_n": 0}

        done = [r for r in recs if r["ok"] and r["done"] is not None]
        if not done:
            return _empty(len(recs))

        # Drop the fill and drain TRANSIENTS by their PHYSICAL size, not a fixed
        # fraction of the run. By Little's Law the pipeline-fill (and symmetric
        # drain) transient is ~one steady-state concurrency
        #   C = arrival_rate * mean_latency
        # requests at each end. Trimming C - instead of a flat 30%/15% - removes
        # exactly the ramp while KEEPING the many good steady samples a fixed
        # fraction would throw away at low QPS: a 40-req long-decode/MM point with
        # C~2 keeps ~36 steady (not ~22), so it clears MIN_STEADY reliably instead
        # of spuriously failing TOO-FEW near its knee. At high QPS C is large and
        # correctly removes a big ramp (there are plenty of samples left). Capped
        # at the old fractions so a SATURATED point (C>n) still yields a middle
        # slice whose achieved << arrival exposes the saturation. Sorted by
        # completion time so the trim is on the completion order (the fill dead
        # zone - send->first-completion - is excluded by construction).
        done_sorted = sorted(done, key=lambda r: r["done"])
        n_done = len(done_sorted)
        mean_lat_s = sum((r["done"] - r["send"]) for r in done_sorted) / n_done
        conc = max(1, int(round(qps * mean_lat_s)))          # Little's Law in-flight
        fill = min(conc, int(self.FILL_FRAC * n_done))       # cap fill drop at 30%
        drain = min(conc, int(self.DRAIN_FRAC * n_done))     # cap drain drop at 15%
        steady = done_sorted[fill: n_done - drain] if (n_done - drain) > fill else []
        if len(steady) < 2:
            # Report the real completed count (>=1 here); only the steady window
            # is too thin to trust, which MIN_STEADY (steady_n=0) already fails.
            return _empty(len(recs) - len(done), completed=len(done))

        # Achieved = completion rate WITHIN the steady set: (n-1) inter-completion
        # intervals over their own span. Tracks arrival when the server keeps up;
        # falls to server-max (<< arrival) when saturated.
        sd = [r["done"] for r in steady]
        achieved = (len(sd) - 1) / max(1e-6, sd[-1] - sd[0])
        ok_slo = [
            r for r in steady
            if r["ttft"] is not None and r["ttft"] <= self.ttft_threshold_ms
            and (r["tpot"] is None or r["tpot"] <= self.tpot_threshold_ms)
        ]
        slo_attain = len(ok_slo) / len(steady)

        def _p99(vals):
            # Proper (interpolated) P99 via numpy. The old index hack
            # vals[min(int(n*0.99), n-1)] returns the MAX for every n<=100, so at
            # low QPS (steady_n ~15-55) the "P99 TTFT/ITL" gate degenerated to a
            # MAX gate - one slow prefill failed the whole rate point and the knee
            # flipped run-to-run.
            vals = [v for v in vals if v is not None]
            return float(np.percentile(vals, 99)) if vals else 0.0

        # P99 ITL/TBT: the DECODE-CADENCE tail gate. Pool every inter-token gap
        # across the steady set (workers already subsampled each request to
        # ITL_SAMPLE_CAP) and take its P99 - this catches per-token stalls that
        # per-request-mean TPOT smooths over. This is what the ITL SLO gates on.
        steady_itls = []
        for r in steady:
            gaps = r.get("itls")
            if gaps:
                steady_itls.extend(gaps)
        p99_itl = _p99(steady_itls) if steady_itls else 0.0
        # Mean end-to-end latency over the steady set (done-send), for Little's-Law
        # N_users = arrival_rate * mean_e2e at the knee (concurrent-users capacity).
        mean_e2e_ms = float(np.mean([(r["done"] - r["send"]) * 1000.0
                                     for r in steady]))

        return {
            "achieved_qps": achieved,
            "arrival_qps": qps,
            "actual_send_qps": actual_send_qps,
            "client_bound": client_bound,
            "slo_attainment": slo_attain,
            "p99_ttft_ms": _p99([r["ttft"] for r in steady]),
            "p99_tpot_ms": _p99([r["tpot"] for r in steady]),
            "p99_itl_ms": p99_itl,
            "mean_e2e_ms": mean_e2e_ms,
            "completed": len(done),
            "failed": len(recs) - len(done),
            "steady_n": len(steady),
        }

    def _warmup(self, model, format) -> None:
        """A few SEQUENTIAL (concurrency-1) discarded requests to warm the engine.

        Warms text + MM CUDA graphs before the sweep. Sent strictly one-at-a-time,
        NOT the old open-loop 20-reqs-at-1-req/s pass: on a slow single-stream server
        that over-capacity warmup floods the backend's request queue, and the backlog
        then starves the preflight probe and the first sweep points (spurious
        no-stream / TOO-FEW). Sequential warming builds no queue on any hardware.
        """
        try:
            api_url, content_field, payloads = self._build_payloads(
                model, format, self.WARMUP_PROMPTS_GENTLE)
        except Exception as e:
            logger.warning(f"Warmup skipped (payload build failed: {e}).")
            return
        if not payloads:
            return
        logger.info(f"Warmup ({len(payloads)} sequential reqs, discarded)...")
        for p in payloads:
            try:
                _stress_client_worker({
                    "payloads": [p], "qps": 0.0, "seed": 83,
                    "api_url": api_url, "content_field": content_field,
                    "request_timeout": self.REQUEST_TIMEOUT_S,
                    "max_point_s": self.MAX_POINT_S, "itl_cap": self.ITL_SAMPLE_CAP,
                })
            except Exception as e:
                logger.warning(f"Warmup request failed (continuing): {e}")
                return

    def _test_rate(self, qps: float, model, format) -> tuple:
        """Measure one open-loop arrival rate at steady state.

        Returns (passed, p99_ttft_ms, p99_tpot_ms, achieved_qps). A rate is
        SUSTAINABLE iff the server KEEPS UP (achieved >= KEEP_UP_FRAC * arrival -
        the queue isn't growing) AND >= SLO_ATTAINMENT of steady-state requests
        meet BOTH the TTFT and TPOT SLOs. This replaces the old latency-P99-only
        check, which reported arrival-rate knees the server couldn't service
        (e.g. 72 "req/s" when it actually completed ~21).
        """
        if qps in self._tested_rates:
            c = self._tested_rates[qps]
            return c["passed"], c["p99_ttft_ms"], c["p99_tpot_ms"], c["achieved_qps"]

        # Size the point to ~MEASURE_ARRIVAL_S of arrivals so the steady middle
        # has enough samples; clamp so slow campaigns aren't tiny and fast ones
        # don't send unboundedly.
        num_prompts = int(min(max(self._min_prompts, round(qps * self.MEASURE_ARRIVAL_S)), self.max_prompts))

        r = self._measure_rate_point(model, format, qps, num_prompts)

        achieved = float(r.get("achieved_qps", 0.0) or 0.0)
        attain = float(r.get("slo_attainment", 0.0) or 0.0)
        p99_ttft = float(r.get("p99_ttft_ms", 0.0) or 0.0)
        p99_tpot = float(r.get("p99_tpot_ms", 0.0) or 0.0)
        p99_itl = float(r.get("p99_itl_ms", 0.0) or 0.0)
        _srv = r.get("p99_itl_server_ms")   # None when /metrics is absent
        p99_itl_server = float(_srv) if _srv is not None else None
        mean_e2e = float(r.get("mean_e2e_ms", 0.0) or 0.0)
        steady_n = int(r.get("steady_n", 0) or 0)
        completed = int(r.get("completed", 0) or 0)
        failed = int(r.get("failed", 0) or 0)
        client_bound = bool(r.get("client_bound", False))
        actual_send = float(r.get("actual_send_qps", 0.0) or 0.0)
        sched_lag = float(r.get("client_sched_lag_s", 0.0) or 0.0)

        keep_up = achieved >= self.KEEP_UP_FRAC * qps
        # Decode-cadence SLO gates on the P99 ITL TAIL (per-token stalls). Prefer
        # the SERVER's own P99 ITL (from /metrics) = TRUE decode smoothness; the
        # client-measured P99 ITL is confounded by the multiproc-async client's SSE
        # read scheduling under bursty load and under-reports the knee (~7x for MM:
        # client 195-1193ms vs server ~25ms). Fall back to client ITL, then to the
        # per-request-mean TPOT (single-token outputs). TTFT gates on its own P99.
        if p99_itl_server is not None and p99_itl_server > 0:
            itl_metric = p99_itl_server
            itl_source = "server"
        elif p99_itl > 0:
            itl_metric = p99_itl
            itl_source = "client"
        else:
            itl_metric = p99_tpot
            itl_source = "tpot"
        slo_ok = (p99_ttft <= self.ttft_threshold_ms) and (itl_metric <= self.tpot_threshold_ms)
        enough = steady_n >= self._min_steady
        # Failure gate: a point where a non-trivial fraction of OFFERED load
        # errored/dropped is NOT sustainable. Failures never lower achieved or
        # slo_attainment (both computed over successful requests only), so without
        # this gate a partially-failing point passes and inflates the knee.
        offered = completed + failed
        low_failure = (failed / offered) <= self.MAX_FAILURE_FRAC if offered > 0 else True
        # A CLIENT-BOUND point is inconclusive - the load generator, not the server,
        # was the bottleneck - so it must NOT count as a sustainable server-capacity
        # result, or the reported knee reflects the client's max send rate rather
        # than server capacity. Excluding it makes the search move DOWN (conservative:
        # under-, never over-reports), and it is surfaced prominently in the log.
        passed = keep_up and slo_ok and enough and low_failure and (not client_bound)

        self._tested_rates[qps] = {
            "request_rate_qps": qps,
            "arrival_qps": qps,
            "num_prompts": num_prompts,
            "num_client_procs": int(r.get("num_client_procs", 1)),  # ACTUAL procs used
            "achieved_qps": achieved,
            "actual_send_qps": actual_send,
            "client_sched_lag_s": sched_lag,
            "client_bound": client_bound,
            "request_throughput": achieved,   # back-compat alias for run()/reports
            "slo_attainment": attain,
            "completed": completed,
            "failed": failed,
            "p99_ttft_ms": p99_ttft,
            "p99_tpot_ms": p99_tpot,
            "p99_itl_ms": p99_itl,                    # client-measured (confounded)
            "p99_itl_server_ms": p99_itl_server,      # server /metrics (None if absent)
            "itl_slo_source": itl_source,             # which ITL gated this point
            "mean_e2e_ms": mean_e2e,
            "steady_n": steady_n,
            "passed": passed,
        }

        if not enough:
            status = "⚠ TOO-FEW"       # not enough steady completions to trust
        elif not low_failure:
            status = "⚠ FAILURES"      # too many offered requests errored/dropped
        elif client_bound:
            status = "⚑ CLIENT-BOUND"  # load-gen bottleneck: inconclusive, NOT a pass
        elif not keep_up:
            status = "⚠ SATURATED"     # server can't keep up (queue growing)
        elif not slo_ok:
            status = "⚠ SLO-MISS"      # keeps up but latency SLO not met
        else:
            status = "✓ PASSED"
        logger.info(
            f"{qps:>7.1f} req/s   ach {achieved:>6.1f}   TTFT {p99_ttft:>7.0f}ms   "
            f"ITL {itl_metric:>6.0f}ms ({itl_source})   SLO {attain*100:>4.0f}%   {status}"
        )
        return passed, p99_ttft, p99_tpot, achieved

    def _sweep_start_qps(self) -> float:
        """Starting arrival rate for the ascending sweep.

        Seed it near HALF the single-stream capacity implied by the preflight's e2e
        latency (capacity ~= 1000/e2e_ms), so it starts comfortably INSIDE the
        sustainable zone on a slow server (few requests -> the start almost always
        passes, avoiding the downward fallback) while still being high enough to reach
        a fast server's knee in a few doublings. Falls back to initial_qps when no
        probe latency is available."""
        e2e_ms = getattr(self, "_probe_e2e_ms", None)
        if e2e_ms and e2e_ms > 0:
            start = 0.5 * (1000.0 / e2e_ms)   # half the single-stream capacity
            return float(min(self.max_qps, max(self.MIN_DOWN_QPS, start)))
        return float(min(self.max_qps, self.initial_qps))

    def _sweep_once(self, model, format) -> dict:
        """One ASCENDING open-loop QPS sweep against an already-up, warm server.

        Probes LOW -> HIGH and STOPS at the first rate that misses the SLO, so every
        sustainable rate is measured BEFORE the server is ever pushed over capacity.
        This is what makes a slow single-stream server measurable: an over-capacity
        probe floods the backend's request queue, and a client-side timeout/cancel
        does NOT cancel that already-queued work - so a descending sweep's early
        over-rate points leave a backlog that makes every subsequent lower point read
        a spurious TOO-FEW. Ascending avoids that by construction (the only over-rate
        probe is the last one, and nothing is probed after it). The knee is the last
        passing rate; the SLO crossing is interpolated into the final [pass, fail]
        bracket. Start is seeded from the preflight latency (see _sweep_start_qps)."""
        self._tested_rates = {}
        start = round(self._sweep_start_qps(), 3)
        last_pass = 0.0
        first_fail = None

        passed, *_ = self._test_rate(start, model, format)
        if passed:
            last_pass = start
            # Ramp UP x2 until the first SLO miss, then STOP (never probe higher, so
            # the server is never flooded beyond the one boundary point).
            nxt = round(last_pass * 2, 3)
            while nxt <= self.max_qps:
                p, *_ = self._test_rate(nxt, model, format)
                if not p:
                    first_fail = nxt
                    break
                last_pass = nxt
                nxt = round(last_pass * 2, 3)
            else:
                logger.info(f"Reached safety cap {self.max_qps} req/s - server handles extreme load.")
        else:
            # Seeded start already missed the SLO (server slower than the single-stream
            # estimate implied). Ramp DOWN by halving to find its small sustainable
            # rate. This fallback probes one over-rate first; seeding at half-capacity
            # keeps it rare.
            first_fail = start
            probe = round(start / 2, 3)
            while probe >= self.MIN_DOWN_QPS:
                p, *_ = self._test_rate(probe, model, format)
                if p:
                    last_pass = probe
                    break
                first_fail = probe
                probe = round(probe / 2, 3)

        knee = last_pass
        best = self._tested_rates.get(knee, {})
        # Interpolate the crossing inside the final [pass, fail] bracket, but only
        # against a failing point with a REAL steady window - a TOO-FEW failure's P99
        # is a handful of samples (~= max) and would pull the interp toward noise.
        knee_interp = knee
        if (first_fail is not None and knee > 0 and first_fail in self._tested_rates
                and int(self._tested_rates[first_fail].get("steady_n", 0) or 0) >= self._min_steady):
            knee_interp = _interpolate_knee(
                self._tested_rates[knee], self._tested_rates[first_fail],
                self.ttft_threshold_ms, self.tpot_threshold_ms, self.KEEP_UP_FRAC,
            )
        return {
            "knee_qps": knee,
            "knee_qps_interp": knee_interp,
            "achieved_at_knee": best.get("achieved_qps", 0.0),
            "best": best,
            "tested_rates": sorted(self._tested_rates.values(),
                                   key=lambda r: r["request_rate_qps"]),
        }

    def _save_stress_result(self, final_result: dict, model, format, multimodal: bool) -> None:
        """Persist a stress result to results/performance/stress_<model>_<fmt>_<mode>.json."""
        lm = self.config.log_manager
        results_dir = lm.results_dir / "performance"
        results_dir.mkdir(parents=True, exist_ok=True)
        mode_suffix = "mm" if multimodal else "text"
        output_file = results_dir / f"stress_{model.short_name}_{format.value}_{mode_suffix}.json"
        with open(output_file, "w") as f:
            json.dump(final_result, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_file}")

    def _preflight_floor_check(self, model, format, multimodal: bool) -> Optional[dict]:
        """Fast-fail probe run BEFORE the open-loop sweep on slow hardware.

        The sweep is an OPEN-LOOP arrival test: arrivals fire on a schedule
        regardless of whether the server keeps up, so queueing only ADDS to the
        single-stream latency. If even ONE request at a time already exceeds the
        P99 TTFT SLO, no arrival rate can be sustainable - the whole descending
        ramp would just report TOO-FEW for tens of minutes. So send a few STRICTLY
        SEQUENTIAL requests, take the best-case (min) single-stream TTFT, and if it
        already blows the SLO, return an early-exit result (sustainable QPS = 0).
        Return None to proceed with the normal sweep (the common case; on capable
        hardware the probe is a handful of quick requests).
        """
        try:
            api_url, content_field, payloads = self._build_payloads(
                model, format, self.PREFLIGHT_PROBES)
        except Exception as e:
            logger.warning(f"Stress preflight skipped (payload build failed: {e}); running sweep.")
            return None
        if not payloads:
            return None
        ttfts, e2es = [], []
        for p in payloads[:self.PREFLIGHT_PROBES]:
            # One payload => the worker sends exactly one request and awaits it:
            # a true single-stream (concurrency-1) latency sample. qps is irrelevant
            # for a single request (no inter-arrival wait is taken).
            try:
                out = _stress_client_worker({
                    "payloads": [p], "qps": 0.0, "seed": 83,
                    "api_url": api_url, "content_field": content_field,
                    "request_timeout": self.REQUEST_TIMEOUT_S,
                    "max_point_s": self.MAX_POINT_S, "itl_cap": self.ITL_SAMPLE_CAP,
                })
            except Exception as e:
                logger.warning(f"Stress preflight probe errored ({e}); running sweep.")
                return None
            for rec in out.get("records", []):
                if rec.get("ok") and rec.get("ttft") is not None:
                    ttfts.append(float(rec["ttft"]))
                    if rec.get("done") and rec.get("send"):
                        e2es.append((rec["done"] - rec["send"]) * 1000.0)
        if not ttfts:
            # Not one clean single-stream request produced output, even after warmup.
            # The sweep would then report 0 completions / infinite TTFT for every rate
            # (a doomed ~20-min grind), so bail now with an honest diagnostic instead.
            logger.warning(
                "\n" + "=" * 78 + "\n"
                "  STRESS TEST SKIPPED - endpoint returned no usable stream\n"
                + "=" * 78 + "\n"
                "  No single-stream probe produced any output tokens (after warmup), so the\n"
                "  open-loop sweep cannot measure capacity. Likely the endpoint does not\n"
                "  stream chat/completions as expected, or the model emits an immediate EOS\n"
                "  for this workload. Check the endpoint, or use --no-stress-test.\n"
                + "=" * 78)
            _ds, _in, _out = self._workload_shape()
            return {
                "stress_test": True,
                "below_stress_floor": True,
                "preflight_no_stream": True,
                "slo_met": False,
                "max_sustainable_qps": 0.0,
                "max_sustainable_throughput": 0.0,
                "preflight_probes": 0,
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "itl_threshold_ms": self.tpot_threshold_ms,
                "tpot_threshold_ms": self.tpot_threshold_ms,
                "reps": 0,
                "workload_dataset": "multimodal" if multimodal else _ds,
                "workload_input_length": _in,
                "workload_output_length": _out,
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
                "note": ("stress requests produced no output tokens against this endpoint "
                         "(0 completions); the sweep cannot measure capacity. Verify the "
                         "endpoint streams chat/completions, or use --no-stress-test."),
            }
        best_ttft = min(ttfts)
        # Stash the single-stream latency so the ascending sweep can seed its start
        # rate near half the implied capacity (1000/e2e), on both slow and fast servers.
        self._probe_ttft_ms = best_ttft
        if e2es:
            self._probe_e2e_ms = min(e2es)
        logger.info(
            f"Stress preflight: best single-stream TTFT {best_ttft:.0f}ms over "
            f"{len(ttfts)} probe(s) vs P99 TTFT SLO {self.ttft_threshold_ms}ms.")
        if best_ttft <= self.ttft_threshold_ms:
            return None   # single stream can meet the SLO -> the sweep is meaningful
        # Below the floor: no open-loop rate can pass. Bail with an honest result.
        logger.warning(
            "\n" + "=" * 78 + "\n"
            f"  STRESS TEST SKIPPED - hardware below the stress floor\n"
            + "=" * 78 + "\n"
            f"  A single request already takes {best_ttft:.0f}ms to first token, above the\n"
            f"  P99 TTFT SLO of {self.ttft_threshold_ms}ms. Open-loop arrivals only add queueing,\n"
            f"  so the sustainable QPS is ~0 and the sweep would only report TOO-FEW.\n"
            f"  -> Use --no-stress-test on this machine (the serving latency sweep already\n"
            f"     captures its single-stream TTFT/TPOT), or raise the SLO with\n"
            f"     --stress-threshold <ms> if you want a capacity number at this latency.\n"
            + "=" * 78)
        _ds, _in, _out = self._workload_shape()
        return {
            "stress_test": True,
            "below_stress_floor": True,
            "slo_met": False,
            "max_sustainable_qps": 0.0,
            "max_sustainable_throughput": 0.0,
            "preflight_single_stream_ttft_ms": best_ttft,
            "preflight_single_stream_e2e_ms": (min(e2es) if e2es else None),
            "preflight_probes": len(ttfts),
            "ttft_threshold_ms": self.ttft_threshold_ms,
            "itl_threshold_ms": self.tpot_threshold_ms,
            "tpot_threshold_ms": self.tpot_threshold_ms,
            "reps": 0,
            "workload_dataset": "multimodal" if multimodal else _ds,
            "workload_input_length": _in,
            "workload_output_length": _out,
            "multimodal": multimodal,
            "model": model.short_name,
            "format": format.value,
            "note": (
                f"single-stream TTFT {best_ttft:.0f}ms exceeds the {self.ttft_threshold_ms}ms "
                f"P99 TTFT SLO; sustainable open-loop QPS is ~0 on this hardware. Use "
                f"--no-stress-test, or raise --stress-threshold to get a capacity number."),
        }

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> dict:
        """Sweep the open-loop arrival rate to find max sustainable QPS under SLO.

        Args:
            model: Model configuration
            format: Model format (hf, gguf)
            multimodal: Whether to test multimodal (sends real images)

        Returns:
            Dictionary with stress test results
        """
        mode_str = "Multimodal" if multimodal else "Text"
        self._tested_rates = {}  # Reset cache
        self._multimodal = multimodal
        # Per-image token count for this model (model-specific; gemma-4 default 280).
        self._vision_tokens_per_image = (
            getattr(model, "vision_tokens_per_image", None) or self.PER_IMAGE_SOFT_TOKENS)

        if self.config.dry_run:
            logger.info("[DRY RUN] Would run open-loop QPS stress sweep")
            return {
                "dry_run": True,
                "stress_test": True,
                "model": model.short_name,
                "format": format.value,
                "multimodal": multimodal,
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "tpot_threshold_ms": self.tpot_threshold_ms,
            }

        try:
            # Set up multimodal images if needed
            mm_media_path = None
            if multimodal:
                mm_media_path = self._setup_mm_images(num_images=50)

            # Start vLLM server once (reused across all rate points)
            self._serving_runner = ServingBenchmarkRunner(self.config)
            if not self.config.remote_endpoint:
                if not self._serving_runner._start_server(
                    model, format, allowed_local_media_path=mm_media_path,
                    mm_limit_images=self._images_per_request() if multimodal else 1,
                ):
                    # A server that never comes ready must fail the campaign, not
                    # silently sweep a dead endpoint and record a phantom 0-QPS
                    # "measurement". The raise is caught below -> failed dict.
                    raise RuntimeError("Failed to start vLLM server for stress test")

            self._api_model_id = self._serving_runner._resolve_model_id(model, format)
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            # Tokenizer is needed for text request generation AND for sizing the
            # MM padded-text prompt, so load it in both modes.
            from gbench.utils import safe_get_tokenizer
            self._tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            logger.info(f"Tokenizer loaded safely for: {self.config.tokenizer or tokenizer_path}")

            # Preflight: prove the server ingests images before sweeping, so an MM
            # stress knee can never be measured on silently-text-only requests.
            if multimodal:
                self._serving_runner._mm_image_paths = self._mm_image_paths
                base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self._serving_runner.server_port}"
                measured_per_img = self._serving_runner._assert_mm_images_processed(
                    self._api_model_id, base_url, per_img_tokens=self._vision_tokens_per_image)
                # Size + report the MM workload against the SERVER-MEASURED per-image
                # soft-token count, not the 280 class default. The deployed gemma-4
                # server charges 1120 tok/img (--hf-overrides), so the old default
                # under-reported the effective MM prefill ~3.7x (128+4*280=1248 vs
                # the true 128+4*1120=4608) and mis-sized the padded text budget.
                if measured_per_img and measured_per_img >= self.MIN_MM_TEXT_TOKENS:
                    if int(measured_per_img) != int(self._vision_tokens_per_image):
                        logger.info(
                            f"MM stress: using server-measured "
                            f"{int(measured_per_img)} vision tokens/image "
                            f"(default was {int(self._vision_tokens_per_image)}).")
                    self._vision_tokens_per_image = int(measured_per_img)

            logger.info(
                f"\n{'='*66}\n"
                f"  STRESS TEST - open-loop QPS sweep (capacity under SLO)\n"
                f"{'='*66}\n"
                f"  Mode:      {mode_str}\n"
                f"  Model:     {model.short_name} ({format.value})\n"
                f"  Workload:  {self._workload_desc()}\n"
                f"  Sustainable: server keeps up (achieved >= {self.KEEP_UP_FRAC:.0%} of arrival) AND\n"
                f"               P99 TTFT <= {self.ttft_threshold_ms}ms AND P99 ITL <= {self.tpot_threshold_ms}ms,\n"
                f"               measured at steady state; reported over {self._reps} reps (knee mean+CI).\n"
                f"  Client:    up to {self._num_client_procs} load-gen processes (multi-proc; 1 at low QPS or if spawn is unavailable)\n"
                f"  Arrivals:  Poisson open-loop; ASCENDING ramp (stop at first SLO miss), cap {self.max_qps} req/s\n"
                f"{'='*66}"
            )
            # Warmup once (discarded) so the first measured point isn't cold; the
            # server is reused across every rep, so one warmup suffices.
            self._warmup(model, format)

            # Fast-fail floor check, AFTER warmup so it probes a warm (not cold-loading)
            # model: if even a single stream can't meet the TTFT SLO, no open-loop
            # arrival rate can (arrivals only add queueing), so bail with a clear result
            # instead of grinding every rep's descending ramp into TOO-FEW.
            pf = self._preflight_floor_check(model, format, multimodal)
            if pf is not None:
                self._save_stress_result(pf, model, format, multimodal)
                return pf

            # Run the whole sweep STRESS_REPS times with varied (deterministic)
            # seeds so the knee is reported as a distribution + CI, not a single
            # run-to-run-variable point.
            reps = max(1, self._reps)
            rep_results, knees, knees_interp = [], [], []
            all_best = None
            for rep in range(reps):
                self._rep_seed = 83 + rep * 101   # varied, deterministic per rep
                logger.info("=" * 78)
                logger.info(f"Sweep rep {rep + 1}/{reps} (seed base {self._rep_seed}, "
                            f"{self._num_client_procs} client procs)")
                logger.info(
                    f"{'Rate':>9}   {'Achieved':>9}   {'P99 TTFT':>9}   "
                    f"{'P99 ITL':>9}   {'%SLO':>5}   Status"
                )
                logger.info("-" * 78)
                sw = self._sweep_once(model, format)
                rep_results.append(sw)
                knees.append(sw["knee_qps"])
                knees_interp.append(sw["knee_qps_interp"])
                if sw["best"]:
                    all_best = sw["best"]
                logger.info(
                    f"  rep {rep + 1}: knee {sw['knee_qps']:.2f} req/s "
                    f"(interp {sw['knee_qps_interp']:.2f}); "
                    f"achieved {sw['achieved_at_knee']:.2f}"
                )

            kstats = _knee_stats(knees)
            kstats_interp = _knee_stats(knees_interp)
            # Headline = MEDIAN knee (robust to one outlier rep); mean/CI reported too.
            max_qps = kstats.get("median", knees[0] if knees else 0.0)
            # Secondary metrics (P99s, mean_e2e, N_users) MUST come from the SAME rep
            # as the reported (median) knee - otherwise little_law_n_users mixes the
            # median knee with the LAST rep's mean_e2e, and the tail-latency headline
            # is a different rep than the throughput headline (inconsistent by up to
            # the per-rep CV, ~16%). Pick the rep whose knee is closest to the median.
            if rep_results:
                med_idx = min(range(len(knees)), key=lambda i: abs(knees[i] - max_qps))
                best = rep_results[med_idx].get("best") or all_best or {}
            else:
                best = all_best or {}
            max_tp = best.get("request_throughput", 0.0)

            logger.info("-" * 78)
            logger.info(
                f"✅ Max sustainable arrival rate (goodput) over {reps} reps: "
                f"{max_qps:.2f} req/s (median; mean {kstats.get('mean', max_qps):.2f} "
                f"[{kstats.get('ci_low', max_qps):.2f}, {kstats.get('ci_high', max_qps):.2f}], "
                f"CV {kstats.get('cv_percent', 0.0):.1f}%, reps {knees})"
            )

            _ds, _in, _out = self._workload_shape()
            # For MM, the reported input length is the ACTUAL prefill (clamped text
            # + image placeholder tokens), not the campaign's nominal text input -
            # otherwise small-input campaigns (chat-like/decode-heavy in=128) are
            # labeled 128 while the server actually prefills ~1136 tokens/req.
            _in_reported = self._mm_effective_input_len(_in, _out) if multimodal else _in
            # Report the ACTUAL max client workers observed across the sweep, not the
            # configured cap: low-QPS points intentionally use 1, and if 'spawn' is ever
            # blocked the run falls back to a single in-process client. engaged=False means
            # multi-proc never ran (so the number is not a contention-free measurement).
            _all_points = [pt for rr in rep_results for pt in rr.get("tested_rates", [])]
            actual_max_procs = max((int(pt.get("num_client_procs", 1)) for pt in _all_points),
                                   default=1)
            multiproc_engaged = actual_max_procs > 1
            final_result = {
                "stress_test": True,
                "method": ("open_loop_steady_state_goodput_multiproc" if multiproc_engaged
                           else "open_loop_steady_state_goodput_singleproc"),
                "reps": reps,
                "num_client_procs": actual_max_procs,            # ACTUAL max workers used
                "num_client_procs_configured": self._num_client_procs,
                "client_multiproc_engaged": multiproc_engaged,
                "keep_up_frac": self.KEEP_UP_FRAC,
                "client_keepup_frac": self.CLIENT_KEEPUP_FRAC,
                "slo_attainment_target": self.SLO_ATTAINMENT,
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "itl_threshold_ms": self.tpot_threshold_ms,
                "tpot_threshold_ms": self.tpot_threshold_ms,  # back-compat alias
                "workload_dataset": "multimodal" if multimodal else _ds,
                "workload_input_length": _in_reported,
                "workload_input_length_nominal": _in,
                "workload_mm_images": self._images_per_request() if multimodal else 0,
                "workload_output_length": _out,
                "max_sustainable_qps": max_qps,
                "max_sustainable_qps_stats": kstats,
                "max_sustainable_qps_interp_stats": kstats_interp,
                "max_sustainable_throughput": max_tp,
                "per_rep_knees": knees,
                "per_rep_knees_interp": knees_interp,
                "sweep_points": rep_results[0]["tested_rates"] if rep_results else [],
                "sweep_points_all_reps": [rr["tested_rates"] for rr in rep_results],
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
                "slo_met": max_qps > 0,
            }
            if best:
                final_result["p99_ttft_ms"] = best.get("p99_ttft_ms", 0)
                final_result["p99_tpot_ms"] = best.get("p99_tpot_ms", 0)
                final_result["p99_itl_ms"] = best.get("p99_itl_ms", 0)   # client-measured
                # Server-side P99 ITL (from /metrics) that actually gated the knee, and
                # which ITL source was used. None/"client" when /metrics was absent.
                final_result["p99_itl_server_ms"] = best.get("p99_itl_server_ms")
                final_result["itl_slo_source"] = best.get("itl_slo_source", "client")
                final_result["mean_e2e_ms"] = best.get("mean_e2e_ms", 0)
                final_result["slo_attainment"] = best.get("slo_attainment", 0)
                final_result["request_throughput"] = best.get("request_throughput", 0)
                # Little's Law: concurrent-users capacity ≈ arrival_rate * mean_e2e
                # at the knee (the closed-loop equivalent of the open-loop knee).
                mean_e2e_s = float(best.get("mean_e2e_ms", 0.0)) / 1000.0
                final_result["little_law_n_users"] = (
                    max_qps * mean_e2e_s if mean_e2e_s > 0 else 0.0)

            # Save result
            self._save_stress_result(final_result, model, format, multimodal)

            return final_result

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stress_test": True,
                "failed": True,
                "error": str(e),
                # Carry the mode so cli.py stamps benchmark_type correctly
                # (stress_test_multimodal vs stress_test). Without it a FAILED
                # MM stress is mis-stamped as a text failure, which the
                # exit-code guard would treat as fatal.
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
            }

        finally:
            # Tear down the reusable client pool before the server so no worker
            # processes linger between campaigns.
            self._shutdown_client_pool()
            if self._serving_runner and not self.config.remote_endpoint:
                self._serving_runner._cleanup_server()
            # Clean up MM images
            if multimodal:
                self._cleanup_mm_images()

    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        include_text: bool = True,
        include_multimodal: bool = True,
    ) -> list[dict]:
        """Run stress tests - text and/or multimodal per the caller's gates.

        Both use the open-loop QPS sweep; text drives vLLM's benchmark engine,
        multimodal sends image+text chat requests with Poisson arrivals.

        Args:
            include_text: run the text stress pass (gated off by --multimodal-only).
            include_multimodal: run the multimodal pass - but only for MM-capable
                models (gated off by --text-only).
        """
        results = []

        if include_text:
            logger.info("Running text stress test...")
            text_result = self.run(model, format, multimodal=False)
            results.append(text_result)

        # Image-based MM stress requires VISION -> gate on supports_multimodal
        # (has_vision), not category==MULTIMODAL (which is vision OR audio).
        if include_multimodal and model.supports_multimodal:
            logger.info("Running multimodal stress test...")
            mm_result = self.run(model, format, multimodal=True)
            results.append(mm_result)

        return results
