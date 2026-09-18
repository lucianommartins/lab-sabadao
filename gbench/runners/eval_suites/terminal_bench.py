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

"""Official Terminal-Bench 2.1 evaluation suite via Harbor framework & Docker sandboxes.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_TERMINAL_BENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.

Sharding (`--shard I/N`, transported as `GBENCH_SHARD`): this suite bypasses
`base.run_eval_suite`, so it applies the shard itself instead of inheriting the native
partition. Harbor's `harbor run` supports `--include-task-name` (an fnmatch filter over each
task's `org/name`, repeatable), so the shard IS honored: the runner enumerates the dataset's
full task-id list via Harbor's own package resolver (`PackageDatasetClient.get_dataset_metadata`,
the exact names `--include-task-name` matches at run time), sorts it deterministically, selects
shard `I` of `N` by round-robin (`sorted[I-1::N]`, matching `sampling.shard_select`), and passes
each selected task name as `--include-task-name` (so `--n-tasks` is NOT used in shard mode). The
`--eval-limit` cap is composed AFTER the shard (limit caps within the shard), matching the native
order. A sharded run covers a non-overlapping subset by construction, so `total_questions` reflects
the shard size (missing/errored tasks still count as 0, no-skip) and `leaderboard_comparable` stays
False. If the registry cannot be reached to enumerate the task list, the run hard-errors rather than
silently running the full set (no fabricated shard coverage).
"""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import repetition_onset, repetition_run, resolve_temperature
from .sampling import shard_select, stratified_sample
from .swe_thread_cap import THREAD_VARS
from .swebench_common import infra_required
from . import terminus_salvage_patch

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=0, desc="", **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def refresh(self):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/terminal_bench.md"
DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"



#: Cap on the transcript kept per trial. Enough to see a cycle without carrying megabytes
#: of terminal output into every result file.
_MAX_TRANSCRIPT_CHARS = 40000

#: Per-turn output cap handed to the agent's LLM calls. Terminus-2 and LiteLLM both leave
#: `max_tokens` unset, so vLLM falls back to `max_model_len - prompt` and a single turn can
#: consume the entire agent budget - ~250k tokens, 56 minutes at the rate measured here.
#:
#: Sized from the TAIL of the observed distribution, not near it. The first attempt used
#: 8192, which was 7% above the largest turn seen at the time (7,644) and livelocked 2 of 3
#: trials on 2026-08-18: a truncated turn is re-asked, truncates again, and no step is ever
#: recorded, so the trial burns to its agent timeout having made zero progress. Measured
#: per-turn tokens over the one trial that stayed healthy (22 turns): median 1,665, p90
#: 3,413, max 7,644. 32768 is 4.3x the observed max, so it binds only the pathological tail
#: while still bounding a turn to ~7 minutes rather than ~56.
#:
#: The cap is a BACKSTOP, not a schedule - the median turn is 1,665 tokens and never sees
#: it. Set `GBENCH_TB_TURN_MAX_TOKENS=0` to restore the uncapped behaviour.
_TURN_MAX_TOKENS_THINKING = 32768
_TURN_MAX_TOKENS_PLAIN = 8192

#: Terminus-2 parses the model's action out of each turn. The `json` parser (its default)
#: has no `salvage_truncated_response`, so a turn that hits the cap is UNRECOVERABLE and
#: gets re-asked - the livelock above. `terminus_xml_plain_parser` implements salvage, so
#: the same truncated turn yields an action and the agent moves on. Overridable because it
#: changes the action format the model is asked to emit.
_PARSER = os.environ.get("GBENCH_TB_PARSER", "xml")

#: A turn that truncates and is re-asked, repeatedly, with no step recorded in between.
#: This is what a too-small cap looks like from the outside, and it is invisible in the
#: result unless counted: the trial simply reports `AgentTimeoutError`.
_LIVELOCK_TRUNCATIONS = 3

#: Threads to allow inside the task container.
#:
#: Terminal-Bench pins each container's CPU via `task.toml` - measured across the cached
#: dataset on 2026-08-18 (an EARLIER, larger task revision: 167 of 178 tasks declared
#: `cpus = 1`, 6 declared 2, 5 declared 4, none omitted it). The canonical DEFAULT_DATASET is
#: now terminal-bench-2-1 (89 tasks); the exact per-task distribution there has not been
#: re-measured, but `cpus = 1` was the modal AND minimum value, so a thread cap of 1 stays
#: correct-or-conservative regardless of the count. Docker enforces the pin as a cgroup quota
#: (`cpu.max` = "100000 100000"),
#: but nothing rewrites `/proc/cpuinfo`, so `nproc` inside a 1-CPU container still reports
#: the host's 96. Anything that sizes a thread pool off the visible core count therefore
#: spawns 96 threads to share one CPU's worth of quota - the same oversubscription
#: pathology that cost swebench 1800s per instance before `swe_thread_cap` (see the table
#: in that module).
#:
#: 1 is the modal AND minimum declared value, so it is right for 94% of tasks and merely
#: conservative for the rest - single-threaded is slower than a 4-CPU task could manage,
#: but nothing like 24:1 thrashing. A single value is required because Harbor takes agent
#: kwargs once per run, not per task.
#:
#: The CPU LIMIT itself is deliberately untouched: it is part of the benchmark's published
#: environment spec, and raising it would make these numbers incomparable with every other
#: Terminal-Bench result.
_CONTAINER_THREADS_DEFAULT = 1


def container_thread_env() -> Dict[str, str]:
    """Thread-count env for the task container, or {} when disabled.

    Shares `swe_thread_cap.THREAD_VARS` so the two harnesses cap the same runtimes.
    `GBENCH_TB_CONTAINER_THREADS=0` (or "off") restores the unbounded behaviour.
    """
    raw = (os.environ.get("GBENCH_TB_CONTAINER_THREADS") or "").strip()
    threads = _CONTAINER_THREADS_DEFAULT
    if raw:
        if raw.lower() in ("off", "none", "false"):
            return {}
        try:
            threads = int(raw)
        except ValueError:
            logger.warning(
                "GBENCH_TB_CONTAINER_THREADS=%r is not an integer; using %d.",
                raw, _CONTAINER_THREADS_DEFAULT)
    if threads <= 0:
        return {}
    return {var: str(threads) for var in THREAD_VARS}


def _turn_token_cap(enable_thinking: bool) -> int:
    """Per-turn `max_tokens` for the agent's LLM calls (0 disables the cap)."""
    raw = os.environ.get("GBENCH_TB_TURN_MAX_TOKENS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("GBENCH_TB_TURN_MAX_TOKENS=%r is not an integer; ignoring", raw)
    return _TURN_MAX_TOKENS_THINKING if enable_thinking else _TURN_MAX_TOKENS_PLAIN


def _timeout_multiplier(enable_thinking: bool) -> str:
    """Agent-timeout multiplier, overridable with `GBENCH_TB_TIMEOUT_MULTIPLIER`.

    Capping the turn is the lever that buys TURNS; this only buys wall-clock. Left at 4.0
    for thinking runs so a slow-but-progressing agent is not cut off, but exposed because
    it sets the run's worst case: 4.0 against the dataset's longest task is 13.3 hours.
    """
    raw = os.environ.get("GBENCH_TB_TIMEOUT_MULTIPLIER")
    if raw:
        try:
            return str(float(raw))
        except ValueError:
            logger.warning("GBENCH_TB_TIMEOUT_MULTIPLIER=%r is not a number; ignoring", raw)
    return "4.0" if enable_thinking else "2.0"


#: Shortest agent budget observed before the multiplier: 900s (the large majority of tasks sat
#: at 900s in the 2026-08-18 measurement, taken on an earlier/larger task revision - see the
#: _CONTAINER_THREADS note; the canonical terminal-bench-2-1 set is 89 tasks and its exact
#: distribution has not been re-measured). Used to express the turn cap as "how many turns
#: actually fit", the quantity that decides whether a run was winnable on THIS hardware.
_SHORTEST_TASK_AGENT_TIMEOUT_S = 900


def _turn_budget_report(trials: List[Dict[str, Any]], turn_cap: int,
                        enable_thinking: bool) -> Optional[Dict[str, Any]]:
    """What this endpoint's speed implies for the turn budget, and what it wasted.

    Derived from the trials themselves rather than from a constant, because the constants
    in this file were tuned on one 8xA100 deployment and are meaningless elsewhere.

    Both inputs need correcting for truncation first. Terminus appends to
    `api_request_times_msec` only on the SUCCESS path and Harbor's `n_output_tokens`
    likewise counts only completed turns, so a turn that hits the cap and raises
    `OutputLengthExceededError` is invisible in both. On the 2026-08-18 run that made a
    trial look 22% model-bound when it was ~99%: 4 truncated turns had silently burned
    ~131k tokens and ~1840s of the 2385s agent phase. Uncorrected, this report understates
    model time and OVERSTATES throughput on exactly the trials it exists to diagnose.

    A truncated turn generates the full cap by definition, so it can be priced exactly.
    """
    done_tokens = sum(t.get("output_tokens") or 0 for t in trials)
    done_seconds = sum((t.get("mean_turn_latency_s") or 0) * (t.get("agent_turns") or 0)
                       for t in trials)
    if not done_tokens or done_seconds <= 0:
        return None
    # Rate must come from COMPLETED turns only - that is the pairing we can trust.
    rate = done_tokens / done_seconds
    truncated = sum(t.get("truncated_turns") or 0 for t in trials)
    wasted_tokens = truncated * turn_cap if turn_cap else 0
    wasted_seconds = wasted_tokens / rate if rate else 0
    per_turn = turn_cap / rate if turn_cap and rate else None
    budget = _SHORTEST_TASK_AGENT_TIMEOUT_S * float(_timeout_multiplier(enable_thinking))
    total_tokens = done_tokens + wasted_tokens
    return {
        "observed_tok_s": round(rate, 1),
        "turn_max_tokens": turn_cap or None,
        "seconds_per_turn": round(per_turn, 1) if per_turn else None,
        "shortest_agent_budget_s": int(budget),
        "turns_in_shortest_budget": int(budget / per_turn) if per_turn else None,
        "observed_turns": [t.get("agent_turns") for t in trials],
        # Every token a truncated turn produced was discarded. This is the number that
        # says whether the cap is set anywhere near the right place.
        "truncated_turns": truncated,
        "tokens_useful": done_tokens,
        "tokens_discarded": wasted_tokens,
        "discarded_token_pct": (round(100.0 * wasted_tokens / total_tokens, 1)
                                if total_tokens else 0.0),
        "seconds_lost_to_truncation": int(wasted_seconds),
    }


@contextlib.contextmanager
def _jobs_dir():
    """Harbor's jobs directory, kept when the run produced nothing to show for itself.

    It used to be an unconditional `TemporaryDirectory`, which deleted Harbor's
    `result.json`, `trial.log` and `agent/trajectory.json` on the way out. On 2026-08-18
    that destroyed the only evidence of why a completed 2h40 job scored `total_questions:
    0`, and the same deletion had already made two `AgentTimeoutError` trials
    undiagnosable earlier the same day. Twice is a design fault, not bad luck.

    Kept only on failure so healthy runs still clean up after themselves; point
    `GBENCH_TB_KEEP_JOBS_DIR` at a path (or set it to 1) to always keep.
    """
    keep = (os.environ.get("GBENCH_TB_KEEP_JOBS_DIR") or "").strip()
    base = keep if (keep and keep not in ("1", "true", "yes")) else None
    path = tempfile.mkdtemp(prefix="gbench_tb_", dir=base)
    failed = False
    try:
        yield path
    except BaseException:
        failed = True
        raise
    finally:
        if keep or failed or not _JOBS_DIR_OK.get(path, False):
            logger.warning(
                "[terminal_bench] keeping Harbor's jobs dir for diagnosis: %s "
                "(result.json / trial.log / agent/trajectory.json live here). Delete it "
                "when done; set GBENCH_TB_KEEP_JOBS_DIR= to disable this.", path)
        else:
            shutil.rmtree(path, ignore_errors=True)
        _JOBS_DIR_OK.pop(path, None)


#: Set true by the parser once a jobs dir has yielded at least one usable trial, so the
#: directory is only preserved when there is something to investigate.
_JOBS_DIR_OK: Dict[str, bool] = {}


def _extract_reward(rdata: Dict[str, Any]) -> Optional[float]:
    """Canonical Harbor per-trial reward, or None when none is present.

    Mirrors Harbor's aggregation (metrics/base.py aggregate_reward_dicts): prefer the 'reward'
    key, else the SOLE value in the rewards dict (a task may name its single reward differently);
    coerce bool -> 1.0/0.0. gbench previously did rewards.get("reward", 0.0), which scored any
    non-'reward'-keyed task as a hard 0 (fail) that Harbor would have counted. Returning None (not
    0.0) keeps a missing reward distinguishable from a real 0.0 so it is never silently a pass/fail.
    """
    vr = rdata.get("verifier_result") or {}
    rewards = vr.get("rewards") if isinstance(vr, dict) else None
    val: Any = None
    if isinstance(rewards, dict) and rewards:
        val = rewards["reward"] if "reward" in rewards else next(iter(rewards.values()), None)
    elif "reward" in rdata:
        val = rdata.get("reward")
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _job_stats_totals(job_stats: Dict[str, Any]) -> Tuple[int, int]:
    """Trial and pass counts from Harbor's JOB-level result.json.

    This is the fallback for when per-trial parsing yields nothing - precisely the case
    that happened on 2026-08-18 - and it was reading three keys Harbor does not emit
    (`n_trials`, `n_passed`, `n_success`) plus a `mean_reward` that lives elsewhere. It
    therefore returned 0 and the run published "nothing was measured" over a completed
    2h40 job. Harbor 0.20.0 puts the totals at `n_completed_trials` and the score under
    `evals[<name>].metrics[0].mean`.
    """
    total = int(job_stats.get("n_completed_trials")
                or job_stats.get("n_trials") or 0)
    evals = job_stats.get("evals")
    means: List[float] = []
    if isinstance(evals, dict):
        for ev in evals.values():
            if not isinstance(ev, dict):
                continue
            if not total:
                total += int(ev.get("n_trials") or 0)
            for m in ev.get("metrics") or []:
                if isinstance(m, dict) and isinstance(m.get("mean"), (int, float)):
                    means.append(float(m["mean"]))
    correct = int(round(sum(means) / len(means) * total)) if (means and total) else 0
    return total, correct


def _truncation_livelock(trial_dir: str) -> Dict[str, Any]:
    """Consecutive truncated turns with no step recorded between them.

    A turn that hits the output cap is re-asked. If it truncates again, and again, the
    agent makes no progress and the trial dies as a bare `AgentTimeoutError` - the result
    looks identical to a hard task. On 2026-08-18 two trials sat at 9 and 7 consecutive
    truncations having recorded 2 and 10 steps, and nothing in the result said so; the
    evidence was only in Harbor's `trial.log`, inside a jobs-dir that is deleted on exit.
    """
    path = os.path.join(trial_dir, "trial.log")
    if not os.path.isfile(path):
        return {}
    run = worst = total = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Output length exceeded" in line:
                    run += 1
                    total += 1
                    worst = max(worst, run)
                elif "Trajectory dumped" in line:
                    run = 0
    except OSError as e:
        logger.debug("Could not read %s: %s", path, e)
        return {}
    if not total:
        return {}
    return {"truncated_turns": total, "longest_truncation_run": worst,
            "unresolved_truncations": run,
            "livelocked": max(worst, run) >= _LIVELOCK_TRUNCATIONS}


def _trajectory_text(trial_dir: str) -> Tuple[str, Optional[int]]:
    """Agent transcript and turn count from Terminus-2's ATIF trajectory files.

    Harbor writes the agent's conversation to `<trial_dir>/agent/trajectory.json` (plus
    `trajectory.cont-N.json` when context summarization splits it), NOT into result.json -
    `AgentContext` carries only token counts, cost and rollout details, which is why
    probing result.json for `messages`/`n_turns` returned nothing on the 2026-08-18 run and
    left two `AgentTimeoutError` trials unexplained. Terminus dumps after every episode and
    again in a `finally`, so the file exists even when the agent times out.
    """
    agent_dir = os.path.join(trial_dir, "agent")
    if not os.path.isdir(agent_dir):
        return "", None
    names = sorted(n for n in os.listdir(agent_dir)
                   if n == "trajectory.json" or n.startswith("trajectory.cont-"))
    parts: List[str] = []
    steps = 0
    for name in names:
        try:
            with open(os.path.join(agent_dir, name), "r", encoding="utf-8") as f:
                traj = json.load(f)
        except Exception as e:
            logger.debug("Could not parse %s: %s", name, e)
            continue
        for step in traj.get("steps") or []:
            if not isinstance(step, dict):
                continue
            steps += 1
            for chunk in (step.get("reasoning_content"), step.get("message")):
                if isinstance(chunk, str) and chunk.strip():
                    parts.append(chunk)
            for call in step.get("tool_calls") or []:
                if isinstance(call, dict):
                    parts.append(json.dumps(call.get("arguments") or call, default=str))
    return "\n".join(parts)[:_MAX_TRANSCRIPT_CHARS], (steps or None)


def _agent_transcript(agent: dict, data: dict) -> str:
    """Best-effort agent transcript from Harbor's result.json.

    Only reachable when the agent was run with `store_all_messages`; the normal path is
    `_trajectory_text`. Kept as a fallback because Harbor's schema varies by agent and
    version. Returns "" when none is present.
    """
    meta = agent.get("metadata") if isinstance(agent.get("metadata"), dict) else {}
    for src in (meta.get("all_messages"), agent.get("messages"), agent.get("transcript"),
                agent.get("history"), data.get("messages"), data.get("transcript")):
        if isinstance(src, list) and src:
            parts = []
            for m in src:
                if isinstance(m, dict):
                    parts.append(str(m.get("content") or m.get("text") or ""))
                else:
                    parts.append(str(m))
            return "\n".join(p for p in parts if p)[:_MAX_TRANSCRIPT_CHARS]
        if isinstance(src, str) and src.strip():
            return src[:_MAX_TRANSCRIPT_CHARS]
    for k in ("agent_output", "output", "stdout", "logs"):
        v = agent.get(k) or data.get(k)
        if isinstance(v, str) and v.strip():
            return v[:_MAX_TRANSCRIPT_CHARS]
    return ""


def check_terminal_bench_prerequisites() -> Tuple[bool, str]:
    """Check if Harbor CLI and Docker daemon are available and accessible."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."

    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            return False, f"Docker daemon is not running or accessible: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"

    if not shutil.which("harbor"):
        return False, "Harbor CLI is not installed. Install via: pip install harbor (or uv tool install harbor)"

    return True, ""


def _parse_shard() -> Optional[Tuple[int, int]]:
    """`GBENCH_SHARD="I/N"` -> `(I, N)` (1-indexed shard I of N), or None when unset/empty.

    Mirrors `sampling.parse_shard` exactly, including its fail-fast validation: a malformed or
    out-of-range spec raises ValueError so a typo surfaces instead of silently running the whole
    set. Parsed here (not imported) because this suite never enters `base.run_eval_suite`.
    """
    spec = (os.environ.get("GBENCH_SHARD") or "").strip()
    if not spec:
        return None
    if "/" not in spec:
        raise ValueError(f"GBENCH_SHARD must be I/N (e.g. 1/8), got {spec!r}")
    i_str, n_str = spec.split("/", 1)
    try:
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise ValueError(f"GBENCH_SHARD must be I/N with integers, got {spec!r}")
    if n < 1 or i < 1 or i > n:
        raise ValueError(f"GBENCH_SHARD I/N needs 1 <= I <= N and N >= 1, got {spec!r}")
    return (i, n)


def _dataset_task_names(dataset: str) -> List[str]:
    """Full, deterministically SORTED list of task names (`org/name` each) in `dataset`.

    Enumerated through Harbor's OWN package-dataset resolver so the strings are byte-identical to
    the ones `harbor run`'s `--include-task-name` filter fnmatches at run time (it compares
    `PackageTaskId.get_name()`, which is `f"{org}/{name}"`). `harbor run -d <org/name>` resolves the
    bare name to `<org/name>@latest` (see cli/jobs.py), so this queries the same `@latest` version.

    Sorted+deduped here because Harbor returns task ids in DB order; `sampling.shard_select` requires
    a stable, reproducible order so the round-robin partition (`sorted[I-1::N]`) is non-overlapping
    and unions back to the full set across machines. Harbor is imported lazily so importing this
    module never requires Harbor to be installed.
    """
    import asyncio
    from harbor.registry.client.package import PackageDatasetClient

    async def _fetch() -> List[str]:
        client = PackageDatasetClient()
        metadata = await client.get_dataset_metadata(f"{dataset}@latest")
        return [tid.get_name() for tid in metadata.task_ids]

    return sorted(set(asyncio.run(_fetch())))


def run_terminal_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run official Terminal-Bench 2.1 evaluation suite via Harbor framework."""
    ok, reason = check_terminal_bench_prerequisites()
    if not ok:
        # No-skip policy: a missing prerequisite must HARD-ERROR, never emit a fabricated 0%/skip
        # row. evals.py isolates the raise into a status:"error" row and continues the sweep.
        raise infra_required("terminal_bench", reason, DOCS_URL)

    start_time = time.time()
    limit = kwargs.get("limit")
    total = 0
    correct = 0
    category_acc: Dict[str, Dict[str, Any]] = {}
    sample_traces: List[Dict[str, Any]] = []

    # --shard I/N: this suite bypasses base.run_eval_suite, so honor the shard here. Harbor's
    # `--include-task-name` names the exact subset (see the module docstring), so enumerate the
    # dataset's deterministically-sorted full task list and select shard I of N by round-robin -
    # matching sampling.shard_select - composed BEFORE the limit (the limit caps WITHIN the shard).
    shard = _parse_shard()
    include_names: Optional[List[str]] = None
    shard_size: Optional[int] = None
    if shard:
        try:
            full = _dataset_task_names(DEFAULT_DATASET)
        except Exception as e:
            # No fabrication: without the real task list the shard subset cannot be built, and
            # running the full set (or a no-op) would misreport coverage. Hard-error instead.
            raise infra_required(
                "terminal_bench",
                f"--shard {shard[0]}/{shard[1]} was requested but Harbor's registry could not be "
                f"queried to enumerate {DEFAULT_DATASET}'s task list ({type(e).__name__}: {e}); the "
                f"shard subset cannot be built without fabricating shard coverage",
                DOCS_URL) from e
        i, n = shard
        include_names = shard_select(full, (i, n))    # shard FIRST (sampling.shard_select)
        if limit and limit > 0:
            # cap WITHIN the shard, stratified (a seeded subset, not a contiguous head - audit RC-1)
            include_names = stratified_sample(include_names, limit, seed="terminal_bench")
        shard_size = len(include_names)
        logger.info("[terminal_bench] shard %d/%d: %d of %d tasks%s", i, n, shard_size, len(full),
                    f" (capped at --eval-limit {limit})" if (limit and limit > 0) else "")
        if shard_size == 0:
            # A legitimately empty shard (N > task count) must NOT invoke Harbor: passing zero
            # `--include-task-name` filters would run the FULL set - a fabrication. Return an honest
            # empty subset result instead (nothing measured here, so nothing to hard-error on).
            logger.warning("[terminal_bench] shard %d/%d is empty (%d tasks across %d shards); "
                           "nothing to run on this node.", i, n, len(full), n)
            temperature, temperature_source = resolve_temperature(
                "terminal_bench", kwargs.get("temperature"), thinking=enable_thinking)
            return {
                "benchmark_type": "eval",
                "eval_name": "terminal_bench",
                "model_name": model_name,
                "thinking": enable_thinking,
                "total_questions": 0,
                "correct_answers": 0,
                "failed_requests": 0,
                "accuracy": 0.0,
                "category_accuracy": {},
                "sample_traces": [],
                "turn_budget": None,
                "temperature": temperature,
                "temperature_source": temperature_source,
                "shard": f"{i}/{n}",
                "is_full": False,
                "partial": False,
                "leaderboard_comparable": False,
                "leaderboard_comparable_reason": (
                    f"empty shard {i}/{n} (N > {len(full)} tasks): a non-overlapping subset by "
                    "construction"),
                "status": "success",
                "duration_s": round(time.time() - start_time, 2),
            }

    with _jobs_dir() as tmp_dir:
        env = os.environ.copy()
        # Direct Harbor to target OpenAI-compatible endpoint
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/v1"
        env["OPENAI_BASE_URL"] = endpoint
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "EMPTY")

        # This suite shells out to Harbor rather than going through `run_eval_suite`, so it
        # never reached `resolve_temperature` - `kwargs["temperature"]` is None unless the
        # operator passed --temperature, and Harbor's kwarg parser turns the literal "None"
        # into None, which terminus-2 skips. The header has been promising a 1.0 default
        # this whole time and silently handing the server whatever its own default was.
        temperature, temperature_source = resolve_temperature(
            "terminal_bench", kwargs.get("temperature"),
            thinking=enable_thinking)  # named param; kwargs never carries enable_thinking
        agent_name = kwargs.get("agent", "terminus-2")
        harbor_model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
        # Terminus-2 drives LiteLLM, and neither sets `max_tokens`. Unset, vLLM lets a turn
        # run to `max_model_len - prompt` - up to ~250k tokens on this deployment. Measured
        # 2026-08-18 at 74 tok/s per stream, that is 56 MINUTES for a single turn, and even
        # an ordinary 30k-token thinking turn costs 6.8 min. Against `write-compressor`'s
        # 900s x 4.0 = 3600s agent budget that is at most 8 turns; Terminus needs far more,
        # so the trial dies as `AgentTimeoutError` having barely started. Capping the turn
        # is what buys turns - the timeout multiplier only buys wall-clock (the 2026-08-18
        # run spent 9652s, exactly `schemelike-metacircular-eval`'s 2400x4=9600s budget,
        # and still timed out). The cap must be paired with the XML parser: with the `json`
        # default a truncated turn cannot be salvaged, so it is re-asked and truncates
        # again forever - "costs one turn rather than the trial" was only true with salvage
        # available, and 2 of 3 trials livelocked before that was understood.
        turn_cap = _turn_token_cap(enable_thinking)
        llm_call_kwargs: Dict[str, Any] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        }
        if turn_cap:
            llm_call_kwargs["max_tokens"] = turn_cap
        # Two separate channels, because the agent and the verifier run through different
        # code paths in Harbor and neither inherits the other's environment:
        #   agent    -> `--agent-env` -> config.env -> terminus-2's `extra_env` ->
        #               `tmux new-session -e K=V`, so every command the agent types
        #               inherits it (tmux 3.4 in these images; -e needs >= 3.2)
        #   verifier -> `--verifier-env` -> config.verifier.env -> `override_env` on the
        #               verifier's exec
        # See the note at the flags themselves for why the agent side must NOT go through
        # `--agent-kwarg`.
        thread_env = container_thread_env()
        cmd = [
            "harbor",
            "run",
            "-d",
            DEFAULT_DATASET,
            "--agent",
            agent_name,
            "--model",
            harbor_model,
            "--n-concurrent",
            str(concurrency),
            "--jobs-dir",
            tmp_dir,
            "--agent-kwarg",
            f"api_base={endpoint}",
            "--agent-kwarg",
            f"temperature={temperature}",
            "--agent-kwarg",
            f"parser_name={_PARSER}",
            "--agent-kwarg",
            f"llm_call_kwargs={json.dumps(llm_call_kwargs)}",
        ]
        for key, value in thread_env.items():
            # `--agent-env`, NOT `--agent-kwarg extra_env={...}`. Terminus-2 does take an
            # `extra_env` constructor kwarg, but AgentFactory already fills it from
            # `config.env` and then splats the agent kwargs on top:
            #     create_agent_from_name(..., extra_env=extra_env, **agent_kwargs)
            # so supplying it through kwargs is a guaranteed
            # `TypeError: got multiple values for keyword argument 'extra_env'`, which
            # killed the whole job 8 seconds in on 2026-08-19. `--agent-env` populates
            # `config.env`, which is the same destination by the supported route.
            cmd.extend(["--agent-env", f"{key}={value}"])
            # The verifier is a separate exec with its own env; without this the SCORING
            # run still thrashes and a timeout there fails a correct solution.
            cmd.extend(["--verifier-env", f"{key}={value}"])

        # Terminal-Bench's per-task timeouts are calibrated for fast hosted models. A local
        # reasoning model is far slower per turn - measured 2026-08-18, every task in a
        # 3-task --thinking run ended in `AgentTimeoutError` after 10-25 minutes, scoring a
        # structural 0% that says nothing about the model's ability.
        #
        # Follows from --thinking, like the output-token floor: a thinking turn spends most
        # of its time in the reasoning channel before the agent can act, so it needs the
        # larger allowance. Only the AGENT's clock is scaled - the verifier timeout is a
        # correctness bound and must not move.
        cmd.extend(["--agent-timeout-multiplier", _timeout_multiplier(enable_thinking)])

        if shard:
            # Name the shard's exact tasks; `--include-task-name` fnmatches each `org/name`. Do NOT
            # also pass `--n-tasks`: the limit was already applied within the shard when building
            # `include_names`, so the two must not compose twice.
            for task_name in include_names:
                cmd.extend(["--include-task-name", task_name])
        elif limit:
            cmd.extend(["--n-tasks", str(limit)])

        import threading

        full_output_lines = []
        total_expected = shard_size if shard else (limit if limit else 89)
        seen_trials = set()
        correct_trials = 0

        # Harbor's XML-salvage path returns a bare str where the caller expects an
        # LLMResponse, killing the trial with AttributeError. Install the coercion inside
        # the harbor subprocess - patching gbench's own process would do nothing.
        cmd = terminus_salvage_patch.wrap_command(cmd)
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Consume any direct stdout from Harbor asynchronously
            def _read_stdout():
                if proc.stdout:
                    line_iter = iter(proc.stdout.readline, "") if hasattr(proc.stdout, "readline") else iter(proc.stdout or [])
                    for line in line_iter:
                        full_output_lines.append(line)

            stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
            stdout_thread.start()

            with tqdm(total=total_expected, desc="Eval [TERMINAL_BENCH]") as pbar:
                # Monitor trial progress via result.json files on disk
                while proc.poll() is None:
                    time.sleep(2)
                    for root, _, files in os.walk(tmp_dir):
                        if "result.json" in files:
                            rf_path = os.path.join(root, "result.json")
                            try:
                                with open(rf_path, "r", encoding="utf-8") as rf:
                                    rdata = json.load(rf)
                                if isinstance(rdata, dict):
                                    if "n_total_trials" in rdata and rdata["n_total_trials"] and pbar.total != rdata["n_total_trials"]:
                                        pbar.total = rdata["n_total_trials"]
                                        pbar.refresh()
                                    if "verifier_result" in rdata or "trial_name" in rdata:
                                        trial_id = rdata.get("trial_name", os.path.basename(root))
                                        if trial_id not in seen_trials:
                                            seen_trials.add(trial_id)
                                            r_val = _extract_reward(rdata)
                                            if r_val is not None and r_val >= 1.0:
                                                correct_trials += 1
                                            pbar.update(1)
                                            acc = (correct_trials / len(seen_trials) * 100.0) if seen_trials else 0.0
                                            pbar.set_postfix(correct=f"{correct_trials}/{len(seen_trials)} ({acc:.1f}%)")
                            except Exception:
                                pass

                proc.wait()
                stdout_thread.join(timeout=1.0)

                # Final sweep to catch any remaining results on completion
                for root, _, files in os.walk(tmp_dir):
                    if "result.json" in files:
                        rf_path = os.path.join(root, "result.json")
                        try:
                            with open(rf_path, "r", encoding="utf-8") as rf:
                                rdata = json.load(rf)
                            if isinstance(rdata, dict) and ("verifier_result" in rdata or "trial_name" in rdata):
                                trial_id = rdata.get("trial_name", os.path.basename(root))
                                if trial_id not in seen_trials:
                                    seen_trials.add(trial_id)
                                    r_val = _extract_reward(rdata)
                                    if r_val is not None and r_val >= 1.0:
                                        correct_trials += 1
                                    pbar.update(1)
                                    acc = (correct_trials / len(seen_trials) * 100.0) if seen_trials else 0.0
                                    pbar.set_postfix(correct=f"{correct_trials}/{len(seen_trials)} ({acc:.1f}%)")
                        except Exception:
                            pass

            full_output = "".join(full_output_lines)
            if proc.returncode != 0:
                logger.error(f"Harbor CLI exited with code {proc.returncode}")
        except Exception as e:
            # A crashed external harness produced no trustworthy number: hard-error rather than
            # record a fabricated 0.0 row (no-skip/no-partial policy). The _jobs_dir context keeps
            # Harbor's logs for diagnosis (failed=True) on the way out.
            logger.error(f"Failed to execute Harbor CLI: {e}")
            raise infra_required(
                "terminal_bench",
                f"the Harbor harness failed to execute ({type(e).__name__}: {e}) - a harness/infra "
                "failure, not a model result",
                DOCS_URL) from e

        # Specifically scan for Harbor result.json files (ignoring config.json, lock.json).
        # The sibling `agent/trajectory.json` IS read - see `_trajectory_text`.
        trial_results = []
        parse_failures: List[str] = []
        job_stats = {}

        for root, _, files in os.walk(tmp_dir):
            for file in files:
                if file == "result.json":
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r", encoding="utf-8") as rf:
                            data = json.load(rf)
                        if isinstance(data, dict):
                            if "stats" in data and isinstance(data["stats"], dict):
                                job_stats = data["stats"]
                            elif "verifier_result" in data or "trial_name" in data:
                                reward_val = _extract_reward(data)
                                is_passed = reward_val is not None and reward_val >= 1.0
                                trial_name = data.get("trial_name", os.path.basename(root))
                                exc = data.get("exception_info", {}).get("exception_type") if data.get("exception_info") else None
                                # `AgentTimeoutError` on its own does not say WHY the agent
                                # ran out of time - a hard task and a model looping on the
                                # same command look identical in the result. Harbor's
                                # jobs-dir is a TemporaryDirectory that is deleted on exit,
                                # so unless the evidence is lifted out here it is gone:
                                # after the 2026-08-18 run there was no way to tell whether
                                # the two timed-out trials were stuck in a loop.
                                agent = data.get("agent_result") if isinstance(
                                    data.get("agent_result"), dict) else {}
                                transcript, traj_steps = _trajectory_text(root)
                                if not transcript:
                                    transcript = _agent_transcript(agent, data)
                                meta = agent.get("metadata") if isinstance(
                                    agent.get("metadata"), dict) else {}
                                # `n_episodes` and `api_request_times_msec` separate the two
                                # explanations a timeout cannot: few slow turns (the model
                                # is generating too much per turn) vs many fast ones (the
                                # agent is thrashing). Terminus records both on the context
                                # in a `finally`, so they survive the timeout.
                                times = meta.get("api_request_times_msec") or []
                                times = [t for t in times if isinstance(t, (int, float))]
                                trial_results.append({
                                    "trial_name": trial_name,
                                    "reward": reward_val,  # None when the trial recorded no reward
                                    "passed": is_passed,
                                    "exception": exc,
                                    **_truncation_livelock(root),
                                    "agent_turns": meta.get("n_episodes") or traj_steps,
                                    "summarization_count": meta.get("summarization_count"),
                                    "output_tokens": agent.get("n_output_tokens"),
                                    "mean_turn_latency_s": (
                                        round(sum(times) / len(times) / 1000.0, 1)
                                        if times else None),
                                    "max_turn_latency_s": (
                                        round(max(times) / 1000.0, 1) if times else None),
                                    "response_text": transcript,
                                    "repetition_run": repetition_run(transcript),
                                    "repetition_onset": repetition_onset(transcript),
                                })
                    except Exception as e:
                        # WARNING, not debug: on 2026-08-18 Harbor completed 3/3 trials in
                        # 2h40 and gbench published `total_questions: 0` because every
                        # per-trial parse failed silently here. A swallowed exception on
                        # the only path that turns work into a score is not a debug detail.
                        parse_failures.append(f"{fpath}: {type(e).__name__}: {e}")
                        logger.warning("Could not parse %s: %s: %s",
                                       fpath, type(e).__name__, e)

        # Keep the jobs dir when any TRIAL errored too, not just when parsing did. On
        # 2026-08-19 two trials died with a bare `AttributeError` after 3 turns; parsing
        # succeeded, so the dir was cleaned up and the traceback went with it. A trial
        # that crashed is exactly when the logs are wanted.
        trial_errors = [t for t in trial_results if t.get("exception")]
        if trial_errors:
            logger.warning(
                "[terminal_bench] %d/%d trials raised: %s. These are HARNESS failures, not "
                "model results - the accuracy below is a floor, not a measurement.",
                len(trial_errors), len(trial_results),
                ", ".join(f"{t['trial_name']}={t['exception']}" for t in trial_errors))
        _JOBS_DIR_OK[tmp_dir] = (bool(trial_results) and not parse_failures
                                 and not trial_errors)
        if trial_results:
            sample_traces = sorted(trial_results, key=lambda x: x.get("trial_name", ""))
            total = len(trial_results)
            correct = sum(1 for t in trial_results if t.get("passed"))
        elif job_stats:
            total, correct = _job_stats_totals(job_stats)

        # `measured_total` is what Harbor actually produced; it drives the "nothing was measured"
        # hard-error below. For a sharded run the DENOMINATOR is the shard size (the tasks this node
        # was asked to cover), so any task Harbor dropped counts as 0 (no-skip) instead of shrinking
        # the denominator and inflating accuracy.
        measured_total = total
        if shard and shard_size is not None:
            total = shard_size
        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        duration = time.time() - start_time
        # A timed-out trial says nothing about WHY on its own, and the turn cap below was
        # sized on one particular deployment. Derive what this endpoint actually did -
        # tokens the agent generated over the time it spent waiting on the model - and turn
        # it into the number that decides whether the run was ever winnable: how many turns
        # fit in the shortest agent budget. On slower hardware that number collapses, and a
        # collapsed turn budget is the explanation a bare `AgentTimeoutError` withholds.
        stuck = [t for t in trial_results if t.get("livelocked")]
        if stuck:
            logger.warning(
                "[terminal_bench] %d/%d trials LIVELOCKED on the output cap: a turn hit "
                "max_tokens=%s, could not be salvaged, and was re-asked (worst run: %d "
                "consecutive truncations with no step recorded). These trials measure the "
                "cap, not the model. Raise GBENCH_TB_TURN_MAX_TOKENS, and check "
                "GBENCH_TB_PARSER is 'xml' - the 'json' parser cannot salvage a truncated "
                "turn. Affected: %s",
                len(stuck), len(trial_results), turn_cap,
                max(t.get("longest_truncation_run") or 0 for t in stuck),
                ", ".join(t["trial_name"] for t in stuck))
        budget_report = _turn_budget_report(trial_results, turn_cap, enable_thinking)
        if budget_report and budget_report.get("turns_in_shortest_budget", 99) < 10:
            logger.warning(
                "[terminal_bench] this endpoint decodes at ~%.0f tok/s, so a %d-token turn "
                "costs ~%.0fs and only ~%d turns fit the shortest task's agent budget "
                "(%ds). Terminus needs more than that, so timeouts here are a SIZING "
                "result, not a model result. Lower GBENCH_TB_TURN_MAX_TOKENS or raise "
                "GBENCH_TB_TIMEOUT_MULTIPLIER.",
                budget_report["observed_tok_s"], turn_cap,
                budget_report["seconds_per_turn"],
                budget_report["turns_in_shortest_budget"],
                budget_report["shortest_agent_budget_s"])

        # CC6: zero trials means nothing was measured -> hard-error (no-skip/no-partial), never a
        # fabricated 0.0 row. Previously this returned status:"failed" accuracy:0.0, which cli.py
        # still renders as a measured number. Keyed on `measured_total` (what Harbor produced), not
        # the shard-adjusted denominator, so a sharded run whose tasks all vanished still hard-errors.
        if measured_total == 0:
            tail = (full_output.strip() or
                    ("Harbor CLI produced no trials" if proc.returncode == 0
                     else f"Harbor CLI exited {proc.returncode}"))
            raise infra_required(
                "terminal_bench",
                f"the Harbor harness produced zero parseable trials so nothing was measured "
                f"(returncode={proc.returncode}). Tail: {tail[-800:]}",
                DOCS_URL)

        # leaderboard_comparable honesty. gbench's DEFAULTS are deliberately non-canonical for a
        # slow local endpoint (a per-turn max_tokens cap, the xml salvage parser, and an inflated
        # agent-timeout multiplier), and Harbor's own leaderboard validator rejects a non-1.0
        # timeout multiplier. So a run is comparable ONLY when every knob is at its canonical value
        # AND the full task set ran AND no trials were lost to parse/harness errors. With the
        # shipped defaults this is False, which is the honest answer.
        tmult = float(_timeout_multiplier(enable_thinking))
        noncanon: List[str] = []
        if shard:
            noncanon.append(f"sharded run (--shard {shard[0]}/{shard[1]}): a non-overlapping subset "
                            f"of the full task set by construction")
        if limit:
            noncanon.append(f"subset run (--eval-limit {limit})")
        if tmult != 1.0:
            noncanon.append(f"agent-timeout-multiplier={tmult} (canonical requires 1.0)")
        if turn_cap:
            noncanon.append(f"per-turn max_tokens cap={turn_cap} (canonical: uncapped)")
        if _PARSER != "json":
            noncanon.append(f"parser={_PARSER!r} (canonical terminus-2 default is 'json')")
        if parse_failures:
            noncanon.append(f"{len(parse_failures)} trial(s) failed to parse (partial measurement)")
        if trial_errors:
            noncanon.append(f"{len(trial_errors)} trial(s) raised harness exceptions")
        # Harbor runs 1 trial/task here (gbench does not pass -k/--n-attempts); the TB2.1
        # leaderboard requires >=5 trials/task for its pass^k, so a single-trial run is not comparable.
        noncanon.append("1 trial per task (canonical TB2.1 leaderboard requires >=5 trials/task)")

        return {
            "benchmark_type": "eval",
            "eval_name": "terminal_bench",
            "model_name": model_name,
            "thinking": enable_thinking,
            "total_questions": total,
            "correct_answers": correct,
            "failed_requests": 0,
            "attempts": 1,   # Harbor runs 1 trial/task here (-k/--n-attempts not plumbed); surfaced for the CLI contract
            "accuracy": round(accuracy, 2),
            "category_accuracy": category_acc,
            "sample_traces": sample_traces,
            "turn_budget": budget_report,
            "temperature": temperature,
            "temperature_source": temperature_source,
            # Shard transparency: `shard` is "I/N" when a shard was applied (else None), and
            # `is_full` is True only when the entire canonical task set was requested (no shard, no
            # limit). `total_questions` above is the shard size for a sharded run.
            "shard": f"{shard[0]}/{shard[1]}" if shard else None,
            "is_full": not shard and not limit,
            "agent_parser": _PARSER,
            "container_threads": int(thread_env.get("OMP_NUM_THREADS", 0)) or None,
            # A run that lost trials to parse/harness errors is a partial measurement, not a clean
            # score - surface it so it is never read as complete.
            "parse_failures": len(parse_failures),
            "trial_errors": len(trial_errors),
            "partial": bool(parse_failures or trial_errors),
            "leaderboard_comparable": not noncanon,
            "leaderboard_comparable_reason": "; ".join(noncanon) or "full canonical run",
            "status": "success" if proc.returncode == 0 or measured_total > 0 else "failed",
            "duration_s": round(duration, 2),
        }
