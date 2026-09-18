#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# gbench launcher for SkillsBench, run INSIDE the gbench-skillsbench image. It drives the BenchFlow
# 0.6.3 CLI directly (the repo's own open-model runner script targets an OLDER benchflow: it uses
# `--backend`/opencode, but the locked 0.6.3 CLI uses `--sandbox` and ships deepagents/codex/etc.,
# configuring the model via BENCHFLOW_PROVIDER_BASE_URL/API_KEY). For each condition x task it runs
#   uv run bench eval run --tasks-dir <task> --agent <agent> --sandbox docker --jobs-dir <jobs>
#       [--model <model> --agent-env BENCHFLOW_PROVIDER_BASE_URL=.. --agent-env BENCHFLOW_PROVIDER_API_KEY=..]
#       [--skills-dir <task>/environment/skills]      # with-skills condition
# and parses each rollout's result.json `rewards.reward` (the DETERMINISTIC per-task verifier; no
# LLM judge). Scoring is upstream's own reward - gbench adds nothing.
#
# docker-out-of-docker: BenchFlow bind-mounts its working dirs into sibling task containers, so the
# workdir MUST be an identity mount (same path on host + orchestrator) and TMPDIR must point there,
# else the sibling sandbox gets empty mounts and no reward is collected (the entrypoint/harness set
# this up). Env contract (set by the entrypoint / docker run -e):
#   SKILLSBENCH_DIR         repo root (default /app/skillsbench)
#   SKILLSBENCH_WORKDIR     identity-mounted host path for jobs + summary + TMPDIR (default /out)
#   SKILLSBENCH_AGENT       agent (default deepagents; 'oracle' for a model-free pipeline smoke)
#   GBENCH_MODEL_BASE_URL   /v1 reachable FROM task containers (model path only)
#   GBENCH_MODEL_NAME       served model name (model path only)
#   GBENCH_MODEL_API_KEY    api key (dummy for vLLM)
#   SKILLSBENCH_CONDITIONS  comma list from {with-skills,without-skills} (default without-skills)
#   SKILLSBENCH_TASKS       comma list to limit tasks (default all-minus-excludes)
#   SKILLSBENCH_TASK_LIMIT  cap to the first N tasks (sorted) when no explicit subset
#   SKILLSBENCH_CONCURRENCY per-task bench concurrency (default 1)
#   GBENCH_SHARD            "I/N": run only shard I of N of the sorted task list (round-robin
#                           full_sorted[I-1::N]); applied BEFORE SKILLSBENCH_TASK_LIMIT, and within
#                           the SKILLSBENCH_TASKS subset when one is given. Score set == run set.

import glob
import json
import os
import subprocess
import sys

SKILLSBENCH_DIR = os.environ.get("SKILLSBENCH_DIR", "/app/skillsbench")
WORKDIR = os.environ.get("SKILLSBENCH_WORKDIR", "/out")
_EXCLUDED = {"mhc-layer-impl"}


def _shard():
    """Parse GBENCH_SHARD='I/N' -> (i, n) or None (matches gbench sampling.shard_select)."""
    spec = os.environ.get("GBENCH_SHARD", "").strip()
    if not spec or "/" not in spec:
        return None
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        return None
    return (i, n) if (n >= 1 and 1 <= i <= n) else None


def _apply_shard(ids):
    """Round-robin shard I of N over the DETERMINISTIC sorted id list: full_sorted[i-1::n].

    No-op when no shard is set, so non-sharded runs are byte-identical to before.
    """
    sh = _shard()
    if not sh:
        return ids
    i, n = sh
    return sorted(ids)[i - 1::n]


def _task_list():
    explicit = [t.strip() for t in os.environ.get("SKILLSBENCH_TASKS", "").split(",") if t.strip()]
    if explicit:
        # Shard WITHIN the explicit subset (the "full task list" is that subset here).
        return _apply_shard(explicit)
    tasks_dir = os.path.join(SKILLSBENCH_DIR, "tasks")
    allt = sorted(d for d in os.listdir(tasks_dir)
                  if d not in _EXCLUDED and os.path.isfile(os.path.join(tasks_dir, d, "task.md")))
    # Shard FIRST (over the full sorted set), then let the limit cap within the shard.
    allt = _apply_shard(allt)
    lim = os.environ.get("SKILLSBENCH_TASK_LIMIT")
    if lim and int(lim) > 0:
        allt = allt[:int(lim)]
    return allt


def _reward_from_jobs(jobs_dir):
    """Return the reward float from the newest result.json under jobs_dir (or None)."""
    results = sorted(glob.glob(os.path.join(jobs_dir, "**", "result.json"), recursive=True))
    if not results:
        return None
    try:
        with open(results[-1], encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    rewards = d.get("rewards")
    if isinstance(rewards, dict) and isinstance(rewards.get("reward"), (int, float)):
        return float(rewards["reward"])
    if isinstance(rewards, (int, float)):
        return float(rewards)
    if isinstance(d.get("reward"), (int, float)):
        return float(d["reward"])
    return None


def _run_one(task, condition, agent, jobs_dir):
    task_dir = os.path.join(SKILLSBENCH_DIR, "tasks", task)
    cmd = ["uv", "run", "bench", "eval", "run",
           "--tasks-dir", task_dir, "--agent", agent, "--sandbox", "docker",
           "--jobs-dir", jobs_dir,
           "--concurrency", os.environ.get("SKILLSBENCH_CONCURRENCY", "1")]
    if agent != "oracle":
        model = os.environ.get("GBENCH_MODEL_NAME", "")
        if model:
            cmd += ["--model", model]
        base_url = os.environ.get("GBENCH_MODEL_BASE_URL", "")
        api_key = os.environ.get("GBENCH_MODEL_API_KEY", "dummy")
        if base_url:
            cmd += ["--agent-env", f"BENCHFLOW_PROVIDER_BASE_URL={base_url}",
                    "--agent-env", f"BENCHFLOW_PROVIDER_API_KEY={api_key}",
                    "--agent-env", f"OPENAI_BASE_URL={base_url}",
                    "--agent-env", f"OPENAI_API_KEY={api_key}"]
    if condition == "with-skills":
        skills_dir = os.path.join(task_dir, "environment", "skills")
        if os.path.isdir(skills_dir):
            # BenchFlow 0.6.3 defaults --skill-mode to no-skill and RAISES when --skills-dir is
            # set without with-skill, so the with-skills arm (the lift metric) needs both flags.
            cmd += ["--skills-dir", skills_dir, "--skill-mode", "with-skill"]
    env = dict(os.environ)
    env["TMPDIR"] = WORKDIR
    proc = subprocess.run(cmd, cwd=SKILLSBENCH_DIR, env=env, capture_output=True, text=True)
    reward = _reward_from_jobs(jobs_dir)
    if reward is None:
        sys.stderr.write(f"[skillsbench] {condition}/{task}: no reward "
                         f"(rc={proc.returncode}) tail: {(proc.stderr or proc.stdout or '')[-300:]}\n")
    return reward


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    agent = os.environ.get("SKILLSBENCH_AGENT", "deepagents")
    conditions = [c.strip() for c in os.environ.get("SKILLSBENCH_CONDITIONS", "without-skills").split(",") if c.strip()]
    tasks = _task_list()

    out = {"conditions": {}, "agent": agent, "n_tasks_selected": len(tasks)}
    for cond in conditions:
        per_task = {}
        for task in tasks:
            jobs_dir = os.path.join(WORKDIR, "jobs", cond, task)
            os.makedirs(jobs_dir, exist_ok=True)
            per_task[task] = _run_one(task, cond, agent, jobs_dir)
        vals = [v for v in per_task.values() if isinstance(v, (int, float))]
        out["conditions"][cond] = {
            "mean_reward": (sum(vals) / len(vals)) if vals else None,
            "n_scored": len(vals),
            "n_tasks": len(per_task),
            "per_task": per_task,
        }
    ws = out["conditions"].get("with-skills", {}).get("mean_reward")
    ns = out["conditions"].get("without-skills", {}).get("mean_reward")
    if isinstance(ws, (int, float)) and isinstance(ns, (int, float)):
        out["with_skills_lift"] = ws - ns

    with open(os.path.join(WORKDIR, "skillsbench_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
