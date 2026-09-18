#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# gbench launcher for GAIA2 / Meta-ARE, run INSIDE the gbench-gaia2 image. It drives the pinned
# `are-benchmark` CLI (verified against the 1.2.0 wheel: cli.py + cli/shared_params.py) and parses
# the emitted benchmark_stats.json - it NEVER fabricates a score.
#
# Modes (GAIA2_MODE):
#   run       (default) - `are-benchmark run` with NO --hf-config runs ALL 5 GAIA2 capability
#                         configs (adaptability/ambiguity/execution/search/time via
#                         CapabilityTag.gaia2_capabilities()) and writes ONE aggregated
#                         benchmark_stats.json. A single --hf-config (GAIA2_CONFIG) narrows it.
#   gaia2-run           - the full canonical leaderboard submission: 3 phases (Standard / A2A-mini /
#                         Noise-mini) x --num_runs (default 3), Pass@k/Pass^k. Very heavy; opt-in.
#
# Sharding (--shard I/N): NOT expressible here and deliberately NOT consumed. are-benchmark selects
# scenarios ONLY via --limit (first-N per config) and owns selection internally - huggingface_loader
# streams the HF dataset and breaks the enumerate loop at `index > limit`, with no offset /
# scenario-id / shard flag on the CLI. So the gbench runner does not forward GBENCH_SHARD into this
# container, and this launcher intentionally has no shard-parsing/slicing code (adding a no-op would
# fabricate shard coverage). See gaia2.py + docs/evals/gaia2.md.
#
# Model-under-test wiring (litellm_engine.py: provider "local" -> custom_llm_provider=None +
# api_base=endpoint): pass --model openai/<served> so LiteLLM routes to the OpenAI-compat /v1.
# Judge: --judge_provider local --judge_endpoint <cascade proxy or override> --judge_model
# openai/<tag>; ARE's judge PROMPTS are untouched, only the model call is swapped.
#
# Env contract (set by the entrypoint / docker run -e):
#   GAIA2_WORKDIR         output dir (mounted; holds benchmark_stats.json + our summary). default /out
#   GBENCH_MODEL_NAME     served model name (sent as openai/<name>)
#   GBENCH_MODEL_BASE_URL model /v1 endpoint (reachable from the container; --network host -> 127.0.0.1)
#   GBENCH_MODEL_API_KEY  api key for the model endpoint (dummy for vLLM)
#   GAIA2_JUDGE_ENDPOINT  judge /v1 (default: the in-container cascade proxy 127.0.0.1:$GAIA2_JUDGE_PORT/v1)
#   GAIA2_JUDGE_MODEL     judge model tag (default openai/gbench-cascade; proxy ignores it)
#   GAIA2_JUDGE_PORT      cascade proxy port (default 18790)
#   GAIA2_MODE            run | gaia2-run    (default run)
#   GAIA2_CONFIG          single capability config (narrows `run`; e.g. mini/demo for smoke)
#   GAIA2_LIMIT           cap scenarios per config (smoke)
#   GAIA2_NUM_RUNS        gaia2-run repetitions (default 3)
#   GAIA2_HF_DATASET      dataset id (default meta-agents-research-environments/gaia2)
#   GAIA2_SPLIT           split (default validation)
#   GAIA2_SCENARIO_TIMEOUT / GAIA2_MAX_CONCURRENT  passthrough tuning

import glob
import json
import os
import subprocess
import sys

WORKDIR = os.environ.get("GAIA2_WORKDIR", "/out")
DATASET = os.environ.get("GAIA2_HF_DATASET", "meta-agents-research-environments/gaia2")
SPLIT = os.environ.get("GAIA2_SPLIT", "validation")


def _model_arg():
    name = os.environ.get("GBENCH_MODEL_NAME", "served")
    return name if name.startswith("openai/") else f"openai/{name}"


def _judge_args():
    jport = os.environ.get("GAIA2_JUDGE_PORT", "18790")
    endpoint = os.environ.get("GAIA2_JUDGE_ENDPOINT", f"http://127.0.0.1:{jport}/v1")
    model = os.environ.get("GAIA2_JUDGE_MODEL", "openai/gbench-cascade")
    return ["--judge_provider", "local", "--judge_endpoint", endpoint, "--judge_model", model]


def _base_cmd(command):
    cmd = ["are-benchmark", command,
           "--model", _model_arg(),
           "--provider", "local",
           "--endpoint", os.environ.get("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
           "--hf-dataset", DATASET,
           "--hf-split", SPLIT,
           "--output_dir", WORKDIR]
    cmd += _judge_args()
    if os.environ.get("GAIA2_SCENARIO_TIMEOUT"):
        cmd += ["--scenario_timeout", os.environ["GAIA2_SCENARIO_TIMEOUT"]]
    if os.environ.get("GAIA2_MAX_CONCURRENT"):
        cmd += ["--max_concurrent_scenarios", os.environ["GAIA2_MAX_CONCURRENT"]]
    return cmd


def _run_are():
    mode = os.environ.get("GAIA2_MODE", "run").strip()
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = os.environ.get("GBENCH_MODEL_API_KEY", "dummy")  # litellm requires a key
    if mode == "gaia2-run":
        cmd = _base_cmd("gaia2-run") + ["--num_runs", os.environ.get("GAIA2_NUM_RUNS", "3")]
    else:
        cmd = _base_cmd("run")
        # CRITICAL: `are-benchmark run` defaults --agent to None, and ScenarioRunner._run() then takes
        # the `_run_without_agent` path (env.join() + validate ONLY) - the model-under-test is NEVER
        # invoked, so every scenario trivially fails with "Agent count 0" and the suite reports a
        # fabricated 0.0%. The agentic model run requires an explicit agent; "default" is ARE's only
        # agent (AgentBuilder().list_agents() == ['default']) and is what `gaia2-run` uses internally.
        cmd += ["--agent", os.environ.get("GAIA2_AGENT", "default")]
        if os.environ.get("GAIA2_CONFIG"):
            cmd += ["--hf-config", os.environ["GAIA2_CONFIG"]]   # else ALL 5 capability configs
        if os.environ.get("GAIA2_LIMIT"):
            cmd += ["--limit", os.environ["GAIA2_LIMIT"]]
    sys.stderr.write("[gaia2] " + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _summarize():
    """Parse ARE's benchmark_stats.json -> gbench summary. Never invents a scalar."""
    stats_path = os.path.join(WORKDIR, "benchmark_stats.json")
    if not os.path.exists(stats_path):
        return None
    try:
        data = json.load(open(stats_path, encoding="utf-8"))
    except Exception:
        return None
    stats = data.get("statistics") or {}
    g = stats.get("global") or {}
    per_cap = stats.get("per_capability") or {}
    # Guard the agent-less no-op: `are-benchmark run` without --agent runs scenarios agent-less,
    # never invokes the model, and STILL reports a real macro=0.0 that would be published as a fake
    # score. A genuine agentic run leaves at least one per-run lite trace carrying agent LLM usage /
    # interaction history; scan for that so gbench (gaia2.py) can hard-error if the model was never
    # exercised (measured 2026-09-11). Only report True when we positively see agent activity.
    agent_invoked = False
    for p in glob.glob(os.path.join(WORKDIR, "lite", "*.json")):
        try:
            t = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if t.get("per_agent_llm_usage_stats") or t.get("per_agent_interaction_histories"):
            agent_invoked = True
            break
    # Headline = per-capability equal-weight macro success rate (the GAIA2 Overall). micro = per-
    # scenario. Values are fractions in [0,1]. Keep the raw global block for provenance.
    return {
        "macro_success_rate": g.get("macro_success_rate"),
        "micro_success_rate": g.get("micro_success_rate"),
        "pass_at_k_percent": g.get("pass_at_k_percent"),
        "pass_k_percent": g.get("pass_k_percent"),
        "total_scenarios": g.get("total_scenarios"),
        "total_runs": g.get("total_runs"),
        "validated_runs": g.get("validated_runs"),
        "success_runs": g.get("success_runs"),
        "no_validation_runs": g.get("no_validation_runs"),
        "exception_runs": g.get("exception_runs"),
        "agent_invoked": agent_invoked,
        "per_capability": {
            c: {"macro_success_rate": (v or {}).get("macro_success_rate"),
                "micro_success_rate": (v or {}).get("micro_success_rate"),
                "total_scenarios": (v or {}).get("total_scenarios")}
            for c, v in per_cap.items()
        },
        "metadata": data.get("metadata"),
    }


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    rc = _run_are()
    summary = _summarize()
    if summary is None:
        sys.stderr.write(f"[gaia2] no benchmark_stats.json produced (are-benchmark rc={rc}); "
                         f"a harness failure, not a 0%.\n")
        # leave no summary file -> the gbench runner raises infra_required
        return rc or 1
    summary["are_returncode"] = rc
    with open(os.path.join(WORKDIR, "gaia2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    sys.stderr.write(f"[gaia2] summary: macro={summary.get('macro_success_rate')} "
                     f"micro={summary.get('micro_success_rate')} "
                     f"scenarios={summary.get('total_scenarios')} "
                     f"no_validation={summary.get('no_validation_runs')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
