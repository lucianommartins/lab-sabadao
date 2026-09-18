#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# gbench launcher for WildClawBench, run INSIDE the gbench-wildclawbench orchestrator image. It
# drives the pinned InternLM harness (`eval/run_batch.py`) docker-out-of-docker: run_batch spawns
# one sibling task container per task (from `wildclawbench-ubuntu:v1.3`, mounted host docker.sock),
# runs the OpenClaw agent inside pointed at the gbench /v1 model endpoint (via an injected
# `my_api.json` custom provider), then `docker exec`s the task's own `grade()` for the score.
#
# What gbench adds (everything else is upstream, untouched):
#   * writes my_api.json → OpenClaw custom provider pointing at the gbench model endpoint;
#   * sets OPENROUTER_BASE_URL / OPENROUTER_API_KEY / JUDGE_MODEL so the tasks' verbatim
#     OpenAI-SDK judge calls hit the gbench Gemini cascade proxy instead of OpenRouter/gpt-5.4
#     (all 43 judged tasks declare these three keys in their `## Env`, so run_grading injects them);
#   * points OUTPUT_SUBDIR at the identity-mounted workdir so score.json/usage.json land on the
#     host-visible mount, then walks that tree → a single wildclawbench_summary.json with the
#     harness global mean (headline), the modality (multimodal/pure-text) breakdown, avg time/cost,
#     and a count of tasks that silently regex-fell-back (judge outage) so the harness can gate
#     leaderboard_comparable honestly.
#
# DooD identity-mount contract (the skillsbench pitfall): run_batch bind-mounts each task's
# `<workspace>/exec` dir into the sibling container, and `-v <path>` from inside a container is a
# HOST path. So the harness ROOT (with its `workspace/` data) MUST be at an identity path (same on
# host + orchestrator). The gbench runner arranges that by identity-mounting the provisioned host
# checkout; this launcher just runs run_batch from there.
#
# Env contract (set by the entrypoint / docker run -e):
#   WILDCLAW_HARNESS_DIR   harness root, identity-mounted (default /app/WildClawBench)
#   WILDCLAW_WORKDIR       identity-mounted host path for output + summary (default /out)
#   GBENCH_MODEL_BASE_URL  /v1 reachable FROM task containers (bridge gateway)
#   GBENCH_MODEL_NAME      served model name (sent as the OpenAI `model`)
#   GBENCH_MODEL_API_KEY   api key for the model endpoint (dummy for vLLM)
#   WILDCLAW_JUDGE_ENDPOINT  OPENROUTER_BASE_URL the tasks' judge client uses (the cascade proxy)
#   WILDCLAW_CATEGORIES    comma list of categories (default: all six)
#   WILDCLAW_TASK          path to a single task .md (smoke; overrides categories)
#   WILDCLAW_LIMIT         cap the selected task set to the first N ids (--eval-limit; default none)
#   WILDCLAW_PARALLEL      per-run parallelism (default 1)
#   WILDCLAW_THINKING      OpenClaw thinking level (default unset)

import glob
import json
import os
import re
import subprocess
import sys

HARNESS_DIR = os.environ.get("WILDCLAW_HARNESS_DIR", "/app/WildClawBench")
WORKDIR = os.environ.get("WILDCLAW_WORKDIR", "/out")
ALL_CATEGORIES = [
    "01_Productivity_Flow", "02_Code_Intelligence", "03_Social_Interaction",
    "04_Search_Retrieval", "05_Creative_Synthesis", "06_Safety_Alignment",
]
_PROVIDER = "gbench-model"


def _task_index():
    """frontmatter `id` -> {category, modality, path} for every task .md in the harness."""
    out = {}
    for tf in glob.glob(os.path.join(HARNESS_DIR, "tasks", "*", "*task_*.md")):
        try:
            head = open(tf, encoding="utf-8").read(4000)
        except OSError:
            continue
        mid = re.search(r"^id:\s*(.+)$", head, re.M)
        if not mid:
            continue
        mmod = re.search(r"^modality:\s*(.+)$", head, re.M)
        out[mid.group(1).strip()] = {
            "category": os.path.basename(os.path.dirname(tf)),
            "modality": (mmod.group(1).strip() if mmod else "pure-text"),
            "path": tf,
        }
    return out


def _limit():
    """--eval-limit N forwarded as WILDCLAW_LIMIT -> cap the selected task set to the first N ids.
    Unset / non-positive / unparseable means no cap (None)."""
    raw = os.environ.get("WILDCLAW_LIMIT", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _shard():
    """--shard I/N forwarded as GBENCH_SHARD -> (i, n) or None. Parsed inline because launcher
    scripts run INSIDE the container and cannot import gbench (sampling.shard_select). The gbench CLI
    already validated the spec before exporting it, so a malformed value here just means no shard."""
    spec = os.environ.get("GBENCH_SHARD", "").strip()
    if not spec or "/" not in spec:
        return None
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        return None
    return (i, n) if (n >= 1 and 1 <= i <= n) else None


def _selected_task_ids(index):
    """The task ids this run SELECTED - the true denominator (a task that never produced a
    score.json still counts as 0, matching upstream print_global_summary and no-skip honesty).
    GBENCH_SHARD selects shard I of N over the deterministic sorted id list (round-robin
    ids[i-1::n], matching sampling.shard_select) FIRST; WILDCLAW_LIMIT then caps to the first N
    within that shard (same compose order as the native path)."""
    single = os.environ.get("WILDCLAW_TASK", "").strip()
    if single:
        try:
            head = open(single, encoding="utf-8").read(4000)
        except OSError:
            return []
        mid = re.search(r"^id:\s*(.+)$", head, re.M)
        return [mid.group(1).strip()] if mid else []
    cats = [c.strip() for c in os.environ.get("WILDCLAW_CATEGORIES", "").split(",") if c.strip()]
    cats = set(cats or ALL_CATEGORIES)
    ids = sorted(tid for tid, meta in index.items() if meta["category"] in cats)
    shard = _shard()
    if shard is not None:
        i, n = shard
        ids = ids[i - 1::n]           # shard FIRST (over the full sorted set)
    limit = _limit()
    return ids[:limit] if limit is not None else ids   # then cap within the shard


def _write_models_config():
    """Write my_api.json: an OpenClaw custom provider aimed at the gbench model endpoint."""
    base_url = os.environ.get("GBENCH_MODEL_BASE_URL", "").rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url += "/v1"
    model = os.environ.get("GBENCH_MODEL_NAME", "served")
    cfg = {"providers": {_PROVIDER: {
        "baseUrl": base_url,
        "apiKey": os.environ.get("GBENCH_MODEL_API_KEY", "dummy"),
        "api": "openai-completions",
        "models": [{"id": model, "name": model}],
    }}}
    path = os.path.join(WORKDIR, "my_api.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path, f"{_PROVIDER}/{model}"


def _patch_task_network(docker_utils):
    """Force the sibling TASK containers onto the requested docker network (default: host).

    The upstream harness starts each task container on the default bridge and expects the model
    endpoint to be reachable from there (their leaderboard uses OpenRouter over the public
    internet). For a LOCAL gbench /v1 endpoint + the in-orchestrator judge proxy, the task
    container must reach the host: on a normal Linux host the bridge gateway works, but on a
    hardened host (FORWARD policy DROP) it does not, so we default to `--network host` (the task
    then reaches 127.0.0.1). This monkeypatch scopes STRICTLY to the container-start `docker run
    -d` in docker_utils.start_container; docker exec/cp/rm are untouched. Set WILDCLAW_TASK_NETWORK
    to a network name, or empty to keep the upstream bridge default.
    """
    net = os.environ.get("WILDCLAW_TASK_NETWORK", "host").strip()
    label = os.environ.get("WILDCLAW_TASK_LABEL", "").strip()  # run-scoped, for reaping on the host
    if not net and not label:
        return
    orig_run = docker_utils.subprocess.run

    def _run(cmd, *a, **k):
        if isinstance(cmd, list) and cmd[:3] == ["docker", "run", "-d"]:
            inject = []
            if net and "--network" not in cmd:
                inject += ["--network", net]
            if label and "--label" not in cmd:
                inject += ["--label", label]
            if inject:
                cmd = cmd[:3] + inject + cmd[3:]
        return orig_run(cmd, *a, **k)

    docker_utils.subprocess.run = _run


def _run_batch(models_config_path, model_arg):
    output_root = os.path.join(WORKDIR, "output")
    # run_batch + docker_utils read these at IMPORT time, so set them before importing.
    os.environ["OUTPUT_SUBDIR"] = output_root       # absolute -> lands on the identity mount
    os.environ["OPENROUTER_BASE_URL"] = os.environ.get(
        "WILDCLAW_JUDGE_ENDPOINT", os.environ.get("OPENROUTER_BASE_URL", ""))
    os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("WILDCLAW_JUDGE_API_KEY", "gbench-cascade"))
    os.environ.setdefault("JUDGE_MODEL", "gbench-cascade")
    os.environ["MY_PROXY_API_KEY"] = os.environ.get("GBENCH_MODEL_API_KEY", "dummy")
    os.environ["DOCKER_IMAGE"] = os.environ.get("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")

    # Drive the harness IN-PROCESS so the task-network monkeypatch takes effect (a subprocess would
    # not see it). Run from the identity-mounted harness root so `from src...` + workspace/ resolve.
    sys.path.insert(0, HARNESS_DIR)
    os.chdir(HARNESS_DIR)
    from src.utils import docker_utils           # noqa: E402
    _patch_task_network(docker_utils)
    from eval import run_batch as RB             # noqa: E402

    parallel = os.environ.get("WILDCLAW_PARALLEL", "1")
    common = ["--agent-backend", "openclaw", "--model", model_arg,
              "--models-config", models_config_path, "--parallel", str(parallel)]
    if os.environ.get("WILDCLAW_THINKING"):
        common += ["--thinking", os.environ["WILDCLAW_THINKING"]]

    single = os.environ.get("WILDCLAW_TASK", "").strip()
    invocations = []
    if single:
        invocations.append(["--task", single])
    elif _limit() is not None or _shard() is not None:
        # --eval-limit N and/or --shard I/N. Upstream run_batch has no --limit or --shard flag (only
        # --task / --category), so drive its single-task mode over the SELECTED task ids.
        # _selected_task_ids applies the SAME shard-then-cap, so the tasks we run == the tasks
        # _summarize uses as the denominator (run-set == score-set; n_tasks reflects the shard).
        index = _task_index()
        for tid in _selected_task_ids(index):
            path = index.get(tid, {}).get("path")
            if path:
                invocations.append(["--task", path])
    else:
        cats = [c.strip() for c in os.environ.get("WILDCLAW_CATEGORIES", "").split(",") if c.strip()]
        cats = cats or ALL_CATEGORIES
        for c in cats:
            invocations.append(["--category", c])

    for inv in invocations:
        argv = ["run_batch.py"] + inv + common
        sys.stderr.write(f"[wildclawbench] run_batch {' '.join(inv + common)}\n")
        sys.argv = argv
        # run_batch.main() sys.exit(1)s on a single-task failure; a category run never aborts. Don't
        # let that abort the launcher - the score tree is the source of truth.
        try:
            RB.main()
        except SystemExit:
            pass
        except Exception as exc:  # one category failing must not sink the rest
            sys.stderr.write(f"[wildclawbench] run_batch {inv} raised: {exc}\n")
    return output_root


def _score_of(score_json):
    """Per-task headline = overall_score, else mean of numeric metrics (harness semantics)."""
    numeric = {k: v for k, v in score_json.items() if isinstance(v, (int, float))}
    if not numeric:
        return None
    if "overall_score" in numeric:
        return float(numeric["overall_score"])
    return sum(numeric.values()) / len(numeric)


# The upstream tasks record a judge outage in several shapes: a `*_judge_method` OR bare
# `judge_method` set to a deterministic fallback ({regex,rule,keyword,heuristic}_fallback / "failed"),
# or a `*_judge_error` / bare `judge_error` key. Match all of them so judge_fallback_tasks is honest.
_FALLBACK_VALUES = {"regex_fallback", "rule_fallback", "keyword_fallback",
                    "heuristic_fallback", "failed"}


def _judge_fell_back(score_json):
    """True if any judge metric silently used a deterministic fallback (a judge outage)."""
    for k, v in score_json.items():
        kl = k.lower()
        if kl == "judge_error" or kl.endswith("judge_error"):
            return True
        if kl == "judge_method" or kl.endswith("judge_method"):
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in _FALLBACK_VALUES or "fallback" in vl or vl == "failed":
                    return True
    return False


def _collect(output_root, index):
    """Walk output/openclaw/<category>/<task_id>/<suffix>/{score.json,usage.json} -> rows by task_id.

    One row per task_id (the frontmatter id; newest suffix run wins), joined with modality.
    """
    oc_root = os.path.join(output_root, "openclaw")
    rows = {}
    for score_path in glob.glob(os.path.join(oc_root, "*", "*", "*", "score.json")):
        run_dir = os.path.dirname(score_path)
        task_id = os.path.basename(os.path.dirname(run_dir))
        category = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        try:
            score_json = json.load(open(score_path, encoding="utf-8"))
        except Exception:
            continue
        usage = {}
        up = os.path.join(run_dir, "usage.json")
        if os.path.exists(up):
            try:
                usage = json.load(open(up, encoding="utf-8"))
            except Exception:
                usage = {}
        mtime = os.path.getmtime(score_path)
        prev = rows.get(task_id)
        if prev and prev["_mtime"] >= mtime:
            continue
        rows[task_id] = {
            "_mtime": mtime,
            "category": index.get(task_id, {}).get("category", category),
            "task_id": task_id,
            "modality": index.get(task_id, {}).get("modality", "pure-text"),
            "overall_score": _score_of(score_json),
            "grading_error": score_json.get("error"),
            "judge_fallback": _judge_fell_back(score_json),
            "missing": False,
            "elapsed_time_s": usage.get("elapsed_time"),
            "cost_usd": usage.get("cost_usd"),
            "output_tokens": usage.get("output_tokens"),
        }
    return rows


def _summarize(output_root, index):
    """Roll the score tree into a summary. The denominator is the SELECTED task set: a task that
    produced no score.json (agent/setup crash, or a grade() that exited non-zero) counts as 0 -
    matching upstream print_global_summary (total_score / len(results)) and the no-skip policy."""
    found = _collect(output_root, index)
    selected = _selected_task_ids(index)
    rows = []
    for tid in selected:
        if tid in found:
            r = found[tid]
        else:
            r = {"category": index.get(tid, {}).get("category", "?"), "task_id": tid,
                 "modality": index.get(tid, {}).get("modality", "pure-text"),
                 "overall_score": None, "grading_error": None, "judge_fallback": False,
                 "missing": True, "elapsed_time_s": None, "cost_usd": None, "output_tokens": None}
        rows.append({k: v for k, v in r.items() if k != "_mtime"})
    # If the selection could not be resolved (no index / unknown task), fall back to whatever scored
    # so a smoke run still reports - but this should not happen for a normal categories/all run.
    if not rows:
        rows = [{k: v for k, v in r.items() if k != "_mtime"} for r in found.values()]

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    def _overall(subset):
        n = len(subset)
        s = sum(r["overall_score"] for r in subset if isinstance(r["overall_score"], (int, float)))
        return (s / n) if n else None      # missing/errored contribute 0 to the numerator, 1 to n

    n_tasks = len(rows)
    scored = [r for r in rows if isinstance(r["overall_score"], (int, float))]
    mm_rows = [r for r in rows if r["modality"] == "multimodal"]
    pt_rows = [r for r in rows if r["modality"] != "multimodal"]
    mm_overall = _overall(mm_rows)
    pt_overall = _overall(pt_rows)
    weighted = None
    if mm_overall is not None and pt_overall is not None:
        weighted = 0.5 * mm_overall + 0.5 * pt_overall
    return {
        "n_tasks": n_tasks,
        "n_scored": len(scored),
        "n_missing": sum(1 for r in rows if r.get("missing")),
        "n_grading_errors": sum(1 for r in rows if r.get("grading_error")),
        "global_mean": _overall(rows),                   # harness global_avg (headline), missing=0
        "multimodal_mean": mm_overall,
        "pure_text_mean": pt_overall,
        "weighted_overall": weighted,                    # 0.5*MM + 0.5*pure-text (approx of leaderboard)
        "n_multimodal": len(mm_rows),
        "n_pure_text": len(pt_rows),
        "avg_time_min": (_mean([r["elapsed_time_s"] for r in rows]) / 60.0)
                        if _mean([r["elapsed_time_s"] for r in rows]) is not None else None,
        "avg_cost_usd": _mean([r["cost_usd"] for r in rows]),
        "judge_fallback_tasks": sum(1 for r in rows if r["judge_fallback"]),
        "per_task": sorted(rows, key=lambda r: (r["category"], r["task_id"])),
    }


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    models_config_path, model_arg = _write_models_config()
    output_root = _run_batch(models_config_path, model_arg)
    index = _task_index()
    summary = _summarize(output_root, index)
    with open(os.path.join(WORKDIR, "wildclawbench_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    sys.stderr.write(f"[wildclawbench] summary: n_tasks={summary['n_tasks']} "
                     f"scored={summary['n_scored']} global_mean={summary['global_mean']} "
                     f"judge_fallback={summary['judge_fallback_tasks']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
