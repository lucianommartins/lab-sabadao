#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# gbench launcher for MCP-Bench, run INSIDE the gbench-mcp-bench image. It:
#   1. injects the gbench Gemini cascade judge into BenchmarkRunner (monkeypatch on __init__,
#      so it fires on the CLI path too and the hard-coded Azure o4-mini judge never runs);
#   2. filters the canonical task GROUPS down to the servers actually provisioned in this
#      environment (offline servers always; key-gated servers only if their key is set;
#      network servers only if egress works) and writes a manifest of what ran vs was dropped
#      - "canonical-when-provisioned", never a silent skip or a fabricated number;
#   3. points the model-under-test at the gbench /v1 endpoint via the openai_compatible provider;
#   4. delegates to upstream benchmark.runner.main() (reusing all its glue) and writes the
#      averaged-metrics JSON to /out. The gbench harness computes the 0-1 Overall Score from it.
#
# Env contract (set by the entrypoint / docker run -e):
#   MCPBENCH_DIR            repo root (default /app/mcp-bench)
#   MCP_MODEL_CONFIG_NAME   the --models key hijacked for the endpoint (default llama-3-1-8b)
#   MCP_TASK_FILES          comma-sep source task files (default the 3 canonical files)
#   MCP_OUTPUT              output JSON path (default /out/mcp_bench_results.json)
#   MCP_MANIFEST            subset manifest path (default /out/subset_manifest.json)
#   MCP_DISTRACTION_COUNT   pass-through to --distraction-count (optional)
#   MCP_DISABLE_STABILITY   if set, adds --disable-judge-stability
#   MCP_NO_SUBSET_FILTER    if set, run every task regardless of provisioning
#   MCP_TASK_LIMIT          cap the number of tasks run (applied AFTER the shard)
#   GBENCH_SHARD            "I/N" round-robin shard over the DETERMINISTIC sorted full task-id list
#                           (ids[I-1::N], 1-indexed); applied BEFORE MCP_TASK_LIMIT, matching the
#                           native path (base.run_eval_suite: shard then limit)
#   GOOGLE_MAPS_API_KEY / NCI_API_KEY / HF_TOKEN / NPS_API_KEY / NASA_API_KEY  (optional keys)

import json
import os
import socket
import sys

MCPBENCH_DIR = os.environ.get("MCPBENCH_DIR", "/app/mcp-bench")
ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MCPBENCH_DIR)
sys.path.insert(0, ADAPTER_DIR)

# --- server provisioning model (names match mcp_servers/commands.json + task group `servers`) ---
_OFFLINE = {"Bibliomantic", "Math MCP", "Medical Calculator", "Scientific Computing",
            "Time MCP", "Unit Converter"}
# key-gated servers that ALSO need network (available iff key set AND egress up)
_KEY_REQUIRED = {"Google Maps": "GOOGLE_MAPS_API_KEY", "Hugging Face": "HF_TOKEN",
                 "National Parks": "NPS_API_KEY", "NASA Data": "NASA_API_KEY"}
# BioMCP needs network; its NCI_API_KEY is OPTIONAL (most tools work without it) -> network-gated only.


def _network_up() -> bool:
    if os.environ.get("MCP_ASSUME_NETWORK"):
        return True
    for host in ("api.github.com", "8.8.8.8"):
        try:
            s = socket.create_connection((host, 443 if host != "8.8.8.8" else 53), timeout=5)
            s.close()
            return True
        except Exception:
            continue
    return False


def _available_servers(all_servers):
    net = _network_up()
    avail = set()
    for s in all_servers:
        if s in _OFFLINE:
            avail.add(s)
        elif not net:
            continue
        elif s in _KEY_REQUIRED:
            if os.environ.get(_KEY_REQUIRED[s]):
                avail.add(s)
        else:  # network server (incl. BioMCP)
            avail.add(s)
    return avail, net


# fastmcp-based servers run from their OWN venv (they need mcp>=2, conflicting with the system
# mcp<2 the v1 servers need). Repoint their commands.json cmd at that venv's python.
_VENV_SERVERS = {
    "Unit Converter": ("unit-converter-mcp", "-m unit_converter_mcp.server"),
    "Game Trends": ("game-trends-mcp", "server.py"),
    "OSINT Intelligence": ("mcp-osint-server", "mcp_osint_server/mcp_osint_server/main.py"),
    "Paper Search": ("paper-search-mcp", "-m paper_search_mcp.server"),
}


def _patch_venv_commands(mcpbench_dir):
    """Repoint the fastmcp servers at their per-server venv python (idempotent). Only patches a
    server whose .venv actually exists, so a server that failed to build stays as-is (and is then
    dropped by the build/connectivity check)."""
    cj_path = os.path.join(mcpbench_dir, "mcp_servers", "commands.json")
    try:
        with open(cj_path, encoding="utf-8") as f:
            cj = json.load(f)
    except Exception:
        return
    changed = False
    for name, (d, tail) in _VENV_SERVERS.items():
        if name not in cj:
            continue
        vpy = os.path.join(mcpbench_dir, "mcp_servers", d, ".venv", "bin", "python")
        if os.path.isfile(vpy):
            new_cmd = vpy + " " + tail
            if cj[name].get("cmd") != new_cmd:
                cj[name]["cmd"] = new_cmd
                changed = True
    if changed:
        with open(cj_path, "w", encoding="utf-8") as f:
            json.dump(cj, f, indent=2)


def _built_servers(mcpbench_dir):
    """Servers whose entrypoint artifact (and, for `uv run` servers, .venv) actually exists in the
    image = the ones that BUILT and can be spawned. Returns None if commands.json is unreadable
    (then no build-based filtering is applied). This keeps the run honest when a server fails to
    build: its tasks are dropped and reported, never scored as model failures."""
    cj_path = os.path.join(mcpbench_dir, "mcp_servers", "commands.json")
    try:
        with open(cj_path, encoding="utf-8") as f:
            cj = json.load(f)
    except Exception:
        return None
    base = os.path.join(mcpbench_dir, "mcp_servers")
    built = set()
    for name, spec in cj.items():
        cwd = (spec.get("cwd") or "").replace("../", "")
        d = os.path.join(base, cwd)
        if not os.path.isdir(d):
            continue
        cmd = spec.get("cmd", "")
        art = next((p for p in cmd.split() if p.endswith(".js") or p.endswith(".py")), None)
        # explicit-file entrypoints must exist; `python -m`/`uv run <pkg>` invocations have no
        # file to check (their deps live in system python / the uv venv) so are assumed present.
        art_ok = os.path.isfile(os.path.join(d, art)) if art else True
        venv_ok = os.path.isdir(os.path.join(d, ".venv")) if cmd.startswith("uv run") else True
        if art_ok and venv_ok:
            built.add(name)
    return built


def _all_task_servers(task_files):
    servers = set()
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("server_tasks", []):
            servers.update(g.get("servers") or [])
    return servers


def _shard():
    """Parse GBENCH_SHARD="I/N" (1-indexed shard I of N). Returns (i, n) or None (unset/malformed).
    Launchers run inside the container and cannot import gbench, so this mirrors sampling.parse_shard
    inline."""
    spec = os.environ.get("GBENCH_SHARD", "").strip()
    if not spec or "/" not in spec:
        return None
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        return None
    return (i, n) if (n >= 1 and 1 <= i <= n) else None


def _sharded_task_ids(task_files, shard):
    """The set of task_ids selected for `shard` = (i, n), via round-robin over the DETERMINISTIC
    SORTED full task-id list across ALL task files: ids[i-1::n] (1-indexed, non-overlapping; the N
    shards union back to the full set, matching sampling.shard_select). Returns None when `shard`
    is None (keep every task). Each mcp_bench task carries a stable, globally-unique task_id, so an
    interleaved subset is exactly expressible."""
    if shard is None:
        return None
    ids = []
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("server_tasks", []):
            for t in g.get("tasks", []):
                tid = t.get("task_id")
                if tid is not None:
                    ids.append(tid)
    ids = sorted(set(ids))
    i, n = shard
    return set(ids[i - 1::n])


def _filter_file(path, allowed, out_dir, budget=None, keep_ids=None):
    """Rewrite `path` down to the tasks that will actually run; return
    (out_path, kept, dropped, remaining_budget).

    Composition matches the native path (base.run_eval_suite: shard FIRST, then the limit):
      * SHARD: if `keep_ids` is not None, only tasks whose `task_id` is in it are in scope for this
        shard. Tasks outside the shard belong to another shard and are neither kept nor counted as
        dropped (so the denominator reflects only the shard).
      * PROVISIONING: a group is runnable only if every server it needs is provisioned (`servers`
        subset of `allowed`); an in-shard task in an unprovisioned group is dropped.
      * LIMIT: with a `budget` (from MCP_TASK_LIMIT) the run caps at `budget` tasks in file order
        WITHIN the shard; in-shard, provisioned tasks past the budget are dropped.
    A dropped task counts against the denominator (reported in the manifest) - never a silent skip."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kept_groups, kept, dropped = [], 0, 0
    for g in data.get("server_tasks", []):
        need = set(g.get("servers") or [])
        # SHARD first: restrict to this shard's tasks (all tasks when not sharding).
        in_shard = [t for t in g.get("tasks", [])
                    if keep_ids is None or t.get("task_id") in keep_ids]
        if not in_shard:
            continue
        if not need.issubset(allowed):
            dropped += len(in_shard)          # in-shard but unprovisioned -> dropped (counts as 0)
            continue
        sel = []
        for t in in_shard:
            if budget is None or budget > 0:
                sel.append(t)
                if budget is not None:
                    budget -= 1
            else:
                dropped += 1                  # in-shard, provisioned, past the limit
        if sel:
            g = dict(g)                       # copy before slicing so the loaded data is untouched
            g["tasks"] = sel
            kept_groups.append(g)
            kept += len(sel)
    data["server_tasks"] = kept_groups
    data["total_tasks"] = kept
    out_path = os.path.join(out_dir, "filtered_" + os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return out_path, kept, dropped, budget


def main():
    import benchmark.runner as R
    from mcp_bench_cascade_judge import CascadeGeminiJudge

    # 0) repoint the fastmcp servers at their isolated venvs (before the runner reads commands.json)
    _patch_venv_commands(MCPBENCH_DIR)

    # 1) inject the cascade judge (fires even on the CLI path; neutralizes the Azure default)
    _orig_init = R.BenchmarkRunner.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        if getattr(self, "_judge_provider", None) is None:
            self._judge_provider = CascadeGeminiJudge()
    R.BenchmarkRunner.__init__ = _patched_init

    out = os.environ.get("MCP_OUTPUT", "/out/mcp_bench_results.json")
    manifest_path = os.environ.get("MCP_MANIFEST", "/out/subset_manifest.json")
    out_dir = os.path.dirname(out) or "/out"

    default_tasks = ",".join(os.path.join(MCPBENCH_DIR, "tasks", f) for f in (
        "mcpbench_tasks_single_runner_format.json",
        "mcpbench_tasks_multi_2server_runner_format.json",
        "mcpbench_tasks_multi_3server_runner_format.json"))
    task_files = [t.strip() for t in os.environ.get("MCP_TASK_FILES", default_tasks).split(",") if t.strip()]

    # 2) SHARD (round-robin over the DETERMINISTIC sorted full task-id list) is applied FIRST, then
    #    the provisioning subset filter, then the MCP_TASK_LIMIT budget - matching the native path's
    #    compose order. keep_ids is None (keep every task) unless a shard is active.
    shard = _shard()
    shard_manifest = {"i": shard[0], "n": shard[1]} if shard else None
    if shard:
        print(f"mcp_bench: shard {shard[0]}/{shard[1]} over the sorted full task-id list",
              file=sys.stderr)
    keep_ids = _sharded_task_ids(task_files, shard)

    # 3) subset filter by provisioned servers (unless disabled)
    manifest = {"filtered": False}
    if os.environ.get("MCP_NO_SUBSET_FILTER"):
        if keep_ids is None:
            # No shard + no subset filter: pass the files through unchanged (original behavior).
            run_files = task_files
            manifest = {"filtered": False, "task_files": task_files, "shard": None}
        else:
            # Shard active: still select the shard subset, but never drop for provisioning
            # (allowed = every server) and honour no explicit task limit here (as before).
            all_servers = _all_task_servers(task_files)
            run_files, per_file = [], {}
            total_kept = total_dropped = 0
            for path in task_files:
                fp, kept, dropped, _ = _filter_file(path, all_servers, out_dir, None, keep_ids)
                per_file[os.path.basename(path)] = {"kept": kept, "dropped": dropped}
                total_kept += kept
                total_dropped += dropped
                if kept:
                    run_files.append(fp)
            manifest = {"filtered": False, "shard": shard_manifest, "per_file": per_file,
                        "tasks_kept": total_kept, "tasks_dropped": total_dropped}
    else:
        all_servers = _all_task_servers(task_files)
        allowed, net = _available_servers(all_servers)
        # Intersect provisioning (key/network) with what actually built in the image.
        built = _built_servers(MCPBENCH_DIR)
        unbuilt = sorted(all_servers - built) if built is not None else []
        if built is not None:
            allowed = allowed & built
        lim = os.environ.get("MCP_TASK_LIMIT")
        budget = int(lim) if (lim and int(lim) > 0) else None
        run_files, per_file = [], {}
        total_kept = total_dropped = 0
        for path in task_files:
            fp, kept, dropped, budget = _filter_file(path, allowed, out_dir, budget, keep_ids)
            per_file[os.path.basename(path)] = {"kept": kept, "dropped": dropped}
            total_kept += kept
            total_dropped += dropped
            if kept:
                run_files.append(fp)
        manifest = {
            "filtered": True,
            "network_up": net,
            "available_servers": sorted(allowed),
            "unavailable_servers": sorted(all_servers - allowed),
            "unbuilt_servers": unbuilt,
            "per_file": per_file,
            "tasks_kept": total_kept,
            "tasks_dropped": total_dropped,
            "task_limit": (int(lim) if (lim and int(lim) > 0) else None),
            "shard": shard_manifest,
        }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not run_files:
        print("mcp_bench: no tasks are runnable with the provisioned servers.", file=sys.stderr)
        # still write an (empty) output so the harness reports a clean provisioning error
        with open(out, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return

    # 3) build argv (model-under-test env is set by the entrypoint) + delegate to upstream main()
    argv = ["run_benchmark.py",
            "--models", os.environ.get("MCP_MODEL_CONFIG_NAME", "llama-3-1-8b"),
            "--tasks-file", ",".join(run_files),
            "--output", out]
    dc = os.environ.get("MCP_DISTRACTION_COUNT")
    if dc is not None and dc != "":
        argv += ["--distraction-count", str(dc)]
    if os.environ.get("MCP_DISABLE_STABILITY"):
        argv += ["--disable-judge-stability"]
    sys.argv = argv

    import asyncio
    asyncio.run(R.main())


if __name__ == "__main__":
    main()
