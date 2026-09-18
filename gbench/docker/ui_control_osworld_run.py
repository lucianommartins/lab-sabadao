#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# gbench launcher for OSWorld, run INSIDE the gbench-ui-control-osworld image. It drives the pinned
# OSWorld harness (run.py's test loop) IN-PROCESS so the model-wiring monkeypatch takes effect, then
# parses the per-example result.txt tree -> a gbench summary. It NEVER fabricates a score.
#
# Model wiring: OSWorld's mm_agents.PromptAgent.call_llm routes by model-name PREFIX; the "gpt"
# branch honours OPENAI_BASE_URL. But vLLM strictly rejects an unknown model name, so we (1) pass
# --model gpt-4o (a routing alias -> the OpenAI-compat path), (2) set OPENAI_BASE_URL to the served
# /v1 endpoint, and (3) monkeypatch mm_agents.agent.requests.post to rewrite the outgoing payload
# `model` field to the real served name. Scoring is upstream's DETERMINISTIC per-task evaluators
# (result.txt in [0,1]) - no judge.
#
# docker-out-of-docker IDENTITY-MOUNT contract (the skillsbench/wildclaw pitfall): OSWorld's docker
# provider bind-mounts the Ubuntu qcow2 into the sibling VM container via os.path.abspath(path_to_vm),
# and VMS_DIR is "./docker_vm_data" relative to CWD - so the launcher chdir's to the identity-mounted
# WORKDIR (which holds docker_vm_data/Ubuntu.qcow2 + results/), and that path must be identical on the
# host daemon (the gbench runner arranges the identity mount).
#
# Env contract (set by the entrypoint / docker run -e):
#   OSWORLD_DIR            harness root (default /app/OSWorld)
#   OSWORLD_WORKDIR        identity-mounted workdir: docker_vm_data/ (qcow2) + results/ (default /out)
#   GBENCH_MODEL_NAME      served model name (rewritten into the outgoing payload)
#   GBENCH_MODEL_BASE_URL  model /v1 endpoint (OPENAI_BASE_URL; --network host -> 127.0.0.1)
#   GBENCH_MODEL_API_KEY   api key (dummy for vLLM)
#   OSWORLD_DOMAIN         single domain, or "all" (default all)
#   GBENCH_SHARD           "I/N" round-robin shard over the DETERMINISTIC sorted (domain, example_id)
#                          full task list (applied BEFORE OSWORLD_LIMIT); unset/malformed => full set
#   OSWORLD_LIMIT          cap total examples (smoke; applied WITHIN the shard)
#   OSWORLD_MAX_STEPS      per-task step budget (default 15)
#   OSWORLD_OBS_TYPE       screenshot | a11y_tree | screenshot_a11y_tree | som (default screenshot)
#   OSWORLD_TEST_META      manifest filename under evaluation_examples/ (default test_all.json)

import json
import os
import sys

OSWORLD_DIR = os.environ.get("OSWORLD_DIR", "/app/OSWorld")
WORKDIR = os.environ.get("OSWORLD_WORKDIR", "/out")
_ROUTING_ALIAS = "gpt-4o"   # takes PromptAgent.call_llm's OPENAI_BASE_URL path


def _install_model_rewrite(served: str):
    """Rewrite the outgoing OpenAI-compat payload `model` (routing alias) -> the real served name,
    since vLLM rejects an unknown model id. Scoped to chat/completions POSTs only."""
    import mm_agents.agent as A
    orig_post = A.requests.post

    def _post(url, *args, **kwargs):
        j = kwargs.get("json")
        if isinstance(j, dict) and j.get("model") == _ROUTING_ALIAS and "chat/completions" in str(url):
            j = dict(j)
            j["model"] = served
            kwargs["json"] = j
        return orig_post(url, *args, **kwargs)

    A.requests.post = _post


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


def _build_subset_meta():
    """Load evaluation_examples/<manifest> (domain -> [example_ids]); optionally filter by domain;
    apply the shard (round-robin over the DETERMINISTIC sorted (domain, example_id) full task list)
    FIRST, then the limit caps WITHIN the shard; rebuild + write the per-domain subset used as
    --test_all_meta_path. Compose order matches the native path (base.run_eval_suite: shard then
    limit). Returns (path, selected {domain: [ids]})."""
    manifest = os.environ.get("OSWORLD_TEST_META", "test_all.json")
    src = os.path.join(OSWORLD_DIR, "evaluation_examples", manifest)
    with open(src, encoding="utf-8") as f:
        meta = json.load(f)
    domain = os.environ.get("OSWORLD_DOMAIN", "all").strip()
    if domain and domain != "all":
        meta = {domain: meta.get(domain, [])}

    # Flatten to a DETERMINISTIC, SORTED full task list so the shard partition is reproducible across
    # machines and matches sampling.shard_select's "already-ordered sequence" contract.
    flat = sorted((d, ex) for d, ids in meta.items() for ex in ids)

    # Shard FIRST: shard I of N = flat[I-1::N] (non-overlapping; the N shards union back to the full set).
    shard = _shard()
    if shard:
        i, n = shard
        flat = flat[i - 1::n]

    # Then the limit caps WITHIN the shard (head of the sorted+sharded list).
    limit = os.environ.get("OSWORLD_LIMIT", "").strip()
    if limit and int(limit) > 0:
        flat = flat[:int(limit)]

    # Rebuild the per-domain manifest from the selected set, preserving deterministic order. Empty
    # domains fall out naturally (only selected pairs are added).
    selected: dict = {}
    for d, ex in flat:
        selected.setdefault(d, []).append(ex)

    out = os.path.join(WORKDIR, "test_subset.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(selected, f)
    return out, selected


def _run_osworld(subset_path):
    import run as OS  # OSWorld run.py (safe to import: no __main__ side effects)
    obs = os.environ.get("OSWORLD_OBS_TYPE", "screenshot")
    results_dir = os.path.join(WORKDIR, "results")
    sys.argv = ["run.py",
                "--provider_name", "docker",
                "--headless",
                "--action_space", "pyautogui",
                "--observation_type", obs,
                "--model", _ROUTING_ALIAS,
                "--result_dir", results_dir,
                "--test_all_meta_path", subset_path,
                "--max_steps", os.environ.get("OSWORLD_MAX_STEPS", "15"),
                "--domain", "all"]   # subset already narrowed by _build_subset_meta
    args = OS.config()
    with open(subset_path, encoding="utf-8") as f:
        test_all_meta = json.load(f)
    test_file_list = OS.get_unfinished(args.action_space, args.model, args.observation_type,
                                       args.result_dir, test_all_meta)
    OS.test(args, test_file_list)
    return args.action_space, args.model, obs, results_dir


def _summarize(action_space, use_model, obs, results_dir, selected):
    """Parse result_dir/<action_space>/<obs>/<use_model>/<domain>/<example>/result.txt (float in
    [0,1]). Denominator = SELECTED examples; a missing result.txt counts as 0 (no-skip honesty)."""
    base = os.path.join(results_dir, action_space, obs, use_model)

    def _read(domain, ex):
        p = os.path.join(base, domain, ex, "result.txt")
        if not os.path.exists(p):
            return None
        try:
            return float(open(p, encoding="utf-8").read().strip())
        except Exception:
            try:
                return float(eval(open(p, encoding="utf-8").read().strip()))  # some tasks write exprs
            except Exception:
                return None

    per_domain, all_scores, n_selected, n_scored = {}, [], 0, 0
    for domain, ids in selected.items():
        dvals = []
        for ex in ids:
            n_selected += 1
            v = _read(domain, ex)
            if isinstance(v, (int, float)):
                n_scored += 1
                dvals.append(v)
                all_scores.append(v)
            else:
                dvals.append(0.0)          # missing/errored task counts as 0
                all_scores.append(0.0)
        per_domain[domain] = {"success_rate": (sum(dvals) / len(dvals)) if dvals else None,
                              "n": len(ids), "n_scored": sum(1 for e in ids if _read(domain, e) is not None)}
    overall = (sum(all_scores) / len(all_scores)) if all_scores else None
    return {
        "success_rate": overall,
        "n_selected": n_selected,
        "n_scored": n_scored,
        "n_missing": n_selected - n_scored,
        "per_domain": per_domain,
        "action_space": action_space,
        "observation_type": obs,
        "model_label": use_model,
    }


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)                       # docker_vm_data/ + results/ under the identity mount
    os.environ["OPENAI_BASE_URL"] = os.environ.get("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    os.environ["OPENAI_API_KEY"] = os.environ.get("GBENCH_MODEL_API_KEY", "dummy")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, OSWORLD_DIR)
    _install_model_rewrite(os.environ.get("GBENCH_MODEL_NAME", "served"))

    subset_path, selected = _build_subset_meta()
    action_space, use_model, obs, results_dir = _run_osworld(subset_path)
    summary = _summarize(action_space, use_model, obs, results_dir, selected)
    with open(os.path.join(WORKDIR, "ui_control_osworld_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    sys.stderr.write(f"[ui_control_osworld] summary: success_rate={summary['success_rate']} "
                     f"n_selected={summary['n_selected']} n_scored={summary['n_scored']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
