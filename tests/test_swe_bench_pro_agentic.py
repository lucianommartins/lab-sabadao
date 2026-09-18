# -*- coding: utf-8 -*-
"""Shared agentic SWE-bench Pro runner (built-in `swe_bench_pro`).

Docker/gated-image execution is validated by the operator on a subset; these cover the parts
that must be right before spending compute: config wiring, the instance filter, orchestration/
result assembly, the mini-swe -> harness preds bridge, and log routing.
"""
import json
import os
import sys
import types

import pytest

os.environ.setdefault("SWE_BENCH_PRO_HARNESS_DIR", os.path.expanduser("~/.cache/gbench/swe-bench-pro-os"))
from gbench.runners.eval_suites import swe_bench_pro_agentic as A  # noqa: E402


def test_build_agent_config_points_at_vllm_and_pro_image_quirks():
    # build_agent_config loads the canonical mini-swe-agent config (minisweagent.config),
    # so this test needs the optional dep; skip cleanly when it is not installed.
    pytest.importorskip("minisweagent")
    cfg = A.build_agent_config("google/gemma-4-26B-A4B-it", "http://127.0.0.1:8000/v1",
                               step_limit=80, cost_limit=5.0, temperature=1.0, max_tokens=8192)
    m = cfg["model"]
    assert m["model_name"] == "openai/google/gemma-4-26B-A4B-it"
    assert m["model_kwargs"]["api_base"] == "http://127.0.0.1:8000/v1"
    assert m["model_kwargs"]["api_key"]
    assert m["model_kwargs"]["max_tokens"] == 8192            # per-turn output cap (from run flag)
    assert m["model_kwargs"]["temperature"] == 1.0           # honors gbench --temperature
    assert m["cost_tracking"] == "ignore_errors"             # local model has no price
    assert cfg["agent"]["step_limit"] == 80
    assert "cwd" not in cfg["agent"]                         # not an AgentConfig field
    assert cfg["environment"]["cwd"] == "/app"               # Pro repo path
    assert cfg["environment"]["run_args"] == ["--rm", "--entrypoint", ""]   # keep container alive
    joined = " ".join(v for v in cfg["agent"].values() if isinstance(v, str))
    assert "/testbed" not in joined and "/app" in joined     # prompt paths rewritten


def test_build_instances_respects_the_id_filter_and_limit(monkeypatch):
    fake = types.ModuleType("generate_sweagent_instances")
    fake.generate_instances = lambda user: [
        {"instance_id": f"i{i}", "image_name": f"img{i}", "problem_statement": "x"} for i in range(5)]
    monkeypatch.setitem(sys.modules, "generate_sweagent_instances", fake)
    monkeypatch.setattr(A, "_add_harness_to_path", lambda: "/x")
    # full set
    assert len(A.build_instances(only_instance_ids=None)) == 5
    # filtered
    got = A.build_instances(only_instance_ids={"i1", "i3"})
    assert {g["instance_id"] for g in got} == {"i1", "i3"}
    # limited
    assert len(A.build_instances(only_instance_ids=None, limit=2)) == 2


def test_limit_is_stratified_across_repos_not_a_contiguous_head(monkeypatch):
    # Pool stored grouped by repo (RC-1 shape): 5 repos x 20 instances each, repo A first.
    ids = [f"instance_org__repo{r}-{'%040x' % (r * 100 + i)}-vnan"
           for r in range(5) for i in range(20)]
    fake = types.ModuleType("generate_sweagent_instances")
    fake.generate_instances = lambda user: [
        {"instance_id": iid, "image_name": "x", "problem_statement": "x"} for iid in ids]
    monkeypatch.setitem(sys.modules, "generate_sweagent_instances", fake)
    monkeypatch.setattr(A, "_add_harness_to_path", lambda: "/x")
    got = A.build_instances(limit=5, seed="swe_bench_pro")
    repos = {A._repo_of(g) for g in got}
    assert len(got) == 5
    assert len(repos) == 5, f"limit collapsed to {repos} instead of spanning all 5 repos"


def test_repo_of_recovers_org_repo_even_with_hyphenated_repo():
    assert A._repo_of({"instance_id": "instance_NodeBB__NodeBB-%s-vnan" % ("0" * 40)}) == "NodeBB__NodeBB"
    # element-web has a '-' in the repo name; the 40-hex commit is the split point.
    assert A._repo_of({"instance_id": "instance_element-hq__element-web-%s-v%s" % ("a" * 40, "b" * 40)}) \
        == "element-hq__element-web"


def test_execute_agentic_assembles_result_with_eval_name(monkeypatch):
    fake = types.ModuleType("minisweagent.run.extra.swebench")
    fake.process_instance = lambda inst, out, cfg, pm: None
    monkeypatch.setitem(sys.modules, "minisweagent.run.extra.swebench", fake)
    monkeypatch.setattr(A, "build_instances", lambda only_instance_ids=None, limit=None, seed="x": [
        {"instance_id": i, "image_name": f"img:{i}", "problem_statement": "x"} for i in ("a", "b", "c")])
    monkeypatch.setattr(A, "build_agent_config", lambda *a, **k: {"model": {}, "agent": {}, "environment": {}})
    monkeypatch.setattr(A, "_score_patches", lambda preds, num_workers, eval_name: {
        "results": {"a": True, "b": False, "c": True}, "total_instances": 3,
        "resolved_instances": 2, "empty_patch_instances": 0, "submitted_with_patch": 3})

    res = A.execute_agentic("m", "http://x", concurrency=2, limit=3, eval_name="swe_bench_pro")
    assert res["status"] == "success" and res["eval_name"] == "swe_bench_pro"
    assert res["total_questions"] == 3 and res["correct_answers"] == 2 and res["accuracy"] == 66.67
    assert res["mode"] == "agentic_mini_swe_agent"


def test_execute_agentic_no_instances_is_error_not_fake_zero(monkeypatch):
    monkeypatch.setattr(A, "build_instances", lambda only_instance_ids=None, limit=None, seed="x": [])
    res = A.execute_agentic("m", "http://x", concurrency=1, eval_name="swe_bench_pro")
    assert res["status"] == "error" and res["total_questions"] == 0


def test_score_patches_bridges_preds_and_labels_prefix(monkeypatch, tmp_path):
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps({
        "i1": {"model_name_or_path": "m", "model_patch": "diff --git a/x b/x\n..."},
        "i2": {"model_name_or_path": "m", "model_patch": "   "},
    }))
    import gbench.runners.eval_suites.swe_bench_pro as SBP
    from gbench.runners.eval_suites import swe_thread_cap
    monkeypatch.setattr(SBP, "_harness_dir", lambda: str(tmp_path))
    monkeypatch.setattr(SBP, "raw_sample_path", lambda: str(tmp_path / "raw.csv"))
    monkeypatch.setattr(swe_thread_cap, "apply", lambda cmd, w, name: (cmd, None))
    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        captured["cmd"] = cmd
        out = next(c.split("=", 1)[1] for c in cmd if c.startswith("--output_dir="))
        (__import__("pathlib").Path(out) / "eval_results.json").write_text(json.dumps({"i1": True, "i2": False}))
        captured["preds"] = json.loads(open(next(c.split("=", 1)[1] for c in cmd if c.startswith("--patch_path="))).read())
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    rep = A._score_patches(preds, num_workers=1, eval_name="swe_bench_pro")
    assert rep["total_instances"] == 2 and rep["resolved_instances"] == 1
    assert rep["empty_patch_instances"] == 1 and rep["submitted_with_patch"] == 1
    assert captured["preds"][0]["prefix"] == "gbench__swe_bench_pro_agentic"
    # scorer must run under THIS interpreter (pandas/docker live in the gbench venv), never bare
    # "python" off PATH - otherwise a wrong PATH silently zeros every instance.
    import sys as _sys
    assert captured["cmd"][0] == _sys.executable


def test_agent_logs_are_redirected_to_file_and_restored(monkeypatch, tmp_path):
    import logging
    import litellm
    lg = logging.getLogger("minisweagent")
    lg.propagate = True
    orig_handlers = lg.handlers[:]
    logging.getLogger("LiteLLM").setLevel(logging.INFO)
    litellm.suppress_debug_info = False
    during = {}
    fake = types.ModuleType("minisweagent.run.extra.swebench")
    fake.process_instance = lambda inst, out, cfg, pm: during.update(
        propagate=logging.getLogger("minisweagent").propagate,
        to_file=any(isinstance(h, logging.FileHandler) for h in logging.getLogger("minisweagent").handlers),
        litellm_level=logging.getLogger("LiteLLM").level, suppress=litellm.suppress_debug_info)
    monkeypatch.setitem(sys.modules, "minisweagent.run.extra.swebench", fake)
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(A, "build_instances", lambda only_instance_ids=None, limit=None, seed="x": [
        {"instance_id": "a", "image_name": "i", "problem_statement": "x"}])
    monkeypatch.setattr(A, "build_agent_config", lambda *a, **k: {"model": {}, "agent": {}, "environment": {}})
    monkeypatch.setattr(A, "_score_patches", lambda preds, num_workers, eval_name: {
        "results": {"a": True}, "total_instances": 1, "resolved_instances": 1,
        "empty_patch_instances": 0, "submitted_with_patch": 1})

    A.execute_agentic("m", "http://x", concurrency=1, eval_name="swe_bench_pro")
    assert during["propagate"] is False and during["to_file"] is True
    assert during["litellm_level"] == logging.WARNING and during["suppress"] is True
    assert lg.propagate is True and lg.handlers == orig_handlers
    assert logging.getLogger("LiteLLM").level == logging.INFO
    assert litellm.suppress_debug_info is False
    assert (tmp_path / "swe_bench_pro_agent.log").exists()   # named per eval


def test_builtin_swe_bench_pro_routes_to_agentic_when_flagged(monkeypatch):
    import gbench.runners.eval_suites.swe_bench_pro as B
    monkeypatch.setattr(B, "check_swe_bench_pro_prerequisites", lambda: (True, ""))
    seen = {}
    monkeypatch.setattr(A, "execute_agentic",
                        lambda *a, **k: seen.update(kwargs=k, args=a) or {"ok": 1})
    monkeypatch.setenv("GBENCH_SWEBENCH_PRO_AGENTIC", "1")
    B.run_swe_bench_pro("m", "http://x", concurrency=1, limit=3)
    # built-in runs the FULL set (no id filter) under its own eval_name
    assert seen["kwargs"]["only_instance_ids"] is None
    assert seen["kwargs"]["eval_name"] == "swe_bench_pro"
