# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Tier-4 agentic / execution-environment suites: swe_lancer, gaia2, tau2/tau3 env.

- swe_lancer: gated OpenAI SWELancer Docker harness (substring heuristic removed).
- gaia2: canonical Meta-ARE harness delegated to a LOCAL container (are-benchmark); missing
  Docker / orchestrator image / GEMINI judge hard-error via infra_required (no skip, no proxy).
  Deterministic parse/gate helpers are covered in tests/test_gaia2.py.
- tau2/tau3: a plain single-turn /v1 endpoint CANNOT score tau-bench (assertion-only
  tasks + environment-dependent gold args), so the default reports a clean skip instead
  of a misleading 0%; the only scored path is the gated full environment (TAU2_ENV_RUN=1
  + the tau2-bench simulator), which itself skips cleanly when opted-in-but-unavailable.
- gdpval: grades produced file deliverables, so a text endpoint can't be scored -> skips.

All harnesses are absent on CI, so these lock in the gates, skip dicts, loaders (no
fabrication), and the env result-parsing/aggregation that runs fully offline.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from gbench.runners.eval_suites import SUITES


# --------------------------------------------------------------------------- #
# registration + sandbox wiring
# --------------------------------------------------------------------------- #
def test_tier4_suites_registered():
    # swe_lancer is DEREGISTERED (roadmap-only: upstream Expensify image drift) - see test_missing_evals.
    for name in ("gaia2", "tau2", "tau3"):
        assert name in SUITES, f"{name} not registered in SUITES"


@pytest.mark.parametrize("name", ["tau2", "tau3"])
def test_tier4_execution_suites_are_sandbox_evals(name):
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    config = BenchmarkConfig()
    config.sandboxes = 5
    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES",
                    {name: MagicMock(return_value={"status": "success"})}) as mock_suites:
        runner._run_single_eval(name, "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=128)
        assert mock_suites[name].call_args[1]["concurrency"] == 5


# --------------------------------------------------------------------------- #
# swe_lancer: gated Docker harness (no heuristic scoring)
# --------------------------------------------------------------------------- #
def test_swe_lancer_hard_errors_on_missing_prereqs():
    # No-skip policy: missing Docker / swelancer image / harness / opt-in must hard-error, not skip.
    from gbench.runners.eval_suites.swe_lancer import run_swe_lancer
    with pytest.raises(RuntimeError, match="docs/evals/swe_lancer.md"):
        run_swe_lancer("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


def test_swe_lancer_gated_behind_opt_in():
    from gbench.runners.eval_suites import swe_lancer as SL
    with tempfile.TemporaryDirectory() as hd:
        open(os.path.join(hd, "run_swelancer_eval.py"), "w").close()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch.dict("sys.modules", {"docker": MagicMock()}), \
             patch.dict(os.environ, {"SWELANCER_HARNESS_DIR": hd}, clear=False):
            os.environ.pop("SWELANCER_RUN", None)
            ok, reason = SL.check_swe_lancer_prerequisites()
            assert ok is False and "SWELANCER_RUN=1" in reason


def test_swe_lancer_ships_and_installs_the_predictions_adapter():
    # gbench provides run_swelancer_eval.py + predictions_solver.py (upstream has none) and installs
    # them into the harness dir - so the user never hand-writes an adapter.
    import py_compile
    from gbench.runners.eval_suites import swe_lancer as SL
    src = os.path.join(os.path.dirname(SL.__file__), "_swelancer_adapter")
    for fn in SL._ADAPTER_FILES:
        py_compile.compile(os.path.join(src, fn), doraise=True)   # syntactically valid
    with tempfile.TemporaryDirectory() as d:
        SL._install_adapter(d)
        assert set(os.listdir(d)) == set(SL._ADAPTER_FILES)


def test_swe_lancer_loader_and_patch_extractor():
    from gbench.runners.eval_suites import swe_lancer as SL
    # The loader now sources tasks from the harness IMAGE's own issue manifest (id parity with the
    # harness); the prediction key MUST be the image issue id so run.sh's setup finds
    # /app/tests/issues/<id>/. (The old DCAgent2 mirror keyed prompts by ids absent from the image,
    # which crashed every task container before any report.)
    fake_manifest = [{"id": "42", "title": "Login screen crash",
                      "issue_repo_steps": "Fix the login bug.", "price": 250.0, "issue_id": 55827}]
    with patch.object(SL, "_extract_issue_manifest", return_value=fake_manifest):
        samples = SL._load_swe_lancer_samples()
    assert len(samples) == 1 and samples[0][1] == "42"        # keyed by the image issue id
    assert samples[0][2]["task_id"] == "42"
    assert "Fix the login bug." in samples[0][0][0]["content"]
    assert SL._extract_patch("```diff\ndiff --git a/x b/x\n@@\n```").startswith("diff --git")
    assert SL._extract_patch("nothing here") == ""


def test_swe_lancer_results_parser():
    from gbench.runners.eval_suites import swe_lancer as SL
    with tempfile.TemporaryDirectory() as out:
        with open(os.path.join(out, "r.json"), "w") as f:
            json.dump({"resolved_ids": ["task_42"]}, f)
        assert SL._parse_results(out) == {"task_42": True}
    with tempfile.TemporaryDirectory() as out:
        with open(os.path.join(out, "r.json"), "w") as f:
            json.dump({"task_1": True, "task_2": False}, f)
        assert SL._parse_results(out) == {"task_1": True, "task_2": False}


# --------------------------------------------------------------------------- #
# gaia2: canonical Meta-ARE harness (no-skip; hard-errors when un-provisioned)
# --------------------------------------------------------------------------- #
def test_gaia2_hard_errors_on_missing_prereqs():
    # No-skip policy: missing Docker / orchestrator image / GEMINI judge must hard-error, not skip.
    from gbench.runners.eval_suites.gaia2 import run_gaia2
    with pytest.raises(RuntimeError, match="docs/evals/gaia2.md"):
        run_gaia2("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


# --------------------------------------------------------------------------- #
# tau2 / tau3: proxy default + gated full environment
# --------------------------------------------------------------------------- #
def test_tau_env_not_requested_by_default():
    from gbench.runners.eval_suites import tau_common as TC
    old = os.environ.pop("TAU2_ENV_RUN", None)
    try:
        assert TC.env_requested() is False
    finally:
        if old is not None:
            os.environ["TAU2_ENV_RUN"] = old


def test_tau2_and_tau3_default_hard_error_not_a_misleading_zero():
    """Without the env, tau2/tau3 must HARD-ERROR (not score 0, not silently skip) since
    single-turn can't measure them (no-skip policy)."""
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    old = os.environ.pop("TAU2_ENV_RUN", None)
    try:
        for run, name in ((T2.run_tau2, "tau2"), (T3.run_tau3, "tau3")):
            with pytest.raises(RuntimeError, match="TAU2_ENV_RUN=1"):
                run("m", "http://x", 2)
            with pytest.raises(RuntimeError, match=f"docs/evals/{name}.md"):
                run("m", "http://x", 2)
    finally:
        if old is not None:
            os.environ["TAU2_ENV_RUN"] = old


def test_gdpval_hard_errors_without_a_judge(monkeypatch):
    """gdpval needs the Gemini judge; per the no-skip policy a missing GEMINI_API_KEY hard-errors
    (the judge gate runs before the file/text-endpoint check). (gdpval's own text-endpoint
    handling is a Group-D item; this only pins the no-judge hard-error.)"""
    import pytest
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from gbench.runners.eval_suites.gdpval import run_gdpval
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY|docs/evals/gdpval.md"):
        run_gdpval("m", "http://x", 2)


def test_tau2_and_tau3_env_opt_in_but_missing_hard_errors():
    # No-skip policy: opted into the real simulator but the tau2 package is missing ->
    # hard-error telling the operator to install it, not a silent skip.
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"TAU2_ENV_RUN": "1"}, clear=False), \
         patch.object(TC, "_import_tau2",
                      return_value=(False, "ModuleNotFoundError: No module named 'tau2'")):
        with pytest.raises(RuntimeError, match="docs/evals/tau2.md"):
            T2.run_tau2("m", "http://x", 2)
        with pytest.raises(RuntimeError, match="docs/evals/tau3.md"):
            T3.run_tau3("m", "http://x", 2)


def test_tau_env_prereq_requires_gemini_key():
    """With tau2 importable but no GEMINI_API_KEY, the gemini user/judge defaults -> skip reason."""
    from gbench.runners.eval_suites import tau_common as TC
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        with patch.object(TC, "_import_tau2", return_value=(True, "")):
            ok, why = TC.check_tau_env_prerequisites()
        assert ok is False and "GEMINI_API_KEY" in why
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


def test_tau_env_prereq_surfaces_import_error_with_audioop_hint():
    """A findable-but-broken tau2 (e.g. py3.13 audioop) yields the real reason + backport hint."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "_import_tau2",
                      return_value=(False, "ModuleNotFoundError: No module named 'audioop'")), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False):
        ok, why = TC.check_tau_env_prerequisites()
    assert ok is False and "audioop-lts" in why and "audioop" in why


def test_tau_common_run_env_aggregates_reward_across_domains():
    """run_tau_env aggregates per-domain (total, reward_sum, perfect) into the result dict.

    Mocks the tau2 Python-API layer (_run_one_domain) so the gbench-side aggregation +
    result schema are what's under test, not the external simulator.
    """
    from gbench.runners.eval_suites import tau_common as TC
    canned = {"airline": (2, 1.0, 1, 0), "retail": (2, 2.0, 2, 0)}  # (total, reward_sum, perfect, infra)

    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "_install_robustness_patches", lambda *a, **k: None), \
         patch.object(TC, "_configure_evaluator_llm", lambda *a, **k: None), \
         patch.object(TC, "_run_one_domain", side_effect=lambda d, *a, **k: canned[d]):
        r = TC.run_tau_env("tau2", ["airline", "retail"], "m", "http://x", 4, None, "docs/evals/tau2.md")

    assert r["status"] == "success" and r["mode"] == "full_environment"
    # total=4, reward_sum=3.0 -> mean reward 0.75 -> accuracy 75.0; perfect=3
    assert r["total_questions"] == 4 and r["correct_answers"] == 3 and r["accuracy"] == 75.0
    assert r["category_accuracy"]["airline"]["accuracy"] == 50.0    # 1.0/2
    assert r["category_accuracy"]["retail"]["accuracy"] == 100.0    # 2.0/2
    assert r["tau2_report"]["mean_reward"] == 0.75 and r["tau2_report"]["perfect_tasks"] == 3


# --------------------------------------------------------------------------- #
# tau3 == τ³-bench banking_knowledge (RAG) repurpose
# --------------------------------------------------------------------------- #
def test_tau3_default_hard_error_names_banking_knowledge():
    """tau3's default hard-error should describe the τ³ banking_knowledge RAG domain, not telecom."""
    from gbench.runners.eval_suites import tau3 as T3
    old = os.environ.pop("TAU2_ENV_RUN", None)
    try:
        with pytest.raises(RuntimeError) as ei:
            T3.run_tau3("m", "http://x", 2)
        msg = str(ei.value)
        assert "banking_knowledge" in msg
        assert "telecom" not in msg.lower()
    finally:
        if old is not None:
            os.environ["TAU2_ENV_RUN"] = old


def test_tau3_env_runs_banking_knowledge_domain():
    """With the env opt-in, tau3 must drive the banking_knowledge domain (not telecom)."""
    from gbench.runners.eval_suites import tau3 as T3
    with patch.dict(os.environ, {"TAU2_ENV_RUN": "1"}, clear=False), \
         patch.object(T3, "run_tau_env", return_value={"status": "success"}) as m:
        T3.run_tau3("m", "http://x", 8, enable_thinking=False, limit=5)
    args, kwargs = m.call_args
    assert args[0] == "tau3" and args[1] == ["banking_knowledge"]


def test_check_banking_prerequisites_requires_rank_bm25():
    """Missing rank-bm25 -> clean skip with the pip hint + docs pointer (like audioop)."""
    import sys as _sys
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(_sys.modules, {"rank_bm25": None}):   # makes `import rank_bm25` raise
        ok, why = TC.check_banking_prerequisites()
    assert ok is False and "rank-bm25" in why and "docs/evals/tau3.md" in why


def test_check_banking_prerequisites_default_needs_embed_key():
    """rank-bm25 present but no key and no config override -> skip about the embeddings key."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {}, clear=False):
        for k in ("TAU2_RETRIEVAL_CONFIG", "GEMINI_API_KEY", "TAU2_EMBED_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        ok, why = TC.check_banking_prerequisites()
        assert ok is False and "GEMINI_API_KEY" in why
        os.environ["GEMINI_API_KEY"] = "k"
        ok2, _ = TC.check_banking_prerequisites()
        assert ok2 is True


def test_check_banking_prerequisites_override_bypasses_key_check():
    """A TAU2_RETRIEVAL_CONFIG override (e.g. bm25, no embeddings) shouldn't demand a Gemini key."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"TAU2_RETRIEVAL_CONFIG": "bm25"}, clear=False):
        for k in ("GEMINI_API_KEY", "TAU2_EMBED_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        ok, _ = TC.check_banking_prerequisites()
    assert ok is True


def test_run_tau_env_gates_on_banking_prereq():
    """run_tau_env must apply the banking prereq gate for the banking_knowledge domain
    as a hard-error (no-skip policy)."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "check_banking_prerequisites",
                      return_value=(False, "needs rank-bm25 ... docs/evals/tau3.md")):
        with pytest.raises(RuntimeError, match="rank-bm25"):
            TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x", 4, None, "docs/evals/tau3.md")


def test_setup_banking_retrieval_wires_gemini_and_honors_override():
    """Default banking retrieval registers a Gemini alltools variant + points OpenAI SDK at Gemini;
    an explicit TAU2_RETRIEVAL_CONFIG override is used verbatim."""
    pytest.importorskip("tau2.domains.banking_knowledge.retrieval")
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=False):
        for k in ("TAU2_RETRIEVAL_CONFIG", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            os.environ.pop(k, None)
        name, kw = TC._setup_banking_retrieval()
        assert name == "alltools-gemini" and kw is None
        assert "generativelanguage.googleapis.com" in os.environ.get("OPENAI_BASE_URL", "")
        assert os.environ.get("OPENAI_API_KEY") == "k"
    with patch.dict(os.environ, {"TAU2_RETRIEVAL_CONFIG": "bm25"}, clear=False):
        name2, _ = TC._setup_banking_retrieval()
        assert name2 == "bm25"


def _fake_tau2_modules():
    """Fake tau2 submodules that _quiet_tau2_noise patches (functions + rich console)."""
    import types
    llm = types.ModuleType("tau2.utils.llm_utils")
    llm.get_response_cost = lambda r: 999.0
    utils = types.ModuleType("tau2.utils.utils")
    utils.get_commit_hash = lambda: "realhash"
    helpers = types.ModuleType("tau2.runner.helpers")
    helpers.get_commit_hash = lambda: "realhash"
    display = types.ModuleType("tau2.utils.display")

    class ConsoleDisplay:
        console = "loud-console"  # a placeholder tau2 would set to a rich Console()
    display.ConsoleDisplay = ConsoleDisplay
    return {
        "tau2": types.ModuleType("tau2"),
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm,
        "tau2.utils.utils": utils,
        "tau2.utils.display": display,
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.helpers": helpers,
    }


def test_tau_common_quiet_noops_functions_and_mutes_console():
    """_quiet_tau2_noise no-ops the two noisy functions (incl. the by-value import in
    runner.helpers) AND mutes tau2's rich console by default."""
    import sys
    from gbench.runners.eval_suites import tau_common as TC
    fake = _fake_tau2_modules()
    TC._QUIETED = False
    old = os.environ.pop("TAU2_VERBOSE", None)
    try:
        with patch.dict(sys.modules, fake):
            TC._quiet_tau2_noise()
        assert fake["tau2.utils.llm_utils"].get_response_cost("x") == 0.0
        assert fake["tau2.utils.utils"].get_commit_hash() == "unknown"
        assert fake["tau2.runner.helpers"].get_commit_hash() == "unknown"
        # rich console replaced with a quiet one
        console = fake["tau2.utils.display"].ConsoleDisplay.console
        assert getattr(console, "quiet", False) is True
        # litellm's noisy loggers raised to ERROR (kills the per-call Gemini deprecation spam)
        import logging as _logging
        assert _logging.getLogger("LiteLLM").level == _logging.ERROR
    finally:
        TC._QUIETED = False
        if old is not None:
            os.environ["TAU2_VERBOSE"] = old


def test_tau_common_verbose_env_keeps_console():
    """TAU2_VERBOSE=1 leaves tau2's rich console untouched (full panels)."""
    import sys
    from gbench.runners.eval_suites import tau_common as TC
    fake = _fake_tau2_modules()
    TC._QUIETED = False
    try:
        with patch.dict(sys.modules, fake), patch.dict(os.environ, {"TAU2_VERBOSE": "1"}):
            TC._quiet_tau2_noise()
        assert fake["tau2.utils.display"].ConsoleDisplay.console == "loud-console"
    finally:
        TC._QUIETED = False


def test_tau_common_hard_errors_when_no_domain_produces_results():
    # No-skip policy: if the simulator produces no results, hard-error (not a silent skip).
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "_install_robustness_patches", lambda *a, **k: None), \
         patch.object(TC, "_configure_evaluator_llm", lambda *a, **k: None), \
         patch.object(TC, "_run_one_domain", side_effect=RuntimeError("simulator boom")):
        with pytest.raises(RuntimeError, match="docs/evals/tau2.md"):
            TC.run_tau_env("tau2", ["airline"], "m", "http://x", 2, None, "docs/evals/tau2.md")
