# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Canonical AgentDojo: delegation to the agentdojo package (utility + attack-success-rate)."""
import warnings

warnings.filterwarnings("ignore")

import pytest
from gbench.runners.eval_suites import agent_dojo as AD


def test_agent_dojo_pct():
    assert AD._pct([True, True]) == 100.0
    assert AD._pct([True, False]) == 50.0
    assert AD._pct([]) == 0.0


class _Suite:
    def __init__(self, users, injs):
        self.user_tasks = dict.fromkeys(users)
        self.injection_tasks = dict.fromkeys(injs)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Pipe:
    def __init__(self, _elements):
        self.name = None


def _fake_agentdojo():
    """A stand-in for the agentdojo package: 2 user tasks, both benign-pass; the attack succeeds
    on the first user task and fails on the second."""
    def fake_without(pipe, suite, ld, force, user_tasks=None, benchmark_version=None):
        return {"utility_results": {(u, ""): True for u in user_tasks}}

    def fake_with(pipe, suite, attack, ld, force, user_tasks=None, injection_tasks=None,
                  benchmark_version=None):
        util, sec = {}, {}
        for k, u in enumerate(user_tasks):
            for j in injection_tasks:
                util[(u, j)] = (k != 0)        # task breaks under attack on the first user task
                sec[(u, j)] = (k == 0)         # attack SUCCEEDS on the first user task
        return {"utility_results": util, "security_results": sec}

    return {
        "openai": type("O", (), {"OpenAI": staticmethod(lambda **k: object())}),
        "OpenAILLM": lambda *a, **k: object(),
        "AgentPipeline": _Pipe,
        "SystemMessage": lambda *a, **k: object(),
        "InitQuery": lambda *a, **k: object(),
        "ToolsExecutionLoop": lambda *a, **k: object(),
        "ToolsExecutor": lambda *a, **k: object(),
        "load_system_message": lambda *a, **k: "sys",
        "get_suites": lambda v: {"banking": _Suite(["user_task_0", "user_task_1"],
                                                   ["injection_task_0"])},
        "load_attack": lambda *a, **k: object(),
        "OutputLogger": lambda *a, **k: _Ctx(),
        "benchmark_without": fake_without,
        "benchmark_with": fake_with,
    }


def test_agent_dojo_delegates_utility_and_asr(monkeypatch):
    """Headline accuracy = benign utility; attack_success_rate = mean(security_results) where
    True means the attack SUCCEEDED (NOT inverted into accuracy)."""
    monkeypatch.setenv("GBENCH_AGENTDOJO_MODEL", "m")     # skip the endpoint /models query
    monkeypatch.setattr(AD, "_load_agentdojo", _fake_agentdojo)
    r = AD.run_agent_dojo("m", "http://x", limit=None, concurrency=1)
    assert r["eval_name"] == "agent_dojo" and r["status"] == "success"
    assert r["accuracy"] == 100.0                 # both benign tasks pass
    assert r["attack_success_rate"] == 50.0       # 1 of 2 injections succeeded
    assert r["utility_under_attack"] == 50.0      # task broke on the attacked-successfully one
    assert r["attack"] == "important_instructions" and r["benchmark_version"] == "v1.2.2"
    assert r["leaderboard_comparable"] is True    # full set (no --eval-limit)
    # ASR is reported separately and must NOT be folded into accuracy.
    assert r["accuracy"] != r["attack_success_rate"]


def test_agent_dojo_limit_is_not_comparable(monkeypatch):
    monkeypatch.setenv("GBENCH_AGENTDOJO_MODEL", "m")
    monkeypatch.setattr(AD, "_load_agentdojo", _fake_agentdojo)
    r = AD.run_agent_dojo("m", "http://x", limit=1, concurrency=1)
    assert r["leaderboard_comparable"] is False   # a subset is not comparable


def test_agent_dojo_hard_errors_without_the_package():
    """No-skip policy: a missing agentdojo package must hard-error (infra_required), not skip."""
    import inspect
    src = inspect.getsource(AD._load_agentdojo)
    assert "infra_required" in src and "pip install agentdojo" in src
    # run_agent_dojo has no skipped_result path
    assert "skipped_result" not in inspect.getsource(AD)
