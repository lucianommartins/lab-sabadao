# -*- coding: utf-8 -*-
"""Deterministic unit tests for the skillsbench canonical harness (no Docker / model / judge)."""

import json
import os
import sys

import pytest

from gbench.runners.eval_suites import skillsbench as S

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import skillsbench_run as RUN  # noqa: E402


# --------------------------------------------------------------------------- #
# Headline scoring.
# --------------------------------------------------------------------------- #
def test_score_single_condition():
    summary = {"conditions": {"without-skills": {"mean_reward": 0.42, "n_tasks": 87, "n_scored": 87}}}
    s = S.compute_skillsbench_score(summary)
    assert s["primary_condition"] == "without-skills"
    assert s["mean_reward"] == pytest.approx(0.42)


def test_score_both_conditions_primary_and_lift():
    summary = {
        "conditions": {
            "with-skills": {"mean_reward": 0.55, "n_tasks": 87},
            "without-skills": {"mean_reward": 0.40, "n_tasks": 87},
        },
        "with_skills_lift": 0.15,
    }
    s = S.compute_skillsbench_score(summary)
    assert s["primary_condition"] == "without-skills"   # raw capability is the headline
    assert s["mean_reward"] == pytest.approx(0.40)
    assert s["with_skills_lift"] == pytest.approx(0.15)
    assert set(s["condition_means"]) == {"with-skills", "without-skills"}


def test_score_none_when_no_means():
    assert S.compute_skillsbench_score({"conditions": {}}) is None
    assert S.compute_skillsbench_score({"conditions": {"without-skills": {"mean_reward": None}}}) is None


# --------------------------------------------------------------------------- #
# Task-reachable endpoint translation.
# --------------------------------------------------------------------------- #
def test_task_endpoint_rewrites_localhost(monkeypatch):
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_HOST", raising=False)
    assert S._task_reachable_endpoint("http://127.0.0.1:8000/v1") == "http://172.17.0.1:8000/v1"
    assert S._task_reachable_endpoint("http://localhost:8000") == "http://172.17.0.1:8000/v1"


def test_task_endpoint_custom_host_and_override(monkeypatch):
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_HOST", "host.docker.internal")
    assert S._task_reachable_endpoint("http://127.0.0.1:9000/v1") == "http://host.docker.internal:9000/v1"
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", "http://gw:1/v1")
    assert S._task_reachable_endpoint("http://127.0.0.1:8000/v1") == "http://gw:1/v1"


def test_task_endpoint_remote_host_unchanged(monkeypatch):
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", raising=False)
    # a non-localhost host is left alone (only /v1 is ensured)
    assert S._task_reachable_endpoint("http://10.0.0.5:8000/v1") == "http://10.0.0.5:8000/v1"
    assert S._task_reachable_endpoint("http://10.0.0.5:8000") == "http://10.0.0.5:8000/v1"


# --------------------------------------------------------------------------- #
# Launcher reward parsing from a rollout's result.json.
# --------------------------------------------------------------------------- #
def test_reward_from_jobs_rewards_dict(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / "r1").mkdir(parents=True)
    (jobs / "r1" / "result.json").write_text(json.dumps({"rewards": {"reward": 1.0}}))
    assert RUN._reward_from_jobs(str(jobs)) == pytest.approx(1.0)


def test_reward_from_jobs_scalar_and_flat_reward(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / "r").mkdir(parents=True)
    (jobs / "r" / "result.json").write_text(json.dumps({"reward": 0.0}))
    assert RUN._reward_from_jobs(str(jobs)) == pytest.approx(0.0)


def test_reward_from_jobs_none_cases(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    assert RUN._reward_from_jobs(str(jobs)) is None            # no result.json at all
    (jobs / "r").mkdir()
    (jobs / "r" / "result.json").write_text(json.dumps({"rewards": None, "verifier_error": "x"}))
    assert RUN._reward_from_jobs(str(jobs)) is None            # verifier crashed -> no reward
