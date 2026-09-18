# -*- coding: utf-8 -*-
"""WS6: swe_bench_live isolates the SWE-bench-Live `swebench` fork in a LOCAL image (no host venv).

The fork is the same package name as upstream `swebench` at an incompatible version, so scoring runs
inside the fork image while the host env keeps upstream for swe_bench_multilingual / copilot_bench_swe.
"""

import asyncio
import sys

from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import swe_bench_live as S


class _Proc:
    returncode = 0
    stdout = ""
    stderr = ""


def _run_scorer(harness_image, monkeypatch):
    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return _Proc()

    monkeypatch.setattr(C.subprocess, "run", _fake_run)
    metrics = {}
    scorer = C.make_swebench_scorer(
        "swe_bench_live", "gemma", "SWE-bench-Live/SWE-bench-Live", "lite",
        "starryzhang", 1, metrics, harness_image=harness_image)
    traces = [{"extra_payload": {"instance_id": "repo__1"}, "response_text": "diff --git a b\n+x\n"}]
    asyncio.run(scorer(traces))
    return captured["cmd"], metrics


def test_scoring_runs_inside_the_fork_image_not_the_host(monkeypatch):
    cmd, metrics = _run_scorer("gbench-swe-bench-live", monkeypatch)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "gbench-swe-bench-live" in cmd
    assert "swebench.harness.run_evaluation" in cmd and "-m" in cmd
    # docker-out-of-docker: task containers spawn on the host daemon
    assert "/var/run/docker.sock:/var/run/docker.sock" in cmd
    # the fork must NOT run on the host interpreter (that is upstream swebench)
    assert sys.executable not in cmd
    assert metrics.get("harness_image") == "gbench-swe-bench-live"


def test_host_path_unchanged_when_no_image(monkeypatch):
    # harness_image=None (swe_bench_multilingual / copilot / pro / multi): host interpreter, no docker run
    cmd, metrics = _run_scorer(None, monkeypatch)
    assert cmd[0] != "docker"
    assert sys.executable in cmd
    # the host path may wrap the module in a thread-cap `-c` shim, so match anywhere in the argv
    assert any("swebench.harness.run_evaluation" in str(c) for c in cmd)
    assert "harness_image" not in metrics


def test_hard_errors_without_the_fork_image(monkeypatch):
    monkeypatch.setattr(S.shutil, "which", lambda _x: "/usr/bin/docker")

    def _fake_run(cmd, **kw):
        class R:
            # docker info OK, but `docker image inspect <fork image>` fails (image not built)
            returncode = 0 if cmd[:2] == ["docker", "info"] else 1
        return R()

    monkeypatch.setattr(S.subprocess, "run", _fake_run)
    ok, reason = S.check_swe_bench_live_prerequisites()
    assert not ok
    assert "swe_bench_live.Dockerfile" in reason and "docker build" in reason


def test_run_forwards_the_image_as_harness_image(monkeypatch):
    seen = {}

    def _fake_execute(eval_name, model_name, base_url, concurrency, dataset, split, namespace, **kwargs):
        seen.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(S, "check_swe_bench_live_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(S, "execute_swebench", _fake_execute)
    S.run_swe_bench_live("gemma", "http://x/v1", 1)
    assert seen.get("harness_image") == S._image()
