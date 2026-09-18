# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Canonical LiveBench: containerized delegation to LiveBench's own scorers."""
import types
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.livebench as L


def test_livebench_hard_errors_without_docker(monkeypatch):
    # No-skip policy: absent docker/image -> hard-error with build instructions.
    monkeypatch.setattr(L.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError, match="docs/evals/livebench.md"):
        L._check_prereqs("gbench-livebench")


def test_livebench_hard_errors_without_image(monkeypatch):
    monkeypatch.setattr(L.shutil, "which", lambda _x: "/usr/bin/docker")

    def fake_run(cmd, **_kw):
        rc = 0 if cmd[:2] == ["docker", "info"] else 1  # daemon ok, image inspect fails
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not found"):
        L._check_prereqs("gbench-livebench")


def test_build_command_covers_six_core_categories_and_endpoint():
    cmd = L._build_command("img", "/out", "openai/m", "m-slug", "http://h:8000/v1", 8, 0.0, "", "")
    assert cmd[0] == "docker" and "--network" in cmd and "host" in cmd
    joined = " ".join(cmd)
    # The full default set runs as the WHOLE-SUITE bench-name `live_bench`, NOT per-category:
    # LiveBench's per-category loader truncates a multi-word category at '_' (data_analysis ->
    # the non-existent HF dataset livebench/data) and crashes the run. The whole-suite path uses
    # the correct full-name code path. So the run step must NOT enumerate individual categories.
    assert "--bench-name live_bench --question-source" in joined
    assert "live_bench/data_analysis" not in joined and "live_bench/coding" not in joined
    assert len(L._CATEGORIES) == 6 and "agentic" not in joined  # agentic_coding excluded
    assert "run_livebench.py" in joined and "show_livebench_result.py" in joined
    assert "--api-base http://h:8000/v1" in joined and "--use-litellm" in joined
    assert "--mode single" in joined              # not sequential/parallel (those need tmux)
    assert "--ignore-missing-judgments" in joined  # score completed categories, don't drop the model
    assert "all_groups.csv" in joined


def test_parse_all_groups(tmp_path):
    p = tmp_path / "all_groups.csv"
    p.write_text("model,average,coding,math,reasoning\n"
                 "m-slug,50.0,40.0,60.0,50.0\n"
                 "other,10.0,10.0,10.0,10.0\n", encoding="utf-8")
    s = L._parse_all_groups(str(p), "m-slug")
    assert s["average"] == 50.0 and s["coding"] == 40.0 and s["math"] == 60.0
    assert "reasoning" in s and s["reasoning"] == 50.0
