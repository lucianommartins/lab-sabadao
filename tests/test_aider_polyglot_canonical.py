# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Canonical Aider Polyglot: containerized delegation to aider's own benchmark.py (pass@2)."""
import json
import types
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.aider_polyglot as A


def test_aider_hard_errors_without_docker(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError, match="docs/evals/aider_polyglot.md"):
        A._check_prereqs("aider-benchmark", "/tmp/bench")


def test_aider_hard_errors_without_image(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")

    def fake_run(cmd, **_kw):
        rc = 0 if cmd[:2] == ["docker", "info"] else 1   # daemon ok, image inspect fails
        return types.SimpleNamespace(returncode=rc, stdout=b"", stderr=b"")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not found"):
        A._check_prereqs("aider-benchmark", "/tmp/bench")


def test_aider_hard_errors_without_exercises(monkeypatch, tmp_path):
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(A.subprocess, "run",
                        lambda cmd, **k: types.SimpleNamespace(returncode=0, stdout=b"", stderr=b""))
    # docker+image OK, but the exercises dir is absent -> hard-error (never a silent skip)
    with pytest.raises(RuntimeError, match="polyglot"):
        A._check_prereqs("aider-benchmark", str(tmp_path))


def test_build_command_drives_aider_benchmark_natively():
    cmd = A._build_command("aider-benchmark", "/bench", "gbench", "openai/m",
                           "http://127.0.0.1:8000/v1", 8, A._LANGUAGES, "", None, "")
    joined = " ".join(cmd)
    assert cmd[0] == "docker" and "--network" in cmd and "host" in cmd
    assert "-v" in cmd and "/bench:/benchmarks" in cmd
    assert "AIDER_DOCKER=1" in joined and "AIDER_BENCHMARK_DIR=/benchmarks" in joined
    assert "OPENAI_API_BASE=http://127.0.0.1:8000/v1" in joined
    assert "benchmark.py gbench" in joined and "--model openai/m" in joined
    assert "--tries 2" in joined                      # canonical pass@2
    assert "--exercises-dir polyglot-benchmark" in joined
    assert "--edit-format" not in joined              # native default when unset
    for lang in A._LANGUAGES:
        assert lang in joined
    # --eval-limit maps to --num-tests
    cmd2 = A._build_command("i", "/b", "gbench", "openai/m", "http://h/v1", 1,
                            A._LANGUAGES, "diff", 5, "")
    j2 = " ".join(cmd2)
    assert "--num-tests 5" in j2 and "--edit-format diff" in j2


def test_parse_results_computes_pass_at_1_and_2(tmp_path):
    """pass@1 = passed on try 1; pass@2 = passed within two tries (aider stops early on pass)."""
    def _write(lang, name, outcomes):
        d = tmp_path / lang / "exercises" / "practice" / name
        d.mkdir(parents=True)
        (d / ".aider.results.json").write_text(json.dumps(
            {"language": lang, "testcase": name, "tests_outcomes": outcomes}))

    _write("python", "p1", [True])            # pass@1 and pass@2
    _write("python", "p2", [False, True])     # pass@2 only
    _write("rust", "r1", [False, False])      # neither
    _write("go", "g1", [])                    # no outcomes -> counted, not passed
    agg = A._parse_results(str(tmp_path))
    assert agg["n"] == 4
    assert agg["pass1"] == 1 and agg["pass2"] == 2
    assert agg["per_lang"]["python"] == {"total": 2, "pass1": 1, "pass2": 2}
    assert agg["per_lang"]["rust"]["pass2"] == 0


def test_aider_leaderboard_comparable_only_for_full_native_run(monkeypatch, tmp_path):
    """A full 6-language native-edit-format run (no --eval-limit) is comparable; a subset is not.
    The mocked `docker run` simulates aider's `--new` by creating a fresh timestamped results dir
    each call (so run_aider_polyglot's before/after snapshot discovers it)."""
    import os
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")   # skip the served-id network lookup
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    calls = {"n": 0}

    import re
    langs = ["cpp", "go", "java", "javascript", "python", "rust"]

    def fake_run(cmd, **k):
        calls["n"] += 1
        # A full run scores the whole 225-exercise set; a --num-tests run scores that many. The
        # no-partial gate in run_aider_polyglot requires the full run to actually reach 225.
        m = re.search(r"--num-tests (\d+)", " ".join(cmd))
        count = int(m.group(1)) if m else A._EXPECTED_FULL
        base = tmp_path / f"2026-run{calls['n']}--gbench"
        for i in range(count):
            lang = langs[i % len(langs)]
            d = base / lang / "exercises" / "practice" / f"x{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(
                json.dumps({"language": lang, "tests_outcomes": [True]}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(A.subprocess, "run", fake_run)

    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert r["eval_name"] == "aider_polyglot" and r["accuracy"] == 100.0
    assert r["pass_rate_2"] == 100.0 and r["tries"] == 2
    assert r["leaderboard_comparable"] is True           # full, native, no limit

    r2 = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1, limit=5)
    assert r2["leaderboard_comparable"] is False          # subset
