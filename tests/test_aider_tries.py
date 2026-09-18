# -*- coding: utf-8 -*-
"""Canonical Aider Polyglot pass@2 is delegated to aider's own benchmark.py `--tries 2` inside
the aider-benchmark container (the gbench hand-rolled tries loop was retired 2026-09-08 in favour
of the canonical native-edit-format harness). The build-command + result-parsing contract lives
in test_aider_polyglot_canonical.py; this file just guards that pass@2 stays the delegated
headline."""
import os
import sys

_S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench")
if _S not in sys.path:
    sys.path.insert(0, _S)

from gbench.runners.eval_suites import aider_polyglot as A  # noqa: E402


def test_pass_at_2_is_delegated_to_aider_benchmark_tries_2():
    cmd = A._build_command("aider-benchmark", "/bench", "gbench", "openai/m",
                           "http://h/v1", 4, A._LANGUAGES, "", None, "")
    joined = " ".join(cmd)
    assert "--tries 2" in joined, "pass@2 = aider benchmark.py --tries 2 (its native retry loop)"
    assert "benchmark.py" in joined


def test_headline_is_pass_at_2():
    import inspect
    src = inspect.getsource(A.run_aider_polyglot)
    assert 'result["accuracy"] = ' not in src or 'pass@2' in src or 'pass2' in src
    assert '"pass_rate_2"' in src and '"pass_rate_1"' in src
