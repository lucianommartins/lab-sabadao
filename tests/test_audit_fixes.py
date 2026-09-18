# -*- coding: utf-8 -*-
# Regression tests for the think-sweep audit fixes:
#  #3 codeforces: grade the LAST fenced code block, not the first (canonical open-r1).
import pytest


def test_codeforces_grades_last_code_block_not_first():
    """Models draft then emit a corrected final block; canonical extraction takes the LAST."""
    try:
        from gbench.runners.eval_suites.codeforces import _eval_codeforces
    except ImportError:
        pytest.skip("codeforces suite not importable")

    gold = {"official_tests": [{"input": "", "output": "RIGHT"}],
            "generated_tests": [], "checker": "", "time_limit": 5}

    # first block WRONG, last block RIGHT -> must pass (grades the last)
    two_blocks = ("Attempt 1:\n```python\nprint('WRONG')\n```\n"
                  "Corrected:\n```python\nprint('RIGHT')\n```")
    assert _eval_codeforces(two_blocks, gold) is True

    # only the wrong block -> must fail (sanity: not silently passing)
    assert _eval_codeforces("```python\nprint('WRONG')\n```", gold) is False

    # single correct block still works
    assert _eval_codeforces("```python\nprint('RIGHT')\n```", gold) is True


def test_resolve_judge_model_default_env_and_self_judge(monkeypatch):
    """Judge model must resolve loudly, track its source, and flag the model judging itself."""
    try:
        from gbench.runners.eval_suites import base
    except ImportError:
        pytest.skip("base not importable")

    UT = "google/gemma-4-26B-A4B-it"  # the model under test (served locally)

    # default when unset
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    base._JUDGE_RESOLVE_LOGGED.clear()
    m, src, selfj = base.resolve_judge_model(UT)
    assert (m, src, selfj) == ("gemini-3.6-flash", "default", False)

    # env override is honored AND its source is recorded (not silently swallowed)
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-3.5-flash")
    base._JUDGE_RESOLVE_LOGGED.clear()
    m, src, selfj = base.resolve_judge_model(UT)
    assert (m, src, selfj) == ("gemini-3.5-flash", "env:GBENCH_JUDGE_MODEL", False)

    # THE FOOTGUN: judge == model under test -> flagged as self-judge (the gemma-4-26b leak)
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemma-4-26b")
    base._JUDGE_RESOLVE_LOGGED.clear()
    m, src, selfj = base.resolve_judge_model(UT)
    assert selfj is True

    # family matcher: catches the gemini-hosted gemma vs the local hf id, and provider prefixes
    assert base._same_model_family("gemma-4-26b", UT) is True
    assert base._same_model_family("gemini/gemma-4-26b", UT) is True
    assert base._same_model_family("gemini-3.6-flash", UT) is False
    assert base._same_model_family("", UT) is False
