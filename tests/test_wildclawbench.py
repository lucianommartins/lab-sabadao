# -*- coding: utf-8 -*-
"""Deterministic unit tests for the wildclawbench canonical harness (no Docker / model / real judge)."""

import json
import os
import sys

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import wildclawbench_run as RUN            # noqa: E402
import wildclawbench_cascade_judge as J    # noqa: E402


# --------------------------------------------------------------------------- #
# Harness-side headline scoring.
# --------------------------------------------------------------------------- #
def test_compute_score_valid_and_none():
    s = W.compute_wildclawbench_score({"global_mean": 0.42, "multimodal_mean": 0.3,
                                       "pure_text_mean": 0.5, "weighted_overall": 0.4,
                                       "avg_time_min": 6.4, "avg_cost_usd": 0.0})
    assert s["global_mean"] == pytest.approx(0.42)
    assert s["multimodal_mean"] == pytest.approx(0.3)
    assert W.compute_wildclawbench_score({"n_tasks": 3}) is None      # no global_mean
    assert W.compute_wildclawbench_score({"global_mean": None}) is None


# --------------------------------------------------------------------------- #
# Task-reachable endpoint translation (model + judge).
# --------------------------------------------------------------------------- #
def test_task_endpoint_rewrites_localhost(monkeypatch):
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_HOST", raising=False)
    assert W._task_reachable_endpoint("http://127.0.0.1:8000/v1",
                                      "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") == "http://172.17.0.1:8000/v1"
    assert W._task_reachable_endpoint("http://localhost:8000",
                                      "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") == "http://172.17.0.1:8000/v1"


def test_task_endpoint_custom_host_and_override(monkeypatch):
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_TASK_HOST", "host.docker.internal")
    assert W._task_reachable_endpoint("http://127.0.0.1:9000/v1",
                                      "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") == "http://host.docker.internal:9000/v1"
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", "http://gw:1/v1")
    assert W._task_reachable_endpoint("http://127.0.0.1:8000/v1",
                                      "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") == "http://gw:1/v1"


def test_task_endpoint_remote_host_unchanged(monkeypatch):
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", raising=False)
    assert W._task_reachable_endpoint("http://10.0.0.5:8000/v1",
                                      "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") == "http://10.0.0.5:8000/v1"
    assert W._task_reachable_endpoint("http://10.0.0.5:8000",
                                      "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") == "http://10.0.0.5:8000/v1"


# --------------------------------------------------------------------------- #
# Prerequisite gate (raises infra_required, never skips).
# --------------------------------------------------------------------------- #
def test_gate_missing_host_dir(monkeypatch):
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_HOST_DIR", raising=False)
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GBENCH_WILDCLAWBENCH_HOST_DIR" in reason


def test_gate_missing_docker():
    # No docker CLI on PATH under a scrubbed env -> gate fails with a build hint (never skips).
    import unittest.mock as mock
    with mock.patch.object(W.shutil, "which", return_value=None):
        ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "docker CLI not found" in reason


def test_gate_requires_gemini_then_brave(tmp_path, monkeypatch):
    # Everything present except the keys: GEMINI first, then BRAVE (needed for the gateway to start).
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_batch.py").write_text("x")
    (tmp_path / "workspace").mkdir()
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_HOST_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GEMINI_API_KEY" in reason
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "BRAVE_API_KEY" in reason
    # both present -> reaches live validation; stub the pings so the presence-ordering test is offline
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (True, ""))
    monkeypatch.setattr(W, "_brave_key_valid", lambda _k: (True, ""))
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    ok, reason = W.check_wildclawbench_prerequisites()
    assert ok and reason == ""


def _gate_env_ready(tmp_path, monkeypatch):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_batch.py").write_text("x")
    (tmp_path / "workspace").mkdir()
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_HOST_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION", raising=False)


def test_gate_rejects_invalid_gemini_key(tmp_path, monkeypatch):
    _gate_env_ready(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (False, "HTTP 401"))
    monkeypatch.setattr(W, "_brave_key_valid", lambda _k: (True, ""))
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GEMINI_API_KEY was rejected" in reason


def test_gate_rejects_invalid_brave_key(tmp_path, monkeypatch):
    _gate_env_ready(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (True, ""))
    monkeypatch.setattr(W, "_brave_key_valid", lambda _k: (False, "HTTP 422"))
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "BRAVE_API_KEY was rejected" in reason


def test_gate_skip_key_validation_env(tmp_path, monkeypatch):
    _gate_env_ready(tmp_path, monkeypatch)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION", "1")

    def _boom(_k):
        raise AssertionError("validation must be skipped")
    monkeypatch.setattr(W, "_gemini_key_valid", _boom)
    monkeypatch.setattr(W, "_brave_key_valid", _boom)
    ok, reason = W.check_wildclawbench_prerequisites()
    assert ok and reason == ""


def test_key_validators_classify_responses(monkeypatch):
    import urllib.error

    def _mk_urlopen(status=None, http_code=None, body=b"", exc=None):
        def _open(req, timeout=0):
            if exc is not None:
                raise exc
            if http_code is not None:
                raise urllib.error.HTTPError(req.full_url, http_code, "err", {}, __import__("io").BytesIO(body))
            return type("R", (), {"status": status, "__enter__": lambda s: s,
                                  "__exit__": lambda s, *a: False})()
        return _open

    # 200 -> valid
    monkeypatch.setattr(W.urllib.request, "urlopen", _mk_urlopen(status=200))
    assert W._gemini_key_valid("k")[0] is True
    assert W._brave_key_valid("k")[0] is True
    # 401 -> rejected
    monkeypatch.setattr(W.urllib.request, "urlopen", _mk_urlopen(http_code=401))
    assert W._gemini_key_valid("k")[0] is False
    assert W._brave_key_valid("k")[0] is False
    # Brave 422 -> rejected; Gemini 400 API_KEY_INVALID -> rejected
    monkeypatch.setattr(W.urllib.request, "urlopen", _mk_urlopen(http_code=422))
    assert W._brave_key_valid("k")[0] is False
    monkeypatch.setattr(W.urllib.request, "urlopen",
                        _mk_urlopen(http_code=400, body=b'{"error":{"status":"API_KEY_INVALID"}}'))
    assert W._gemini_key_valid("k")[0] is False
    # network error -> inconclusive -> valid (never block a good key on infra flakiness)
    monkeypatch.setattr(W.urllib.request, "urlopen", _mk_urlopen(exc=OSError("timeout")))
    assert W._gemini_key_valid("k")[0] is True
    assert W._brave_key_valid("k")[0] is True
    # Gemini 500 -> inconclusive -> valid
    monkeypatch.setattr(W.urllib.request, "urlopen", _mk_urlopen(http_code=500))
    assert W._gemini_key_valid("k")[0] is True


# --------------------------------------------------------------------------- #
# Launcher: per-task scoring + judge-fallback detection.
# --------------------------------------------------------------------------- #
def test_score_of_prefers_overall_then_mean():
    assert RUN._score_of({"overall_score": 0.8, "a": 0.2}) == pytest.approx(0.8)
    assert RUN._score_of({"a": 0.4, "b": 0.6}) == pytest.approx(0.5)     # mean when no overall_score
    assert RUN._score_of({"error": "x", "reason": "y"}) is None          # no numeric metrics


def test_judge_fell_back_detection():
    # prefixed + regex
    assert RUN._judge_fell_back({"recognized_fact_conflict_judge_method": "regex_fallback"}) is True
    # prefixed judge_error
    assert RUN._judge_fell_back({"llm_judge_error": "boom"}) is True
    # BARE judge_error (04_Search_Retrieval task_1 etc.)
    assert RUN._judge_fell_back({"judge_error": "outage"}) is True
    # BARE judge_method = rule_fallback (06_Safety task_3/task_6)
    assert RUN._judge_fell_back({"judge_method": "rule_fallback"}) is True
    # prefixed keyword_fallback (02_Code task_8/9/12)
    assert RUN._judge_fell_back({"desc_judge_method": "keyword_fallback"}) is True
    # judge_method == failed (02_Code task_10/11)
    assert RUN._judge_fell_back({"judge_method": "failed"}) is True
    # success markers are NOT fallbacks
    assert RUN._judge_fell_back({"recognized_fact_conflict_judge_method": "llm",
                                 "overall_score": 1.0}) is False
    assert RUN._judge_fell_back({"image_judge_method": "vlm", "overall_score": 0.8}) is False


def test_write_models_config(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://172.17.0.1:8000")
    monkeypatch.setenv("GBENCH_MODEL_NAME", "google/gemma-4-E4B-it")
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", "sk-dummy")
    path, model_arg = RUN._write_models_config()
    cfg = json.loads(open(path, encoding="utf-8").read())
    prov = cfg["providers"]["gbench-model"]
    assert prov["baseUrl"] == "http://172.17.0.1:8000/v1"       # /v1 ensured
    assert prov["api"] == "openai-completions"
    assert prov["models"][0]["id"] == "google/gemma-4-E4B-it"
    assert model_arg == "gbench-model/google/gemma-4-E4B-it"


def _make_harness_with_tasks(tmp_path, modalities):
    """Create a minimal tasks/ tree so RUN._modality_map() can resolve id -> modality."""
    for tid, mod in modalities.items():
        cat = tid.split("_task_")[0]
        d = tmp_path / "tasks" / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.md").write_text(
            f"---\nid: {tid}\nname: X\ncategory: {cat}\ntimeout_seconds: 60\nmodality: {mod}\n---\n## Prompt\nx\n",
            encoding="utf-8")
    return str(tmp_path)


def _write_run(out_root, category, task_id, score, usage):
    d = os.path.join(out_root, "openclaw", category, task_id, "m_20260101_0000_abc123")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "score.json"), "w", encoding="utf-8") as f:
        json.dump(score, f)
    with open(os.path.join(d, "usage.json"), "w", encoding="utf-8") as f:
        json.dump(usage, f)


def test_collect_and_summarize(tmp_path, monkeypatch):
    # 4 SELECTED tasks: scored PT, scored+fallback MM, grading-errored PT, and a MISSING PT
    # (no score.json). The denominator must be 4 (missing/errored count as 0), matching upstream.
    harness = _make_harness_with_tasks(tmp_path / "h", {
        "01_Productivity_Flow_task_1": "pure-text",
        "05_Creative_Synthesis_task_2": "multimodal",
        "06_Safety_Alignment_task_3": "pure-text",
        "06_Safety_Alignment_task_4": "pure-text",   # selected but never scored (missing)
    })
    monkeypatch.setattr(RUN, "HARNESS_DIR", harness)
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)   # -> all categories -> all 4 selected
    out_root = str(tmp_path / "out")
    _write_run(out_root, "01_Productivity_Flow", "01_Productivity_Flow_task_1",
               {"overall_score": 1.0}, {"elapsed_time": 120.0, "cost_usd": 0.0})
    _write_run(out_root, "05_Creative_Synthesis", "05_Creative_Synthesis_task_2",
               {"overall_score": 0.5, "desc_judge_method": "keyword_fallback"},
               {"elapsed_time": 240.0, "cost_usd": 0.0})
    _write_run(out_root, "06_Safety_Alignment", "06_Safety_Alignment_task_3",
               {"error": "grade script failed"}, {"elapsed_time": 60.0})
    # task_4: no run dir at all -> missing

    index = RUN._task_index()
    summary = RUN._summarize(out_root, index)
    assert summary["n_tasks"] == 4                 # denominator = SELECTED tasks, not files found
    assert summary["n_scored"] == 2
    assert summary["n_missing"] == 1               # task_4
    assert summary["n_grading_errors"] == 1        # task_3
    # global mean over ALL 4 selected: (1.0 + 0.5 + 0 + 0) / 4
    assert summary["global_mean"] == pytest.approx(1.5 / 4)
    assert summary["multimodal_mean"] == pytest.approx(0.5)               # 1 MM task, scored 0.5
    assert summary["pure_text_mean"] == pytest.approx(1.0 / 3)            # 3 PT: 1.0, 0(err), 0(missing)
    assert summary["n_multimodal"] == 1 and summary["n_pure_text"] == 3
    assert summary["judge_fallback_tasks"] == 1                           # keyword_fallback detected


def test_summarize_single_task_selection(tmp_path, monkeypatch):
    harness = _make_harness_with_tasks(tmp_path / "h", {
        "06_Safety_Alignment_task_7_skill_injection": "pure-text",
        "01_Productivity_Flow_task_1": "pure-text",
    })
    monkeypatch.setattr(RUN, "HARNESS_DIR", harness)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    monkeypatch.setenv("WILDCLAW_TASK",
                       os.path.join(harness, "tasks", "06_Safety_Alignment",
                                    "06_Safety_Alignment_task_7_skill_injection.md"))
    out_root = str(tmp_path / "out")
    _write_run(out_root, "06_Safety_Alignment", "06_Safety_Alignment_task_7_skill_injection",
               {"overall_score": 0.0}, {"elapsed_time": 3.5, "cost_usd": 0.0})
    summary = RUN._summarize(out_root, RUN._task_index())
    assert summary["n_tasks"] == 1                 # only the single selected task, not both
    assert summary["global_mean"] == pytest.approx(0.0)


def test_collect_newest_run_wins(tmp_path, monkeypatch):
    harness = _make_harness_with_tasks(tmp_path / "h", {"01_Productivity_Flow_task_1": "pure-text"})
    monkeypatch.setattr(RUN, "HARNESS_DIR", harness)
    out_root = str(tmp_path / "out")
    base = os.path.join(out_root, "openclaw", "01_Productivity_Flow", "01_Productivity_Flow_task_1")
    old = os.path.join(base, "m_20260101_0000_old000")
    new = os.path.join(base, "m_20260101_0100_new000")
    for d, score in ((old, 0.1), (new, 0.9)):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "score.json"), "w") as f:
            json.dump({"overall_score": score}, f)
    os.utime(os.path.join(new, "score.json"), (2e9, 2e9))   # make 'new' clearly newer
    os.utime(os.path.join(old, "score.json"), (1e9, 1e9))
    rows = RUN._collect(out_root, RUN._task_index())
    assert len(rows) == 1 and rows["01_Productivity_Flow_task_1"]["overall_score"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Cascade judge proxy: config, model override, backoff, outage.
# --------------------------------------------------------------------------- #
def test_cascade_config(monkeypatch):
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    assert J._cascade() == J._DEFAULT_JUDGE_CASCADE
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-x")
    assert J._cascade() == ["gemini-x"]
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "a, b ,c")
    assert J._cascade() == ["a", "b", "c"]                    # csv wins over single
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "5")
    assert J._rounds() == 5


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_cascade_overrides_model_and_temperature(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-primary,gemini-backup")
    seen = {}

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data.decode("utf-8"))
        seen["model"] = body["model"]
        seen["temperature"] = body["temperature"]
        seen["has_stream"] = "stream" in body
        return _FakeResp({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    out = J.cascade_chat_completion({"model": "openai/gpt-5.4", "temperature": 0.7,
                                     "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen["model"] == "gemini-primary"        # requested model ignored, cascade used
    assert seen["temperature"] == 0.0               # forced deterministic
    assert seen["has_stream"] is False              # OpenRouter-only field dropped


def test_cascade_strips_thinking_and_nonfunction_tools(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-primary")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    # 06_Safety tasks add extra_body thinking; 04_Search_task_1 adds an OpenRouter server-side tool.
    J.cascade_chat_completion({
        "model": "openai/gpt-5.4",
        "thinking": {"type": "disabled"},
        "reasoning_effort": "low",
        "tools": [{"type": "openrouter:web_search", "parameters": {}},
                  {"type": "function", "function": {"name": "keep_me"}}],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "hi"}],
    })
    b = seen["body"]
    assert "thinking" not in b and "reasoning_effort" not in b     # Gemini-incompatible fields dropped
    assert [t["type"] for t in b["tools"]] == ["function"]         # only standard function tools kept
    assert b["tools"][0]["function"]["name"] == "keep_me"


def test_cascade_drops_tools_and_tool_choice_when_none_survive(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-primary")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    J.cascade_chat_completion({"tools": [{"type": "openrouter:web_search"}],
                               "tool_choice": "required",
                               "messages": [{"role": "user", "content": "x"}]})
    assert "tools" not in seen["body"] and "tool_choice" not in seen["body"]


def test_cascade_falls_through_then_backoff_then_outage(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1,m2")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "2")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "1.0")
    sleeps = []
    monkeypatch.setattr(J.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(J.random, "uniform", lambda a, b: 0.0)

    def always_fail(req, timeout=0):
        raise RuntimeError("429")
    monkeypatch.setattr(J.urllib.request, "urlopen", always_fail)
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE"):
        J.cascade_chat_completion({"messages": [{"role": "user", "content": "x"}]})
    # 2 rounds -> one backoff sleep BETWEEN them: backoff*2^0 + uniform(0) = 1.0
    assert sleeps == [pytest.approx(1.0)]


def test_cascade_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        J.cascade_chat_completion({"messages": []})
