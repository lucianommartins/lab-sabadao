# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the BFCL v4 agentic wrapper (canonical `bfcl-eval` harness).

Everything here runs offline: the harness itself, the model endpoint and the search
backend are all exercised through their seams, never over the network.
"""

import json
import os

import pytest

from gbench.runners.eval_suites import SUITES
from gbench.runners.eval_suites.bfcl_v4_agentic import (
    WEB_SEARCH_CATEGORIES,
    _parse_scores,
    check_bfcl_prerequisites,
    run_bfcl_v4_agentic,
    search_backend,
)

bfcl_eval = pytest.importorskip("bfcl_eval", reason="canonical BFCL harness not installed")


def test_both_bfcl_suites_are_registered():
    """The v3 LIVE slice keeps running under an honest name; v4 is the agentic track."""
    assert "bfcl_v3_live" in SUITES
    assert "bfcl_v4_agentic" in SUITES


def test_agentic_categories_match_the_harness():
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    cats = set(TEST_COLLECTION_MAPPING["agentic"])
    assert cats == {"memory_kv", "memory_vector", "memory_rec_sum",
                    "web_search_base", "web_search_no_snippet"}
    assert set(WEB_SEARCH_CATEGORIES) <= cats


def test_search_backend_selection(monkeypatch):
    from gbench.runners.eval_suites import base as _base
    # Gemini grounding is now gated on key LIVENESS; stub the network probe so this test
    # exercises selection logic, not a real /v1beta call.
    monkeypatch.setattr(_base, "gemini_key_live_valid", lambda key: (True, ""))
    # explicit request without the key must NOT silently fall back to another backend
    monkeypatch.setenv("BFCL_SEARCH_BACKEND", "serpapi")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert search_backend()[0] is None

    monkeypatch.setenv("BFCL_SEARCH_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert search_backend()[0] == "gemini"
    # a DEAD key must not select gemini even when forced (liveness gate, no fabricated grounding)
    monkeypatch.setattr(_base, "gemini_key_live_valid", lambda key: (False, "401"))
    assert search_backend()[0] is None
    monkeypatch.setattr(_base, "gemini_key_live_valid", lambda key: (True, ""))

    # auto: canonical SerpAPI wins when its key exists, else live Gemini
    monkeypatch.delenv("BFCL_SEARCH_BACKEND", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "s")
    assert search_backend()[0] == "serpapi"
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert search_backend()[0] == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert search_backend()[0] is None


def test_prereq_unreachable_endpoint_hard_errors(tmp_path, monkeypatch):
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, reason, cats = check_bfcl_prerequisites("http://127.0.0.1:9/v1")
    assert ok is False and "not reachable" in reason and cats == []

    with pytest.raises(RuntimeError, match="docs/evals/bfcl_v4_agentic.md"):
        run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)


def test_prereq_unwritable_project_root_is_a_clean_skip(monkeypatch):
    monkeypatch.setenv("BFCL_PROJECT_ROOT", "/proc/definitely/not/writable")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, reason, _ = check_bfcl_prerequisites("http://127.0.0.1:9/v1")
    assert ok is False and "not writable" in reason


def test_missing_search_backend_runs_memory_only(tmp_path, monkeypatch):
    """No search key must degrade to the 3 memory categories, not fake a full-track score."""
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    for k in ("GEMINI_API_KEY", "SERPAPI_API_KEY", "BFCL_SEARCH_BACKEND"):
        monkeypatch.delenv(k, raising=False)

    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    monkeypatch.setattr(B, "_endpoint_ok", lambda *_a, **_k: (True, ""), raising=False)
    # endpoint check is inline, so aim at the live-check seam via a stub urlopen
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())

    ok, _reason, cats = check_bfcl_prerequisites("http://x/v1")
    assert ok is True
    assert cats == ["memory_kv", "memory_vector", "memory_rec_sum"]
    assert not any(c in cats for c in WEB_SEARCH_CATEGORIES)


def test_gemini_search_maps_grounding_to_bfcl_result_shape(monkeypatch):
    """BFCL requires [{title, href, body}]; grounding supports supply the snippet."""
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    payload = {
        "candidates": [{
            "content": {"parts": [{"text": "Canberra is the capital."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"title": "wikipedia.org", "uri": "https://example/1"}},
                    {"web": {"title": "abc.net.au", "uri": "https://example/2"}},
                ],
                "groundingSupports": [
                    {"segment": {"text": "Canberra is the capital"},
                     "groundingChunkIndices": [0]},
                ],
            },
        }]
    }

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _R())
    out = B._gemini_search("capital of australia")
    assert isinstance(out, list) and len(out) == 2
    assert set(out[0]) == {"title", "href", "body"}
    assert out[0]["href"] == "https://example/1"
    assert "Canberra is the capital" in out[0]["body"]      # from groundingSupports
    assert out[1]["body"]                                    # falls back to the answer text


def test_gemini_search_reports_errors_instead_of_raising(monkeypatch):
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    out = B._gemini_search("q")
    assert isinstance(out, dict) and "error" in out          # BFCL's documented error shape


def test_score_parsing_into_gbench_shape(tmp_path):
    """_parse_scores must turn the harness' score files into category_accuracy.

    The harness nests scores by track (`<model>/agentic/memory/kv/..._score.json`) and the
    summary is the FIRST line of a JSONL file whose remaining lines are per-sample records -
    a flat glob or a whole-file json.load both yield nothing.
    """
    model = "google/gemma-4-26B-A4B-it"
    d = tmp_path / model.replace("/", "_") / "agentic" / "memory" / "kv"
    d.mkdir(parents=True)
    (d / "BFCL_v4_memory_kv_score.json").write_text(
        json.dumps({"accuracy": 0.14838709677419354, "correct_count": 23, "total_count": 155})
        + "\n" + json.dumps({"id": "memory_kv_0", "valid": False}) + "\n",
        encoding="utf-8")
    per_cat = _parse_scores(tmp_path, model, ["memory_kv", "memory_vector"])
    assert per_cat["memory_kv"] == {"correct": 23, "total": 155, "accuracy": 14.84}
    assert "memory_vector" not in per_cat          # categories that did not run are absent


def test_child_bootstrap_registers_model_and_patches_search():
    """Patches must be re-applied inside the harness subprocess, not just the parent."""
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    code = B._CHILD_BOOTSTRAP.format(model="m", backend="gemini", args=["generate"])
    assert "_register_model('m')" in code or '_register_model("m")' in code
    assert "_patch_search_backend" in code
    assert "from bfcl_eval.__main__ import cli" in code


def test_registered_model_uses_native_function_calling():
    """GemmaHandler's prompt-mode decode fails on a vLLM-served gemma-4; FC mode is required."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    name = "test-only/gemma-4-fake-it"
    try:
        B._register_model(name)
        cfg = MODEL_CONFIG_MAPPING[name]
        assert cfg.is_fc_model is True
        # the registered handler is the OpenAI completions handler, wrapped by the shim
        # that keeps a no-tool-call turn from raising during decode
        assert issubclass(cfg.model_handler, OpenAICompletionsHandler)
        assert cfg.model_handler.__name__ == "_GbenchOpenAICompletionsHandler"
    finally:
        MODEL_CONFIG_MAPPING.pop(name, None)


def test_openai_handler_shim_survives_a_prose_turn():
    """A turn with no tool call must decode to "no calls", not raise.

    bfcl-eval's OpenAICompletionsHandler falls back to the raw `message.content` string
    when the model does not call a tool, then iterates it CHARACTER BY CHARACTER and calls
    `.items()` on each character. Every prose turn therefore raised
    `'str' object has no attribute 'items'` and was logged "Failed to decode the model
    response" - 1225 of ~1425 decode errors in the memory_kv run came from this one path,
    for any model, which made the reported score a harness artifact.
    """
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from gbench.runners.eval_suites.bfcl_v4_agentic import _patched_openai_handler

    patched_cls = _patched_openai_handler()
    patched = patched_cls.__new__(patched_cls)
    patched.is_fc_model = True
    upstream = OpenAICompletionsHandler.__new__(OpenAICompletionsHandler)
    upstream.is_fc_model = True

    prose = "I do not have your first name stored in my memory records."
    with pytest.raises(AttributeError):
        upstream.decode_execute(prose, False)
    assert patched.decode_execute(prose, False) == []

    # a call written as text is still recovered rather than lost
    assert patched.decode_execute("[get_user_info(user_id=7)]", False) == \
        ["get_user_info(user_id=7)"]

    # and genuine tool calls decode exactly as before
    tool_calls = [{"get_user_info": '{"user_id": 7, "verbose": true}'}]
    assert patched.decode_execute(tool_calls, False) == \
        upstream.decode_execute(tool_calls, False)


def test_openai_handler_shim_is_registered():
    """The registration must use the patched handler, not the upstream one."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from gbench.runners.eval_suites.bfcl_v4_agentic import _register_model

    name = "test/gbench-shim-probe"
    MODEL_CONFIG_MAPPING.pop(name, None)
    try:
        _register_model(name)
        handler = MODEL_CONFIG_MAPPING[name].model_handler
        assert handler.__name__ == "_GbenchOpenAICompletionsHandler"
        assert "decode_execute" in vars(handler)
    finally:
        MODEL_CONFIG_MAPPING.pop(name, None)


def test_generate_regenerates_by_default(monkeypatch, tmp_path):
    """`bfcl generate` must not silently re-score stale generations.

    Upstream defaults `--allow-overwrite=False`, which loads whatever is already in
    <BFCL_PROJECT_ROOT>/result/, subtracts those ids from the work list and generates
    nothing when the file is complete. Observed live: a 665-question track "completed" in
    19 s against generations from two days earlier. gbench's --skip-existing is a
    different mechanism entirely and does not affect it.
    """
    import gbench.runners.eval_suites.bfcl_v4_agentic as B

    calls = []

    def fake_cli(argv, env, timeout, model=None, backend=None):
        calls.append(argv)
        return 0, ""

    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "_run_cli", fake_cli)
    monkeypatch.setattr(B, "check_bfcl_prerequisites", lambda url: (True, "", ["memory_kv"]))
    monkeypatch.setattr(B, "_register_model", lambda m: m)
    monkeypatch.setattr(B, "_parse_scores", lambda *a, **k: {
        "memory_kv": {"correct": 23, "total": 155, "accuracy": 14.84}})

    monkeypatch.delenv("BFCL_REUSE_GENERATIONS", raising=False)
    res = B.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    gen = next(c for c in calls if c[0] == "generate")
    assert "--allow-overwrite" in gen
    assert res["bfcl_report"]["generations"] == "regenerated"

    # explicit opt-in re-scores instead, and says so on the result
    calls.clear()
    monkeypatch.setenv("BFCL_REUSE_GENERATIONS", "1")
    res = B.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    gen = next(c for c in calls if c[0] == "generate")
    assert "--allow-overwrite" not in gen
    assert "reused" in res["bfcl_report"]["generations"]


# --------------------------------------------------------------------------- #
# --eval-limit (audit P2-8): `bfcl generate` has no --limit, so gbench writes an id file
# --------------------------------------------------------------------------- #
AGENTIC_CATEGORIES = ["memory_kv", "memory_vector", "memory_rec_sum",
                      "web_search_base", "web_search_no_snippet"]


def _scenario(cat, sid):
    tail = sid[len(cat) + 1:] if sid.startswith(f"{cat}_") else sid
    parts = tail.split("-")
    return parts[1] if len(parts) >= 2 else cat


def test_eval_limit_selects_exactly_that_many_scored_questions():
    """The 2026-08-15 sweep ran all 665 questions under `--eval-limit 20`."""
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 20)
    scored = [i for ids in sel.values() for i in ids if "_prereq" not in i]
    assert len(scored) == 20


def test_eval_limit_spreads_across_categories_and_scenarios():
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 20)
    per_cat = {c: [i for i in ids if "_prereq" not in i] for c, ids in sel.items()}
    assert sorted(per_cat) == sorted(AGENTIC_CATEGORIES), "every category must be sampled"
    assert all(len(v) == 4 for v in per_cat.values()), per_cat
    # memory_kv's first 12 ids are all the `customer` scenario; a head would collapse.
    kv = {_scenario("memory_kv", i) for i in per_cat["memory_kv"]}
    assert len(kv) >= 3, f"memory_kv collapsed to {kv}"


def test_eval_limit_pulls_in_memory_prerequisites():
    """A memory question scored without its `depends_on` setup measures nothing."""
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 20)
    assert any("_prereq" in i for i in sel["memory_kv"])
    assert not any("_prereq" in i for i in sel["web_search_base"]), \
        "web_search has no prerequisites and must not gain any"


def test_eval_limit_is_deterministic():
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    assert _select_limited_ids(AGENTIC_CATEGORIES, 12) == \
           _select_limited_ids(AGENTIC_CATEGORIES, 12)


def test_limit_above_the_dataset_returns_everything():
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 100000)
    scored = [i for ids in sel.values() for i in ids if "_prereq" not in i]
    assert len(scored) == 665


def test_limited_run_uses_run_ids_and_a_separate_result_dir(tmp_path, monkeypatch):
    """`--run-ids` updates result files in place, so a limited run must not share the
    full run's directory - the harness would score the leftovers too (audit RC-4)."""
    from gbench.runners.eval_suites import bfcl_v4_agentic as B

    calls = []

    def _fake_cli(argv, env, timeout, model=None, backend=None):
        calls.append(list(argv))
        return 0, "ok"

    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "_run_cli", _fake_cli)
    monkeypatch.setattr(B, "check_bfcl_prerequisites",
                        lambda url: (True, "", AGENTIC_CATEGORIES))
    monkeypatch.setattr(B, "search_backend", lambda: ("gemini", ""))
    monkeypatch.setattr(B, "_register_model", lambda m: "gbench-model")
    monkeypatch.setattr(B, "_patch_search_backend", lambda b: None)
    monkeypatch.setattr(B, "_parse_scores",
                        lambda root, model, cats: {"memory_kv": {"correct": 1, "total": 2}})

    res = B.run_bfcl_v4_agentic("m", "http://x", concurrency=2, limit=20)

    gen, ev = calls[0], calls[1]
    assert "--run-ids" in gen and "--result-dir" in gen
    assert "--partial-eval" in ev, "a partial category set raises without --partial-eval"
    assert gen[gen.index("--result-dir") + 1] == ev[ev.index("--result-dir") + 1]
    assert gen[gen.index("--result-dir") + 1] != "result"

    id_file = tmp_path / "test_case_ids_to_generate.json"
    assert id_file.exists(), "the harness reads the id list from this file"
    written = json.loads(id_file.read_text())
    assert sum(1 for ids in written.values() for i in ids if "_prereq" not in i) == 20
    assert res["bfcl_report"]["eval_limit"] == 20


def test_unlimited_run_keeps_the_default_directories(tmp_path, monkeypatch):
    from gbench.runners.eval_suites import bfcl_v4_agentic as B

    calls = []
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "_run_cli",
                        lambda argv, env, t, model=None, backend=None: (calls.append(list(argv)), (0, "ok"))[1])
    monkeypatch.setattr(B, "check_bfcl_prerequisites", lambda url: (True, "", ["memory_kv"]))
    monkeypatch.setattr(B, "search_backend", lambda: (None, "no backend"))
    monkeypatch.setattr(B, "_register_model", lambda m: "gbench-model")
    monkeypatch.setattr(B, "_parse_scores",
                        lambda root, model, cats: {"memory_kv": {"correct": 1, "total": 2}})

    res = B.run_bfcl_v4_agentic("m", "http://x", concurrency=2)
    assert not any("--run-ids" in c for c in calls)
    assert not (tmp_path / "test_case_ids_to_generate.json").exists()
    assert res["bfcl_report"]["eval_limit"] is None
