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

"""Contract: a reported number must say how it was produced, and knobs must arrive.

Three defects from the 2026-08-15 sweep are locked in here:

* RC-5 - seven suites quietly swapped the canonical LLM judge for a substring match when
  `GEMINI_API_KEY` was missing and reported the result as the benchmark's metric.
* RC-2 - `--temperature` was forwarded by 15 of 122 suites; the other 107 dropped it, so
  the run's headline knob silently did nothing.
* P1-7 - long-answer suites hardcoded 2048-8192 output tokens and were scored on truncated
  patches (swe_bench_multilingual 10/20, codeforces 5/6, copilot_bench_swe 5/20).
"""

import asyncio
import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

SUITE_DIR = Path(__file__).resolve().parent.parent / "gbench" / "runners" / "eval_suites"

#: Suites with a documented no-judge fallback path (they degrade to substring matching).
#: NB: skillsbench and wildclawbench were removed - they are now canonical container harnesses
#: (the upstream runner docker-out-of-docker + per-task grading). skillsbench uses a DETERMINISTIC
#: verifier (no judge). wildclawbench's judge (gbench Gemini cascade) runs INSIDE each task
#: container via the task's own grade(); its regex fallback is the upstream task code's, surfaced by
#: the launcher as `judge_fallback_tasks` in the summary, not an in-process scoring_mode marker.
FALLBACK_SUITES = ["aa_lcr", "beam_128k", "frames", "simpleqa",
                   "cybergym", "cimemories"]


# --------------------------------------------------------------------------- #
# RC-5: a fallback must never be presented as canonical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", FALLBACK_SUITES)
def test_fallback_branch_marks_itself(suite):
    """Every no-key branch tags its traces, so the aggregate can tell them apart."""
    src = (SUITE_DIR / f"{suite}.py").read_text(encoding="utf-8")
    assert 'GEMINI_API_KEY' in src, f"{suite}: expected a judge-availability check"
    assert 'trace["scoring_mode"] = "judge_fallback"' in src, (
        f"{suite}: the no-judge branch scores with a substring match but does not mark "
        f'the traces. Set trace["scoring_mode"] = "judge_fallback" so the result cannot '
        f"claim to be the canonical metric.")


def _run_with_traces(traces, **kw):
    """Drive run_eval_suite over canned replies and return its result dict."""
    samples = [([{"role": "user", "content": "q"}], "gold", {}) for _ in traces]

    async def _fake_send(*a, **k):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    def _apply(sample_traces):
        for t, patch in zip(sample_traces, traces):
            t.update(patch)

    async def _async_eval(sample_traces):
        _apply(sample_traces)

    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(
            eval_name="unit_test_suite", model_name="m", base_url="http://x",
            concurrency=1, samples=samples, async_eval_fn=_async_eval, **kw)


def test_result_reports_judge_fallback_when_any_trace_fell_back():
    res = _run_with_traces([
        {"is_correct": True, "scoring_mode": "judge_fallback"},
        {"is_correct": True},
    ])
    assert res["scoring_mode"] == "judge_fallback", (
        "one fallback-scored sample is enough: the suite's number is a lower bound, "
        "not the canonical metric")


def test_result_reports_judge_when_nothing_fell_back():
    res = _run_with_traces([{"is_correct": True}, {"is_correct": False}])
    assert res["scoring_mode"] == "judge"


def test_judge_outage_is_excluded_from_accuracy_not_scored_as_wrong():
    # 2 correct, 1 wrong, 1 judge outage -> accuracy is over the 3 scored samples (2/3),
    # NOT 2/4. An exhausted judge cascade is infra, not a wrong answer.
    res = _run_with_traces([
        {"is_correct": True},
        {"is_correct": True},
        {"is_correct": False},
        {"judge_grade": "JUDGE_OUTAGE"},
    ])
    assert res["judge_outages"] == 1
    assert res["judge_outage_rate"] == 0.25
    assert res["accuracy"] == round(2 / 3 * 100.0, 2)     # 66.67, not 50.0
    assert res["status"] == "completed_with_errors"       # an outage is not a clean run


def test_all_judge_outages_is_a_failed_run_not_zero_percent():
    res = _run_with_traces([{"judge_grade": "JUDGE_OUTAGE"},
                            {"judge_grade": "JUDGE_OUTAGE"}])
    assert res["judge_outages"] == 2
    assert res["status"] == "failed"


# --------------------------------------------------------------------------- #
# judge cascade (mirrors search_tool's grounding cascade)
# --------------------------------------------------------------------------- #
class _FakeJudgeClient:
    def __init__(self, fail_models):
        self._fail = set(fail_models)
        outer = self

        class _Models:
            async def generate_content(self, model, contents, config):
                if model in outer._fail:
                    raise RuntimeError("429 quota")

                class _R:
                    text = "graded"
                return _R()

        class _Aio:
            models = _Models()
        self.aio = _Aio()


def test_judge_cascade_falls_through_to_the_next_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1,m2")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    _fake = _FakeJudgeClient(fail_models={"m1"})
    monkeypatch.setattr(base, "_judge_clients", lambda: [_fake])   # non-empty: skip genai build
    monkeypatch.setattr(base, "_next_judge_client", lambda: _fake)  # rotation returns the fake
    text, used = asyncio.run(base.judge_generate_cascade("prompt"))
    assert (text, used) == ("graded", "m2")     # first model 429'd, next served


def test_judge_cascade_exhausted_is_a_judge_outage(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1,m2")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    _fake = _FakeJudgeClient(fail_models={"m1", "m2"})
    monkeypatch.setattr(base, "_judge_clients", lambda: [_fake])   # non-empty: skip genai build
    monkeypatch.setattr(base, "_next_judge_client", lambda: _fake)  # rotation returns the fake
    text, used = asyncio.run(base.judge_generate_cascade("prompt"))
    assert text is None and used == "JUDGE_OUTAGE"


def test_judge_cascade_without_api_key_is_an_outage(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    text, used = asyncio.run(base.judge_generate_cascade("prompt"))
    assert text is None and used == "JUDGE_OUTAGE"


# --------------------------------------------------------------------------- #
# trajectories save BOTH the thinking content and its length (all base-path suites)
# --------------------------------------------------------------------------- #
def test_trace_saves_both_thinking_content_and_length():
    """Every suite that generates through run_eval_suite must record the thinking CONTENT
    and its length on the trace - reply.reasoning was captured but only len() was stored."""
    async def _fake_send(*a, **k):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop",
                          reasoning="let me think carefully about this")

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="unit_test_suite", model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": "q"}], "gold", {})],
            eval_fn=lambda r, g: True)
    tr = res["sample_traces"][0]
    assert tr["reasoning"] == "let me think carefully about this"
    assert tr["reasoning_chars"] == len("let me think carefully about this")


def test_result_reports_deterministic_without_a_judge():
    samples = [([{"role": "user", "content": "q"}], "answer", {})]

    async def _fake_send(*a, **k):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="unit_test_suite", model_name="m", base_url="http://x",
            concurrency=1, samples=samples, eval_fn=lambda r, g: r == g)
    assert res["scoring_mode"] == "deterministic"


def test_fallback_is_logged_loudly(caplog):
    with caplog.at_level("WARNING"):
        _run_with_traces([{"is_correct": True, "scoring_mode": "judge_fallback"}])
    assert any("fallback" in r.message.lower() or "fallback" in str(r.args).lower()
               for r in caplog.records), "the downgrade must be visible in the run log"


def test_cimemories_has_no_length_proxy():
    """`len(resp) >= 30` scored 20/20 including an answer about a different person."""
    src = (SUITE_DIR / "cimemories.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"len\(\s*resp\w*\s*\)\s*[<>]=?\s*\d", code), (
        "cimemories must not grade response length; contextual integrity needs a judge")
    assert "_async_judge_cimemories" in src


# --------------------------------------------------------------------------- #
# RC-2: run-level knobs must reach every suite
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_knobs():
    base._RUN_KNOBS.clear()
    yield
    base._RUN_KNOBS.clear()


def _capture_payload(**suite_kw):
    """Run one sample and return the payload run_eval_suite would have sent."""
    seen = {}
    samples = [([{"role": "user", "content": "q"}], "gold", {})]

    async def _fake_send(**k):
        seen.update(k)
        return base.Reply(text="x", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", _fake_send):
        base.run_eval_suite(
            eval_name=suite_kw.pop("eval_name", "unit_test_suite"),
            model_name="m", base_url="http://x", concurrency=1, samples=samples,
            eval_fn=lambda r, g: True, **suite_kw)
    return seen


def test_temperature_knob_reaches_a_suite_that_never_forwarded_it():
    base.set_run_knobs(temperature=0.7)
    seen = _capture_payload()
    assert seen.get("temperature") == 0.7, (
        "a suite that does not forward --temperature must still receive it from the "
        "run-level knobs (audit RC-2)")


def test_the_cli_temperature_wins_over_a_suite_default():
    """REVERSED 2026-08-17. This used to assert the opposite - that a suite's own
    temperature beat `--temperature` - on the grounds that a per-suite value is a
    measurement decision. In practice that made `--temperature` a no-op for the ~20
    suites that encoded one, while being documented as applying "across all benchmarks".

    The suite value is still honoured; it just sits BELOW the operator in the chain:
        GBENCH_<EVAL>_TEMPERATURE  >  --temperature  >  suite value  >  1.0
    """
    base.set_run_knobs(temperature=0.7)
    seen = _capture_payload(temperature=0.0)
    assert seen.get("temperature") == 0.7


def test_a_suite_default_still_applies_when_the_operator_is_silent():
    """Reversing the precedence must not throw the suite's protocol away entirely."""
    base._RUN_KNOBS.pop("temperature", None)
    seen = _capture_payload(temperature=0.0)
    assert seen.get("temperature") == 0.0


def test_a_per_suite_env_override_beats_even_the_cli():
    import os
    from unittest import mock
    base.set_run_knobs(temperature=0.7)
    with mock.patch.dict(os.environ, {"GBENCH_GSM8K_TEMPERATURE": "0.25"}):
        seen = _capture_payload(eval_name="gsm8k", temperature=0.0)
    assert seen.get("temperature") == 0.25


def test_set_run_knobs_ignores_none():
    base.set_run_knobs(temperature=0.7)
    base.set_run_knobs(temperature=None)
    assert base.get_run_knob("temperature") == 0.7, (
        "an unset CLI flag arrives as None and must not erase a knob")


# --------------------------------------------------------------------------- #
# P1-7: long-answer suites get an output-token floor
# --------------------------------------------------------------------------- #
def test_floor_raises_a_hardcoded_suite_default():
    """multipl_e hardcodes 2048; a program does not fit in 2048 tokens."""
    seen = _capture_payload(eval_name="multipl_e", max_output_tokens=2048)
    got = seen.get("max_output_tokens")
    assert got == base.DEFAULT_MIN_OUTPUT_TOKENS, (
        f"expected the floor {base.SUITE_MIN_OUTPUT_TOKENS['multipl_e']}, got {got}: a "
        "suite-hardcoded default must not undercut the floor, or the truncation stays")


def test_explicit_operator_knob_beats_the_floor():
    base.set_run_knobs(max_output_tokens=512)
    seen = _capture_payload(eval_name="multipl_e", max_output_tokens=2048)
    got = seen.get("max_output_tokens")
    assert got == 512, "--max-output-tokens is an explicit choice and always wins"


def test_floor_does_not_lower_a_generous_suite():
    seen = _capture_payload(eval_name="multipl_e", max_output_tokens=65536)
    got = seen.get("max_output_tokens")
    assert got == 65536


def test_every_floor_names_a_real_suite():
    names = {p.stem for p in SUITE_DIR.glob("*.py")}
    unknown = sorted(set(base.SUITE_MIN_OUTPUT_TOKENS) - names)
    assert not unknown, f"floor table names suites that do not exist: {unknown}"


def test_truncation_prone_suites_are_covered():
    """The suites the sweeps measured hitting the cap must all clear the old 8192 default."""
    measured = ["aime", "arc_agi", "codeforces", "copilot_bench_swe", "culer", "gpqa",
                "gpqa_diamond", "hmmt", "humanitys_last_exam", "ifeval", "imo_answer_bench",
                "livebench", "loft_x_arxiv", "amc_aime", "putnam", "swe_bench_multilingual"]
    for s_ in measured:
        budget = base.SUITE_MIN_OUTPUT_TOKENS.get(s_, base.DEFAULT_MIN_OUTPUT_TOKENS)
        assert budget > 8192, s_


# --------------------------------------------------------------------------- #
# P1-5: no dead knobs. Every eval flag reaches a suite, or says whose it is.
# --------------------------------------------------------------------------- #
#: flag -> (kwargs key, who consumes it). A knob nothing reads is a lie in --help.
EVAL_KNOBS = {
    "--eval-thinking": ("enable_thinking", None),          # every suite takes it
    "--max-output-tokens": ("max_output_tokens", None),    # resolved centrally in base
    "--temperature": ("temperature", None),                # resolved centrally in base
    "--eval-limit": ("limit", None),
    "--eval-n-shot": ("eval_n_shot", "MMLU-Pro"),
    "--eval-categories": ("eval_categories", "BFCL"),
    "--eval-max-soft-tokens": ("eval_max_soft_tokens", "vision"),
    "--sandboxes": ("sandboxes", "sandboxed"),
    "--suite-timeout": ("suite_timeout", None),
}


def _cli_source():
    return (Path(__file__).resolve().parent.parent / "gbench" / "cli.py").read_text("utf-8")


@pytest.mark.parametrize("flag,spec", sorted(EVAL_KNOBS.items()))
def test_every_eval_knob_is_actually_consumed(flag, spec):
    key, _ = spec
    cli, runner = _cli_source(), (
        Path(__file__).resolve().parent.parent / "gbench" / "runners" / "evals.py"
    ).read_text("utf-8")
    assert f'"{flag}"' in cli, f"{flag} is documented here but no longer exists"

    if key in ("temperature", "max_output_tokens"):
        # Not forwarded by most suites; base.run_eval_suite resolves them from the
        # run knobs, which is the whole point of the RC-2 fix.
        base_src = (SUITE_DIR / "base.py").read_text("utf-8")
        assert f'get_run_knob("{key}"' in base_src
        assert "set_run_knobs(" in runner
        return
    if key == "suite_timeout":
        assert "suite_timeout" in runner
        return
    assert f'"{key}"' in runner or f"'{key}'" in runner, (
        f"{flag} is parsed but never put in the suite kwargs")

    if key in ("eval_n_shot", "eval_categories", "eval_max_soft_tokens"):
        consumers = [p.stem for p in SUITE_DIR.glob("*.py")
                     if key in p.read_text("utf-8")]
        assert consumers, f"{flag} reaches no suite at all"


@pytest.mark.parametrize("flag,spec", sorted(
    (f, s) for f, s in EVAL_KNOBS.items() if s[1]))
def test_narrow_knobs_say_so_in_help(flag, spec):
    """A flag only one suite reads must not read as global in --help."""
    _, who = spec
    src = _cli_source()
    start = src.index(f'"{flag}"')
    block = src[start:start + 700]
    assert who.lower() in block.lower(), (
        f"{flag} is only honoured by {who}; --help must say so or operators will assume "
        f"it applied to the whole run")


# --------------------------------------------------------------------------- #
# --skip-existing must stay OFF by default
# --------------------------------------------------------------------------- #
def _parse(*argv):
    import sys
    from gbench.cli import create_parser
    old = sys.argv
    sys.argv = ["gbench", *argv]
    try:
        return create_parser().parse_args()
    finally:
        sys.argv = old


def test_skip_existing_is_off_unless_asked_for():
    """Reusing a cached result silently mixes two methodologies.

    It defaulted to True, so a run that changed how a suite is scored would quietly
    report the *old* number for every suite that already had a file anywhere under
    --results-dir. Resuming is a deliberate act, not the default.
    """
    assert _parse("--evals", "all").skip_existing is False


def test_skip_existing_can_still_be_requested():
    assert _parse("--evals", "all", "--skip-existing").skip_existing is True
    assert _parse("--evals", "all", "--no-skip-existing").skip_existing is False
    # last flag wins, so an alias pair cannot leave it ambiguous
    assert _parse("--evals", "all", "--skip-existing", "--no-skip-existing").skip_existing is False


def test_skip_existing_help_warns_about_mixing():
    src = _cli_source()
    start = src.index('"--skip-existing"')
    block = src[start:start + 700]
    assert "default" not in block.lower() or "off by default" in block.lower(), (
        "the help text still advertises the old default")
    assert "resum" in block.lower(), (
        "--help must say what --skip-existing is for, or operators will treat it as a "
        "speed knob and reuse stale-methodology results")


# --------------------------------------------------------------------------- #
# context-window fit: clamp before sending, and say why when it cannot fit
# --------------------------------------------------------------------------- #
def test_generous_budget_fits_a_short_prompt_untouched():
    base.set_run_knobs(max_model_len=262144)
    assert base.clamp_to_context([{"role": "user", "content": "hi"}], 65536) == (65536, None)


def test_long_prompt_shrinks_the_budget_instead_of_400ing():
    base.set_run_knobs(max_model_len=262144)
    got, why = base.clamp_to_context([{"role": "user", "content": "x" * 600_000}], 65536)
    assert why is None
    assert 0 < got < 65536
    assert got + 600_000 / 3.0 <= 262144


def test_prompt_larger_than_the_window_is_reported_not_retried():
    """mrcr sent six 1.3-4.2 MB prompts into a 262144 window: three silent retries each,
    recorded as `request_failed` with `error: None`."""
    base.set_run_knobs(max_model_len=262144)
    _, why = base.clamp_to_context([{"role": "user", "content": "x" * 4_252_036}], 65536)
    assert why and "context window" in why
    assert "1,417," in why, why      # ~1.4M tokens; not pinned to the exact digit


def test_clamp_is_a_noop_without_a_known_window():
    base._RUN_KNOBS.clear()
    assert base.clamp_to_context([{"role": "user", "content": "x" * 10_000}], 8192) == (8192, None)


def test_every_suite_gets_the_measured_floor():
    """One default, set from measurement: across 32 suites the largest HEALTHY completion
    was 8,020 tokens (hmmt), while every loop ran 10,035-56,201. 16384 is 2x the former and
    below the latter, so it cannot truncate a real answer but does cap a spiral."""
    assert base.DEFAULT_MIN_OUTPUT_TOKENS == 16384
    for suite in ("copilot_bench_swe", "aime", "codeforces", "putnam", "hmmt", "mmlu"):
        seen = _capture_payload(eval_name=suite, max_output_tokens=2048)
        assert seen.get("max_output_tokens") == 16384, suite


def test_exceptions_require_a_measurement_and_override_the_default():
    """The dict is the documented escape hatch; an entry beats the global floor."""
    from unittest import mock
    with mock.patch.dict(base.SUITE_MIN_OUTPUT_TOKENS, {"unit_test_suite": 65536}, clear=False):
        seen = _capture_payload(eval_name="unit_test_suite", max_output_tokens=2048)
    assert seen.get("max_output_tokens") == 65536


def test_no_exception_is_carried_without_evidence():
    """The old 65536 tier rested on a char count recorded before repetition detection
    existed, so it was never known whether that response was a patch or a loop."""
    assert base.SUITE_MIN_OUTPUT_TOKENS == {}, (
        "add an exception only when a run reports `genuinely_truncated` for that suite")


def test_global_default_was_raised():
    import inspect
    src = inspect.getsource(base._run_suite_async)
    assert "32768 if thinking else 16384" in src


# --------------------------------------------------------------------------- #
# request timeout must scale with the token budget, and be visible when it fires
# --------------------------------------------------------------------------- #
def test_timeout_scales_with_the_budget():
    """A fixed 1200 s was fine at 8192 tokens and drops the longest answers at 65536."""
    t8, t64 = base.request_timeout_s(8192), base.request_timeout_s(65536)
    assert t64 > t8 > 0
    assert t64 >= 65536 / base.MIN_DECODE_TOK_S
    assert base.request_timeout_s(None) >= base.REQUEST_TIMEOUT_FLOOR_S


def test_timeout_never_drops_below_the_floor():
    assert base.request_timeout_s(1) >= base.REQUEST_TIMEOUT_FLOOR_S


def test_a_timeout_is_classified_apart_from_other_failures():
    """Otherwise a dropped long answer is indistinguishable from a dead endpoint."""
    assert base.classify_reply(base.Reply(None, None, None, None, "timeout after 9392s")) \
        == "request_timeout"
    assert base.classify_reply(base.Reply(None, None, None, None, "OSError: refused")) \
        == "request_failed"
    assert base.classify_reply(base.Reply(None, None, None, None)) == "request_failed"


def test_reply_error_field_is_optional_for_existing_callers():
    r = base.Reply(text="x", tool_calls=None, finish_reason="stop")
    assert r.error is None and base.classify_reply(r) == "ok"


def test_timeouts_are_counted_and_warned_about(caplog):
    from unittest import mock

    async def _timing_out(**k):
        return base.Reply(None, None, None, None, "timeout after 9392s")

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _timing_out), \
         caplog.at_level("WARNING"):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1, samples=samples,
                                  eval_fn=lambda r, g: True)
    assert res["timed_out_requests"] == 2
    assert res["request_timeout_s"] > 0
    assert all(t["status"] == "TIMEOUT" for t in res["sample_traces"])
    assert all("timeout" in (t["error"] or "") for t in res["sample_traces"])
    assert any("timed out" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# generated-but-discarded output (gemma-4 <|tool_response|> + tool-call parser)
# --------------------------------------------------------------------------- #
def test_generated_but_empty_is_not_the_same_as_empty():
    """Measured live: completion_tokens=34, content=null, tool_calls=[], stop_reason=50.

    The model emitted a tool call and `<|tool_response>` (token 50, a stop token); vLLM's
    parser lifted the call out of `content` and, with no `tools` on the request, discarded
    it. 28 samples across bfcl_v3_live / mcp_atlas / skillsbench were scored as wrong
    answers when the model had in fact answered.
    """
    discarded = base.Reply(text=None if False else "", tool_calls=None, finish_reason="stop",
                           completion_tokens=34, stop_reason=50)
    assert base.classify_reply(discarded) == "output_discarded"
    truly_empty = base.Reply(text="", tool_calls=None, finish_reason="stop",
                             completion_tokens=0)
    assert base.classify_reply(truly_empty) == "empty"


def test_discarded_outputs_are_counted_and_explained(caplog):
    from unittest import mock

    async def _discarding(**k):
        return base.Reply("", None, "stop", None, None, 34, 50)

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _discarding), \
         caplog.at_level("WARNING"):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1, samples=samples,
                                  eval_fn=lambda r, g: True)
    assert res["discarded_outputs"] == 2
    assert res["sample_traces"][0]["completion_tokens"] == 34
    assert res["sample_traces"][0]["stop_reason"] == 50
    assert any("DISCARDED" in r.message for r in caplog.records)


@pytest.mark.parametrize("suite,loader", [
    ("bfcl_v3_live", "_load_bfcl_v3_live_samples"),
    ("mcp_atlas", "_load_mcp_atlas_samples"),
])
def test_tool_advertising_suites_declare_their_tools(suite, loader):
    """Advertising tools in the prompt but not on the request is what loses the answer."""
    import inspect, importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    src = inspect.getsource(getattr(mod, loader))
    assert '"tools"' in src, f"{suite} advertises tools in the prompt but never declares them"


def test_swe_lancer_wrong_model_hazard_is_documented():
    """It cannot be run without an adapter, and running it as shipped would benchmark
    GPT-4o - a real, plausible number attributed to the wrong model.

    This used to read docs/PENDING.md, which is an operator scratchpad that lives outside
    the repo. A published repo's test must not depend on an unpublished file, so the guard
    moved to the suite's own doc, which ships.
    """
    doc = (Path(__file__).resolve().parent.parent
           / "docs" / "evals" / "swe_lancer.md").read_text(encoding="utf-8")
    assert "CRITICAL" in doc, "the hazard must be flagged, not just described"
    assert "gpt-4o" in doc.lower(), "the wrong-model hazard must be stated"
    assert "OPENAI_BASE_URL" in doc, "keep the one detail that makes an adapter feasible"


# --------------------------------------------------------------------------- #
# looping vs truncation: "raise the budget" is wrong advice for a repeating model
# --------------------------------------------------------------------------- #
def test_repeating_response_at_the_cap_is_looping_not_truncated():
    """Measured 2026-08-15: gpqa_diamond produced 163,976 chars with a 20-gram repeated
    379 times; arc_agi 76,763 chars with 328. More budget buys more of the same."""
    looping_text = " ".join(["the same twenty word window repeated over and over again to "
                             "make a cycle that never terminates here"] * 30)
    r = base.Reply(text=looping_text, tool_calls=None, finish_reason="length")
    assert base.classify_reply(r) == "looping"


def test_a_genuinely_long_answer_at_the_cap_is_still_truncated():
    """Real code/patches don't repeat a 20-word window; they must keep the old label."""
    varied = " ".join(f"line{i} def f{i}(x): return x * {i} + compute_{i}(x)" for i in range(2000))
    r = base.Reply(text=varied, tool_calls=None, finish_reason="length")
    assert base.classify_reply(r) == "truncated"


def test_repetition_is_only_judged_when_the_cap_was_hit():
    """A short repetitive answer that finished on its own is a normal answer."""
    looping_text = " ".join(["the same twenty word window repeated over and over again to "
                             "make a cycle that never terminates here"] * 30)
    r = base.Reply(text=looping_text, tool_calls=None, finish_reason="stop")
    assert base.classify_reply(r) == "ok"


def test_looping_is_counted_and_advised_correctly(caplog):
    from unittest import mock
    looping_text = " ".join(["the same twenty word window repeated over and over again to "
                             "make a cycle that never terminates here"] * 30)

    async def _loops(**k):
        return base.Reply(looping_text, None, "length")

    samples = [([{"role": "user", "content": "q"}], "g", {})]
    with mock.patch.object(base, "_send_single_request", _loops), caplog.at_level("WARNING"):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1, samples=samples,
                                  eval_fn=lambda r, g: False)
    assert res["looping_responses"] == 1
    msgs = " ".join(r.message for r in caplog.records)
    assert "REPEATING THEMSELVES" in msgs
    assert "buys more of the same" in msgs, "must not tell the operator to raise the budget"


def test_repetition_run_is_recorded_even_when_not_classified():
    """Unbounded enumeration ("Step 375: Let's try k=-283") has a low n-gram count and
    cannot be classified; the raw figure must still reach the trace so it is visible."""
    from unittest import mock

    async def _long(**k):
        return base.Reply(" ".join(f"step {i} try k={i}" for i in range(3000)), None, "length")

    with mock.patch.object(base, "_send_single_request", _long):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1,
                                  samples=[([{"role": "user", "content": "q"}], "g", {})],
                                  eval_fn=lambda r, g: False)
    assert "repetition_run" in res["sample_traces"][0]


# --------------------------------------------------------------------------- #
# decoding penalties: available, off by default, and never silent
# --------------------------------------------------------------------------- #
def test_penalties_are_off_by_default():
    """Canonical protocol is greedy with no penalty; turning one on changes the metric."""
    import os
    from unittest import mock
    with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if not k.startswith("GBENCH_")}, clear=True):
        assert base.decoding_penalties() == {}


def test_penalties_are_read_from_the_environment():
    import os
    from unittest import mock
    with mock.patch.dict(os.environ, {"GBENCH_REPETITION_PENALTY": "1.15",
                                      "GBENCH_FREQUENCY_PENALTY": "0.2"}, clear=False):
        got = base.decoding_penalties()
    assert got["repetition_penalty"] == 1.15 and got["frequency_penalty"] == 0.2


def test_a_bad_penalty_value_is_ignored_not_crashed_on(caplog):
    import os
    from unittest import mock
    with mock.patch.dict(os.environ, {"GBENCH_REPETITION_PENALTY": "high"}, clear=False), \
         caplog.at_level("WARNING"):
        assert base.decoding_penalties() == {}
    assert any("not a number" in r.message for r in caplog.records)


def test_penalties_reach_the_request_and_are_recorded_and_warned(caplog):
    import os
    from unittest import mock
    seen = {}

    async def _fake_send(**k):
        seen.update(k.get("extra_payload") or {})
        return base.Reply("x", None, "stop")

    with mock.patch.dict(os.environ, {"GBENCH_REPETITION_PENALTY": "1.15"}, clear=False), \
         mock.patch.object(base, "_send_single_request", _fake_send), \
         caplog.at_level("WARNING"):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1,
                                  samples=[([{"role": "user", "content": "q"}], "g", {})],
                                  eval_fn=lambda r, g: True)
    assert seen.get("repetition_penalty") == 1.15
    assert res["decoding_penalties"] == {"repetition_penalty": 1.15}
    assert any("NOT comparable" in r.message for r in caplog.records), \
        "a run with penalties must say its numbers are not comparable"


def test_repetition_onset_locates_where_the_cycle_starts():
    """So a run can distinguish 'degenerates immediately' from 'cannot terminate a search'."""
    prefix = " ".join(f"reasoning step {i} about the problem at hand here now" for i in range(60))
    cycle = " ".join(["the identical twenty word window that keeps coming back again and "
                      "again without any variation at all here"] * 25)
    o = base.repetition_onset(prefix + " " + cycle)
    assert o is not None
    assert o["onset_word"] > 0 and 0 < o["onset_frac"] < 1
    assert o["period_words"] and o["repeat_count"] >= base.REPETITION_THRESHOLD
    assert "identical twenty word window" in o["cycle_preview"]


def test_repetition_onset_is_none_for_healthy_text():
    assert base.repetition_onset("short text") is None
    varied = " ".join(f"unique line {i} with distinct content {i*7}" for i in range(500))
    assert base.repetition_onset(varied) is None


def test_onset_reaches_the_trace_and_the_warning(caplog):
    from unittest import mock
    cycle = " ".join(["the identical twenty word window that keeps coming back again and "
                      "again without any variation at all here"] * 25)

    async def _loops(**k):
        return base.Reply(cycle, None, "length")

    with mock.patch.object(base, "_send_single_request", _loops), caplog.at_level("WARNING"):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1,
                                  samples=[([{"role": "user", "content": "q"}], "g", {})],
                                  eval_fn=lambda r, g: False)
    o = res["sample_traces"][0]["repetition_onset"]
    assert o and "cycle_preview" in o and "period_words" in o
    assert any("Onset at" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# the budget that is REPORTED must be the budget that was SENT
# --------------------------------------------------------------------------- #
def _run_one(eval_name, meta, **kw):
    from unittest import mock
    seen = {}

    async def _f(**k):
        seen["max_tokens"] = k.get("max_output_tokens")
        return base.Reply("x" * 100, None, "length")

    with mock.patch.object(base, "_send_single_request", _f):
        res = base.run_eval_suite(eval_name=eval_name, model_name="m", base_url="http://x",
                                  concurrency=1,
                                  samples=[([{"role": "user", "content": "q"}], "g", meta)],
                                  eval_fn=lambda a, b: False, **kw)
    return seen["max_tokens"], res


def test_a_sample_level_max_tokens_override_is_honoured_and_reported():
    """culer puts `max_tokens: 512` in every sample's meta. That lands in the payload after
    the resolved budget, so the server saw 512 while the run logged 16384 and reported
    "8/20 truncated" against a number that was never applied."""
    sent, res = _run_one("culer", {"category": "c", "max_tokens": 512})
    assert sent == 512
    assert res["sample_traces"][0]["max_tokens_sent"] == 512
    assert res["max_tokens_sent"] == [512]


def test_without_an_override_the_default_is_sent_and_reported():
    sent, res = _run_one("mmlu", {"category": "c"})
    assert sent == base.DEFAULT_MIN_OUTPUT_TOKENS
    assert res["max_tokens_sent"] == [base.DEFAULT_MIN_OUTPUT_TOKENS]


def test_truncation_warning_names_the_budget_actually_sent(caplog):
    with caplog.at_level("WARNING"):
        _run_one("culer", {"category": "c", "max_tokens": 512})
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "max_tokens=512" in msgs
    assert "max_tokens=16384" not in msgs, "reporting a budget that was never applied"


def test_suites_that_override_max_tokens_are_known():
    """Only these do it; a new one appearing silently changes what that suite measures.

    `ruler` sets canonical per-task generation caps (niah 128 / vt 30 / cwe 120 / fwe 50 /
    qa 32) from RULER's own constants - a deliberate, measured override.

    `complexfuncbench` drives its own multi-step tool-calling loop (it does not go through
    base.run_eval_suite) and caps each turn at 2048 tokens, the upstream ComplexFuncBench
    protocol (`GBENCH_COMPLEXFUNCBENCH_MAX_TOKENS`) - also deliberate and measured.

    REMOVED (WS10 audit): culer/mmmu_pro/screenspot used to set a small per-sample `max_tokens`
    (512/8192/128) for short answers, but a per-sample payload key OVERRIDES the sovereign
    --max-output-tokens and truncated the reasoning under --thinking (a false 0%/low). They now pass
    `kwargs.get("max_output_tokens")` and let base.py's thinking-aware default govern when unset."""
    import re
    from pathlib import Path as _P
    found = set()
    for path in SUITE_DIR.glob("*.py"):
        if path.stem == "base":   # base DEFINES the key; it does not override it
            continue
        src = path.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        if re.search(r'"max_tokens"\s*:', code):
            found.add(path.stem)
    assert found == {"ruler", "complexfuncbench"}, (
        f"per-sample max_tokens overrides changed: {sorted(found)}. Each one silently "
        f"replaces the sovereign --max-output-tokens, so it must be deliberate and measured "
        f"(only ruler's canonical per-task caps + complexfuncbench's upstream per-turn cap qualify).")


# --------------------------------------------------------------------------- #
# context estimation must not count a base64 image as text
# --------------------------------------------------------------------------- #
def _image_msg(nbytes=2_000_000, text="Transcribe this page."):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * nbytes}},
        {"type": "text", "text": text}]}]


def test_an_image_costs_soft_tokens_not_its_base64_length():
    """Counting the payload as text made a 2 MB page look like ~2.9M tokens and blocked the
    request as over-context. On 2026-08-17 that dropped 56 requests across 10 suites
    (omnidocbench 9/20, screenspot 9/20, mrcr 7/20)."""
    est = base._estimate_prompt_tokens(_image_msg())
    assert est < 2 * base.IMAGE_SOFT_TOKENS, f"image counted as {est} tokens"


def test_a_vision_prompt_is_not_rejected_as_over_context():
    base.set_run_knobs(max_model_len=262144)
    got, why = base.clamp_to_context(_image_msg(), 16384)
    assert why is None, why
    assert got == 16384


def test_a_genuinely_oversized_text_prompt_is_still_rejected():
    base.set_run_knobs(max_model_len=262144)
    _, why = base.clamp_to_context([{"role": "user", "content": "x" * 900_000}], 16384)
    assert why and "context window" in why


def test_plain_text_and_string_messages_still_estimate():
    assert base._estimate_prompt_tokens([{"role": "user", "content": "hello there"}]) > 0
    assert base._estimate_prompt_tokens("a plain string prompt") > 0
    assert base._estimate_prompt_tokens(None) is None
