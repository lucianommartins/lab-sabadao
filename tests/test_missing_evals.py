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

"""Unit tests for the 4 sandbox/agentic evaluation suites with prerequisite checks."""

import pytest
from unittest.mock import patch, MagicMock

from gbench.runners.eval_suites.terminal_bench import (
    check_terminal_bench_prerequisites,
    run_terminal_bench,
)
from gbench.runners.eval_suites.ui_control_osworld import (
    check_ui_control_osworld_prerequisites,
    run_ui_control_osworld,
)
from gbench.runners.eval_suites.copilot_bench_swe import (
    check_copilot_bench_swe_prerequisites,
    run_copilot_bench_swe,
    _load_copilot_bench_swe_dataset,
)
from gbench.runners.eval_suites import SUITES


def test_suites_registration():
    """Verify runnable suites are registered, and roadmap-only suites are NOT."""
    assert "terminal_bench" in SUITES
    assert "copilot_bench_swe" in SUITES
    # DEREGISTERED / roadmap-only (see docs/evals_roadmap.md): not runnable, so they must NOT appear
    # in the registry or `--evals all` (a suite that cannot run must not be registered):
    #   lmarena_web_agent -> canonical WebArena (never built)
    #   ui_control_osworld -> needs /dev/kvm; not available on accelerator hosts
    #   swe_lancer -> gbench-side fixed but blocked by upstream Expensify/App image drift
    for name in ("lmarena_web_agent", "ui_control_osworld", "swe_lancer"):
        assert name not in SUITES, f"{name} is roadmap-only and must not be registered"


def test_terminal_bench_hard_errors_on_missing_prereqs():
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    with patch("gbench.runners.eval_suites.terminal_bench.check_terminal_bench_prerequisites", return_value=(False, "Harbor CLI not found")):
        with pytest.raises(RuntimeError, match="docs/evals/terminal_bench.md"):
            run_terminal_bench(model_name="gemma-4-E4B-it", base_url="http://localhost:8000/v1")


def test_terminal_bench_zero_trials_hard_errors():
    # No-partial policy: a Harbor run that completes but yields zero parseable trials measured
    # nothing, so it must hard-error rather than publish a fabricated 0.0 row. (This mock produces
    # no result.json under the jobs dir, so total==0.)
    with patch("gbench.runners.eval_suites.terminal_bench.check_terminal_bench_prerequisites", return_value=(True, "")), \
         patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ["Running trial 1...\n", "Completed trial 1.\n"]
        mock_proc.wait.return_value = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        with pytest.raises(RuntimeError, match="docs/evals/terminal_bench.md"):
            run_terminal_bench(model_name="gemma-4-E4B-it", base_url="http://localhost:8000/v1", limit=2)


def test_ui_control_osworld_hard_errors_on_missing_prereqs():
    # No-skip policy: an un-provisioned canonical OSWorld harness must hard-error, not skip/fake.
    with patch("gbench.runners.eval_suites.ui_control_osworld.check_ui_control_osworld_prerequisites",
               return_value=(False, "/dev/kvm not present")):
        with pytest.raises(RuntimeError, match="docs/evals/ui_control_osworld.md"):
            run_ui_control_osworld(model_name="gemma-4-E4B-it", base_url="http://localhost:8000/v1")


def test_copilot_bench_swe_hard_errors_on_missing_prereqs():
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    with patch("gbench.runners.eval_suites.copilot_bench_swe.check_copilot_bench_swe_prerequisites", return_value=(False, "Docker not found")):
        with pytest.raises(RuntimeError, match="docs/evals/copilot_bench_swe.md"):
            run_copilot_bench_swe(model_name="gemma-4-E4B-it", base_url="http://localhost:8000/v1")


def test_sample_loaders_and_eval_matchers():
    # OSWorld + WebArena: the single-shot keyword loaders/matchers were removed. OSWorld is now the
    # canonical execution harness (see tests/test_ui_control_osworld.py); WebArena (formerly
    # lmarena_web_agent) is DEREGISTERED, roadmap-only pending its canonical harness (docs/evals_roadmap.md).

    # SWE-bench Verified (now execution-scored): check the loader + shared patch extractor
    from gbench.runners.eval_suites.swebench_common import extract_patch
    with patch("datasets.load_dataset", return_value=[{"repo": "astropy/astropy", "instance_id": "astropy__astropy-1", "problem_statement": "Fix the parser bug."}]):
        swe_samples = _load_copilot_bench_swe_dataset(limit=1)
    assert len(swe_samples) == 1
    assert swe_samples[0][2]["instance_id"] == "astropy__astropy-1"
    assert "diff --git" in extract_patch("see:\n```diff\ndiff --git a/x b/x\n@@\n```")
    assert extract_patch("no patch here") == ""


def test_sandboxes_cli_argument_and_concurrency_override():
    from gbench.cli import create_parser
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    parser = create_parser()
    args = parser.parse_args(["--evals-only", "--evals", "terminal_bench", "--batch-sizes", "128", "--sandboxes", "64"])
    assert args.sandboxes == 64
    assert args.batch_sizes == [128]

    config = BenchmarkConfig()
    config.sandboxes = 64
    config.batch_sizes = [128]

    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES", {"terminal_bench": MagicMock(return_value={"status": "success"})}) as mock_suites:
        res = runner._run_single_eval("terminal_bench", "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=128)
        mock_suites["terminal_bench"].assert_called_once()
        call_kwargs = mock_suites["terminal_bench"].call_args[1]
        assert call_kwargs["concurrency"] == 64


def test_one_failing_suite_does_not_abort_the_sweep():
    """A suite that raises must be recorded as status='error' and the sweep continue."""
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig
    from gbench.core.models import ModelFormat

    config = BenchmarkConfig()
    config.evals = ["good1", "boom", "good2"]
    config.remote_endpoint = "http://127.0.0.1:8000/v1"
    config.batch_sizes = [4]
    config.skip_existing = False
    config.initialize()

    runner = EvalsBenchmarkRunner(config)
    model = MagicMock()
    model.name = "m"
    model.short_name = "m"
    model.hf_model_id = "m"

    def fake_single(eval_name, model_name, base_url, num_threads):
        if eval_name == "boom":
            raise RuntimeError("simulated suite crash")
        return {"status": "success", "total_questions": 2, "correct_answers": 2,
                "accuracy": 100.0, "category_accuracy": {}, "thinking": False}

    with patch.object(EvalsBenchmarkRunner, "_run_single_eval", side_effect=fake_single), \
         patch("gbench.runners.eval_suites.discover_and_register_plugins", return_value={}), \
         patch("gbench.runners.evals._resolve_remote_model_id", return_value="m"):
        results = runner.run(model, ModelFormat.HF)

    by_name = {r["eval_name"]: r["status"] for r in results}
    assert by_name == {"good1": "success", "boom": "error", "good2": "success"}
    assert len(results) == 3  # the failing suite did not swallow the others


def test_temperature_cli_argument_and_runner_propagation():
    from gbench.cli import create_parser
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    parser = create_parser()
    args = parser.parse_args(["--evals-only", "--evals", "terminal_bench", "--temperature", "0.7"])
    assert args.temperature == 0.7

    config = BenchmarkConfig()
    config.temperature = 0.7

    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES", {"terminal_bench": MagicMock(return_value={"status": "success"})}) as mock_suites:
        res = runner._run_single_eval("terminal_bench", "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=4)
        mock_suites["terminal_bench"].assert_called_once()
        call_kwargs = mock_suites["terminal_bench"].call_args[1]
        assert call_kwargs["temperature"] == 0.7
