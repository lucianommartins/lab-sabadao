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

"""Evaluation benchmarks runner for native gbench execution.

Executes aime, bfcl, codeforces, gpqa_diamond, gsm8k, lcb, mmlu_pro, mmmu_pro, mrcr, putnam, ruler, screenspot, and textvqa natively
within gbench using asynchronous HTTP requests and real-time tqdm logging.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from ..core.config import BenchmarkConfig
from ..core.models import ModelConfig, ModelFormat

logger = logging.getLogger(__name__)


def _resolve_remote_model_id(base_url: str, fallback_id: str) -> str:
    """Query /v1/models from server to obtain the registered model ID."""
    import urllib.request
    import json
    try:
        url = f"{base_url.rstrip('/')}/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        logger.warning(f"Could not query /v1/models from {base_url}: {e}")
    return fallback_id


def _fmt_accuracy_pct(v) -> str:
    """Format an accuracy as `NN.NN%`, degrading a present-but-None (or otherwise non-numeric)
    value to "-". A suite whose composite is undefined for the sampled data (e.g. omnidocbench when
    the sampled pages carry no scorable annotation) returns accuracy=None; formatting that with
    `{v:.2f}` raises "unsupported format string passed to NoneType.__format__". Because the per-eval
    result print runs at the MODEL level, an unguarded format aborts the ENTIRE model run and loses
    every suite after it (WS10: the fast stream died at omnidocbench, dropping 12 later suites)."""
    return f"{v:.2f}%" if isinstance(v, (int, float)) else "-"


BUILTIN_PILLARS = [
    (
        "1. GENERAL KNOWLEDGE & SCIENTIFIC REASONING",
        [
            "arc_agi",
            "causalbench",
            "cimemories",
            "cyberseceval",
            "cybergym",
            "frames",
            "gpqa",
            "gpqa_diamond",
            "healthbench",
            "humanitys_last_exam",
            "i18n_translate",
            "ifeval",
            "lab_bench",
            "livebench",
            "lmsys_noncoding_hard",
            "medxpertqa",
            "mmlu",
            "mmlu_pro",
            "mmlu_redux",
            "multilingual_mmlu",
            "simpleqa",
            "wmdp",
        ],
    ),
    (
        "2. MATHEMATICS & PROOFS",
        [
            "aime",
            "gsm8k",
            "hmmt",
            "imo_answer_bench",
            "amc_aime",
            "putnam",
            "putnam_formal",
        ],
    ),
    (
        "3. CODING & ALGORITHMIC DESIGN",
        [
            "acebench",
            "aider_polyglot",
            "bigcodebench",
            "codeforces",
            "copilot_bench_swe",
            "cruxeval",
            "lcb",
            "multi_swe_bench",
            "multipl_e",
            "ojbench",
            "scicode",
            "swe_bench_live",
            "swe_bench_multilingual",
            "swe_bench_pro",
        ],
    ),
    (
        "4. LONG CONTEXT & RETRIEVAL",
        [
            "aa_lcr",
            "beam_128k",
            "culer",
            "loft_x_arxiv",
            "mrcr",
            "ruler",
        ],
    ),
    (
        "5. TOOL USE & AGENTIC WORKFLOWS",
        [
            "agent_dojo",
            "api_bank",
            "bfcl",
            "bfcl_v3_live",
            "bfcl_v4_agentic",
            "browsecomp",
            "complexfuncbench",
            "deepsearch_qa",
            "gaia",
            "gaia2",
            "gdpval",
            "gorilla_apibench",
            "mcp_atlas",
            "mcp_bench",
            "nestful",
            "nexus_function_calling",
            "seal_tools",
            "skillsbench",
            "spider2",
            "t_eval",
            "tau2",
            "tau3",
            "terminal_bench",
            "toolbench",
            "wildclawbench",
        ],
    ),
    (
        "6. MULTIMODAL VISION & GROUNDING",
        [
            "bundled_detection",
            "chartqa",
            "charxiv",
            "coco_caption",
            "docvqa",
            "infographicvqa",
            "mmmu_pro",
            "omnidocbench",
            "screenspot",
            "semantic_keypoint",
            "textvqa",
        ],
    ),
]



def _run_with_timeout(eval_name, runner_fn, kwargs, configured, model_name):
    """Run one suite, optionally under a wall-clock budget. OFF unless --suite-timeout is set.

    Why it exists: one wedged suite stalls the whole sweep. In the 2026-08-15 run `tau3`
    sat ~45 minutes on a single conversation while its Gemini user-simulator returned
    PREFILL_QUEUE_OVERLOADED, and nothing behind it could start.

    Why it is OPT-IN rather than default-on - a total-runtime cap cannot tell "wedged"
    from "legitimately slow", and cutting a suite is destructive:

      * partial results are LOST. A suite 90% finished at the cap reports `timeout` and
        contributes nothing, which is worse than letting it run.
      * child processes are NOT reaped. The worker thread is abandoned, so the suite's
        Docker containers and harness subprocesses keep running and must be cleaned up by
        hand (terminal_bench leaves one container per task).
      * legitimately long harnesses (terminal_bench ~40 min, the SWE-bench family hours)
        would need per-suite budgets that go stale as hardware changes.

    So: unset -> no budget, exactly the previous behaviour. Set it when you want a sweep
    to make progress unattended and accept losing a suite to get that.
    """
    import concurrent.futures

    budget = configured or 0
    if budget <= 0:
        return runner_fn(**kwargs)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"suite-{eval_name}")
    future = executor.submit(runner_fn, **kwargs)
    try:
        return future.result(timeout=budget)
    except concurrent.futures.TimeoutError:
        logger.error("[%s] exceeded its %ds wall-clock budget; marking it timed out and "
                     "continuing. Raise --suite-timeout if this suite legitimately needs "
                     "longer.", eval_name, budget)
        return {
            "benchmark_type": "eval",
            "eval_name": eval_name,
            "model_name": model_name,
            "status": "timeout",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": (f"exceeded the {budget}s per-suite wall-clock budget and was "
                            f"abandoned so the sweep could continue "
                            f"(raise --suite-timeout to allow longer)"),
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


_SERVER_MAX_LEN: Dict[str, Any] = {}


def _server_max_model_len(base_url: str) -> Optional[int]:
    """The endpoint's advertised context window, queried once per URL."""
    if base_url in _SERVER_MAX_LEN:
        return _SERVER_MAX_LEN[base_url]
    value = None
    try:
        import json as _json
        import urllib.request
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as r:
            for entry in (_json.loads(r.read()).get("data") or []):
                if entry.get("max_model_len"):
                    value = int(entry["max_model_len"])
                    break
    except Exception as e:                                    # offline / non-vLLM endpoint
        logger.debug("could not read max_model_len from %s (%s)", base_url, e)
    _SERVER_MAX_LEN[base_url] = value
    return value


class EvalsBenchmarkRunner:
    """Runner for self-contained evaluation suites in gbench."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, model: ModelConfig, format: ModelFormat) -> List[Dict[str, Any]]:
        """Run requested evaluation suites for a model.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)

        Returns:
            List of standardized evaluation result dictionaries
        """
        if not self.config.evals and not getattr(self.config, "eval_custom_jsonl", None):
            return []

        # Auto-discover any custom eval plugins from configured directory / env
        from .eval_suites import discover_and_register_plugins
        plugin_dirs = getattr(self.config, "eval_plugins_dir", None)
        discover_and_register_plugins(plugin_dirs)

        evals_to_run = []
        if self.config.evals:
            for item in self.config.evals:
                for sub in str(item).split(","):
                    sub_clean = sub.strip()
                    if sub_clean and sub_clean not in evals_to_run:
                        evals_to_run.append(sub_clean)
        if getattr(self.config, "eval_custom_jsonl", None) and "custom_jsonl" not in evals_to_run:
            evals_to_run.append("custom_jsonl")

        builtin_all = []
        for _, suites in BUILTIN_PILLARS:
            for s in suites:
                if s not in builtin_all:
                    builtin_all.append(s)

        final_evals = []
        for e in evals_to_run:
            if e == "all":
                for b in builtin_all:
                    if b not in final_evals:
                        final_evals.append(b)
                # If custom plugins configured, 'all' includes them
                custom_discovered = discover_and_register_plugins(plugin_dirs)
                for p in custom_discovered.keys():
                    if p not in final_evals:
                        final_evals.append(p)
            elif e == "plugins":
                custom_discovered = discover_and_register_plugins(plugin_dirs)
                for p in custom_discovered.keys():
                    if p not in final_evals:
                        final_evals.append(p)
            else:
                if e not in final_evals:
                    final_evals.append(e)
        evals_to_run = final_evals

        num_threads = self.config.batch_sizes[0] if self.config.batch_sizes else 8
        results = []

        started_local_server = False
        serving_runner = None
        if not self.config.remote_endpoint:
            from .serving import ServingBenchmarkRunner
            serving_runner = ServingBenchmarkRunner(self.config)
            logger.info("No remote endpoint specified. Launching local vLLM server for evals...")
            if not serving_runner._start_server(model, format):
                logger.error("Failed to start local vLLM server for evals.")
                return [{"status": "error", "error": "Failed to start local server"}]
            started_local_server = True

        # Build base_url AFTER the server is up so a local endpoint reflects the port actually bound
        # (GBENCH_SERVER_PORT, via serving_runner.server_port), not a hardcoded 8000; every eval
        # (native, and containerized via --network host) receives this base_url. Remote wins.
        if self.config.remote_endpoint:
            base_url = self.config.remote_endpoint
        else:
            base_url = f"http://127.0.0.1:{serving_runner.server_port}/v1"
        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
            base_url = f"{base_url.rstrip('/')}/v1"

        target_model_id = _resolve_remote_model_id(base_url, model.hf_model_id or model.name)
        logger.info(f"Using server model ID for evals: '{target_model_id}'")

        try:
            for eval_name in evals_to_run:
                output_file = (
                    self.config.log_manager.results_dir
                    / f"eval_{eval_name}_{model.short_name}_{format.value}.json"
                )
                found_existing_file = None
                if output_file.exists():
                    found_existing_file = output_file
                elif getattr(self.config, "skip_existing", False):
                    base_dir = self.config.log_manager.base_dir
                    target_filename = f"eval_{eval_name}_{model.short_name}_{format.value}.json"
                    for prev_file in sorted(base_dir.rglob(target_filename), reverse=True):
                        if prev_file.is_file() and prev_file.stat().st_size > 0:
                            found_existing_file = prev_file
                            break

                if getattr(self.config, "skip_existing", False) and found_existing_file:
                    logger.info(
                        f"Skipping {eval_name.upper()} (found existing results in {found_existing_file})"
                    )
                    import json
                    try:
                        with open(found_existing_file, "r") as f:
                            cached_res = json.load(f)
                        results.append(cached_res)
                        if found_existing_file != output_file:
                            self.config.log_manager.save_result(output_file, cached_res)
                    except Exception:
                        pass
                    continue

                logger.info(f"\n{'='*60}\nRunning eval: {eval_name.upper()} against {target_model_id}\n{'='*60}")
                start_time = time.time()
                try:
                    res = self._run_single_eval(
                        eval_name=eval_name,
                        model_name=target_model_id,
                        base_url=base_url,
                        num_threads=num_threads,
                    )
                except Exception as e:
                    duration_s = time.time() - start_time
                    # A plugin whose private dataset the operator has not provided is a
                    # *prerequisite* gap, not a failure of the model or the harness: emit a clean
                    # skip with fetch instructions instead of an error/0% row (audit CC1/CC6).
                    # (Plugins are optional/user-supplied; this is not a built-in canonical suite.)
                    if type(e).__name__ == "DatasetUnavailableError":
                        from .eval_suites.swebench_common import skipped_result
                        skip_res = skipped_result(eval_name, model.short_name, str(e),
                                                  "docs/evals/plugins.md")
                        skip_res["duration_s"] = round(duration_s, 2)
                        skip_res["format"] = format.value
                        results.append(skip_res)
                        try:
                            self.config.log_manager.save_result(output_file, skip_res)
                        except Exception:
                            pass
                        continue
                    # Isolate per-suite failures: one raising suite must never abort the
                    # rest of the sweep. Record it as an error result and continue.
                    logger.error(f"❌ Eval '{eval_name}' failed: {e}", exc_info=True)
                    from .eval_suites.canonical_sync import canonical_sync_for
                    err_res = {
                        "benchmark_type": "eval",
                        "eval_name": eval_name,
                        "model_name": model.name,
                        "model_short": model.short_name,
                        "format": format.value,
                        "status": "error",
                        "error": str(e),
                        "duration_s": round(duration_s, 2),
                        "total_questions": 0,
                        "correct_answers": 0,
                        "accuracy": 0.0,
                        "canonical_sync": canonical_sync_for(eval_name),
                    }
                    results.append(err_res)
                    try:
                        self.config.log_manager.save_result(output_file, err_res)
                    except Exception:
                        pass
                    continue
                duration_s = time.time() - start_time
                res["duration_s"] = round(duration_s, 2)
                res["benchmark_type"] = "eval"
                res["eval_name"] = eval_name
                res["model_name"] = model.name
                res["model_short"] = model.short_name
                res["format"] = format.value
                from .eval_suites.canonical_sync import canonical_sync_for
                res["canonical_sync"] = canonical_sync_for(eval_name)
                results.append(res)

                self.config.log_manager.save_result(output_file, res)
                logger.info(
                    f"Saved {eval_name.upper()} evaluation traces to {output_file}"
                )

                acc = res.get("accuracy", 0.0)
                tot = res.get("total_questions", 0)
                corr = res.get("correct_answers", 0)
                status = res.get("status", "ok")
                cat_acc = res.get("category_accuracy", {})
                cat_lines = ""
                if 1 < len(cat_acc) <= 25:
                    cat_lines = "\n--- By Category ---\n" + "\n".join(
                        f"  {c:<22} {s.get('correct', 0)}/{s.get('total', 0)} "
                        f"({_fmt_accuracy_pct(s.get('accuracy'))})"
                        for c, s in sorted(cat_acc.items())
                    )
                logger.info(
                    f"\n============ {eval_name.upper()} Eval Result ============\n"
                    f"Model:                   {model.short_name} ({format.value})\n"
                    f"Status:                  {status}\n"
                    f"Thinking enabled:        {res.get('thinking', False)}\n"
                    f"Duration (s):            {res['duration_s']}\n"
                    f"Total questions:         {tot}\n"
                    f"Correct answers:         {corr}\n"
                    f"Accuracy:                {_fmt_accuracy_pct(acc)}"
                    f"{cat_lines}\n"
                    f"=================================================="
                )
        finally:
            if started_local_server and serving_runner:
                logger.info("Cleaning up local vLLM server after evals...")
                serving_runner._cleanup_server()

        return results

    def _run_single_eval(
        self,
        eval_name: str,
        model_name: str,
        base_url: str,
        num_threads: int,
    ) -> Dict[str, Any]:
        """Run single eval suite natively via internal runners."""
        from .eval_suites import SUITES

        if eval_name not in SUITES:
            logger.error(f"Unsupported eval suite: {eval_name}")
            return {"status": "error", "error": f"Unsupported eval suite: {eval_name}"}

        # Suites that execute model-written code require bubblewrap isolation by default.
        # Missing/blocked isolation is a real EXTERNAL prerequisite gap, so it HARD-ERRORS
        # (infra_required) rather than skipping or running unsandboxed - the no-skip policy.
        # Set GBENCH_SANDBOX=none to opt into unsandboxed execution, or =bwrap to fail at the
        # exec call. Keep this set in sync with the suites that call sandbox.run_sandboxed;
        # multipl_e is deliberately NOT here - it executes via Docker (not bubblewrap) and
        # carries its own Docker infra_required.
        CODE_EXEC_EVALS = {"codeforces", "lcb", "scicode"}
        if eval_name in CODE_EXEC_EVALS:
            from .eval_suites.sandbox import sandbox_skip_reason
            from .eval_suites.swebench_common import infra_required
            _blocked = sandbox_skip_reason()
            if _blocked:
                raise infra_required(eval_name, _blocked, f"docs/evals/{eval_name}.md")

        # For container/sandboxed evaluations (Docker via Harbor or Lean), use --sandboxes override if specified
        # Evals whose concurrency is bounded by --sandboxes. NOTE: this only bounds
        # parallelism; true OS-level isolation (nsjail/container) is a separate follow-up.
        SANDBOX_EVALS = {"terminal_bench", "putnam_formal", "swe_bench_live", "aider_polyglot",
                         "copilot_bench_swe", "swe_bench_multilingual", "bigcodebench", "multipl_e",
                         "spider2", "multi_swe_bench", "ojbench", "swe_bench_pro",
                         "swe_lancer", "tau2", "tau3", "skillsbench", "wildclawbench",
                         "codeforces", "lcb", "scicode"}
        concurrency = num_threads
        if eval_name in SANDBOX_EVALS and getattr(self.config, "sandboxes", None) is not None:
            concurrency = self.config.sandboxes

        runner_fn = SUITES[eval_name]
        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "base_url": base_url,
            "concurrency": concurrency,
            "enable_thinking": self.config.eval_thinking,
            "max_output_tokens": self.config.eval_max_output_tokens,
            "eval_max_soft_tokens": self.config.eval_max_soft_tokens,
            "eval_n_shot": self.config.eval_n_shot,
            "eval_categories": self.config.eval_categories,
            "limit": self.config.eval_limit,
            "tokenizer": getattr(self.config, "tokenizer", None),  # ruler needs the target tokenizer
            "temperature": getattr(self.config, "temperature", None),
            "attempt_count": getattr(self.config, "attempt_count", 1),
            # Expose --sandboxes to every runner (not just the hardcoded SANDBOX_EVALS names) so
            # containerized *plugin* evals can honor it without their name being baked into core.
            "sandboxes": getattr(self.config, "sandboxes", None),
        }
        if eval_name == "custom_jsonl" and getattr(self.config, "eval_custom_jsonl", None):
            kwargs["jsonl_path"] = self.config.eval_custom_jsonl

        # Run-level generation knobs resolved centrally: a suite that does not forward
        # `temperature`/`max_output_tokens` no longer silently drops them (audit RC-2).
        from .eval_suites.base import set_run_knobs
        set_run_knobs(temperature=kwargs.get("temperature"),
                      attempt_count=kwargs.get("attempt_count"),
                      max_output_tokens=kwargs.get("max_output_tokens"),
                      # Lets base clamp `prompt + max_tokens` to the window BEFORE sending,
                      # instead of paying a 400-and-retry (or failing silently when the
                      # prompt alone is over the limit).
                      max_model_len=_server_max_model_len(base_url))

        return _run_with_timeout(eval_name, runner_fn, kwargs,
                                 getattr(self.config, "suite_timeout", None), model_name)
