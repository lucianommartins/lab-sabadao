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

"""Native LiveCodeBench algorithmic coding evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_LCB_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from .base import run_eval_suite, suite_env
from .sampling import limit_dataset, stratified_sample
from .sandbox import run_sandboxed, sandbox_skip_reason
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

#: Which LiveCodeBench release window to evaluate. The point of the benchmark is to test
#: on problems published AFTER the model was trained, so this SHOULD be set per model;
#: `release_v1` (2023-05..2024-03) is contaminated for anything current and is not the
#: default for that reason. Window tags like `v5_v6` select only newly-added problems.
_LCB_VERSION = os.environ.get("GBENCH_LCB_VERSION", "release_v6")

#: Optional hard floor on `contest_date` (ISO, e.g. "2024-10"). The release tags are
#: cumulative - `release_v6` still contains the 2023 problems from v1 - so a tag alone
#: cannot express "only problems published after this model was trained". Setting this is
#: the precise contamination control; a window tag like `v5_v6` is the coarse one.
_LCB_MIN_DATE = os.environ.get("GBENCH_LCB_MIN_DATE", "").strip()


#: `version_tag` -> shard files, copied from ALLOWED_FILES in `code_generation_lite.py`
#: (scaleapi-style loading script) at scaleapi... see LiveCodeBench/code_generation_lite.
#: Each release is cumulative; a WINDOW tag like `v4_v5` selects only the shards added
#: between two releases, which is how you evaluate strictly past a model's cutoff.
_LCB_RELEASE_FILES = {f"release_v{i}": [f"test{j}.jsonl" if j > 1 else "test.jsonl"
                                        for j in range(1, i + 1)]
                      for i in range(1, 7)}
_LCB_RELEASE_FILES["release_latest"] = _LCB_RELEASE_FILES["release_v6"]
for _i in range(1, 7):
    _LCB_RELEASE_FILES[f"v{_i}"] = [f"test{_i}.jsonl" if _i > 1 else "test.jsonl"]
for _a in range(1, 7):
    for _b in range(_a + 1, 7):
        _LCB_RELEASE_FILES[f"v{_a}_v{_b}"] = [
            f"test{_j}.jsonl" if _j > 1 else "test.jsonl" for _j in range(_a, _b + 1)]


def _load_lcb_lite(version_tag: str):
    """Rows for a LiveCodeBench release window, read straight from the repo's JSONL shards.

    `load_dataset(..., version_tag=...)` went through the repo's loading script, and
    `datasets` 5.x refuses to execute dataset scripts at all ("Dataset scripts are no longer
    supported"). The shard-per-release layout is stable and documented on the card, so the
    mapping is replicated here and the files are fetched directly. This keeps the one
    property that matters: being able to name a window past the model's training cutoff.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    files = _LCB_RELEASE_FILES.get(version_tag)
    if not files:
        raise RuntimeError(
            f"lcb: unknown GBENCH_LCB_VERSION {version_tag!r}. "
            f"Known: {sorted(_LCB_RELEASE_FILES)}")
    paths = [hf_hub_download("livecodebench/code_generation_lite", f,
                             repo_type="dataset") for f in files]
    return load_dataset("json", data_files=paths, split="train")


def _decode_test_cases(raw: Any, field: str, qid: Any) -> List[Dict[str, Any]]:
    """Test cases from either LCB encoding, or raise.

    `code_generation` ships plain JSON; `code_generation_lite` ships
    base64 -> zlib -> pickle. The previous `except: tc_data = []` silently produced a
    problem with no private tests, which does not fail the run - it makes the problem
    EASIER and inflates the score, invisibly. An undecodable payload is a broken input, so
    it stops the run instead.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    text = str(raw)
    try:
        return [t for t in json.loads(text) if isinstance(t, dict)]
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        import base64
        import pickle
        import zlib
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(
            text.encode("utf-8")))))
        return [t for t in decoded if isinstance(t, dict)]
    except Exception as e:                                              # noqa: BLE001
        raise RuntimeError(
            f"lcb: could not decode {field} for question {qid!r} as JSON or as "
            f"base64+zlib+pickle ({e}). Refusing to continue: an empty test list would "
            f"silently make this problem free rather than fail it."
        ) from e


def _load_lcb_samples(enable_thinking: bool = False, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load LiveCodeBench samples covering Code Generation (public + private tests), Code Execution, and Test Generation."""
    import json
    from datasets import load_dataset

    samples = []

    # 1. Code Generation (with both public and private test cases)
    # `code_generation_lite`, NOT `code_generation`. The latter is the frozen v1 dump:
    # 400 problems dated 2023-05..2024-03, every one of them inside any current model's
    # training window - which defeats the single thing LiveCodeBench exists to do. "lite"
    # is not a reduced benchmark: verified 2026-08-20 on problem 1873_A, both ship 1 public
    # + 4 private cases; lite just stores `private_test_cases` base64+zlib+pickle'd, which
    # is why it is 1.25GB against 9.4GB. Only lite carries v2..v6 and accepts window tags
    # (`v4_v5` = problems added between those releases), so only lite can be pointed past a
    # model's cutoff. Set GBENCH_LCB_VERSION to the window you want.
    ds_gen = _load_lcb_lite(_LCB_VERSION)
    if _LCB_MIN_DATE:
        before = len(ds_gen)
        ds_gen = [r for r in ds_gen if str(r.get("contest_date") or "") >= _LCB_MIN_DATE]
        logger.info("lcb: contest_date >= %s keeps %d of %d problems.",
                    _LCB_MIN_DATE, len(ds_gen), before)
        if not ds_gen:
            raise RuntimeError(
                f"lcb: GBENCH_LCB_MIN_DATE={_LCB_MIN_DATE!r} excluded every problem in "
                f"{_LCB_VERSION!r}. Pick a later release tag or an earlier date.")
    # Stratified by difficulty, not a contiguous head (audit RC-1): the split is
    # ordered, so a head skewed the easy/medium/hard mix.
    ds_gen = limit_dataset(ds_gen, limit, "difficulty", seed="lcb")
    raw_gen = list(ds_gen)
    for item in raw_gen:
        title = item.get("question_title", "")
        content = item.get("question_content") or item.get("question", "")
        starter = item.get("starter_code", "")
        difficulty = item.get("difficulty", "coding")

        # Collect both public and private test cases
        all_tests = []
        for test_field in ["public_test_cases", "private_test_cases"]:
            all_tests.extend(_decode_test_cases(item.get(test_field), test_field,
                                                item.get("question_id")))

        prompt = f"Problem: {title}\n\n{content}\n\n"
        if starter and starter.strip():
            prompt += f"Starter code:\n```python\n{starter}\n```\n\n"
        prompt += (
            "Write a complete Python 3 solution to solve this problem. "
            "Your solution should read from standard input if required, or implement the starter code function. "
            "Provide your solution code in a single markdown ```python ... ``` code block."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, {"task": "generation", "tests": all_tests},
                        {"category": f"code_gen_{difficulty}",
                         "contest_date": str(item.get("contest_date") or "")}))
    logger.info(f"Loaded {len(raw_gen)} LCB Code Generation samples (public + private tests).")

    # If limit reached for overall run, stop early
    if limit is not None and len(samples) >= limit:
        return samples[:limit]

    # 2. Code Execution
    try:
        ds_exec = load_dataset("livecodebench/execution", split="test")
        if limit is not None:
            exec_limit = limit - len(samples)
            # Seeded subset, not `.select(range(n))` - a contiguous head of an ordered
            # split is the audit RC-1 bias the generation branch already avoids.
            ds_exec = (stratified_sample(list(ds_exec), exec_limit, None, seed="lcb_exec")
                       if exec_limit > 0 else [])
        for item in list(ds_exec):
            code = item.get("code", "")
            call_input = item.get("input", "")
            expected_output = str(item.get("output", "")).strip()

            prompt = (
                "Predict the exact return value of executing the following Python code:\n\n"
                f"```python\n{code}\n```\n\n"
                f"Execution call: `{call_input}`\n\n"
                "State your final predicted return value on the last line in the format: 'Final Output: <output>'."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, {"task": "execution", "expected": expected_output}, {"category": "code_execution"}))
        logger.info("Loaded LCB Code Execution samples.")
    except Exception as e:
        logger.warning(f"Could not load LCB 'execution' ({e}).")

    if limit is not None and len(samples) >= limit:
        return samples[:limit]

    # 3. Test Generation - EXCLUDED by default: faithful scoring needs the model's generated
    # tests to be run against a reference solution (not available in this harness), and the
    # cheap length proxy fakes ~100%. Set GBENCH_LCB_INCLUDE_TEST_GEN=1 (legacy alias
    # LCB_INCLUDE_TEST_GEN=1 still works) to load it anyway (it will not be credited by the
    # scorer; see _verify_single_lcb_sample).
    if suite_env("GBENCH_LCB_INCLUDE_TEST_GEN", "LCB_INCLUDE_TEST_GEN") != "1":
        logger.info("LCB: skipping test_generation (needs reference-solution execution to "
                    "score faithfully; set GBENCH_LCB_INCLUDE_TEST_GEN=1 to include).")
        return samples[:limit] if (limit is not None) else samples
    try:
        ds_test = load_dataset("livecodebench/test_generation", split="test")
        if limit is not None:
            test_limit = limit - len(samples)
            if test_limit > 0:
                ds_test = stratified_sample(list(ds_test), test_limit, None,
                                            seed="lcb_testgen")
            else:
                ds_test = []
        for item in list(ds_test):
            title = item.get("question_title", "")
            content = item.get("question_content", "")
            fn_name = item.get("function_name", "")
            starter = item.get("starter_code", "")
            difficulty = item.get("difficulty", "medium")
            gold_test = item.get("test", "")

            prompt = (
                f"Problem: {title}\n\n{content}\n\n"
                f"Function Name: `{fn_name}`\n"
                f"Starter code:\n```python\n{starter}\n```\n\n"
                "Generate valid inputs and expected outputs that test this function correctly. "
                "Provide your test case in the format: 'Test: input=<input>, output=<output>'."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, {"task": "test_gen", "expected": gold_test}, {"category": f"test_gen_{difficulty}"}))
        logger.info("Loaded LCB Test Generation samples.")
    except Exception as e:
        logger.warning(f"Could not load LCB 'test_generation' ({e}).")

    return samples


#: Injected into the sandboxed test script for `functional` cases. Kept as a string so the
#: whole thing runs inside the sandbox with the submitted solution, and so the entry point
#: is resolved from the code the model actually wrote rather than from dataset fields that
#: do not exist (see the comment at the call site).
_FUNCTIONAL_DRIVER = '''
def _gbench_entry():
    """The callable under test: a Solution method if present, else a module-level function.

    LiveCodeBench is LeetCode-derived, so the overwhelmingly common shape is
    `class Solution: def someName(self, ...)`. Fall back to a lone top-level function for
    the handful of problems written that way.
    """
    g = dict(globals())
    sol = g.get("Solution")
    if sol is not None:
        meths = [n for n in vars(sol)
                 if not n.startswith("_") and callable(getattr(sol, n))]
        if len(meths) == 1:
            return getattr(sol(), meths[0])
        if meths:                      # prefer the longest name; helpers are usually short
            return getattr(sol(), sorted(meths, key=len)[-1])
    fns = [v for k, v in g.items()
           if callable(v) and getattr(v, "__module__", None) == "__main__"
           and not k.startswith("_")]
    if len(fns) == 1:
        return fns[0]
    raise RuntimeError("no unambiguous entry point")


def _gbench_parse_args(raw):
    """`input` is one JSON literal per line, one per positional argument."""
    args = []
    for line in str(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except Exception:
            args.append(line)
    return args


def _gbench_eq(got, want_raw):
    try:
        want = json.loads(want_raw)
    except Exception:
        want = want_raw
    if got == want:
        return True
    # LeetCode writes true/false; Python prints True/False.
    return str(got).strip().lower() == str(want).strip().lower()


def _gbench_check(raw_in, raw_out):
    try:
        fn = _gbench_entry()
        return 0 if _gbench_eq(fn(*_gbench_parse_args(raw_in)), raw_out) else 1
    except Exception:
        return 1


def _gbench_check_named(fn_name, args, raw_out):
    """Payloads that DO carry an explicit fn_name/args pair.

    Still resolve through Solution() when the submission is a class, since a LeetCode-style
    answer defines the method there rather than at module level.
    """
    try:
        g = dict(globals())
        fn = g.get(fn_name)
        if fn is None:
            sol = g.get("Solution")
            fn = getattr(sol(), fn_name) if sol is not None else None
        if fn is None:
            return 1
        args = args if isinstance(args, list) else [args]
        return 0 if _gbench_eq(fn(*args), raw_out) else 1
    except Exception:
        return 1
'''


def _verify_single_lcb_sample(resp_text: str, gold_payload: Any) -> bool:
    """Verify single LCB sample in an isolated process with strict timeout."""
    import subprocess
    import sys
    import json

    if not resp_text:
        return False

    if isinstance(gold_payload, list):
        gold_payload = {"task": "generation", "tests": gold_payload}
    elif not isinstance(gold_payload, dict):
        # An unparseable payload means nothing was executed against the submission.
        # Returning True auto-passed those rows.
        return False

    task = gold_payload.get("task", "generation")

    # Task 1: Code Generation
    if task == "generation":
        gold_tests = gold_payload.get("tests", [])
        code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", resp_text, re.DOTALL)
        code = code_match.group(1) if code_match else resp_text

        if not gold_tests:
            # A problem whose tests could not be decoded is unverified, not solved.
            return False

        # Test string assertions
        if isinstance(gold_tests[0], str):
            test_script = f"{code}\n" + "\n".join(gold_tests)
            try:
                # Program on STDIN (`python -`), NOT a `-c <script>` argument: an LCB test harness
                # can exceed Linux MAX_ARG_STRLEN (128KB) -> OSError('Argument list too long'), which
                # the except below would silently score as a wrong answer (a false negative).
                r = run_sandboxed(
                    [sys.executable, "-"],
                    input=test_script,
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                return r.returncode == 0
            except (subprocess.TimeoutExpired, Exception):
                return False

        for tc in gold_tests:
            if not isinstance(tc, dict):
                # Skipping a case let a problem pass on the cases that DID run. Fail it:
                # an unrunnable test is not a passed test. (0 occurrences on 2026-08-20.)
                return False
            test_type = tc.get("testtype", "stdin")
            if test_type not in ("stdin", "functional"):
                return False
            inp = str(tc.get("input") or "")
            expected = str(tc.get("output") or "").strip()

            if test_type == "stdin":
                try:
                    r = run_sandboxed(
                        [sys.executable, "-c", code],
                        input=inp,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if r.returncode != 0 or r.stdout.strip() != expected:
                        return False
                except (subprocess.TimeoutExpired, Exception):
                    return False
            elif test_type == "functional":
                # LCB functional cases carry ONLY {input, output, testtype} - there is no
                # `fn_name` and no `args` field, at test level or problem level. The old
                # code read both, got None and [], and emitted `res = None()`, so every
                # functional problem failed with `TypeError: 'NoneType' object is not
                # callable` regardless of the model. Measured 2026-08-18: all 308 test
                # cases across the 3 sampled problems are `functional`, and a verifiably
                # correct solution scored 0.
                #
                # So: derive the entry point from the submitted code, parse `input` (a
                # newline-separated list of JSON literals, one per argument), and compare
                # against `output` in a JSON-aware way - LeetCode writes `true`/`false`
                # where Python prints `True`/`False`.
                # Both payload shapes exist. Some carry an explicit `fn_name`/`args` pair;
                # the real LiveCodeBench export carries neither, only `input`/`output`.
                fn_name = tc.get("fn_name")
                args = tc.get("args")
                if fn_name and args is not None:
                    test_script = (
                        f"{code}\n\n"
                        f"import sys, json\n"
                        f"{_FUNCTIONAL_DRIVER}\n"
                        f"sys.exit(_gbench_check_named({fn_name!r}, {args!r}, {expected!r}))\n"
                    )
                else:
                    test_script = (
                        f"{code}\n\n"
                        f"import sys, json\n"
                        f"{_FUNCTIONAL_DRIVER}\n"
                        f"sys.exit(_gbench_check({inp!r}, {expected!r}))\n"
                    )
                try:
                    # Program on STDIN (`python -`): functional inputs can be >100KB (embedded via
                    # {inp!r}/{args!r} above); as a `-c` argument that hits MAX_ARG_STRLEN and the
                    # OSError would be silently scored wrong. Same fix as the string-assert path.
                    r = run_sandboxed(
                        [sys.executable, "-"],
                        input=test_script,
                        text=True,
                        capture_output=True,
                        timeout=5,
                    )
                    if r.returncode != 0:
                        return False
                except (subprocess.TimeoutExpired, Exception):
                    return False

        return True

    # Task 2: Code Execution - the prompt asks for 'Final Output: <output>' on the last
    # line. Extract the model's *stated* answer and compare to the gold, instead of a bare
    # substring test (`expected in resp`), which passed whenever the value appeared anywhere
    # in the model's reasoning and heavily inflated this category.
    elif task == "execution":
        expected = str(gold_payload.get("expected", "")).strip()
        if not expected:
            return False
        resp = resp_text.strip()

        def _norm(s: str) -> str:
            s = str(s).strip()
            if s.startswith("```") and s.endswith("```"):
                s = s.strip("`").strip()
            s = s.strip().strip("`").strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
                s = s[1:-1]
            return s.strip()

        m = re.search(r"(?i)final\s*output\s*[:=]\s*(.+)", resp)
        if m and _norm(m.group(1)) == _norm(expected):
            return True
        # fallbacks: whole response is just the value, or the last non-empty line is
        if _norm(resp) == _norm(expected):
            return True
        lines = [ln for ln in resp.splitlines() if ln.strip()]
        if lines and _norm(lines[-1]) == _norm(expected):
            return True
        return False

    # Task 3: Test Generation - the canonical LCB metric runs the model's generated tests
    # against a reference solution to check they discriminate correct vs. buggy code, which
    # this lightweight harness cannot do. The previous `len(resp) > 10` check passed almost
    # everything (fake ~100%), so test_gen is EXCLUDED from the loader by default
    # (GBENCH_LCB_INCLUDE_TEST_GEN=1, legacy alias LCB_INCLUDE_TEST_GEN=1, to force-load it).
    # If it is loaded, we do not credit it here rather than report an unvalidated pass.
    elif task == "test_gen":
        return False

    # Unrecognised task. This used to `return True`, i.e. auto-pass anything the verifier
    # did not understand. Measured 2026-08-20 it never fired (0 of 400 problems), but a
    # scorer whose default is "correct" turns a future schema change into silent inflation.
    return False


async def _async_judge_lcb(sample_traces: List[Dict[str, Any]]) -> Tuple[int, int, Dict[str, Dict[str, Any]]]:
    """Execute all LCB test suites in parallel using ProcessPoolExecutor post-generation."""
    import asyncio
    import concurrent.futures

    loop = asyncio.get_running_loop()
    max_workers = min(32, (os.cpu_count() or 4))
    
    with tqdm(total=len(sample_traces), desc="Judging [LCB]") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            tasks = []
            for sample in sample_traces:
                resp = sample.get("response_text") or sample.get("response") or ""
                gold = sample.get("gold_answer")
                fut = loop.run_in_executor(pool, _verify_single_lcb_sample, resp, gold)
                tasks.append((sample, fut))

            correct_count = 0
            category_stats = {}
            for sample, fut in tasks:
                verdict = await fut
                sample["is_correct"] = verdict
                sample["correct"] = verdict
                sample["status"] = "OK" if verdict else "FAILED"
                cat = sample.get("category")
                if cat:
                    if cat not in category_stats:
                        category_stats[cat] = {"correct": 0, "total": 0}
                    category_stats[cat]["total"] += 1
                    if verdict:
                        category_stats[cat]["correct"] += 1
                if verdict:
                    correct_count += 1
                pbar.update(1)

    return correct_count, len(sample_traces), category_stats


def run_lcb(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native LiveCodeBench algorithmic coding evaluation suite."""
    # Executes model-written code: require bubblewrap isolation. Missing/blocked isolation is an
    # EXTERNAL prerequisite gap -> hard-error (no-skip), even when invoked directly (bypassing the
    # central evals.py gate). GBENCH_SANDBOX=none opts into unsandboxed execution.
    _blocked = sandbox_skip_reason()
    if _blocked:
        raise infra_required("lcb", _blocked, "docs/evals/lcb.md")
    samples = _load_lcb_samples(enable_thinking, limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="lcb",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_lcb,
        declared_scoring_mode="execution",  # sandboxed test execution, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )

    # Report the scenarios SEPARATELY. LiveCodeBench never combines them, and blending is
    # not cosmetic: on the 2026-08-20 run the headline read 87.37% off 73.25% generation
    # and 99.16% execution, with execution supplying 479 of 879 rows. That execution set
    # has 49 distinct expected answers across 479 problems (median 1 character, the six
    # commonest values covering 60%), so it is close to free and it was more than half the
    # number being quoted.
    buckets: Dict[str, List[bool]] = {}
    for tr in result.get("sample_traces", []):
        cat = str(tr.get("category") or "")
        scen = ("code_execution" if cat == "code_execution"
                else "test_generation" if cat == "test_gen"
                else "code_generation")
        buckets.setdefault(scen, []).append(bool(tr.get("is_correct")))
    scenarios = {k: {"correct": sum(v), "total": len(v),
                     "accuracy": round(100.0 * sum(v) / len(v), 2)}
                 for k, v in buckets.items() if v}
    result["scenarios"] = scenarios

    # `accuracy` becomes code generation alone - the scenario "LiveCodeBench pass@1" means
    # - so the headline is a number that exists upstream. The blend is still reported, but
    # under a name that cannot be mistaken for a published figure.
    if "code_generation" in scenarios:
        cg = scenarios["code_generation"]
        result["blended_all_scenarios_accuracy"] = result.get("accuracy")
        # Keep the blended counts, but under names that don't masquerade as the headline.
        result["all_scenarios_correct"] = result.get("correct_answers")
        result["all_scenarios_total"] = result.get("total_questions")
        # Headline accuracy AND its counts are all code_generation, so the top-line trio
        # (correct/total/accuracy) is self-consistent instead of showing the blended
        # 746/933 next to a code-gen-only 59.69%.
        result["accuracy"] = cg["accuracy"]
        result["correct_answers"] = cg["correct"]
        result["total_questions"] = cg["total"]
        result["metric"] = f"code_generation pass@1 ({_LCB_VERSION}); execution reported separately"
    result["lcb_version"] = _LCB_VERSION
    result["lcb_min_contest_date"] = _LCB_MIN_DATE or None
    result["dataset"] = f"livecodebench/code_generation_lite:{_LCB_VERSION}"
    # The problem dates the score actually covers. LiveCodeBench is only contamination-free
    # to the extent this window sits past the model's cutoff, and a release tag alone does
    # not say - the tags are cumulative, so `release_v6` still carries v1's 2023 problems.
    dates = sorted(str((t.get("extra_payload") or {}).get("contest_date") or "")[:7]
                   for t in result.get("sample_traces", [])
                   if (t.get("extra_payload") or {}).get("contest_date"))
    if dates:
        result["contest_date_range"] = [dates[0], dates[-1]]
    # v1 is 2023-05..2024-03 and predates every current model, so a v1 run cannot be
    # contamination-free; a --eval-limit subset is also not the full-set leaderboard number.
    _noncanon = []
    if _LCB_VERSION in ("release_v1", "v1"):
        _noncanon.append(f"contamination-prone version {_LCB_VERSION}")
    if kwargs.get("limit"):
        _noncanon.append(f"subset run (--eval-limit {kwargs.get('limit')})")
    result["leaderboard_comparable"] = not _noncanon
    if _noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(_noncanon)
    return result
