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

"""Native Codeforces competitive programming evaluation suite.

Follows the protocol the `open-r1/codeforces` dataset card prescribes, which is the only
reproducible open Codeforces protocol there is. (Frontier labs report a Codeforces **Elo
percentile** obtained by simulating real contests; that needs the live platform and is not
reproducible here, so this is a pass-rate over a held-out split, not an Elo.)

    config  `verifiable`        - `executable` AND (`official_tests_complete` OR has
                                  `generated_tests`); 8,338 train / 422 test
    split   `test`              - "problems from late 2024 and early 2025"
    prompts `verifiable-prompts`- the dataset's own prompt, Python variant
    tests   `official_tests` + the separately-downloaded `generated_tests/` shards
    checker `generated_checker` - ~30% of problems accept MULTIPLE valid outputs
    limits  `time_limit` (s) / `memory_limit` (MB), per problem

What this replaced, on 2026-08-20, and why each mattered:

* `open-r1/codeforces-cots` **train** split. That is a chain-of-thought *distillation* set
  carrying five generations per problem, so the previous "500 samples" were **100 distinct
  problems counted five times each**. It also starts at 2010, well inside any current
  model's training window; the `test` split here is late-2024/2025 and the card says
  "Please avoid training on these."
* A hardcoded **40/30/30 CF/ICPC/IOI quota**, invented here and described as "balanced".
  IOI is 85 of 10,024 problems (0.85%), so the quota over-represented it ~35x, and 46% of
  IOI problems ship no `input_format` at all because they are grader-based - unrunnable as
  stdin/stdout, which is what produced IOI 17.33% against CF 68% / ICPC 76%. The natural
  distribution is used instead; no suite should invent a category mix.
* Grading against `examples`, the 1-2 sample cases printed in the problem statement
  (mean 1.95; 205 of 500 rows had exactly one). The card is explicit that platforms
  truncate visible tests to ~400 chars, so those cases "can be solved with an easy brute
  force solution".
* **Exact stdout match on every problem.** ~30% of Codeforces problems accept multiple
  valid answers and need a checker program; exact match marks correct solutions wrong.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_CODEFORCES_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .sandbox import run_sandboxed
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/codeforces.md"

#: The dataset, and the subset/split its card prescribes for evaluation.
_REPO = "open-r1/codeforces"
_CONFIG = os.environ.get("GBENCH_CODEFORCES_CONFIG", "verifiable")
_SPLIT = os.environ.get("GBENCH_CODEFORCES_SPLIT", "test")

#: `generated_tests` live in the SAME repo under `generated_tests/test_cases_<contest>.parquet`
#: but are NOT part of the dataset load - the full set is ~110GB. Point this at a directory
#: populated with:
#:     hf download open-r1/codeforces --repo-type dataset \
#:         --include 'generated_tests/*.parquet' --max-workers 8 --local-dir <DIR>
#: Only the shards for contests in the split are read, so a partial mirror is fine.
_GENERATED_TESTS_DIR = prereqs_path("codeforces", os.environ.get("GBENCH_CODEFORCES_GENERATED_TESTS")) or ""

#: Where the warning tells the operator to put the mirror. Sibling tools already live under
#: gbench-prereqs/ (aider, OJBench, SWE-bench-Live, ...), so the shards go beside them.
_SUGGESTED_MIRROR = os.environ.get(
    "GBENCH_PREREQS_DIR", os.path.expanduser("~/.cache/gbench/gbench-prereqs")) + "/codeforces"

#: Cap on generated test cases executed per problem. Canonical open-r1 verification runs ALL
#: generated tests (a solution that fails only a capped-out case must be marked wrong), so the
#: default is 0 = NO CAP. The shards carry a plain `test_i` index (NOT a hardness ordering), so a
#: cap would drop arbitrary cases, not "keep the hardest". Set the env var to bound wall-clock on
#: very large shards if you must, but that makes the run non-canonical (a subset of the tests).
_MAX_GENERATED_TESTS = int(os.environ.get("GBENCH_CODEFORCES_MAX_GENERATED_TESTS", "0"))

#: Below this many executed test cases a problem's verdict is called out as weak. Measured
#: 2026-08-20 on the full mirror: 32 of 422 problems, against a suite median of 29.
_THIN_TEST_THRESHOLD = 5

#: Fallback per-case wall-clock when a problem declares no `time_limit`, and the multiple
#: applied to the declared limit (CPython is far slower than the C++ the limits assume).
_DEFAULT_TIME_LIMIT_S = 10.0
_TIME_LIMIT_MULTIPLIER = float(os.environ.get("GBENCH_CODEFORCES_TIME_MULTIPLIER", "3.0"))


def _as_int(v: Any) -> int:
    """Ordering key, tolerant of the column arriving as str or int."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _load_generated_tests(contest_id: str, problem_id: str) -> List[Dict[str, str]]:
    """Generated cases for one problem, from a local `generated_tests/` mirror.

    Returns [] when no mirror is configured or the shard is absent - the caller then grades
    on `official_tests` alone and the result records the reduced coverage. Never raises: a
    missing shard must degrade the score's *confidence*, not crash the run.
    """
    if not _GENERATED_TESTS_DIR:
        return []
    try:
        cid = int(str(contest_id))
    except (TypeError, ValueError):
        return []
    path = os.path.join(_GENERATED_TESTS_DIR, "generated_tests",
                        f"test_cases_{cid:04d}.parquet")
    if not os.path.exists(path):
        path = os.path.join(_GENERATED_TESTS_DIR, f"test_cases_{cid:04d}.parquet")
        if not os.path.exists(path):
            return []
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(path).to_pydict()
    except Exception as e:                                              # noqa: BLE001
        logger.warning("codeforces: could not read %s: %s", path, e)
        return []
    # The card documents the ordering column as `test_case_i`; the shipped shards actually
    # use `test_i`. Accept either, and treat its absence as "unordered" rather than
    # indexing a one-element default, which raised IndexError on the second row.
    ids = tbl.get("problem_id") or []
    inputs = tbl.get("input") or []
    outputs = tbl.get("output") or []
    order = tbl.get("test_i") or tbl.get("test_case_i") or []
    if not (len(ids) == len(inputs) == len(outputs)):
        logger.warning("codeforces: %s has ragged columns (%d/%d/%d); skipping it. See %s",
                       os.path.basename(path), len(ids), len(inputs), len(outputs), DOCS_URL)
        return []
    rows = []
    pid = str(problem_id)
    for i, pid_i in enumerate(ids):
        if str(pid_i) != pid:
            continue
        rows.append({"input": str(inputs[i]), "output": str(outputs[i]),
                     "i": _as_int(order[i]) if i < len(order) else i})
    rows.sort(key=lambda r: r.get("i") or 0)
    if _MAX_GENERATED_TESTS > 0:
        rows = rows[:_MAX_GENERATED_TESTS]
    return [{"input": r["input"], "output": r["output"]} for r in rows]


def _load_codeforces_samples(enable_thinking: bool = False,
                             limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load the `verifiable` / `test` split with its official prompts and full test sets."""
    from datasets import load_dataset
    ds = list(load_dataset(_REPO, _CONFIG, split=_SPLIT))

    # The dataset ships its own prompt (Python and C++ variants). Use it rather than a bespoke
    # one so the number is about the model, not about our phrasing. It is REQUIRED: silently
    # swapping in a built-in prompt would change the measured quantity, so a load failure
    # hard-errors (never a silent degradation).
    prompts: Dict[str, str] = {}
    try:
        for r in load_dataset(_REPO, "verifiable-prompts", split=_SPLIT):
            if str(r.get("language", "")).lower().startswith("py"):
                prompts[str(r.get("id"))] = str(r.get("prompt") or "")
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(
            "codeforces",
            f"the canonical prompts ('{_REPO}' config 'verifiable-prompts', split '{_SPLIT}') "
            f"could not be loaded ({e}). They ship in the same dataset repo; ensure it is "
            "downloaded/cached (needs network on first load). Refusing to substitute a "
            "non-canonical built-in prompt.",
            DOCS_URL) from e
    if not prompts:
        raise infra_required(
            "codeforces",
            f"'{_REPO}' 'verifiable-prompts'/{_SPLIT} loaded no Python prompts; cannot run "
            "canonically without the dataset's own prompts.",
            DOCS_URL)

    # Seeded random subset, NOT stratified by contest_type. Stratifying would balance the
    # categories, which is a milder version of the 40/30/30 quota this replaced: IOI is
    # 0.85% of the corpus and 1 of the 422 test problems, so any key that spreads across
    # contest types over-represents it again. The dataset's own distribution IS the
    # measurement. `stratified_sample` with no key degrades to a seeded subset, which still
    # beats a contiguous head (dataset order correlates with contest date and difficulty).
    ds = stratified_sample(ds, limit, None, seed="codeforces")

    samples = []
    n_gen = n_checker = 0
    for item in ds:
        pid = str(item.get("id"))
        official = [t for t in (item.get("official_tests") or []) if isinstance(t, dict)]
        generated = _load_generated_tests(item.get("contest_id"), pid)
        if generated:
            n_gen += 1
        checker = item.get("generated_checker") or ""
        if checker:
            n_checker += 1

        prompt = prompts.get(pid)
        if not prompt:
            # Every problem must carry the dataset's own canonical prompt; a bespoke substitute
            # would change the measured quantity. A missing one is a data gap, not something to
            # paper over.
            raise RuntimeError(
                f"codeforces: no canonical Python prompt for problem {pid!r} in "
                f"'{_REPO}' verifiable-prompts/{_SPLIT}; refusing to substitute a built-in "
                f"prompt. See {DOCS_URL}")

        gold = {
            "official_tests": official,
            "generated_tests": generated,
            "checker": checker,
            "time_limit": item.get("time_limit"),
            "memory_limit": item.get("memory_limit"),
            "testset_size": item.get("testset_size"),
            "official_tests_complete": bool(item.get("official_tests_complete")),
        }
        meta = {"category": str(item.get("contest_type") or "CF"),
                "rating": item.get("rating"),
                "year": item.get("contest_start_year")}
        samples.append(([{"role": "user", "content": prompt}], gold, meta))

    logger.info("Loaded %d codeforces problems (%s/%s); %d with generated tests, "
                "%d needing a checker.", len(samples), _CONFIG, _SPLIT, n_gen, n_checker)
    return samples


def _require_generated_tests_mirror() -> None:
    """The generated_tests mirror is REQUIRED. `official_tests` alone is a median 9.7% of each
    problem's real suite (the card: the visible cases 'solved with an easy brute force
    solution'), so grading on them is a near-meaningless partial - hard-error instead."""
    if _GENERATED_TESTS_DIR and os.path.isdir(_GENERATED_TESTS_DIR):
        return
    raise infra_required(
        "codeforces",
        "the generated_tests mirror is required (official_tests alone is ~9.7% coverage). "
        "Download the shards and point GBENCH_CODEFORCES_GENERATED_TESTS at the directory:\n"
        "  hf download open-r1/codeforces --repo-type dataset "
        f"--include 'generated_tests/*.parquet' --max-workers 8 --local-dir {_SUGGESTED_MIRROR}\n"
        f"  export GBENCH_CODEFORCES_GENERATED_TESTS={_SUGGESTED_MIRROR}",
        DOCS_URL)


#: Driver that runs a problem's official checker, executed INSIDE the sandbox (bwrap mounts a
#: fresh tmpfs over /tmp, so the checker + its files must be created in-sandbox).
#:
#: open-r1's checkers (verified against real `generated_checker` sources): argv is
#:     checker.py  <input>  <reference/jury answer>  <submission/model output>
#: and the verdict is PRINTED TO STDOUT - `0` for wrong, the checker's max (`1` or `100`) for
#: correct - while the process ALWAYS exits 0. Reading the return code (the old bug) therefore
#: passed EVERY checker-graded problem regardless of the model's answer (~30% of the suite).
#:
#: We read the last numeric token from stdout as the score, and calibrate "full credit" by also
#: scoring the REFERENCE answer as if it were the submission (it is correct by construction, so
#: its score is the max) - this makes both 1-scale and 100-scale (and partial) checkers work
#: without hardcoding the maximum.
_CHECKER_DRIVER = r"""
import subprocess, sys, tempfile, os, re
def _run_checker(checker_src, test_input, submission_out, reference_out, timeout):
    d = tempfile.mkdtemp()
    cpath = os.path.join(d, "checker.py")
    with open(cpath, "w") as f:
        f.write(checker_src)
    paths = {}
    # argv order: input, reference (jury answer), submission (model output)
    for name, data in (("input", test_input), ("reference", reference_out),
                       ("submission", submission_out)):
        p = os.path.join(d, name + ".txt")
        with open(p, "w") as f:
            f.write(data)
        paths[name] = p
    try:
        r = subprocess.run([sys.executable, cpath, paths["input"], paths["reference"],
                            paths["submission"]], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return "NONE"
    nums = re.findall(r"-?\d+(?:\.\d+)?", (r.stdout or "").strip())
    return nums[-1] if nums else "NONE"
"""


def _outputs_match(actual: str, expected: str) -> bool:
    """Token-wise comparison, the standard Codeforces `wcmp` behaviour.

    Plain string equality fails on trailing whitespace and on line endings, which the
    dataset carries verbatim from the platform (`\r\n`), and would mark correct answers
    wrong.
    """
    return actual.split() == expected.split()


def _run_case(code: str, test: Dict[str, str], timeout: float,
              checker: str) -> bool:
    """One test case. True iff the submission is accepted for it."""
    import subprocess
    inp = str(test.get("input") or "")
    expected = str(test.get("output") or "")
    try:
        res = run_sandboxed([sys.executable, "-c", code], input=inp,
                            capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, Exception):                      # noqa: BLE001
        return False
    if res.returncode != 0:
        return False
    actual = res.stdout or ""
    if not checker:
        return _outputs_match(actual.strip(), expected.strip())
    # Score the model AND the reference (for full-credit calibration) in one sandboxed driver run.
    script = (f"{_CHECKER_DRIVER}\n"
              f"m = _run_checker({checker!r}, {inp!r}, {actual!r}, {expected!r}, {timeout!r})\n"
              f"r = _run_checker({checker!r}, {inp!r}, {expected!r}, {expected!r}, {timeout!r})\n"
              f"print('GBENCH_MODEL', m)\n"
              f"print('GBENCH_REF', r)\n")
    try:
        # Checker driver on STDIN (`python -`), NOT a `-c <script>` argument: it embeds the test
        # input TWICE ({inp!r}) plus the model output, so a large case (>~40KB) makes the argument
        # exceed Linux MAX_ARG_STRLEN (128KB) -> OSError('Argument list too long'), which the except
        # below would silently score as a failed case (a false negative). The non-checker path above
        # is safe: its input goes via stdin and only the (small) model program is a `-c` argument.
        proc = run_sandboxed([sys.executable, "-"], input=script, capture_output=True, text=True,
                             timeout=2 * timeout + 20)
    except (subprocess.TimeoutExpired, Exception):                      # noqa: BLE001
        return False
    model_score = ref_score = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("GBENCH_MODEL "):
            model_score = _parse_score(line[len("GBENCH_MODEL "):])
        elif line.startswith("GBENCH_REF "):
            ref_score = _parse_score(line[len("GBENCH_REF "):])
    if model_score is None:
        return False
    # Full credit = the score the (correct-by-construction) reference earns; fall back to 1.0 if
    # the checker does not validate its own reference (e.g. it ignores the answer file).
    full = ref_score if (ref_score is not None and ref_score > 0) else 1.0
    return model_score >= full


def _parse_score(token: str) -> Optional[float]:
    token = (token or "").strip()
    if not token or token == "NONE":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _eval_codeforces(response_text: str, gold: Any) -> bool:
    """Accepted iff EVERY official and generated test case passes.

    A problem with no usable tests is reported incorrect, never auto-passed: an untested
    submission measured nothing, and crediting it inflates the suite with rows that carry
    no signal.
    """
    if not response_text:
        return False
    if isinstance(gold, list):          # legacy shape: a bare list of {input, output}
        gold = {"official_tests": gold, "generated_tests": [], "checker": ""}
    if not isinstance(gold, dict):
        return False

    # Canonical open-r1 extraction takes the LAST fenced block. Models draft/iterate and often
    # emit a corrected final solution after earlier attempts (measured: 36/422 responses carried
    # 2-8 blocks, first != last), so grading the FIRST block under-counts. Take the last.
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", response_text, re.DOTALL)
    code = code_blocks[-1] if code_blocks else response_text

    tests = [t for t in (gold.get("official_tests") or []) if isinstance(t, dict)]
    tests += [t for t in (gold.get("generated_tests") or []) if isinstance(t, dict)]
    if not tests:
        return False

    tl = gold.get("time_limit")
    try:
        timeout = float(tl) * _TIME_LIMIT_MULTIPLIER if tl else _DEFAULT_TIME_LIMIT_S
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIME_LIMIT_S
    timeout = max(1.0, min(timeout, 60.0))

    checker = str(gold.get("checker") or "")
    return all(_run_case(code, t, timeout, checker) for t in tests)


def run_codeforces(model_name: str, base_url: str, concurrency: int,
                   enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native Codeforces competitive programming evaluation suite."""
    _require_generated_tests_mirror()   # fail fast: full test coverage is required, not optional
    # Fail fast on a broken sandbox. _run_case() catches every exception and returns False, so a
    # missing/blocked bubblewrap would otherwise be swallowed sample-by-sample and reported as a
    # fake 0% (every problem "incorrect") instead of the infra error it is. GBENCH_SANDBOX=none is
    # the explicit unsandboxed opt-out and is honoured; any other mode requires working isolation.
    from .sandbox import sandbox_mode, sandbox_available, sandbox_skip_reason
    if sandbox_mode() != "none" and not sandbox_available():
        raise infra_required(
            "codeforces",
            "executes model-written code and requires bubblewrap process isolation, which is "
            "unavailable or blocked here: "
            + (sandbox_skip_reason() or "the bubblewrap probe failed")
            + " (set GBENCH_SANDBOX=none to run UNSANDBOXED on purpose)",
            DOCS_URL)
    samples = _load_codeforces_samples(enable_thinking=enable_thinking,
                                       limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="codeforces",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_codeforces,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # Record how much of each problem's real test suite was actually executed. Without it a
    # thin-coverage run is indistinguishable from a rigorous one.
    covs, per_problem, with_gen, with_chk = [], [], 0, 0
    for tr in result.get("sample_traces", []):
        g = tr.get("gold_answer")
        if not isinstance(g, dict):
            continue
        n = len(g.get("official_tests") or []) + len(g.get("generated_tests") or [])
        per_problem.append(n)
        ts = g.get("testset_size") or 0
        if ts:
            covs.append(min(1.0, n / ts))
        if g.get("generated_tests"):
            with_gen += 1
        if g.get("checker"):
            with_chk += 1
    if covs:
        result["test_coverage_pct"] = round(sum(covs) / len(covs) * 100.0, 1)
    # Absolute counts alongside the ratio. `test_coverage_pct` divides by `testset_size`,
    # the real hidden suite, which ranges from a handful to 198 - so a problem run on 29
    # adversarial tests against a 71-test official suite reads as "41% covered" while being
    # thoroughly tested. open-r1 ranked candidate tests by hardness (using wrong solutions
    # that passed the public tests) and kept the hardest, so these are not random samples.
    # The number that says whether a verdict is trustworthy is how many tests actually ran.
    if per_problem:
        per_problem.sort()
        mid = per_problem[len(per_problem) // 2]
        result["tests_per_problem_median"] = mid
        result["tests_per_problem_min"] = per_problem[0]
        thin = sum(1 for n in per_problem if n < _THIN_TEST_THRESHOLD)
        result["problems_under_5_tests"] = thin
        if thin:
            logger.warning(
                "[codeforces] %d/%d problems were graded on fewer than %d test cases "
                "(median across the suite is %d). Those verdicts are weak in both "
                "directions - open-r1 discards generated tests when correct solutions "
                "disagree, so these are the problems where generation mostly failed. "
                "See %s", thin, len(per_problem), _THIN_TEST_THRESHOLD, mid, DOCS_URL)
    result["problems_with_generated_tests"] = with_gen
    result["problems_with_checker"] = with_chk
    result["dataset"] = f"{_REPO}:{_CONFIG}/{_SPLIT}"
    result["generated_tests_mirrored"] = bool(_GENERATED_TESTS_DIR)
    result["leaderboard_comparable"] = False
    result["metric"] = ("all official+generated tests pass (checker-aware); NOT the "
                        "Codeforces Elo percentile frontier labs report")
    return result
