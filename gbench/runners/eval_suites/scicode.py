# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: scicode
# Description: SciCode (Scientific Python Problem Solving & Numerical Code Execution Benchmark)

"""gbench native built-in runner for scicode (Coding & Algorithmic).

Faithful port of SciCode's **sequential, per-sub-step** protocol
(scicode-bench/SciCode `eval/scripts/gencode.py` + `test_generated_code.py`):

Generation
    A problem has N sub-steps. For step k we build a prompt containing every
    PRIOR step's description and the model's OWN generated code for that step
    (separated by "------"), followed by step k's description, function header
    and return line, plus the shared dependency block. The model writes only
    step k's function; its code is fed forward as context for step k+1. This is
    N model calls per problem, not a single whole-problem shot.

Scoring
    Each generated step is executed on its own: dependencies + all prior steps'
    code + this step's code, then the canonical `target` bindings
    (`process_hdf5_to_tuple(step_number, n, test_data.h5)`) and the step's test
    cases. A step passes iff its process exits 0. A PROBLEM is correct iff every
    scored sub-step passes.

Metrics
    The framework surfaces a single headline `accuracy`; we report **sub-step
    accuracy** there (correct sub-steps / total scored sub-steps) and stash
    **problem accuracy** and the raw counts as extra keys (the AIME/aider
    convention). Both are the numbers SciCode reports ("correct steps" /
    "correct problems").

Sampling:
    temperature is resolved by `resolve_temperature("scicode", ...)`:
    `GBENCH_SCICODE_TEMPERATURE` > `--temperature` > gbench default. `--thinking`
    toggles the model's native reasoning channel (chat-template kwarg
    `enable_thinking`). SciCode's own default is greedy (temp 0); gbench's policy
    governs here per project baseline. Set `GBENCH_SCICODE_WITH_BACKGROUND=1` to
    inject the human-written step background (canonical `--with-background`);
    the default is the without-background protocol.

Note on validation: SciCode withholds gold (`ground_truth_code` /
`general_solution` are empty in the HF dataset), so there is no gold self-check.
Validate the wiring with a capable model - scores should be non-zero and track
difficulty; an all-zero run means the h5/keying/plumbing is wrong, not the model.
"""

import ast
import concurrent.futures
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .base import DEFAULT_TEMPERATURE, resolve_temperature, suite_env
from .sampling import stratified_sample
from .sandbox import run_sandboxed, sandbox_mode, sandbox_skip_reason
from .swe_thread_cap import THREAD_VARS
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/scicode.md"

PILLAR = "Coding & Algorithmic"

# --------------------------------------------------------------------------- #
# Canonical prompt templates (verbatim from scicode-bench/SciCode eval/data/,  #
# Apache-2.0). NOTE the upstream naming: `--with-background` uses the          #
# multistep template (human background is injected into the step text), while  #
# the default (without background) uses the background-comment template (the   #
# model must generate its own "Background:" comment).                          #
# --------------------------------------------------------------------------- #
_TEMPLATE_MULTISTEP = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""

_TEMPLATE_BG_COMMENT = """PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""

#: Sub-steps SciCode supplies directly (they depend on prior human scaffolding),
#: keyed by step_number. Not generated, not scored; injected as context for the
#: later steps of their problems. Bytes vendored under scicode_data/ (see the
#: ATTRIBUTION there).
_SPECIAL_STEP_IDS = ("13.6", "62.1", "76.3")
_SCICODE_DATA_DIR = os.path.join(os.path.dirname(__file__), "scicode_data")


def _load_special_gold(step_number: str) -> Optional[str]:
    if step_number not in _SPECIAL_STEP_IDS:
        return None
    path = os.path.join(_SCICODE_DATA_DIR, f"{step_number}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("scicode: could not read vendored gold %s (%s)", path, e)
        return None


#: Third-party mirror of SciCode's reference-output HDF5. The canonical source is a Google
#: Drive folder linked from the SciCode README, which is not scriptable without gdown/auth,
#: and the HF dataset (`SciCode1/SciCode`) ships only the problem JSONLs. Override with
#: GBENCH_SCICODE_TEST_DATA (a local path) or GBENCH_SCICODE_TEST_DATA_REPO (a different
#: mirror). The bare SCICODE_TEST_DATA / SCICODE_TEST_DATA_REPO still work as deprecated aliases.
_TEST_DATA_REPO = suite_env("GBENCH_SCICODE_TEST_DATA_REPO", "SCICODE_TEST_DATA_REPO",
                            default="Srimadh/Scicode-test-data-h5")
_TEST_DATA_FILE = "test_data.h5"


def resolve_test_data() -> Tuple[Optional[str], Optional[str]]:
    """Path to SciCode's `test_data.h5`, fetching it if needed. Returns (path, provenance).

    Every SciCode test binds its expected values as `target` from this file
    (`process_hdf5_to_tuple`); without it every test raises NameError and even a perfect
    solution scores 0 - a structural zero, not a model result. It is ~1 GB and is fetched
    at eval time like any other dataset, then cached by huggingface_hub.
    """
    local = suite_env("GBENCH_SCICODE_TEST_DATA", "SCICODE_TEST_DATA")
    if local:
        return (local, "GBENCH_SCICODE_TEST_DATA") if os.path.exists(local) else (None, None)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_TEST_DATA_REPO, filename=_TEST_DATA_FILE,
                               repo_type="dataset")
    except Exception as e:
        logger.warning("scicode: could not fetch %s/%s (%s)", _TEST_DATA_REPO,
                       _TEST_DATA_FILE, e)
        return None, None
    # The canonical file lives on Google Drive, so this mirror is unverified against it.
    # Say so, and record the digest, rather than let an unchecked artefact decide a score.
    logger.warning(
        "scicode: using the third-party mirror %s for %s. The canonical copy is the Google "
        "Drive folder linked from the SciCode README; verify the digest before quoting a "
        "headline number. Set GBENCH_SCICODE_TEST_DATA to use your own copy.",
        _TEST_DATA_REPO, _TEST_DATA_FILE)
    return path, f"hf:{_TEST_DATA_REPO}/{_TEST_DATA_FILE}"


def _digest(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# SciCode's `target` reference outputs are read from test_data.h5 by the canonical
# `process_hdf5_to_tuple`. The `scicode` PyPI/git package pulls an invasively broad
# dependency tree (numpy 2.x, litellm, inspect-ai, aioboto3, ...) that conflicts with
# gbench's pinned stack, so we VENDOR just this loader (needs only h5py + scipy + numpy,
# already in the eval env). Verbatim from scicode-bench/SciCode src/scicode/parse/parse.py
# (Apache-2.0; attribution retained). Inlined into the sandbox script so the subprocess
# reads the h5 directly, mirroring the canonical harness.
_H5_LOADER_SRC = r'''
import h5py
import scipy
import numpy as np


def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    d = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            d[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            d[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                d[tmp] = obj[()]
            except ValueError:
                d[key] = obj[()]
    return d


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file):
    data_lst = []
    with h5py.File(h5py_file, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst
'''


# --------------------------------------------------------------------------- #
# Generation: canonical sequential per-sub-step protocol.                      #
# --------------------------------------------------------------------------- #
def _extract_python_script(response: Optional[str]) -> str:
    """Port of scicode.gen.models.extract_python_script.

    Pull the code out of a ```python``` (or bare ```) fence, then strip any
    import lines (dependencies are supplied separately and prepended). Falls back
    to the raw text if no fence is present. Robust to malformed fences.
    """
    if not response:
        return ""
    try:
        if "```" in response:
            if "```python" in response:
                python_script = response.split("```python")[1].split("```")[0]
            else:
                python_script = response.split("```")[1].split("```")[0]
        else:
            python_script = response
    except IndexError:
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '',
                           python_script, flags=re.MULTILINE)
    return python_script


def _extract_function_name(function_header: str) -> str:
    """Vendored from scicode.parse.parse.extract_function_name (Apache-2.0).

    Canonical quirk preserved: the `def` pattern is tried BEFORE `class`, so a
    header like `class Maxwell:\\n    def __init__(...)` resolves to `__init__`
    (not `Maxwell`). Kept verbatim so the supplied-step context matches upstream.
    """
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    pattern = r'\bclass\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    raise ValueError('Function name or class name not found.')


def _get_function_from_code(code_string: Optional[str], function_name: str) -> Optional[str]:
    """Vendored from scicode.parse.parse.get_function_from_code (Apache-2.0).

    Returns the AST-unparsed source of the FIRST def/class whose name matches -
    so a class's `__init__` comes back WITHOUT its enclosing class, which is
    exactly how canonical forwards the supplied steps for problems 13 and 62.
    None if no node matches; the raw string on parse error (canonical behavior).
    """
    if code_string is None:
        return None
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception as e:  # noqa: BLE001 - mirror canonical: raw string on parse error
        logger.debug("scicode: get_function_from_code(%s) failed: %s", function_name, e)
        return code_string
    return None


def _process_problem_code(sub_steps: List[Dict[str, Any]], num_steps: int) -> str:
    header = sub_steps[num_steps - 1].get("function_header") or ""
    return_line = sub_steps[num_steps - 1].get("return_line") or ""
    return f"{header}\n\n{return_line}"


def _process_problem_steps(
    sub_steps: List[Dict[str, Any]],
    num_steps: int,
    with_background: bool,
    previous_llm_code: List[str],
) -> Tuple[str, str, str]:
    """Return (problem_steps_str, next_step_str, previous_code_str) for step `num_steps`.

    Mirrors Gencode.process_problem_steps: prior steps contribute their
    description (+ background when enabled) and generated code, separated by
    "------"; the next step contributes its description and header/return line.
    """
    output_lines: List[str] = []
    previous_code: List[str] = []
    for i in range(num_steps - 1):
        desc = sub_steps[i].get("step_description_prompt") or ""
        if with_background:
            desc = desc + "\n" + (sub_steps[i].get("step_background") or "")
        output_lines.append(desc)
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    next_desc = sub_steps[num_steps - 1].get("step_description_prompt") or ""
    if with_background:
        next_desc = next_desc + "\n" + (sub_steps[num_steps - 1].get("step_background") or "")
    next_step = [next_desc, _process_problem_code(sub_steps, num_steps)]

    problem_steps_str = "\n\n".join(output_lines[:-1]) if output_lines else ""
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return problem_steps_str, next_step_str, previous_code_str


def _build_step_prompt(
    prob: Dict[str, Any],
    num_steps: int,
    with_background: bool,
    previous_llm_code: List[str],
) -> str:
    template = _TEMPLATE_MULTISTEP if with_background else _TEMPLATE_BG_COMMENT
    problem_steps_str, next_step_str, _ = _process_problem_steps(
        prob["sub_steps"], num_steps, with_background, previous_llm_code)
    dependencies = prob.get("required_dependencies") or ""
    # .format is safe: only the template carries {placeholders}; substituted
    # values may freely contain braces (code, f-strings) - they aren't re-parsed.
    return template.format(
        problem_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=dependencies,
    )


def _complete(client, model: str, prompt: str, temperature: float,
              max_tokens: int, thinking: bool, retries: int = 5) -> Optional[Dict[str, Any]]:
    """One chat call (single-turn agentic pattern).

    Returns a dict {text, reasoning, finish_reason, completion_tokens}, or None after
    exhausting retries. `reasoning` is the model's thinking channel (vLLM's gemma4 reasoning
    parser separates it into `reasoning_content`; some builds use `reasoning`) - captured so
    the trace records BOTH the thinking content and its length, and so a truncated
    (finish_reason=="length") turn is distinguishable from a wrong answer.
    """
    import time
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg = choice.message
            reasoning = (getattr(msg, "reasoning_content", None)
                         or getattr(msg, "reasoning", None) or "")
            usage = getattr(resp, "usage", None)
            return {
                "text": msg.content or "",
                "reasoning": reasoning,
                "finish_reason": getattr(choice, "finish_reason", None),
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except Exception as e:  # noqa: BLE001 - endpoint/transport errors are retried
            if attempt == retries - 1:
                logger.warning("scicode: generation call failed after %d tries: %s",
                               retries, e)
                return None
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return None


def _score_step(assembled_code: str, step_number: str, test_cases: List[str],
                h5: str, timeout: int) -> Dict[str, Any]:
    """Execute one assembled sub-step against its h5-bound targets.

    Returns {correct, returncode, stdout_tail, stderr_tail} - the execution detail is kept
    for the trajectory so a failed step is readable (traceback / assertion) instead of a bare
    boolean. `correct` is True iff the subprocess exited 0.
    """
    lines = [_H5_LOADER_SRC, assembled_code]
    lines.append(f"targets = process_hdf5_to_tuple({step_number!r}, {len(test_cases)}, {h5!r})")
    for i, tc in enumerate(test_cases):
        lines.append(f"target = targets[{i}]")
        lines.append(str(tc))
    script = "\n".join(lines) + "\n"
    # Cap BLAS/OpenMP threads PER scoring subprocess. numpy/scipy otherwise spawn one thread
    # per core, so at high --sandboxes (e.g. 48) the box runs 48 x N-core processes and
    # thrashes on oversubscription (same pathology as swe_thread_cap). Parallelism comes from
    # running many problems at once, not from threading one step; GBENCH_SCICODE_THREADS
    # raises the per-process cap if a genuinely heavy step needs it.
    n_threads = os.environ.get("GBENCH_SCICODE_THREADS", "1")
    env = {**os.environ, **{v: n_threads for v in THREAD_VARS}}
    try:
        res = run_sandboxed(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return {"correct": res.returncode == 0, "returncode": res.returncode,
                "stdout_tail": (res.stdout or "")[-1200:],
                "stderr_tail": (res.stderr or "")[-1200:]}
    except Exception as e:  # noqa: BLE001 - timeouts / sandbox errors count as a failed step
        return {"correct": False, "returncode": None, "stdout_tail": "",
                "stderr_tail": f"{type(e).__name__}: {e}"[-1200:]}


def _process_one_problem(
    client, model: str, prob: Dict[str, Any], *,
    temperature: float, max_tokens: int, thinking: bool,
    with_background: bool, h5: str, eval_timeout: int,
) -> Dict[str, Any]:
    """Generate every sub-step sequentially, then score each generated step."""
    sub_steps = prob["sub_steps"]
    n = len(sub_steps)
    deps = prob.get("required_dependencies") or ""
    previous_llm_code: List[str] = [""] * n
    steps_report: List[Dict[str, Any]] = []

    for k in range(1, n + 1):
        step_number = str(sub_steps[k - 1].get("step_number") or f"{prob['problem_id']}.{k}")
        test_cases = list(sub_steps[k - 1].get("test_cases") or [])

        gold = _load_special_gold(step_number)
        if gold is not None:
            # Supplied step: use as context, never generate or score it. Match
            # canonical exactly - the header-named node via get_function_from_code
            # (so problem 13/62 forward a bare __init__, not the full class).
            header = sub_steps[k - 1].get("function_header") or ""
            func_name = _extract_function_name(header)
            previous_llm_code[k - 1] = _get_function_from_code(gold, func_name) or ""
            steps_report.append({
                "step_number": step_number, "special": True,
                "gen_failed": False, "correct": False, "prompt": None,
                "response_text": None, "reasoning": None, "reasoning_chars": 0,
                "finish_reason": None, "completion_tokens": None, "error": None,
            })
            continue

        prompt = _build_step_prompt(prob, k, with_background, previous_llm_code)
        result = _complete(client, model, prompt, temperature, max_tokens, thinking)
        gen_failed = result is None
        response = None if gen_failed else result["text"]
        reasoning = "" if gen_failed else (result.get("reasoning") or "")
        finish_reason = None if gen_failed else result.get("finish_reason")
        completion_tokens = None if gen_failed else result.get("completion_tokens")
        code = _extract_python_script(response)
        previous_llm_code[k - 1] = code

        # Assembled program for this step = deps + prior steps' code + this step.
        prior = "\n".join(previous_llm_code[:k - 1])
        assembled = f"{deps}\n{prior}\n{code}"

        execution = None
        if gen_failed or not test_cases:
            correct = False
        else:
            execution = _score_step(assembled, step_number, test_cases, h5, eval_timeout)
            correct = execution["correct"]

        steps_report.append({
            "step_number": step_number, "special": False,
            "gen_failed": gen_failed, "correct": correct,
            "prompt": prompt, "response_text": response,
            "reasoning": reasoning, "reasoning_chars": len(reasoning),
            "finish_reason": finish_reason, "completion_tokens": completion_tokens,
            "error": "generation_failed" if gen_failed else None,
            "execution": execution,   # None for special/gen-failed/no-test steps
        })

    return {"problem_id": str(prob["problem_id"]),
            "problem_name": prob.get("problem_name"),
            "steps": steps_report}


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def _load_scicode_problems(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load full SciCode problems (all sub-step fields) from HF (SciCode1/SciCode, test)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("SciCode1/SciCode", split="test")
        rows = list(ds)
    except Exception as e:
        logger.error("Failed to load dataset for scicode: %s", e)
        raise RuntimeError(f"Could not load dataset for scicode: {e}") from e
    if not rows:
        raise RuntimeError("Dataset for scicode returned empty rows")

    # Stratified, not a contiguous head (audit RC-1). NOTE: --eval-limit caps PROBLEMS, not scored
    # items -- each problem expands into its dependent sub-steps (which build on each other, so they
    # can't be capped mid-problem), so --eval-limit N generates ~N x sub-steps graded items by design.
    rows = stratified_sample(rows, limit, None, seed="scicode")

    problems: List[Dict[str, Any]] = []
    for item in rows:
        sub_steps = item.get("sub_steps") or []
        if not isinstance(sub_steps, list) or not sub_steps:
            continue
        problems.append({
            "problem_id": str(item.get("problem_id") or "").strip(),
            "problem_name": str(item.get("problem_name") or "Scientific Problem"),
            "required_dependencies": str(item.get("required_dependencies") or ""),
            "sub_steps": sub_steps,
        })
    logger.info("Loaded %d scicode problems.", len(problems))
    return problems


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def run_scicode(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run SciCode with the canonical sequential per-sub-step protocol.

    Gated up front on the reference h5 and the vendored loader's deps so a real
    run never reports a structural 0%. Generation is parallel across problems and
    sequential within each problem; scoring binds `target` from the h5.
    """
    # Executes model-written code: require bubblewrap isolation. Missing/blocked isolation is an
    # EXTERNAL prerequisite gap -> hard-error (no-skip), even when invoked directly (bypassing the
    # central evals.py gate). GBENCH_SANDBOX=none opts into unsandboxed execution.
    _blocked = sandbox_skip_reason()
    if _blocked:
        raise infra_required("scicode", _blocked, DOCS_URL)
    h5, provenance = resolve_test_data()
    if not h5:
        raise infra_required(
            "scicode",
            "SciCode tests evaluate against per-problem reference outputs bound as `target` "
            f"from the benchmark's test_data HDF5, and it could not be fetched from "
            f"{_TEST_DATA_REPO}. Set GBENCH_SCICODE_TEST_DATA to a local copy (the canonical file "
            "is the Google Drive folder linked from the SciCode README). Without it every "
            "test errors and the suite would report a structural 0%",
            DOCS_URL)
    import importlib.util
    _missing = [p for p in ("h5py", "scipy") if importlib.util.find_spec(p) is None]
    if _missing:
        raise infra_required(
            "scicode",
            "SciCode scoring binds each test's `target` from test_data.h5 with a vendored "
            f"reader (no scicode package needed) that requires {', '.join(_missing)}. "
            f"Install: `pip install {' '.join(_missing)}` (lightweight; see docs/evals/scicode.md).",
            DOCS_URL)

    temperature, temp_source = resolve_temperature("scicode", thinking=enable_thinking)
    with_background = str(os.environ.get("GBENCH_SCICODE_WITH_BACKGROUND", "")).strip().lower() \
        in ("1", "true", "yes")
    mt = kwargs.get("max_output_tokens")
    max_tokens = int(mt) if mt else (32768 if enable_thinking else 16384)
    # Canonical uses a 1800s per-step subprocess timeout (test_generated_code.py);
    # a tighter bound would false-fail slow numerical steps. Override via env.
    eval_timeout = int(suite_env("GBENCH_SCICODE_EVAL_TIMEOUT_S", "SCICODE_EVAL_TIMEOUT_S",
                                 default="1800"))
    workers = max(1, int(concurrency))

    problems = _load_scicode_problems(limit=limit)
    if not problems:
        return {
            "benchmark_type": "eval", "eval_name": "scicode", "model_name": model_name,
            "thinking": enable_thinking, "status": "error", "accuracy": 0.0,
            "total_questions": 0, "correct_answers": 0,
        }

    # The global "Concurrency: N (from --batch-sizes)" startup line is NOT this suite's
    # concurrency: scicode is in SANDBOX_EVALS, so --sandboxes overrides it. Log the value
    # actually in effect so the parallelism is unambiguous.
    logger.info("scicode: %d problems | concurrency=%d | sandbox=%s | thinking=%s | "
                "temp=%.2f (%s) | max_tokens=%d/step | with_background=%s",
                len(problems), workers, sandbox_mode(), enable_thinking, temperature,
                temp_source, max_tokens, with_background)

    from openai import OpenAI
    model = model_name[len("openai/"):] if model_name.startswith("openai/") else model_name
    client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "dummy"),
                    timeout=float(suite_env("GBENCH_SCICODE_HTTP_TIMEOUT_S", "SCICODE_HTTP_TIMEOUT_S",
                                            default="900")))

    results: List[Dict[str, Any]] = []
    errored_problems = 0
    with tqdm(total=len(problems), desc="scicode") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _process_one_problem, client, model, prob,
                    temperature=temperature, max_tokens=max_tokens,
                    thinking=enable_thinking, with_background=with_background,
                    h5=h5, eval_timeout=eval_timeout,
                ): prob["problem_id"]
                for prob in problems
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    errored_problems += 1
                    logger.error("scicode: problem %s failed: %s", futures[fut], e)
                pbar.update(1)

    # Aggregate: sub-step accuracy (headline) + problem accuracy. Special steps
    # are excluded from both numerator and denominator (canonical: not generated).
    total_steps = correct_steps = failed_gen = 0
    total_problems = correct_problems = 0
    sample_traces: List[Dict[str, Any]] = []
    idx = 0
    for pr in results:
        total_problems += 1
        prob_all_ok = True
        prob_scored_any = False
        for st in pr["steps"]:
            if st["special"]:
                continue
            total_steps += 1
            prob_scored_any = True
            if st["gen_failed"]:
                failed_gen += 1
            if st["correct"]:
                correct_steps += 1
            else:
                prob_all_ok = False
            sample_traces.append({
                "sample_idx": idx,
                "category": "scientific_python",
                "problem_id": pr["problem_id"],
                "step_number": st["step_number"],
                "gold_answer": st["step_number"],
                "response_text": st["response_text"],
                "reasoning": st.get("reasoning"),           # thinking content
                "reasoning_chars": st.get("reasoning_chars", 0),  # thinking length
                "finish_reason": st.get("finish_reason"),
                "completion_tokens": st.get("completion_tokens"),
                "is_correct": bool(st["correct"]),
                "status": "error" if st["gen_failed"] else "ok",
                "error": st["error"],
            })
            idx += 1
        if prob_scored_any and prob_all_ok:
            correct_problems += 1

    substep_accuracy = round(100.0 * correct_steps / total_steps, 2) if total_steps else 0.0
    problem_accuracy = round(100.0 * correct_problems / total_problems, 2) if total_problems else 0.0

    if total_steps == 0 or errored_problems == len(problems):
        status = "error"
    elif failed_gen or errored_problems:
        status = "completed_with_errors"
    else:
        status = "success"

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "scicode",
        "model_name": model_name,
        "thinking": enable_thinking,
        "status": status,
        # Headline = sub-step accuracy; total/correct kept consistent with it.
        "accuracy": substep_accuracy,
        "total_questions": total_steps,
        "correct_answers": correct_steps,
        # Secondary metrics (persisted to JSON; not surfaced in eval_summary.csv).
        "substep_accuracy": substep_accuracy,
        "problem_accuracy": problem_accuracy,
        "correct_steps": correct_steps,
        "total_steps": total_steps,
        "correct_problems": correct_problems,
        "total_problems": total_problems,
        "failed_generations": failed_gen,
        "errored_problems": errored_problems,
        "with_background": with_background,
        "metric": (f"sub-step accuracy {correct_steps}/{total_steps} (headline); "
                   f"problem accuracy {correct_problems}/{total_problems}"),
        "temperature": temperature,
        "temperature_source": temp_source,
        "sample_traces": sample_traces,
    }
    # Provenance of the reference outputs decides every score, so it is part of the result.
    result["test_data"] = {"path": h5, "source": provenance, "sha256": _digest(h5)}
    return result
