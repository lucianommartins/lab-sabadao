# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: nestful
# Description: NESTFUL (IBM Nested Output-to-Input Function Calling Benchmark)

"""gbench native built-in runner for nestful (Tool Use & Function Calling).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_NESTFUL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import functools
import importlib.util
import json
import logging
import os
import re
import signal
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .fc_common import parse_tool_calls, score_sequence, _json_objects
from .metrics import scorable_traces
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/nestful.md"

#: NESTFUL's executable Win Rate needs the reference function library from the IBM/NESTFUL
#: checkout: `data_v2/executable_functions/` (basic_functions.py + func_file_map.json +
#: ~4348 py_code_file_*.py). Point this at that directory.
_FUNC_DIR_ENV = "GBENCH_NESTFUL_FUNC_DIR"

_REF_TOKEN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")

# --------------------------------------------------------------------------- #
# Canonical NESTFUL prompt (src/instruct_data_prep.py + PROMPTS.json). NESTFUL presents the
# tool specs and few-shot examples IN THE PROMPT TEXT and asks for the WHOLE nested sequence as a
# JSON list; it does NOT use the OpenAI tools/function-calling API. (Offering API tools makes the
# model emit ONE structured tool_call and stop with finish_reason=tool_calls, so it never composes
# the multi-step sequence - measured 2026-09-08: 0% full/partial/win. Verified against
# deepseek_prompt_input + get_icl_str.)
# --------------------------------------------------------------------------- #
_SYSTEM_PREAMBLE = (
    "You are a helpful assistant with access to the following function calls. Your task is to "
    "produce a sequence of function calls necessary to generate response to the user utterance. "
    "Here is a list of functions in JSON format that you can invoke:\n"
)
_FORMAT_INSTRUCTION = (
    "\nDO NOT try to answer the user question, just invoke the tools needed to respond to the "
    "user, if any. The output MUST strictly adhere to the following JSON format: "
    "[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": "
    "\"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure "
    "the parameter type is correct and follow the documentation for parameter format. If no "
    "function call is needed, please directly output an empty list.\nHere are some examples:\n"
)

#: The canonical 3 in-context examples (input + gold output sequence), vendored verbatim from
#: IBM/NESTFUL src/icl_examples.json (run.sh uses --icl_count 3). Only input/output are shown -
#: matching get_icl_str's generic branch, which does not include per-example tools.
_ICL_EXAMPLES = [
    {"input": "In objective test a correct ans score 4 marks and on a wrong ans 2 marks are deducted. a student score 480 marks from 150 question. how many ans were correct?",
     "output": [{"name": "multiply", "label": "$var_1", "arguments": {"arg_0": 150, "arg_1": 2}},
                {"name": "add", "label": "$var_2", "arguments": {"arg_0": 480, "arg_1": "$var_1.result$"}},
                {"name": "add", "label": "$var_3", "arguments": {"arg_0": 4, "arg_1": 2}},
                {"name": "divide", "label": "$var_4", "arguments": {"arg_0": "$var_2.result$", "arg_1": "$var_3.result$"}}]},
    {"input": "A student traveled 10 percent of the distance of the trip alone, continued another 30 miles with a friend, and then finished the last half of the trip alone. How many miles long was the trip?",
     "output": [{"name": "inverse", "label": "$var_1", "arguments": {"arg_0": 10}},
                {"name": "subtract", "label": "$var_2", "arguments": {"arg_0": 1, "arg_1": "$var_1.result$"}},
                {"name": "divide", "label": "$var_3", "arguments": {"arg_0": 1, "arg_1": 2}},
                {"name": "subtract", "label": "$var_4", "arguments": {"arg_0": "$var_2.result$", "arg_1": "$var_3.result$"}},
                {"name": "divide", "label": "$var_5", "arguments": {"arg_0": 30, "arg_1": "$var_4.result$"}}]},
    {"input": "Given a string \"Hello world! How are you?\", extract non-whitespace substrings and count the words from the resulting substrings.",
     "output": [{"name": "non_whitespace_substrings", "arguments": {"input_str": "Hello world! How are you?"}, "label": "$var1"},
                {"name": "count_words_from_sentences", "arguments": {"sentences": "$var1.output_0$"}, "label": "$var2"}]},
]


def _icl_str() -> str:
    """The few-shot block, formatted exactly as NESTFUL's get_icl_str (generic branch)."""
    out = ""
    for idx, ex in enumerate(_ICL_EXAMPLES, 1):
        out += f"\n#Example-{idx}\nInput: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n"
    return out


def _system_prompt(raw_tools: List[Dict[str, Any]]) -> str:
    """Canonical NESTFUL system prompt: preamble + the row's function specs (as JSON) + the
    strict-output-format instruction + the 3 in-context examples."""
    return _SYSTEM_PREAMBLE + json.dumps(raw_tools) + _FORMAT_INSTRUCTION + _icl_str()


# --------------------------------------------------------------------------- #
# Executable Win Rate - faithful port of IBM/NESTFUL src/scorer.py's
# calculate_ans / calculate_win_score. The predicted call SEQUENCE is executed against the
# benchmark's reference function library and the final output compared to `gold_answer`.
# --------------------------------------------------------------------------- #
def _func_dir() -> str:
    """The IBM/NESTFUL `data_v2/executable_functions` directory. Hard-errors if absent - the
    executable Win Rate is a canonical NESTFUL metric and cannot be faked."""
    d = prereqs_path("NESTFUL/data_v2/executable_functions", os.environ.get(_FUNC_DIR_ENV))
    if d:
        d = os.path.abspath(os.path.expanduser(d))
    if not d or not os.path.isfile(os.path.join(d, "basic_functions.py")) \
            or not os.path.isfile(os.path.join(d, "func_file_map.json")):
        raise infra_required(
            "nestful",
            f"the executable Win Rate needs IBM/NESTFUL's reference functions. Set {_FUNC_DIR_ENV} "
            "to the checkout's data_v2/executable_functions directory (contains basic_functions.py, "
            "func_file_map.json and py_code_file_*.py). Clone github.com/IBM/NESTFUL to obtain it.",
            DOCS_URL,
        )
    return d


@functools.lru_cache(maxsize=8)
def _basic_func_list(func_dir: str) -> frozenset:
    """Function names defined in basic_functions.py (scorer.py: lines starting `def `)."""
    names = set()
    with open(os.path.join(func_dir, "basic_functions.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("def "):
                names.add(line.strip().replace("def ", "").split("(", 1)[0])
    return frozenset(names)


@functools.lru_cache(maxsize=8)
def _func_file_map(func_dir: str) -> Dict[str, str]:
    with open(os.path.join(func_dir, "func_file_map.json"), encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _load_func_module(func_dir: str, file_name: str):
    """Import a reference-function file once (scorer.py re-imports per call; caching is safe -
    the reference functions are stateless)."""
    file_path = os.path.join(func_dir, file_name)
    spec = importlib.util.spec_from_file_location(file_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[file_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _listit(t: Any) -> Any:
    return list(map(_listit, t)) if isinstance(t, (list, tuple)) else t


def _calculate_ans(func_calls: List[Dict[str, Any]], spec_lib: List[Dict[str, Any]],
                   func_dir: str) -> Any:
    """Execute the nested call sequence and return the final output (or False on any failure).

    Faithful port of scorer.py::calculate_ans: resolves `$label.field$` references from earlier
    outputs, dispatches each call to basic_functions.py or func_file_map.json, and returns the
    last call's single output value. A 10s wall clock caps runaway reference code (main thread
    only - matches upstream's signal.alarm)."""
    use_alarm = threading.current_thread() is threading.main_thread()
    if use_alarm:
        def _handler(signum, frame):
            raise TimeoutError("Time limit exceeded!")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(10)
    try:
        basic = _basic_func_list(func_dir)
        file_map = _func_file_map(func_dir)
        variable_result_map: Dict[str, Dict[str, Any]] = {}
        for f in func_calls:
            label = str(f["label"]).replace("$", "")
            matches = [s for s in spec_lib if s.get("name") == f.get("name")]
            if not matches:
                return False
            output_params = list((matches[0].get("output_parameters") or {}).keys())
            arg_val_list = []
            for _k, v in (f.get("arguments") or {}).items():
                if isinstance(v, str) and v.startswith("$") and v.endswith("$"):
                    vv = v[1:-1]
                    v_l, out_param = vv.split(".", 1)
                    v = variable_result_map[v_l][out_param]
                elif isinstance(v, str) and v.startswith("$var"):
                    vv = v[1:]
                    v_l, out_param = vv.split(".", 1)
                    v = variable_result_map[v_l][out_param]
                arg_val_list.append(v)
            name = f["name"]
            file_name = "basic_functions.py" if name in basic else file_map[name]
            func = getattr(_load_func_module(func_dir, file_name), name)
            try:
                res = func(*arg_val_list)
            except Exception:
                return False
            if len(output_params) == 1:
                variable_result_map[label] = {output_params[0]: res}
            else:
                return False
        final_var = str(func_calls[-1]["label"]).replace("$", "")
        return next(iter(variable_result_map[final_var].values()))
    except TimeoutError:
        return False
    except Exception:
        return False
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _win_score(pred_calls: List[Dict[str, Any]], gold_answer: Any,
               tools: List[Dict[str, Any]], func_dir: str) -> bool:
    """Faithful port of scorer.py::calculate_win_score: execute the predicted calls and compare
    the final value to gold_answer (float rounded to gold's precision; tuple/list coerced)."""
    if not pred_calls:
        return False
    gold_ans = gold_answer
    if isinstance(gold_ans, str):
        try:
            gold_ans = json.loads(gold_ans)
        except Exception:
            pass
    pred_ans = _calculate_ans(pred_calls, tools, func_dir)
    if isinstance(gold_ans, float) and isinstance(pred_ans, float):
        dec = str(gold_ans).split(".")
        pred_ans = round(pred_ans, len(dec[1])) if len(dec) > 1 else pred_ans
    if pred_ans == gold_ans:
        return True
    if isinstance(pred_ans, tuple) and isinstance(gold_ans, list):
        return _listit(pred_ans) == gold_ans
    return False


def _pred_raw_calls(response_text: str) -> List[Dict[str, Any]]:
    """The model's predicted calls WITH their original labels/arguments, for execution.

    (Win Rate must keep the model's own `$label` names so nested `$label.field$` references
    resolve; the positional normalization used for match-scoring would break execution.)"""
    return [o for o in _json_objects(response_text or "")
            if isinstance(o, dict) and o.get("name") and o.get("arguments") is not None]


def _load_nestful_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load nestful benchmark dataset directly from HF Hub (ibm-research/nestful)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('ibm-research/nestful', split='train')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for nestful: {e}")
        raise RuntimeError(f"Could not load dataset for nestful: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for nestful returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="nestful")

    samples = []
    for item in rows:
        prompt = str(item.get("input") or item.get("prompt") or "").strip()
        # NESTFUL scores the SEQUENCE OF NESTED API CALLS (`output`), not the final value.
        # `gold_answer` is the arithmetic result ("40.0"); scoring the call-matching
        # evaluator against that could only ever return 0, which is what the 2026-08-15
        # sweep measured (0/20 on every row).
        # NESTFUL scores the SEQUENCE OF NESTED API CALLS (`output`), not the final value.
        raw_output = item.get("output")
        if isinstance(raw_output, str):
            try:
                gold_output = json.loads(raw_output.strip())
            except Exception:
                gold_output = None
        else:
            gold_output = raw_output
        if not gold_output:
            continue
        if not isinstance(gold_output, list):
            gold_output = [gold_output]
        cat = "math_planning"

        # The function specs go IN THE PROMPT (canonical): NESTFUL asks the model to compose
        # calls to these specific functions, presented as JSON in the system message - not via
        # the OpenAI tools API (that yields one structured tool_call, not the nested sequence).
        # The raw specs (with output_parameters) are also kept in the gold for the Win Rate.
        raw_tools = item.get("tools")
        if isinstance(raw_tools, str):
            try:
                raw_tools = json.loads(raw_tools)
            except Exception:
                raw_tools = []
        raw_tools = [fn for fn in (raw_tools or []) if isinstance(fn, dict)]

        # Gold carries the whole scoring context: the gold call SEQUENCE (match metrics) plus
        # the numeric gold_answer and raw tool specs (executable Win Rate).
        gold = json.dumps({"output": gold_output,
                           "gold_answer": item.get("gold_answer"),
                           "tools": raw_tools})

        # Canonical prompt: function specs + few-shot examples in the SYSTEM message, the
        # question in the USER message (src/instruct_data_prep.py).
        messages = [{"role": "system", "content": _system_prompt(raw_tools)},
                    {"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} nestful samples.")
    return samples


def _normalize_labels(raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite each call's own label and its argument references to POSITIONAL tokens.

    NESTFUL nesting is by label ("$var_1"), but the label NAME the model picks is arbitrary -
    only the POSITION it refers to matters. Mapping every label to ``$pos{i}`` (by definition
    order) and rewriting "$var_1.result$" -> "$pos0.result$" makes a correct sequence match
    regardless of how gold and prediction named their variables.
    """
    label_map: Dict[str, str] = {}
    for i, c in enumerate(raw_calls):
        lab = str(c.get("label") or "").strip().rstrip("$")
        if lab:
            label_map[lab] = f"$pos{i}"

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return _REF_TOKEN.sub(lambda m: label_map.get(m.group(0), m.group(0)), value)
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        return value

    out = []
    for c in raw_calls:
        args = {k: rewrite(v) for k, v in (c.get("arguments") or c.get("args") or {}).items()}
        out.append({"name": c.get("name", ""), "args": args})
    return out


def _gold_sequence(gold_target: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(gold_target)
    except Exception:
        return []
    # New gold format is {"output": [...], "gold_answer", "tools"}; legacy is a bare list.
    if isinstance(obj, dict) and "output" in obj:
        obj = obj["output"]
    return _normalize_labels(obj if isinstance(obj, list) else [obj])


def _pred_sequence(response_text: str) -> List[Any]:
    """The model's call sequence as (name, args) tuples, with labels normalized positionally.

    Prefers labelled JSON objects (so nested references normalize); falls back to
    parse_tool_calls when the model emitted no labelled JSON.
    """
    labelled = [o for o in _json_objects(response_text or "")
                if isinstance(o, dict) and o.get("name") and o.get("label")]
    if labelled:
        norm = _normalize_labels(labelled)
        return [(c["name"], c["args"]) for c in norm]
    return parse_tool_calls(response_text)


def _eval_nestful(response_text: str, gold_target: str) -> bool:
    """Binary pass = Full Sequence Match (secondary; headline metrics are in run_nestful).

    Old path was a single order-insensitive `all(call_matches(...))` that credited wrong
    call order and tolerated hallucinated extra calls. This is position-aligned over the
    whole (label-normalized) sequence.
    """
    if not response_text or not str(gold_target).strip():
        return False
    gold = _gold_sequence(str(gold_target).strip())
    if not gold:
        return False
    return bool(score_sequence(gold, _pred_sequence(response_text))["full"])


# --------------------------------------------------------------------------- #
# Canonical NESTFUL corpus metrics - faithful port of IBM-research/NESTFUL
# src/scorer.py::calculate_scores + src/utils.py (compute_score_sklearn,
# post_process_api_with_args). gbench already parses the model output and gold into call
# dicts, so scorer.py's model-specific JSON-string parsers are skipped; the SCORING MATH is
# verbatim so the numbers are leaderboard-comparable:
#   F1 Intent / F1 Slot = sklearn MACRO f1 over MultiLabelBinarizer(fit on gold) labels
#   Partial Match       = mean per-example sklearn accuracy_score over name-aligned f_name(args)
#   Full Match          = fraction of examples whose Partial Match == 1.0
# Do NOT "improve" this to micro/pooled averaging - that was the pre-2026-09 gbench behaviour
# and it does not match the published NESTFUL leaderboard.
# --------------------------------------------------------------------------- #
def _ibm_dollar_fix(val: Any) -> Any:
    """scorer.py repair: a pred arg like `$var1.x` (missing its closing `$`) becomes `$var1.x$`."""
    if isinstance(val, str) and val.startswith("$") and not val.endswith("$"):
        return val + "$"
    return val


_IBM_LABEL_UNDERSCORE = re.compile(r"\$var_(\d+)")


def _ibm_canon_label(val: Any) -> Any:
    """Canonicalize the variable-label spelling inside a `$ref`. The NESTFUL gold is INTERNALLY
    inconsistent ('$var_1' in most traces, '$var1' in others) while the prompt MANDATES '$var_1', so
    a compliant, functionally-correct answer literal-mismatches the gold and is a false full-match
    negative (WS10 audit). Drop the underscore ('$var_1.output_0$' -> '$var1.output_0$') on BOTH gold
    and pred so a reference compares by position, not by the arbitrary label spelling. True sequence
    mismatches still differ."""
    if isinstance(val, str) and "$var_" in val:
        return _IBM_LABEL_UNDERSCORE.sub(r"$var\1", val)
    return val


def _ibm_is_dummy(f: Dict[str, Any]) -> bool:
    return f.get("name") == "dummy" and (f.get("arguments") or {}) == {}


def _ibm_intent_names(calls: List[Dict[str, Any]], skip_dummy: bool) -> List[str]:
    """Per-example API-name list (scorer.py gold/pred intent). Pred skips the dummy call."""
    names = []
    for f in calls:
        if not isinstance(f, dict) or "name" not in f:
            continue
        if skip_dummy and _ibm_is_dummy(f):
            continue
        names.append(str(f["name"]))
    return names


def _ibm_api_map(calls: List[Dict[str, Any]], fix_dollar: bool) -> Dict[str, List[str]]:
    """scorer.py slot map: {api_name: ['arg = val', ...]}, last occurrence of a name wins."""
    api_map: Dict[str, List[str]] = {}
    for f in calls:
        if not isinstance(f, dict) or "name" not in f:
            continue
        if fix_dollar and _ibm_is_dummy(f):
            continue
        name = f["name"]
        api_map[name] = []  # upstream re-initialises per occurrence -> last wins
        for arg, val in (f.get("arguments") or {}).items():
            if fix_dollar:
                val = _ibm_dollar_fix(val)
            val = _ibm_canon_label(val)   # $var_N <-> $varN (inconsistent gold vs prompt-mandated)
            api_map[name].append(f"{arg} = {val}")
    return api_map


def _ibm_api_with_args(calls: List[Dict[str, Any]], fix_dollar: bool) -> List[str]:
    """scorer.py `api_with_args`: `f_name(sorted 'key = val')` per call (pred repairs `$refs`)."""
    out = []
    for f in calls:
        if not isinstance(f, dict) or "name" not in f:
            continue
        f_name = str(f["name"])
        try:
            parts = []
            for key, val in (f.get("arguments") or {}).items():
                if fix_dollar:
                    val = _ibm_dollar_fix(val)
                val = _ibm_canon_label(val)   # $var_N <-> $varN (inconsistent gold vs prompt-mandated)
                parts.append(f"{key} = {val}")
            args = ", ".join(sorted(parts))
        except Exception:                                      # noqa: BLE001 - upstream falls to {}
            args = "{}"
        out.append(f"{f_name}({args})")
    return out


def _ibm_post_process(api_with_args_gold: List[str],
                      api_with_args_pred: List[str]) -> Tuple[List[str], List[str]]:
    """Verbatim port of utils.post_process_api_with_args: name-align unequal-length call lists."""
    def align_lists(list1, list2):
        aligned1, aligned2, i, j = [], [], 0, 0
        while i < len(list1) or j < len(list2):
            if i < len(list1) and j < len(list2) and list1[i] == list2[j]:
                aligned1.append(list1[i]); aligned2.append(list2[j]); i += 1; j += 1
            elif i < len(list1):
                aligned1.append(list1[i]); aligned2.append(""); i += 1
            else:
                aligned1.append(""); aligned2.append(list2[j]); j += 1
        return aligned1, aligned2

    names_gold = [api.split("(", 1)[0] for api in api_with_args_gold]
    names_pred = [api.split("(", 1)[0] for api in api_with_args_pred]
    if len(names_gold) == len(names_pred):
        return api_with_args_gold, api_with_args_pred
    try:
        names_gold, names_pred = align_lists(names_gold, names_pred)
        g, p = list(api_with_args_gold), list(api_with_args_pred)
        upd_gold = ["" if n == "" else g.pop(0) for n in names_gold]
        upd_pred = ["" if n == "" else p.pop(0) for n in names_pred]
        return upd_gold, upd_pred
    except Exception:                                          # noqa: BLE001
        return api_with_args_gold, api_with_args_pred


def _ibm_macro_f1(gold_output: List[List[str]], pred_output: List[List[str]]) -> float:
    """Verbatim port of utils.compute_score_sklearn's macro F1 (binarizer fit on gold)."""
    if not gold_output:
        return 0.0
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer
    b = MultiLabelBinarizer()
    b.fit(gold_output)
    g = b.transform(gold_output)
    p = b.transform(pred_output)
    return float(f1_score(g, p, average="macro", zero_division=0))


def _ibm_corpus_scores(pred_per_ex: List[List[Dict[str, Any]]],
                       gold_per_ex: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Corpus F1 Intent / F1 Slot / Partial Match / Full Match over all examples."""
    from sklearn.metrics import accuracy_score
    gold_intent, pred_intent, gold_slot, pred_slot = [], [], [], []
    accs, full = [], 0
    for pred_calls, gold_calls in zip(pred_per_ex, gold_per_ex):
        gold_intent.append(_ibm_intent_names(gold_calls, skip_dummy=False))
        pred_intent.append(_ibm_intent_names(pred_calls, skip_dummy=True))
        pmap = _ibm_api_map(pred_calls, fix_dollar=True)
        gmap = _ibm_api_map(gold_calls, fix_dollar=False)
        for key in set(pmap).union(gmap):
            pred_slot.append(pmap.get(key, []))
            gold_slot.append(gmap.get(key, []))
        g_awa = _ibm_api_with_args(gold_calls, fix_dollar=False)
        p_awa = _ibm_api_with_args(pred_calls, fix_dollar=True)
        g_awa, p_awa = _ibm_post_process(g_awa, p_awa)
        try:
            acc = float(accuracy_score(g_awa, p_awa))
        except Exception:                                      # noqa: BLE001 - empty -> 0.0
            acc = 0.0
        accs.append(acc)
        if acc == 1:
            full += 1
    n = len(accs)
    return {
        "f1_intent": _ibm_macro_f1(gold_intent, pred_intent),
        "f1_slot": _ibm_macro_f1(gold_slot, pred_slot),
        "partial_match_accuracy": (sum(accs) / n) if n else 0.0,
        "full_match_accuracy": (full / n) if n else 0.0,
    }


def run_nestful(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute nestful; headline = Full Sequence Match, with Partial-SM, F1-Func/F1-Param and the
    executable Win Rate (final-output accuracy)."""
    # Fail fast if the reference-function library for the executable Win Rate is missing.
    func_dir = _func_dir()
    samples = _load_nestful_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="nestful",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_nestful,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # Canonical NESTFUL (IBM-research/NESTFUL scorer.py): F1 Intent + F1 Slot (sklearn macro) +
    # Partial/Full Match Accuracy over the whole corpus, plus the executable Win Rate (run the
    # composed calls, compare the final value to gold_answer). See _ibm_corpus_scores.
    pred_per_ex: List[List[Dict[str, Any]]] = []
    gold_per_ex: List[List[Dict[str, Any]]] = []
    win_ok = 0
    n = 0
    for t in scorable_traces(result):
        gold_str = str(t.get("gold_answer") or "").strip()
        gold_seq = _gold_sequence(gold_str)
        if not gold_seq:
            continue
        try:
            gspec = json.loads(gold_str)
        except Exception:
            gspec = {}
        # Raw gold calls (name/arguments keys, as scorer.py expects) - NOT _normalize_labels'
        # positional (name, args) form. Modern HF gold is {"output": [...]}; legacy is a bare list.
        if isinstance(gspec, dict):
            gold_calls = gspec.get("output") or []
        elif isinstance(gspec, list):
            gold_calls = gspec
        else:
            gold_calls = []
        if not isinstance(gold_calls, list):
            gold_calls = []
        resp = t.get("response_text") or ""
        pred_calls = _pred_raw_calls(resp)
        pred_per_ex.append(pred_calls)
        gold_per_ex.append(gold_calls)
        n += 1
        # Executable Win Rate: keep the model's original labels/arguments and run the sequence.
        win = _win_score(pred_calls, gspec.get("gold_answer"),
                         gspec.get("tools") or [], func_dir)
        t["nestful_win"] = bool(win)
        win_ok += 1 if win else 0

    scores = _ibm_corpus_scores(pred_per_ex, gold_per_ex)
    result["f1_intent"] = round(scores["f1_intent"] * 100.0, 2)
    result["f1_slot"] = round(scores["f1_slot"] * 100.0, 2)
    result["partial_match_accuracy"] = round(scores["partial_match_accuracy"] * 100.0, 2)
    result["full_match_accuracy"] = round(scores["full_match_accuracy"] * 100.0, 2)
    result["win_rate"] = round(win_ok / n * 100.0, 2) if n else 0.0
    result["metric"] = ("Full Match Accuracy (canonical NESTFUL headline, IBM scorer.py); "
                        "F1 Intent / F1 Slot (sklearn macro) / Partial Match Accuracy and the "
                        "executable Win Rate (final-output accuracy, computed by running the "
                        "composed calls against IBM-research/NESTFUL's reference functions) also "
                        "reported.")
    if n:
        result["accuracy"] = result["full_match_accuracy"]
    return result
