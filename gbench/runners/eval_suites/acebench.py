# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: acebench
# Description: ACEBench (arXiv:2501.12851) - tool-use function calling, LLM-free AST scoring

"""gbench native built-in runner for acebench (Tool Use & Function Calling).

Canonical ACEBench (chenchen0103/ACEBench). Loads the real benchmark via the
MIT-licensed HF reformatting `oliveirabruno01/acebench` (config 'en'), presents
each task with the official ACEBench system prompt (bracket-DSL output), and
scores deterministically by porting the official model_eval/checker.py:
  - normal  : AST parse of [ApiName(k=v,...)] + type/value match vs ground_truth
  - special : the model must emit the exact ACEBench error declaration
The 'agent' category needs a stateful simulated environment + user-simulator LLM
and is intentionally gated (raises) rather than mis-scored.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_ACEBENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"

_REPO = "oliveirabruno01/acebench"

# ---- Official ACEBench system prompts (model_inference/prompt_en.py, verbatim) ----
_SYS_NORMAL = """You are an AI assistant with the role name "assistant." Based on the provided API specifications and conversation history from steps 1 to t, generate the API requests that the assistant should call in step t+1. The API requests should be output in the format [ApiName(key1='value1', key2='value2', ...)], replacing ApiName with the actual API name, key1, key2, etc., with the actual parameter names, and value1, value2, etc., with the actual parameter values. The output should start with a square bracket "[" and end with a square bracket "]".
If there are multiple API requests, separate them with commas, for example: [ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...]. Do not include any other explanations, prompts, or API call results in the output.
If the API parameter description does not specify otherwise, the parameter is optional (parameters mentioned in the user input need to be included in the output; if not mentioned, they do not need to be included).
If the API parameter description does not specify the required format for the value, use the user's original text for the parameter value.
If the API requires no parameters, output the API request directly in the format [ApiName()], and do not invent any nonexistent parameter names.

{time}

Role Descriptions:
user: User
assistant: The AI assistant role that makes API requests
tool: Provides the results returned from tool calls

API Specifications:
{function}"""

_SYS_PREFERENCE = """You are an AI assistant, and your role is called assistant. Based on the given API description, dialogue history 1..t, and character profile, generate the API requests that the assistant should call in step t+1. The API requests should be output in the format [ApiName(key1='value1', key2='value2', ...)], where ApiName is replaced with the actual API name, and key1, key2, etc., are replaced with the actual parameter names, and value1, value2 are replaced with the actual parameter values. The output should start with a "[" and end with a "]".
If there are multiple API requests, they should be separated by commas, e.g., [ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...]. Do not output any other explanations, hints, or results of the API calls in the output.
If the API parameter description does not specify special instructions, the parameter is optional (parameters mentioned in the user input or character profile should be included in the output, and if not mentioned, they should not be included).
If the API parameter description does not specify the format for the parameter value, the parameter value should be taken from the user's original text or character profile.
If the API requires no parameters, the API request should be output as [ApiName()], with no fabricated parameter names.

Character Profile:
{profile}

Role Description:
user: User
assistant: AI assistant performing API calls
tool: Provides the results of tool calls

API Description:
{function}"""

_SYS_SPECIAL = """You are an AI assistant with the role name "assistant". Based on the provided API specifications and conversation history from steps 1 to t, generate the API requests that the assistant should call in step t+1. Below are two specific scenarios:
1. When the information provided by the user is clear and unambiguous, and the problem can be resolved using the list of candidate functions:
   - If the API parameter description does not specify the required format for the value, use the user's original text for the parameter value.
   - When multiple tools in the candidate list can satisfy the user's needs, output all API requests.
   - API requests should be output in the format [ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...], replacing ApiName with the actual API name, key1, key2, etc., with the actual parameter names, and value1, value2, etc., with the actual parameter values. The output should start with a square bracket "[" and end with a square bracket "]". At this time, the output must not contain any other content.

2. When the information provided by the user is unclear, incomplete, or incorrect, or the user's question exceeds the capabilities of the provided functions, you need to clearly point out these issues. The following is your strategy:
   (1) If the user's instructions include the key details required to call the API, but the type or form of the parameter values does not match the API's definitions, ask in-depth questions to clarify and correct the details. The output format should be: ["There is incorrect value (value) for the parameters (key) in the conversation history."]
   (2) If the user's instructions lack the key details required by the API, ask questions to obtain the necessary information. The output format should be: ["Missing necessary parameters (key1, key2, ...) for the api (ApiName)"], replacing key1, key2 with the names of the missing parameters and ApiName with the actual API name.
   (3) If the user's request exceeds the current capabilities of your APIs, inform them that you cannot fulfill the request. The output format should be: ["Due to the limitations of the function, I cannot solve this problem."]
   Note: The above steps have a priority order. You need to first determine whether scenario (1) applies. If it does, output according to the requirements in (1). Pay attention to distinguishing between scenarios (1) and (2).

{time}

Role Descriptions:
user: User
assistant: The AI assistant role that makes API requests

API Specifications:
{function}"""

_USER_PROMPT = "Conversation history 1..t:\n{question}"

# ---- Ported scoring helpers (model_eval/utils.py + checker.py) ----
PYTHON_TYPE_MAPPING = {
    "string": str, "integer": int, "float": float, "boolean": bool, "array": list,
    "tuple": list, "dict": dict, "any": str, "list": list, "list(string)": list,
    "list(enum)": list, "int": int, "enum": str, "number": int, "object": dict,
    "objectArray": list,
}
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple", "list(string)", "list(enum)", "object", "objectArray"]


def _standardize_string(s: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", str(s)).lower().replace("'", '"')


def _possible_answer_type(pa: Any) -> Any:
    return type(pa) if pa != "" else None


def _find_description(func_descriptions: Any, name: str) -> Any:
    if isinstance(func_descriptions, list):
        for fd in func_descriptions:
            if fd["name"] in name:
                return fd
        return None
    return func_descriptions


def _string_checker(value: str, possible_answer: str) -> bool:
    return _standardize_string(possible_answer) in _standardize_string(value)


def _list_checker(value: list, possible_answer: list) -> bool:
    if not isinstance(value, list) or len(value) != len(possible_answer):
        return False
    for v, pa in zip(value, possible_answer):
        if isinstance(pa, str):
            if _standardize_string(v) != _standardize_string(pa):
                return False
        elif isinstance(pa, dict):
            if not _dict_checker(v, pa):
                return False
        elif isinstance(pa, list):
            if not _list_checker(v, pa):
                return False
        elif v != pa:
            return False
    return True


def _dict_checker(value: Any, possible_answer: dict) -> bool:
    if not isinstance(value, dict):
        return False
    if len(value.keys()) != len(possible_answer.keys()):
        return False
    for k, v in value.items():
        if v == "true":
            v = True
        if v == "false":
            v = False
        if k not in possible_answer:
            return False
        expected = possible_answer[k]
        if isinstance(expected, dict):
            if not _dict_checker(v, expected):
                return False
        else:
            if isinstance(expected, str):
                if _standardize_string(expected) not in _standardize_string(v):
                    return False
            elif str(expected) not in str(v):
                return False
    return True


def _value_checker(value: Any, possible_answer: Any, expected_type: str) -> bool:
    """Type-aware value match mirroring simple_function_checker's non-variable branch."""
    conv = PYTHON_TYPE_MAPPING.get(expected_type, str)
    # possible answer given as a different-typed literal => treat as variable/acceptable
    pa_type = _possible_answer_type(possible_answer)
    if value == "true":
        value = True
    if value == "false":
        value = False
    if conv == dict:
        return _dict_checker(value, possible_answer)
    if conv == list:
        if isinstance(possible_answer, list) and possible_answer and isinstance(possible_answer[0], dict):
            if not isinstance(value, list) or len(value) != len(possible_answer):
                return False
            return all(_dict_checker(v, pa) for v, pa in zip(value, possible_answer))
        return _list_checker(value, possible_answer)
    if conv == str:
        return _string_checker(value, possible_answer)
    # numeric / bool: allow int->float, else direct equality (or pool membership)
    if conv == float and isinstance(value, int):
        value = float(value)
    if isinstance(possible_answer, list):
        return value in possible_answer
    return value == possible_answer or (pa_type is not None and type(value) == pa_type and value == possible_answer)


def _simple_function_checker(func_description: dict, call: dict, possible_answer: dict) -> bool:
    func_name = func_description["name"]
    if func_name not in call:
        return False
    model_params = call[func_name]
    pa_params = list(possible_answer.values())[0]
    props = func_description.get("parameters", {}).get("properties", {})
    required = func_description.get("parameters", {}).get("required", [])
    if not model_params and not props:
        return True
    for p in required:
        if p not in model_params:
            return False
    for p, v in model_params.items():
        if p not in props or p not in pa_params:
            return False
        expected_type = props[p]["type"]
        if not _value_checker(v, pa_params[p], expected_type):
            return False
    return True


def _normal_checker(func_descriptions: list, model_output: list, possible_answers: dict) -> bool:
    if not isinstance(model_output, list) or len(model_output) != len(possible_answers):
        return False
    # function name/count set match (strip _N suffix on duplicate keys)
    pa_list = [{re.sub(r"_\d+$", "", k): v} for d in
               [{k: v} for k, v in possible_answers.items()] for k, v in d.items()]
    out_counts = Counter(list(c.keys())[0] for c in model_output)
    ans_counts = Counter(list(c.keys())[0] for c in pa_list)
    if out_counts != ans_counts:
        return False
    func_names = [list(d.keys())[0] for d in pa_list]
    for i, pa in enumerate(pa_list):
        fd = _find_description(func_descriptions, func_names[i])
        if fd is None:
            return False
        matched = False
        for call in model_output:
            if list(call.keys())[0] == func_names[i]:
                if _simple_function_checker(fd, call, pa):
                    matched = True
                    break
        if not matched:
            return False
    return True


def _get_lose_param(text: str) -> Tuple[Optional[str], List[str]]:
    params_match = re.search(r"\((.*?)\)", text)
    api_match = re.findall(r"\(.*?\)", text)
    if params_match and len(api_match) >= 2:
        params = [p.strip() for p in params_match.group(1).split(",")]
        api_name = api_match[1][1:-1]
        return api_name, params
    return None, []


def _special_checker(resp: str, ground_truth: Any, sub_category: str) -> bool:
    s = _standardize_string(resp)
    if sub_category == "data_special_irrelevant":
        return _standardize_string(ground_truth) in s
    if sub_category == "data_special_error_param":
        # ground_truth is {param_key: [incorrect_value, ...]}; the model must identify the
        # incorrect VALUE ("There is incorrect value (1234WrongID) for the parameters ..."),
        # per the official checker. The old code tested the parameter KEY instead of the
        # value, so a response naming the right parameter but the WRONG value still passed.
        values: List[Any] = []
        if isinstance(ground_truth, dict):
            for v in ground_truth.values():
                values.extend(v if isinstance(v, list) else [v])
        return "incorrectvalue" in s and all(_standardize_string(str(v)) in s for v in values)
    if sub_category == "data_special_incomplete":
        if not isinstance(ground_truth, dict):
            return False
        api = list(ground_truth.keys())[0]
        params = [p.strip() for p in list(ground_truth.values())[0]]
        ok = ("missingnecessaryparameters" in s or "missing" in s)
        ok = ok and _standardize_string(api) in s
        return ok and all(_standardize_string(p) in s for p in params)
    return False


def _parse_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse the ACEBench bracket DSL [ApiName(k=v,...), ...] -> [{name:{param:value}}]."""
    if not text:
        return None
    t = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    if "[" not in t or "]" not in t:
        return None
    frag = t[t.find("["): t.rfind("]") + 1]
    try:
        tree = ast.parse(frag, mode="eval")
    except Exception:
        return None
    if not isinstance(tree.body, ast.List):
        return None
    calls: List[Dict[str, Any]] = []
    for el in tree.body.elts:
        if not isinstance(el, ast.Call) or not isinstance(el.func, ast.Name):
            return None
        params: Dict[str, Any] = {}
        for kw in el.keywords:
            if kw.arg is None:
                continue
            try:
                params[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                if isinstance(kw.value, ast.Name) and kw.value.id in ("true", "false", "null", "none"):
                    params[kw.arg] = {"true": True, "false": False}.get(kw.value.id)
                else:
                    return None
        calls.append({el.func.id: params})
    return calls


def _eval_acebench(response_text: str, gold: str) -> bool:
    try:
        g = json.loads(gold)
    except Exception:
        return False
    kind = g.get("kind")
    if kind == "special":
        return _special_checker(response_text or "", g["ground_truth"], g["sub"])
    # normal
    calls = _parse_calls(response_text or "")
    if calls is None:
        return False
    try:
        return _normal_checker(g["functions"], calls, g["ground_truth"])
    except Exception:
        return False


def _load_acebench_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load ACEBench (normal+special) from HF Hub; raises on load/schema failure."""
    categories = categories or ["normal", "special"]
    if "agent" in categories:
        raise NotImplementedError(
            "acebench 'agent' category needs a stateful simulated environment + "
            "user-simulator LLM; not supported. Use categories=['normal','special']."
        )
    try:
        from datasets import load_dataset
        splits = {c: list(load_dataset(_REPO, "en", split=c)) for c in categories}
    except Exception as e:
        logger.error(f"Failed to load dataset for acebench: {e}")
        raise RuntimeError(f"Could not load dataset for acebench: {e}") from e

    samples = []
    for cat, rows in splits.items():
        if not rows:
            raise RuntimeError(f"acebench '{cat}' returned empty rows")
        for item in rows:
            question = item.get("question")
            functions = json.loads(item.get("function") or "[]")
            rubric = json.loads(item.get("rubric") or "{}")
            sub = item.get("sub_category") or cat
            gt = rubric.get("ground_truth")
            if not question or gt is None:
                raise RuntimeError(
                    "acebench: unexpected dataset schema (missing question/ground_truth); "
                    "refusing to fabricate sample data"
                )
            if cat == "special":
                sys_prompt = _SYS_SPECIAL.format(
                    time=item.get("time") or "", function=json.dumps(functions)
                )
                gold = json.dumps({"kind": "special", "ground_truth": gt, "sub": sub})
            else:
                if sub == "data_normal_preference":
                    sys_prompt = _SYS_PREFERENCE.format(
                        profile=item.get("profile") or "", function=json.dumps(functions)
                    )
                else:
                    sys_prompt = _SYS_NORMAL.format(
                        time=item.get("time") or "", function=json.dumps(functions)
                    )
                gold = json.dumps({"kind": "normal", "ground_truth": gt, "functions": functions})
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": _USER_PROMPT.format(question=question)},
            ]
            samples.append((messages, gold, {"category": cat, "sub_category": sub}))

    # Stratified on each sample's own category, not a contiguous head (RC-1).
    samples = stratified_sample(
        samples, limit, lambda s: (s[2] or {}).get("category"), seed="acebench")
    logger.info(f"Loaded {len(samples)} acebench samples ({'+'.join(categories)}).")
    return samples


def run_acebench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute acebench native built-in evaluation suite (ACEBench-en Normal+Special subset).

    Scope: only the English config and only the normal+special categories run (the entire agent
    third of ACEBench and the whole Chinese half are excluded, and the normal is_variable checker
    branch is un-ported), so the headline is a lower bound, not the full ACEBench leaderboard number.
    """
    samples = _load_acebench_samples(
        enable_thinking=enable_thinking, limit=kwargs.get("limit")
    )
    result = run_eval_suite(
        eval_name="acebench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_acebench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "ACEBench-en Normal+Special subset only: agent category and Chinese half excluded, "
        "normal is_variable branch un-ported (lower bound, not the full ACEBench Overall)")
    return result
