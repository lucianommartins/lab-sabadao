# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: complexfuncbench
# Description: ComplexFuncBench (THUDM) - multi-step, multi-constraint tool/function calling

"""gbench native built-in runner for complexfuncbench (Tool Use & Function Calling).

Canonical ComplexFuncBench (THUDM/ComplexFuncBench, arXiv:2501.10132) is NOT an
answer-matching dataset: each of the 1000 samples is a multi-STEP agentic tool-calling
episode (avg ~3, up to ~19 sequential calls, often several PARALLEL calls per step) in
which the model is driven through a conversation while the harness feeds back the
**recorded** observations for the calls it gets right (the "Golden Function Call List
Updating" strategy). Scoring is `ComplexEval`:

  * **Success Rate** - fraction of samples where the model completes the whole golden call
    chain and then stops at the right moment. HEADLINE `accuracy`.
  * **Call Accuracy** - correct calls / total golden calls, pooled.
  * **Completeness** / **Correctness** - a judge scores the final natural-language answer
    (0/1/2) for coverage of the request and factual consistency with the observations.

A predicted call matches a golden call through a 4-tier cascade, ported VERBATIM from
upstream (`utils/compare_method.py`): (1) rule-based exact match, (2) a `value_checker`
over the per-function critical parameters (`utils/exact_match_values.json`, vendored),
(3) an OPTIONAL response-based tie-breaker (live RapidAPI - used only if `RAPID_API_KEY`
is set, otherwise it degrades to the judge), and (4) an LLM equivalence judge. Multiple
parallel calls in one step are aligned to the golden calls with bge-large-en-v1.5
embeddings + a max-weight assignment before comparison.

gbench substitutes its ESTABLISHED Gemini cascade (base.judge_generate_cascade) for the
two GPT-4o judge roles upstream uses (the call-equivalence judge + the response
completeness/correctness judge). Upstream prompts are ported verbatim; only the judge
model changes. The embedding model is reproduced faithfully via `transformers.AutoModel`
(CLS pooling + L2 normalisation) so no `FlagEmbedding`/`sentence-transformers`/`scipy`
dependency is added to the environment; a pure-Python max-weight assignment replaces
`scipy.optimize.linear_sum_assignment` (the per-step matrices are tiny).

Grading uses gbench's standard Gemini cascade by convention (the published leaderboard is
GPT-4o-graded), so a run here is a gbench-internal number rather than a like-for-like
leaderboard entry and is reported with `leaderboard_comparable=False`.

HARD-ERRORS (`infra_required`, never skips, never a fabricated number) if the dataset
cannot be loaded, `GEMINI_API_KEY` (the judges) is unset, or no embedding backend is
available. See docs/evals/complexfuncbench.md.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 with `--thinking`;
the measurement is in base.DEFAULT_TEMPERATURE. Override with `--temperature` or
`GBENCH_COMPLEXFUNCBENCH_TEMPERATURE`. The model driver posts to the gbench endpoint with
native tool-calling (the served model must support tool_calls). The Gemini judges are
pinned at 0.0.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

from .base import resolve_temperature, judge_generate_cascade, gemini_key_live_valid
from .sampling import stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/complexfuncbench.md"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "_complexfuncbench_data")
_EXACT_MATCH_PATH = os.path.join(_DATA_DIR, "exact_match_values.json")

_HF_REPO = "THUDM/ComplexFuncBench"
_HF_FILE = "ComplexFuncBench.jsonl"

#: Location lookups etc. are OPTIONAL golden calls (upstream `free_function_list`): the model
#: may or may not issue them, so they never count for/against it and are auto-consumed.
_FREE_FUNCTION_LIST = [
    "Location_to_Lat_Long", "Search_Hotel_Destination", "Search_Attraction_Location",
    "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location",
]

_EMBED_MODEL = os.environ.get("GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
_EMBED_DEVICE = os.environ.get("GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE", "cpu")
_MAX_ROUNDS = int(os.environ.get("GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS", "40"))
_MODEL_MAX_TOKENS = int(os.environ.get("GBENCH_COMPLEXFUNCBENCH_MAX_TOKENS", "2048"))
_REQUEST_TIMEOUT_S = int(os.environ.get("GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S", "600"))

# Upstream normalises Success Rate per domain (150 each; Cross=400; 1000 overall).
_DOMAIN_DENOM = {"Car-Rental": 150, "Hotels": 150, "Attraction": 150, "Flights": 150, "Cross": 400}
_FULL_TOTAL = 1000

_UNEXPECT_CALL_RESP = {
    "api_status": True,
    "content": "There is a problem with your api call, please double-check for possible problems.",
}


# --------------------------------------------------------------------------- #
# Judge prompts - ported VERBATIM from upstream prompts/compare.py + prompts/response.py.
# Only an explicit output-anchor line is appended so the free-text Gemini judge is reliably
# parseable (upstream forced JSON from GPT-4o; we parse the same fields from text/JSON).
# --------------------------------------------------------------------------- #
_COMPARE_SYSTEM = (
    "You are an assistant for function call comparison. Your task is to determine whether "
    "two function calls are equivalent based on the conversation history and the function "
    "descriptions, and provide specific reasons.\n"
    "# Instructions:\n"
    "You need to determine whether two function calls are equivalent based on the following "
    "criteria:\n"
    "1. The same parameter can be expressed in different languages. For example: `America`, "
    "`美国` and `アメリカ` are equivalent.\n"
    "2. The same parameter can be expressed in different forms as long as the meaning is the "
    "same. For example, `New York` and `NY` are equivalent, `Shanghai` and `Shanghai City` "
    "are equivalent. `Narita International Airport` and `Tokyo Narita International Airport` "
    "are equivalent. \n"
    "3. A location with or without a country suffix is considered equivalent. For example, "
    "Van Gogh Museum, Amsterdam and Van Gogh Museum are equivalent.\n"
    "4. The order of parameters in a function call can differ, for example: `add(1, 2)` and "
    "`add(2, 1)` are equivalent.\n"
    "5. If a parameter in the function description has a default value, and the current "
    "parameter value is equal to that default value, then the parameter can be omitted in the "
    "function call. For example, if the adults parameter in the Search_Hotels function has a "
    "default value of 1, then the following two function calls are equivalent: "
    "`Search_Hotels(New_York, adults=1)` and `Search_Hotels(New_York)`.\n\n"
    "# Output:\n"
    "You need to output the result in JSON format, containing the following fields:\n"
    "- **is_equal**: A boolean indicating whether the two function calls are equivalent.\n"
    "- **reason**: Please provide the reason for your judgment.\n"
)
_COMPARE_ANCHOR = "\n\nEnd your reply with a single line exactly: `is_equal: <true|false>`"

_COMPLETE_SYSTEM = (
    "You are a helpful response completeness detect assistant. Your task is to evaluate the "
    "response based on whether it fully addresses all aspects of the user's query. \n"
    "# Your Task\n"
    "For each user query and corresponding response, you should determine the completeness of "
    "the response using the following criteria:\n"
    "- If the response covers all requested information and addresses all parts of the user's "
    "query, it should be considered complete and receive a score of 2.\n"
    "- If the response addresses some but not all parts of the user's query, it should be "
    "considered partial and receive a score of 1.\n"
    "- If the response does not address any of the requested information in the user's query, "
    "it should be considered incomplete and receive a score of 0.\n\n"
    "# Output Format\n"
    "You should output the score for each user query and corresponding response in JSON format "
    "with following keys:\n"
    "- score: the completeness score for the response (0, 1, or 2)\n"
    "- reason: a string describing the reason for the score\n"
)
_CORRECT_SYSTEM = (
    "You are a helpful response correctness detect assistant. Your task is to evaluate the "
    "response based on its accuracy in matching the details provided by API response.\n"
    "# Your task\n"
    "Give a dialogue history containing user query, function calls and api responses, you "
    "should determine the correctness of the corresponding respons using the following "
    "criteria:\n"
    "- If the response is consistent with the information provided in the API response, it "
    "should be considered entirely correct and receive a score of 2.\n"
    "- If the response partially matches the information provided in the API response (with "
    "some correct and some incorrect details), it should be considered partially correct and "
    "receive a score of 1.\n"
    "- If the response does not match any of the information provided in the API response, it "
    "should be considered incorrect and receive a score of 0.\n\n"
    "# Output Format\n"
    "You should output the score in JSON format with following keys:\n"
    "- score: the correctness score for the response (0, 1, or 2)\n"
    "- reason: a string describing the reason for the score\n"
)
_SCORE_ANCHOR = "\n\nEnd your reply with a single line exactly: `score: <0|1|2>`"


# --------------------------------------------------------------------------- #
# Free-text judge parsing (anchored line first, then JSON, then a scan).
# --------------------------------------------------------------------------- #
def _decode_json_loose(text: str) -> Optional[Dict[str, Any]]:
    """Upstream utils.decode_json, tolerant of ```JSON fences and Python bools."""
    if not text:
        return None
    s = text.strip().strip("`")
    for fence in ("JSON", "json"):
        s = s.replace(fence, "")
    s = s.replace("True", "true").replace("False", "false")
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0).replace("\n", " "))
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def _parse_is_equal(text: Optional[str]) -> Optional[bool]:
    if not text:
        return None
    m = re.search(r"is_equal\s*[:=]\s*[`\"']?\s*(true|false)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    obj = _decode_json_loose(text)
    if obj is not None and "is_equal" in obj:
        return bool(obj["is_equal"])
    return None


def _parse_score(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"score\s*[:=]\s*[`\"']?\s*([012])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    obj = _decode_json_loose(text)
    if obj is not None and obj.get("score") in (0, 1, 2):
        return int(obj["score"])
    return None


# --------------------------------------------------------------------------- #
# Embedding backend: FlagEmbedding -> sentence-transformers -> transformers AutoModel
# (CLS pooling + L2 norm, faithful to bge-large-en-v1.5's FlagModel.encode). Cached module-wide.
# --------------------------------------------------------------------------- #
class _Embedder:
    def __init__(self) -> None:
        self._backend = None
        self._impl = None

    def _lazy(self) -> None:
        if self._impl is not None:
            return
        # 1) FlagEmbedding - bit-exact with upstream.
        try:
            from FlagEmbedding import FlagModel  # type: ignore
            self._impl = FlagModel(_EMBED_MODEL,
                                   query_instruction_for_retrieval="Represent this sentence "
                                   "for searching relevant passages:", use_fp16=False)
            self._backend = "FlagEmbedding"
            return
        except Exception:
            pass
        # 2) sentence-transformers.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._impl = SentenceTransformer(_EMBED_MODEL, device=_EMBED_DEVICE)
            self._backend = "sentence-transformers"
            return
        except Exception:
            pass
        # 3) transformers AutoModel (always available in the gbench environment: torch + transformers).
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(_EMBED_MODEL)
        mdl = AutoModel.from_pretrained(_EMBED_MODEL).to(_EMBED_DEVICE).eval()
        self._impl = (tok, mdl)
        self._backend = "transformers"

    @property
    def backend(self) -> str:
        self._lazy()
        return self._backend or "none"

    def encode(self, texts: List[str]):
        import numpy as np
        self._lazy()
        if self._backend == "FlagEmbedding":
            return np.asarray(self._impl.encode(texts))
        if self._backend == "sentence-transformers":
            return np.asarray(self._impl.encode(texts, normalize_embeddings=True))
        import torch
        tok, mdl = self._impl
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(_EMBED_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            emb = out.last_hidden_state[:, 0]  # CLS token, matching bge FlagModel pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()


_EMBEDDER: Optional[_Embedder] = None


def _get_embedder() -> _Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _Embedder()
    return _EMBEDDER


def _assign_max(matrix) -> Tuple[List[int], List[int]]:
    """Max-weight assignment; replaces scipy.linear_sum_assignment(-matrix).

    Returns (row_ind, col_ind) of length min(n_rows, n_cols) maximising the summed weight.
    Brute-force over the smaller dimension for the tiny per-step matrices; greedy fallback
    for the rare large case. Assignment order does not affect scoring (each pair is compared
    independently and keyed by its original predicted-call index).
    """
    import numpy as np
    m = np.asarray(matrix, dtype=float)
    n_rows, n_cols = m.shape
    k = min(n_rows, n_cols)
    if k == 0:
        return [], []
    if k <= 7:
        if n_rows <= n_cols:
            best_s, best = None, None
            for perm in permutations(range(n_cols), n_rows):
                s = sum(m[i, perm[i]] for i in range(n_rows))
                if best_s is None or s > best_s:
                    best_s, best = s, perm
            return list(range(n_rows)), list(best)
        best_s, best = None, None
        for perm in permutations(range(n_rows), n_cols):
            s = sum(m[perm[j], j] for j in range(n_cols))
            if best_s is None or s > best_s:
                best_s, best = s, perm
        pairs = sorted(zip(best, range(n_cols)))
        return [r for r, _ in pairs], [c for _, c in pairs]
    # greedy (large): pick highest-weight admissible pairs.
    order = sorted(((m[i, j], i, j) for i in range(n_rows) for j in range(n_cols)), reverse=True)
    used_r, used_c, pairs = set(), set(), []
    for _, i, j in order:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
        if len(pairs) == k:
            break
    pairs.sort()
    return [i for i, _ in pairs], [j for _, j in pairs]


# --------------------------------------------------------------------------- #
# CompareFC - ported from upstream utils/compare_method.py. The LLM equivalence judge
# (`llm_based`) is async and uses gbench's Gemini cascade; `response_based` degrades to
# False when RAPID_API_KEY is unset (a documented tie-breaker, not a skip).
# --------------------------------------------------------------------------- #
class _CompareFC:
    def __init__(self, exact_match_dict: Dict[str, List[str]], embedder: _Embedder) -> None:
        self.exact_match_dict = exact_match_dict
        self.embedding = embedder
        self.free_function_list = list(_FREE_FUNCTION_LIST)
        self.free_functions: Dict[str, Dict[str, Any]] = {}
        self.error_message: List[Any] = []
        self._rapid = _RapidAPICall() if os.environ.get("RAPID_API_KEY") else None

    # -- format + free-function bookkeeping --------------------------------- #
    def format_check(self, func_call: Dict[str, Any], functions: List[Dict[str, Any]]):
        name_to_func = {f["name"]: f for f in functions}
        if func_call["name"] not in name_to_func:
            return {"error": f"Function {func_call['name']} is not defined in the function list."}
        used = name_to_func[func_call["name"]]
        required = used["parameters"].get("required", [])
        args = func_call.get("arguments", {})
        if not set(required).issubset(set(args.keys())):
            missing = set(required) - set(args.keys())
            return {"error": f"Function {used['name']} requires parameters {required}, but "
                    f"{list(args.keys())} do not provide {missing}"}
        props = used["parameters"].get("properties", {})
        if not set(args.keys()).issubset(set(props.keys())):
            missing = set(args.keys()) - set(props.keys())
            return {"error": f"Function {used['name']} does not have parameters {missing}"}
        for pname, pval in args.items():
            ptype = props.get(pname, {}).get("type")
            if ptype == "string" and not isinstance(pval, str):
                return {"error": f"Parameter {pname} of function {used['name']} should be a "
                        f"string, but {type(pval)} is provided."}
            if ptype == "number" and not isinstance(pval, (int, float)):
                return {"error": f"Parameter {pname} of function {used['name']} should be a "
                        f"number, but {type(pval)} is provided."}
            if ptype == "boolean" and not isinstance(pval, bool):
                return {"error": f"Parameter {pname} of function {used['name']} should be a "
                        f"boolean, but {type(pval)} is provided."}
            if ptype == "array" and not isinstance(pval, list):
                return {"error": f"Parameter {pname} of function {used['name']} should be an "
                        f"array, but {type(pval)} is provided."}
        return True

    def add_free_function(self, convs: List[Dict[str, Any]]) -> None:
        self.free_functions = {}
        for i, turn in enumerate(convs):
            if "function_call" not in turn:
                continue
            for j, fc in enumerate(turn["function_call"]):
                if fc["name"] in self.free_function_list:
                    key = json.dumps(fc)
                    if key not in self.free_functions:
                        self.free_functions[key] = {"called": False, "obs": convs[i + 1]["content"][j]}

    # -- the 4-tier match cascade ------------------------------------------ #
    def rule_based(self, predict: Dict[str, Any], golden: Dict[str, Any]) -> bool:
        if predict["name"] != golden["name"]:
            return False
        if sorted(predict.get("arguments", {}).keys()) != sorted(golden.get("arguments", {}).keys()):
            return False
        for k, v in predict.get("arguments", {}).items():
            if k == "categories_filter":
                pred_f = [s.strip() for s in str(v).split(",")]
                gold_f = [s.strip() for s in str(golden["arguments"][k]).split(",")]
                if set(gold_f) == set(pred_f):
                    continue
            if v != golden["arguments"][k]:
                return False
        return True

    def value_checker(self, pred_call: Dict[str, Any], golden_call: Dict[str, Any]):
        if pred_call["name"] != golden_call["name"]:
            return False, {"error_type": "func_error", "content": "Do not call the correct function."}
        # Defensive .get: every dataset function is a booking-com15 tool present in the asset;
        # an unknown name yields no critical params to check (upstream would KeyError).
        param_list = self.exact_match_dict.get(pred_call["name"], [])
        for k, v in golden_call.get("arguments", {}).items():
            if k in param_list:
                if k not in pred_call.get("arguments", {}):
                    return False, {"error_type": "param_missing",
                                   "content": f"Missing parameter {k} in prediction."}
                if k != "categories_filter" and pred_call["arguments"][k] != v:
                    return False, {"error_type": "value_error",
                                   "content": f"Parameter {k} value is not correct in prediction."}
                if k == "categories_filter":
                    gold_f = [s.strip() for s in str(v).split(",")]
                    pred_f = [s.strip() for s in str(pred_call["arguments"][k]).split(",")]
                    if set(gold_f) != set(pred_f):
                        return False, {"error_type": "value_error",
                                       "content": f"Parameter {k} value is not correct in prediction."}
        return True, ""

    def response_based(self, predict: Dict[str, Any], golden: Dict[str, Any]) -> bool:
        """Optional live-RapidAPI tie-breaker. Degrades to False without RAPID_API_KEY."""
        if self._rapid is None:
            return False
        try:
            r1 = self._rapid.call(predict)
            if not r1:
                return False
            r2 = self._rapid.call(golden)
        except Exception:
            return False
        if r1 is None or r2 is None:
            return False
        return r1 == r2

    async def llm_based(self, functions, history, predict, golden) -> Optional[bool]:
        user = (
            "Function list:\n```JSON\n" + json.dumps(functions, ensure_ascii=False) + "\n```\n"
            "Conversation history:\n```JSON\n" + json.dumps(history, ensure_ascii=False) + "\n```\n"
            "Function call 1:\n```JSON\n" + json.dumps(predict, ensure_ascii=False) + "\n```\n"
            "Function call 2:\n```JSON\n" + json.dumps(golden, ensure_ascii=False) + "\n```\n"
            "Please determine whether Function call 1 and Function call 2 are equivalent and "
            "provide your reason.\noutput:\n" + _COMPARE_ANCHOR)
        text, _used = await judge_generate_cascade(_COMPARE_SYSTEM + "\n\n" + user)
        return _parse_is_equal(text)

    # -- multi-call alignment (bge + max-assignment) ----------------------- #
    def mapping_call(self, predict, golden, golden_obs):
        def _sort(call_list):
            for value in call_list:
                value["arguments"] = {k: value["arguments"][k] for k in sorted(value.get("arguments", {}))}
        _sort(predict)
        _sort(golden)

        exact_matches, remaining_predict, remaining_golden = [], [], []
        remaining_predict_index, remaining_golden_index = {}, {}
        matched_indices = set()

        for p_index, p_value in enumerate(predict):
            match_found = False
            for g_index, g_value in enumerate(golden):
                if g_index in matched_indices:
                    continue
                if p_value == g_value:
                    exact_matches.append({"idx": p_index, "pred_call": p_value,
                                          "golden_call": g_value, "golden_obs": golden_obs[g_index]})
                    matched_indices.add(g_index)
                    match_found = True
                    break
                if json.dumps(p_value) in self.free_functions:
                    exact_matches.append({"idx": p_index, "pred_call": p_value,
                                          "golden_call": p_value,
                                          "golden_obs": self.free_functions[json.dumps(p_value)]["obs"]})
                    self.free_functions[json.dumps(p_value)]["called"] = True
                    match_found = True
                    break
            if not match_found:
                remaining_predict.append(p_value)
                remaining_predict_index[len(remaining_predict) - 1] = p_index

        for g_index, g_value in enumerate(golden):
            if g_index not in matched_indices:
                remaining_golden.append(g_value)
                remaining_golden_index[len(remaining_golden) - 1] = g_index

        if not remaining_predict or not remaining_golden:
            return exact_matches

        # 1x1 fast-path: a single remaining pred vs a single remaining golden always aligns to
        # each other (a 1x1 assignment is trivial), so skip the embedder entirely - this is the
        # common single-call-per-step mismatch case and keeps it dependency-free.
        if len(remaining_predict) == 1 and len(remaining_golden) == 1:
            return exact_matches + [{
                "idx": remaining_predict_index[0], "pred_call": remaining_predict[0],
                "golden_call": remaining_golden[0], "golden_obs": golden_obs[remaining_golden_index[0]]}]

        pred_embed = self.embedding.encode([json.dumps(v, ensure_ascii=False) for v in remaining_predict])
        gold_embed = self.embedding.encode([json.dumps(v, ensure_ascii=False) for v in remaining_golden])
        matrix = pred_embed @ gold_embed.T
        row_ind, col_ind = _assign_max(matrix)

        embedding_matches = []
        for i, j in zip(row_ind, col_ind):
            embedding_matches.append({"idx": remaining_predict_index[i],
                                      "pred_call": remaining_predict[i],
                                      "golden_call": remaining_golden[j],
                                      "golden_obs": golden_obs[remaining_golden_index[j]]})
        return exact_matches + embedding_matches

    def remove_called_fc(self, golden, golden_obs):
        pop_index = []
        for single in golden:
            key = json.dumps(single)
            if key in self.free_functions and self.free_functions[key]["called"]:
                pop_index.append(golden.index(single))
        for idx in sorted(pop_index, reverse=True):
            golden.pop(idx)
            golden_obs.pop(idx)
        return golden, golden_obs

    def get_error_message(self, pred_call, golden_call):
        for k, v in golden_call.get("arguments", {}).items():
            if k not in pred_call.get("arguments", {}):
                return {"error_type": "param_missing", "content": f"Missing parameter {k} in prediction."}
            if v != pred_call["arguments"][k]:
                return {"error_type": "value_error", "content": f"Parameter {k} value do not equal to golden."}
        for k in pred_call.get("arguments", {}):
            if k not in golden_call.get("arguments", {}):
                return {"error_type": "param_hallucination", "content": f"Parameter {k} is hallucinated."}
        return None

    async def compare_single_call(self, functions, history, pred_call, golden_call):
        if self.rule_based(pred_call, golden_call):
            return True, None
        is_valid, err = self.value_checker(pred_call, golden_call)
        if not is_valid:
            return False, err
        if self.response_based(pred_call, golden_call):
            return True, None
        if await self.llm_based(functions, history, pred_call, golden_call):
            return True, None
        return False, None

    async def compare_turn_prediction(self, functions, history, predict, golden, golden_obs):
        self.error_message = []
        golden, golden_obs = self.remove_called_fc(golden, golden_obs)
        if len(golden) == 0:
            # All golden calls this step were already-consumed free functions; nothing to match.
            return self.error_message, {}, [], {}
        match_list = self.mapping_call(predict, golden, golden_obs)
        format_error, success_map, success_matched = {}, {}, []
        for item in match_list:
            msg = self.format_check(item["pred_call"], functions)
            if msg is True:
                is_match, single = await self.compare_single_call(
                    functions, history, item["pred_call"], item["golden_call"])
                if is_match:
                    success_map[item["idx"]] = item["golden_obs"]
                    success_matched.append(item["golden_call"])
                else:
                    self.error_message.append(single or self.get_error_message(
                        item["pred_call"], item["golden_call"]))
            else:
                format_error[item["idx"]] = msg
        return self.error_message, success_map, success_matched, format_error


# --------------------------------------------------------------------------- #
# ModelRunner - ported from upstream runner/base_runner.py + runner/gpt_runner.py.
# Drives the served model through the golden call chain, feeding recorded observations.
# --------------------------------------------------------------------------- #
class _Runner:
    def __init__(self, compare: _CompareFC) -> None:
        self.CompareClass = compare
        self.error_message: List[Any] = []
        self.unexpect_call_resp = _UNEXPECT_CALL_RESP

    # -- golden state machine ---------------------------------------------- #
    def only_free_function(self, temp_fcs) -> bool:
        for call in temp_fcs:
            if call["name"] == "Search_Hotels" and call.get("arguments", {}).get("search_type") == "hotel":
                return True
        return set(fc["name"] for fc in temp_fcs).issubset(set(self.CompareClass.free_function_list))

    def get_success_turn(self, remain_fcs, total_fcs) -> int:
        remain_ids = []
        for idx, fc_list in enumerate(total_fcs):
            for remain_fc in remain_fcs:
                if remain_fc in fc_list:
                    remain_ids.append(idx)
        if not remain_ids:
            return len(total_fcs)
        return max(min(remain_ids), 0)

    def init_golden(self, convs) -> None:
        self.fc_chain, self.obs_chain = [], []
        for turn in convs:
            if "function_call" in turn:
                self.fc_chain.append(turn["function_call"])
            elif turn["role"] == "observation":
                self.obs_chain.append(turn["content"])
        assert len(self.fc_chain) == len(self.obs_chain), "function call / observation length mismatch"
        self.turn_id, self.correct_count = 0, 0
        self.golden_fcs = copy.deepcopy(self.fc_chain[self.turn_id])
        self.golden_obs = copy.deepcopy(self.obs_chain[self.turn_id])
        if self.only_free_function(self.golden_fcs):
            self.update_current_golden()

    def update_current_golden(self) -> None:
        self.turn_id += 1
        if self.turn_id < len(self.fc_chain):
            self.golden_fcs.extend(copy.deepcopy(self.fc_chain[self.turn_id]))
            self.golden_obs.extend(copy.deepcopy(self.obs_chain[self.turn_id]))

    def process_matches(self, success_matched) -> None:
        for matched in success_matched:
            if matched in self.golden_fcs:
                self.golden_obs.pop(self.golden_fcs.index(matched))
                self.golden_fcs.remove(matched)
        if len(success_matched) > 0:
            self.update_current_golden()
        for k, v in self.CompareClass.free_functions.items():
            if v["called"] and json.loads(k) in self.golden_fcs:
                self.golden_obs.pop(self.golden_fcs.index(json.loads(k)))
                self.golden_fcs.remove(json.loads(k))
        if self.only_free_function(self.golden_fcs):
            self.update_current_golden()

    def return_result(self, messages, error_info=None):
        if error_info:
            success_turn = self.get_success_turn(self.golden_fcs, self.fc_chain)
            return messages, error_info, success_turn, self.correct_count
        # Free-function post-process (iterate a copy: upstream mutates during iteration, which
        # can skip elements; the intent is to drop leftover optional calls before the stop check).
        if len(self.golden_fcs) != 0:
            for call in list(self.golden_fcs):
                if call["name"] == "Search_Hotels" and call.get("arguments", {}).get("search_type") == "hotel":
                    if call in self.golden_fcs:
                        self.golden_fcs.remove(call)
                if call["name"] in ["Search_Hotel_Destination", "Search_Attraction_Location",
                                    "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location"]:
                    if call in self.golden_fcs:
                        self.golden_fcs.remove(call)
        if self.turn_id < len(self.fc_chain) or len(self.golden_fcs) > 0:
            return self.return_result(messages, {"error_type": "stop_early", "content": "Stop early."})
        if len(self.golden_fcs) == 0:
            return messages, "Success.", len(self.fc_chain), self.correct_count
        raise RuntimeError("Unexpected complexfuncbench return_result state.")

    # -- name sanitisation for OpenAI-style tool schemas ------------------- #
    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if re.match(r"[a-zA-Z0-9_-]", c) else "-" for c in name)[:64]

    def get_standard_functions(self, functions):
        self.name_dict = {api["name"]: self._sanitize(api["name"]) for api in functions}
        tools = [{"type": "function", "function": copy.deepcopy(f)} for f in functions]
        for t in tools:
            t["function"]["name"] = self.name_dict[t["function"]["name"]]
        return tools

    def _decode_tool_call(self, tc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fn = tc.get("function", {})
        sanitized = fn.get("name")
        orig = next((k for k, v in self.name_dict.items() if v == sanitized), None)
        if orig is None:
            return None
        raw = fn.get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return None
        return {"name": orig, "arguments": args if isinstance(args, dict) else {}}

    # -- the driving loop -------------------------------------------------- #
    async def run(self, data, chat):
        """`chat(oai_messages, tools) -> message_dict | None` posts to the gbench endpoint."""
        convs, functions = data["conversations"], data["functions"]
        self.CompareClass.add_free_function(convs)
        tools = self.get_standard_functions(functions)

        query = convs[0]["content"]
        messages = [{"role": "user", "content": query}]          # abstract record (for judges)
        oai_messages = [{"role": "user", "content": query}]      # what the model actually sees
        self.init_golden(convs)

        rounds = 0
        while True:
            rounds += 1
            if rounds > _MAX_ROUNDS:
                return self.return_result(messages, {"error_type": "max_rounds",
                                                     "content": f"Exceeded {_MAX_ROUNDS} rounds."})
            msg = await chat(oai_messages, tools)
            if msg is None:
                return self.return_result(messages, {"error_type": "unknown_error",
                                                     "content": "llm_response is None"})
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")

            if tool_calls:
                if self.golden_fcs == []:
                    return self.return_result(messages, {"error_type": "func_hallucination",
                                                         "content": "Expected to stop, but model "
                                                         "continued to output function calls."})
                oai_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                function_calls = []
                for tc in tool_calls:
                    fc = self._decode_tool_call(tc)
                    if fc is None:
                        return self.return_result(messages, {"error_type": "decode_error",
                                                             "content": f"{tc.get('function')} is not valid."})
                    function_calls.append(fc)
                messages.append({"role": "assistant", "function_call": function_calls})

                (self.error_message, success_map, success_matched,
                 format_error) = await self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1], copy.deepcopy(function_calls),
                    self.golden_fcs, self.golden_obs)
                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)
                self.correct_count += len(success_map)

                real_time_obs = []
                for t, fc in enumerate(function_calls):
                    if t in success_map:
                        obs = success_map[t]
                    elif t in format_error:
                        obs = format_error[t]
                    else:
                        obs = self.unexpect_call_resp
                    real_time_obs.append(obs)
                    oai_messages.append({"tool_call_id": tool_calls[t].get("id", f"call_{t}"),
                                         "role": "tool", "name": self.name_dict[fc["name"]],
                                         "content": json.dumps(obs, ensure_ascii=False)})
                self.process_matches(success_matched)
                messages.append({"role": "observation", "content": real_time_obs})

            elif content is not None:
                messages.append({"role": "assistant", "content": content})
                return self.return_result(messages, self.error_message)
            else:
                return self.return_result(messages, {"error_type": "unknown_error",
                                                     "content": "empty model message"})


# --------------------------------------------------------------------------- #
# Optional live RapidAPI (response-based tie-breaker). No-op unless RAPID_API_KEY set.
# --------------------------------------------------------------------------- #
class _RapidAPICall:
    def __init__(self) -> None:
        self._ready = False
        try:
            with open(os.path.join(_DATA_DIR, "tool_info.json")) as f:
                info = json.load(f)["booking-com15"]
            self.name_to_url = info["name_to_url"]
            self.path_params = info["path_params"]
            self.headers = {"X-RapidAPI-Key": os.environ.get("RAPID_API_KEY"),
                            "X-RapidAPI-Host": info["host"]}
            self._ready = True
        except Exception as e:
            logger.warning("complexfuncbench: optional RapidAPI response tie-breaker unavailable "
                           "(%s); falling back to the Gemini equivalence judge. See %s", e, DOCS_URL)

    def call(self, func_call):
        if not self._ready:
            return None
        import requests
        url = self.name_to_url[func_call["name"]]
        params = copy.deepcopy(func_call.get("arguments", {}))
        path = {}
        for p in self.path_params:
            if f"{{{p}}}" in url and p in params:
                path[p] = params.pop(p)
        url = url.format(**path)
        for k, v in params.items():
            if k == "legs":
                params[k] = json.dumps(v, ensure_ascii=False)
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("status") is True:
            data.pop("timestamp", None)
            if "data" in data:
                data = data["data"]
        return data


# --------------------------------------------------------------------------- #
# Response completeness / correctness judges (upstream RespEvalRunner -> Gemini).
# --------------------------------------------------------------------------- #
async def _resp_eval(convs, gen_response: str) -> Optional[Dict[str, Any]]:
    if gen_response == "":
        return {"complete": {"score": -2, "reason": "No response generated."},
                "correct": {"score": -2, "reason": "No response generated."}}
    query = convs[0]["content"]
    complete_user = (f"input:\nquery: {query}\nresponse: {gen_response}\n\noutput:\n" + _SCORE_ANCHOR)
    correct_user = ("dialogue history: " + json.dumps(convs[:-1], ensure_ascii=False)
                    + f"\nresponse: {gen_response}\noutput:\n" + _SCORE_ANCHOR)
    c_text, _ = await judge_generate_cascade(_COMPLETE_SYSTEM + "\n\n" + complete_user)
    k_text, _ = await judge_generate_cascade(_CORRECT_SYSTEM + "\n\n" + correct_user)
    c_score, k_score = _parse_score(c_text), _parse_score(k_text)
    return {
        "complete": {"score": c_score if c_score is not None else -1,
                     "reason": "completeness judge" if c_score is not None else "judge failed"},
        "correct": {"score": k_score if k_score is not None else -1,
                    "reason": "correctness judge" if k_score is not None else "judge failed"},
    }


# --------------------------------------------------------------------------- #
# Per-sample evaluation + metric aggregation.
# --------------------------------------------------------------------------- #
async def _evaluate_one(item, chat, exact_match_dict, embedder) -> Optional[Dict[str, Any]]:
    runner = _Runner(_CompareFC(exact_match_dict, embedder))
    try:
        convs, message, turn_id, correct_count = await runner.run(item, chat)
    except Exception as e:
        logger.warning("complexfuncbench sample %s errored: %s", item.get("id"), e)
        return {"id": item.get("id"), "request_failed": True, "message": {"error_type": "exception",
                "content": str(e)}}

    if isinstance(message, dict) and message.get("error_type") == "unknown_error":
        return {"id": item.get("id"), "request_failed": True, "message": message}

    turn_count = call_count = 0
    for turn in item["conversations"]:
        if turn.get("role") == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])
    real_turn = sum(1 for t in convs if t.get("role") == "assistant" and "function_call" in t)

    resp_eval = None
    if convs and convs[-1].get("role") == "assistant" and "content" in convs[-1]:
        resp_eval = await _resp_eval(item["conversations"], convs[-1]["content"])

    return {
        "id": item.get("id"),
        "message": message,
        "count_dict": {"success_turn_num": turn_id, "total_turn_num": turn_count,
                       "correct_call_num": correct_count, "total_call_num": call_count,
                       "real_turn_num": real_turn},
        "resp_eval": resp_eval,
        "request_failed": False,
    }


def _aggregate(results: List[Dict[str, Any]], n_total: int, is_full: bool) -> Dict[str, Any]:
    scored = [r for r in results if r and not r.get("request_failed")]
    n_failed = sum(1 for r in results if r and r.get("request_failed"))

    success = sum(1 for r in scored if r.get("message") == "Success.")
    corr_calls = sum(r["count_dict"]["correct_call_num"] for r in scored)
    tot_calls = sum(r["count_dict"]["total_call_num"] for r in scored)

    # Success Rate: pooled over the scored set (upstream normalises by 1000 on the full set).
    denom = n_total if is_full else max(1, len(scored))
    success_rate = round(success / denom * 100.0, 2)
    call_accuracy = round(corr_calls / tot_calls * 100.0, 2) if tot_calls else 0.0

    comp = [r["resp_eval"]["complete"]["score"] for r in scored
            if r.get("resp_eval") and r["resp_eval"]["complete"]["score"] in (0, 1, 2)]
    corr = [r["resp_eval"]["correct"]["score"] for r in scored
            if r.get("resp_eval") and r["resp_eval"]["correct"]["score"] in (0, 1, 2)]
    completeness = round(sum(comp) / len(comp), 4) if comp else None      # raw 0..2 (upstream)
    correctness = round(sum(corr) / len(corr), 4) if corr else None

    per_domain = {}
    if is_full:
        by_dom_succ: Dict[str, int] = {}
        for r in scored:
            if r.get("message") == "Success.":
                dom = str(r["id"]).rsplit("-", 1)[0]
                by_dom_succ[dom] = by_dom_succ.get(dom, 0) + 1
        per_domain = {d: round(by_dom_succ.get(d, 0) / _DOMAIN_DENOM[d] * 100.0, 2)
                      for d in _DOMAIN_DENOM}

    return {
        "success_rate": success_rate,
        "call_accuracy": call_accuracy,
        "completeness": completeness,
        "correctness": correctness,
        "n_scored": len(scored),
        "n_request_failed": n_failed,
        "n_success": success,
        "per_domain_success_rate": per_domain or None,
    }


# --------------------------------------------------------------------------- #
# Dataset + gate.
# --------------------------------------------------------------------------- #
def _load_dataset(limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE, repo_type="dataset")
    except Exception as e:
        raise infra_required(
            "complexfuncbench",
            f"could not fetch {_HF_REPO}/{_HF_FILE} from the HF Hub ({e}). Needs network/HF access "
            "(or a pre-populated HF cache).", DOCS_URL)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise infra_required("complexfuncbench", f"{_HF_FILE} loaded but empty.", DOCS_URL)
    rows = stratified_sample(rows, limit, lambda r: str((r or {}).get("id", "")).rsplit("-", 1)[0],
                             seed="complexfuncbench")
    return rows


def _load_exact_match_dict() -> Dict[str, List[str]]:
    with open(_EXACT_MATCH_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_complexfuncbench_prerequisites() -> Tuple[bool, str]:
    if not os.environ.get("GEMINI_API_KEY"):
        return False, ("GEMINI_API_KEY is not set - it powers gbench's Gemini cascade judge that "
                       "scores the call-equivalence + response completeness/correctness roles here.")
    _ok, _why = gemini_key_live_valid(os.environ["GEMINI_API_KEY"])
    if not _ok:
        return False, (f"GEMINI_API_KEY was rejected by the judge endpoint ({_why}); a valid key is "
                       "required for the Gemini cascade judge (fail-fast before the full run).")
    if not os.path.exists(_EXACT_MATCH_PATH):
        return False, (f"vendored exact_match_values.json missing at {_EXACT_MATCH_PATH}.")
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        return False, "huggingface_hub is required to fetch the dataset (pip install gbench[evals])."
    # Embedding backend: FlagEmbedding / sentence-transformers / transformers (any one).
    ok_embed = False
    for mod in ("FlagEmbedding", "sentence_transformers", "transformers"):
        try:
            __import__(mod)
            ok_embed = True
            break
        except Exception:
            continue
    if not ok_embed:
        return False, ("no embedding backend available (need one of FlagEmbedding, "
                       "sentence-transformers, or transformers) for multi-call alignment.")
    return True, ""


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
async def _drive_all(model_name, base_url, concurrency, temperature, limit,
                     max_output_tokens=None, enable_thinking=False):
    import aiohttp

    api = base_url.rstrip("/")
    if not api.endswith("/v1"):
        api += "/v1"
    api_url = api + "/chat/completions"

    items = _load_dataset(limit)
    exact_match_dict = _load_exact_match_dict()
    embedder = _get_embedder()

    is_full = (not limit or limit <= 0 or limit >= _FULL_TOTAL) and len(items) >= _FULL_TOTAL
    n_total = _FULL_TOTAL if is_full else len(items)

    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    sample_sem = asyncio.Semaphore(max(1, concurrency))

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Per-turn output cap = the run's --max-output-tokens (required flag) when given, else the
        # suite default; and the model's reasoning channel follows --thinking. Both previously had
        # no effect on this in-process suite (max_tokens was hardcoded, thinking never injected).
        per_turn_max = int(max_output_tokens) if max_output_tokens else _MODEL_MAX_TOKENS

        async def chat(oai_messages, tools):
            payload = {"model": model_name, "messages": oai_messages, "tools": tools,
                       "tool_choice": "auto", "temperature": temperature,
                       "max_tokens": per_turn_max,
                       "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
            for attempt in range(3):
                try:
                    async with session.post(api_url, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["choices"][0]["message"]
                        logger.warning("complexfuncbench HTTP %s: %s", resp.status,
                                       (await resp.text())[:200])
                except Exception as e:
                    logger.debug("complexfuncbench request error (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(0.5 * (attempt + 1))
            return None

        async def _one(item):
            async with sample_sem:
                return await _evaluate_one(item, chat, exact_match_dict, embedder)

        results = await asyncio.gather(*[_one(it) for it in items])

    agg = _aggregate(results, n_total, is_full)
    agg["_is_full"] = is_full
    agg["_n_items"] = len(items)
    agg["_embed_backend"] = embedder.backend
    return agg


def run_complexfuncbench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical ComplexFuncBench: the multi-step golden-driven loop + ComplexEval,
    with gbench's Gemini cascade in the two judge roles."""
    ok, reason = check_complexfuncbench_prerequisites()
    if not ok:
        raise infra_required("complexfuncbench", reason, DOCS_URL)

    temperature, temperature_source = resolve_temperature(
        "complexfuncbench", kwargs.get("temperature"), thinking=enable_thinking)

    limit = kwargs.get("limit")
    agg = asyncio.run(_drive_all(model_name, base_url, max(1, int(concurrency)),
                                 temperature, limit,
                                 max_output_tokens=kwargs.get("max_output_tokens"),
                                 enable_thinking=enable_thinking))

    if agg["n_scored"] == 0:
        raise infra_required(
            "complexfuncbench",
            "no samples could be driven (all requests failed - is the endpoint serving a "
            "tool-calling model?).", DOCS_URL)

    is_full = agg.pop("_is_full")
    n_items = agg.pop("_n_items")
    embed_backend = agg.pop("_embed_backend")

    leaderboard_comparable = False
    lc_reason = ("graded by gbench's standard Gemini cascade (call-equivalence + response roles; a "
                 "gbench convention); the published ComplexFuncBench leaderboard is GPT-4o-graded, "
                 "so this is a gbench-internal number, not a like-for-like leaderboard entry")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "complexfuncbench",
        "model_name": model_name,
        "thinking": enable_thinking,
        "status": "success",
        "accuracy": agg["success_rate"],           # headline = Success Rate (%)
        "success_rate": agg["success_rate"],
        "call_accuracy": agg["call_accuracy"],
        "completeness": agg["completeness"],        # raw mean 0..2 (upstream Complete Score)
        "correctness": agg["correctness"],          # raw mean 0..2 (upstream Correct Score)
        "per_domain_success_rate": agg["per_domain_success_rate"],
        "n_samples": n_items,
        "n_scored": agg["n_scored"],
        "n_success": agg["n_success"],
        "n_request_failed": agg["n_request_failed"],
        "judge": "gbench-gemini-cascade",
        "embedding_backend": embed_backend,
        "temperature": temperature,
        "temperature_source": temperature_source,
        "metric": ("ComplexEval: Success Rate (headline) + Call Accuracy over the multi-step "
                   "golden-driven tool-calling loop, plus response Completeness/Correctness "
                   "(0-2). Call-equivalence + response judges are gbench's standard Gemini cascade "
                   "(upstream prompts verbatim; upstream's grader is GPT-4o)."),
        "leaderboard_comparable": leaderboard_comparable,
        "leaderboard_comparable_reason": lc_reason,
    }
    if agg["n_request_failed"]:
        result["request_failure_note"] = (
            f"{agg['n_request_failed']} sample(s) dropped on endpoint/request failure; "
            "Success Rate is over the scored set.")
    return result
