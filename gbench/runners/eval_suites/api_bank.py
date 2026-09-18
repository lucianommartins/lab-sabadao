# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: api_bank
# Description: API-Bank (Li et al., EMNLP 2023 - runnable tool-use benchmark)

"""gbench native built-in runner for api_bank (Tool Use & Function Calling).

Canonical API-Bank (Li et al., EMNLP 2023; arXiv:2304.08244; code
AlibabaResearch/DAMO-ConvAI/api-bank; dataset liminghao1630/API-Bank) is a RUNNABLE
benchmark: a simulated pool of ~73 Python APIs backed by a local fake DB. It measures three
progressively harder abilities, each a headline column, and TWO tracks per level:

  * Level-1 Call            (all API descriptions given -> emit the right call)
  * Level-2 Retrieve+Call   (must first ToolSearcher-retrieve the API, then call it)
  * Level-3 Plan+Retrieve+Call
  Correctness (Accuracy) = EXECUTION-BASED: execute the predicted call against the simulated
  backend and compare via each API's `check_api_call_correctness` against the gold result.
  Response quality = ROUGE-L F over the AI-turn generations. Level-3 also reports a dialogue
  sample success rate = (50 - #dialogues-with-any-errored-step)/50. No LLM judge anywhere.

Sourcing: L1/L2 drive from the checkout's raw conversations (lv1-lv2-samples/*) via the
canonical `Sample.from_chat_history` + `Evaluator.evaluate` (reused verbatim from the
checkout for fidelity: fresh ToolManager per call, execution + `check_api_call_correctness`).
L3 drives from the HF dataset (test-data/level-3-batch-inf*.json + level-3.json) against the
checkout's lv3_apis backend. The execution backend is a user-provided CHECKOUT pointed at by
`GBENCH_APIBANK_DIR` (this suite hard-errors with clone instructions if it is absent).

Sampling: temperature defaults to 0.0 (greedy) for no-think and 1.0 with `--thinking`; the
measurement is in base.DEFAULT_TEMPERATURE. Override for a whole run with `--temperature`, or
for this suite with `GBENCH_API_BANK_TEMPERATURE`. No LLM judge, so no GEMINI_API_KEY.
"""

import contextlib
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .base import run_eval_suite
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/api_bank.md"

_ENV = "GBENCH_APIBANK_DIR"
_HF_REPO = "liminghao1630/API-Bank"

# Prompt templates, VERBATIM from api-bank/evaluator.py (the canonical single-turn inputs).
API_CALL_PROMPT = """
Based on the given API description and the existing conversation history 1..t, please generate the API request that the AI should call in step t+1 and output it in the format of [ApiName(key1='value1', key2='value2', ...)], replace the ApiName with the actual API name, and replace the key and value with the actual parameters.
Your output should start with a square bracket "[" and end with a square bracket "]". Do not output any other explanation or prompt or the result of the API call in your output.
This year is 2023.
Input:
User: [User's utterence]
AI: [AI's utterence]

Expected output:
[ApiName(key1='value1', key2='value2', ...)]

API descriptions:
"""

RESPONSE_PROMPT = """
Based on the given API description and the existing conversation history 1..t, please generate the next dialog that the AI should response after the API call t.
This year is 2023.
Input:
User: [User's utterence]
AI: [AI's utterence]
[ApiName(key1='value1', key2='value2', …)]

Expected output:
AI: [AI's utterence]

API descriptions:
"""


# --------------------------------------------------------------------------- #
# Checkout access
# --------------------------------------------------------------------------- #
def _checkout_dir() -> str:
    d = prereqs_path("apibank/api-bank", (os.environ.get(_ENV) or "").strip()) or ""
    hint = (f"Set {_ENV} to the api-bank/ dir of a DAMO-ConvAI checkout, e.g.:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/AlibabaResearch/DAMO-ConvAI <dir>\n"
            "  (cd <dir> && git sparse-checkout set api-bank)\n"
            f"  export {_ENV}=<dir>/api-bank")
    if not d or not os.path.isdir(d):
        raise infra_required("api_bank", f"API-Bank checkout not found ({_ENV}={d!r}). {hint}", DOCS_URL)
    if not os.path.isfile(os.path.join(d, "tool_manager.py")) or not os.path.isdir(os.path.join(d, "apis")):
        raise infra_required("api_bank", f"{_ENV}={d!r} is not the api-bank/ dir (no tool_manager.py/apis/). {hint}", DOCS_URL)
    return d


@contextlib.contextmanager
def _in_checkout(d: str):
    """ToolManager hardcodes ./apis and ./init_database, so run backend work with CWD=checkout
    and the checkout on sys.path. Post-generation scoring is serial, so the global chdir is safe."""
    if d not in sys.path:
        sys.path.insert(0, d)
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def _split_by_uppercase(s: str) -> str:
    return "".join(" " + c if c.isupper() else c for c in s).strip()


def _build_messages(system_prompt: str, chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Replicate evaluator.py's message construction, then normalise for a chat template that
    requires alternating roles: API-result turns (canonically role 'system' mid-conversation)
    are folded into the user side, and consecutive same-role turns are merged."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in chat_history:
        role = item.get("role")
        if role == "User":
            msgs.append({"role": "user", "content": item.get("text", "")})
        elif role == "AI":
            msgs.append({"role": "assistant", "content": item.get("text", "")})
        elif role == "API":
            params = ", ".join("{}='{}'".format(k, v) for k, v in (item.get("param_dict") or {}).items())
            out = str((item.get("result") or {}).get("output"))
            msgs.append({"role": "user", "content": "[{}({})] Response: {}".format(item.get("api_name"), params, out)})
    merged: List[Dict[str, str]] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1]["content"] = (merged[-1]["content"] + "\n\n" + m["content"]).strip()
        else:
            merged.append(dict(m))
    return merged


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_L12 = [(1, "level-1-given-desc"), (2, "level-2-toolsearcher")]


def _load_l12_samples(d: str) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Build L1/L2 samples from the checkout raw conversations (reusing evaluator.Sample)."""
    samples = []
    with _in_checkout(d):
        import evaluator as ev  # noqa: E402  (checkout module)
        from tool_manager import ToolManager  # noqa: E402
        shared_tm = ToolManager()  # read-only description lookups (state not used)
        desc_cache: Dict[str, str] = {}

        def desc(name: str) -> str:
            if name not in desc_cache:
                desc_cache[name] = shared_tm.get_api_description(name)
            return desc_cache[name]

        for level, subdir in _L12:
            ddir = os.path.join("lv1-lv2-samples", subdir)
            for fname in sorted(f for f in os.listdir(ddir) if f.endswith(".jsonl")):
                with open(os.path.join(ddir, fname), encoding="utf-8") as f:
                    history = [json.loads(line) for line in f]
                for sid, s in enumerate(ev.Sample.from_chat_history(history)):
                    role = s.ground_truth.get("role")
                    if role == "API":
                        api_desc = desc("ToolSearcher") if level == 2 else "\n".join(desc(a) for a in s.apis)
                        prompt, track = API_CALL_PROMPT + api_desc, "api"
                    elif role == "AI":
                        api_desc = "\n".join(desc(a) for a in s.apis)
                        prompt, track = RESPONSE_PROMPT + api_desc, "response"
                    else:
                        continue
                    messages = _build_messages(prompt, s.chat_history)
                    gold = json.dumps({"level": level, "track": track, "file": fname, "sample_id": sid})
                    samples.append((messages, gold, {"category": f"api_bank::L{level}_{track}"}))
    return samples


def _hf(fn: str):
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(_HF_REPO, filename=fn, repo_type="dataset"), encoding="utf-8") as f:
        return json.load(f)


def _load_l3_samples() -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """L3 batch-inference steps from HF (call steps + response steps)."""
    samples = []
    for fn, track in [("test-data/level-3-batch-inf.json", "api"),
                      ("test-data/level-3-batch-inf-response.json", "response")]:
        for r in _hf(fn):
            content = (str(r.get("instruction", "")) + "\n" + str(r.get("input", ""))).strip()
            gold = {"level": 3, "track": track, "sample_id": r.get("sample_id")}
            if track == "api":
                gold["api_id"] = r.get("api_id")
            samples.append(([{"role": "user", "content": content}], json.dumps(gold),
                            {"category": f"api_bank::L3_{track}"}))
    return samples


# --------------------------------------------------------------------------- #
# Scoring - reuse the checkout's Evaluator for L1/L2; replicate lv3_evaluator for L3.
# --------------------------------------------------------------------------- #
def _score_traces(d: str, traces: List[Dict[str, Any]]) -> set:
    """Score in place; return the set of L3 dialogue sample_ids that had any errored step."""
    l3_gt = None  # lazily fetched from HF only if there are L3 traces
    l3_errored: set = set()
    with _in_checkout(d):
        import evaluator as ev  # noqa: E402
        from tool_manager import ToolManager  # noqa: E402
        from api_call_extraction import get_api_call, parse_api_call  # noqa: E402

        l12_cache: Dict[Tuple[int, str], Any] = {}
        l3_tm = ToolManager("./lv3_apis")  # shared across L3 samples (matches lv3_evaluator)

        for t in traces:
            try:
                blob = json.loads(t.get("gold_answer") or "{}")
            except Exception:
                blob = {}
            level, track = blob.get("level"), blob.get("track")
            pred = t.get("response_text") or ""
            t["api_bank_level"], t["api_bank_track"] = level, track

            if level in (1, 2):
                key = (level, blob.get("file"))
                if key not in l12_cache:
                    subdir = "level-1-given-desc" if level == 1 else "level-2-toolsearcher"
                    with open(os.path.join("lv1-lv2-samples", subdir, blob["file"]), encoding="utf-8") as f:
                        hist = [json.loads(line) for line in f]
                    l12_cache[key] = ev.Evaluator(ev.Sample.from_chat_history(hist))
                evaluator, sid = l12_cache[key], blob.get("sample_id")
                if track == "api":
                    correct = False
                    call = get_api_call(pred)
                    if call:
                        try:
                            correct, _res = evaluator.evaluate(sid, call)
                        except Exception:
                            correct = False
                    t["is_correct"] = bool(correct)
                    t["api_bank_score"] = 1.0 if correct else 0.0
                else:
                    try:
                        score = float(evaluator.evaluate(sid, pred)) if pred else 0.0
                    except Exception:
                        score = 0.0
                    t["api_bank_score"] = score
                    t["is_correct"] = score >= 0.5

            elif level == 3:
                if l3_gt is None:
                    l3_gt = _hf("test-data/level-3.json")
                sid = blob.get("sample_id")
                if track == "api":
                    aid = blob.get("api_id")
                    try:
                        gt = l3_gt[sid]["apis"][aid]
                        gt_name, gt_out = gt["api_name"], gt["output"]
                    except Exception:
                        gt_name, gt_out = None, None
                    correct = False
                    call = get_api_call(pred)
                    if not call or gt_name is None:
                        l3_errored.add(sid)
                    else:
                        try:
                            pname, pparams = parse_api_call(call)
                            if pname == "ToolSearcher" and "keywords" in pparams:
                                pparams["keywords"] = _split_by_uppercase(pparams["keywords"])
                            pres = l3_tm.api_call(pname, **pparams)
                            gt_api = l3_tm.init_tool(gt_name)
                            correct = bool(gt_api.check_api_call_correctness(pres, gt_out))
                            if not correct:
                                l3_errored.add(sid)
                        except Exception:
                            l3_errored.add(sid)
                            correct = False
                    t["is_correct"] = correct
                    t["api_bank_score"] = 1.0 if correct else 0.0
                else:
                    try:
                        gt_resp = l3_gt[sid].get("response", "")
                    except Exception:
                        gt_resp = ""
                    clean = pred.replace("User:", "").replace("AI:", "").strip()
                    try:
                        score = float(ev.calculate_rouge_l_score(gt_resp, clean)) if clean else 0.0
                    except Exception:
                        score = 0.0
                    t["api_bank_score"] = score
                    t["is_correct"] = score >= 0.5
    return l3_errored


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(100.0 * sum(vals) / len(vals), 2) if vals else None


def run_api_bank(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run API-Bank: 3 levels x (execution Correctness + response ROUGE-L) + L3 success rate."""
    d = _checkout_dir()

    samples = _load_l12_samples(d) + _load_l3_samples()
    limit = kwargs.get("limit")
    if limit:
        samples = stratified_sample(samples, limit,
                                    key_fn=lambda s: (s[2] or {}).get("category"), seed="api_bank")
    if not samples:
        raise RuntimeError("api_bank produced no samples")

    res = run_eval_suite(
        eval_name="api_bank",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=lambda pred, gold: False,  # placeholder; real scoring is execution-based below
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,  # already limited above
        max_output_tokens=kwargs.get("max_output_tokens", 1024),
    )

    traces = res.get("sample_traces", []) or []
    l3_errored = _score_traces(d, traces)

    def pick(level, track):
        return [t.get("api_bank_score") for t in traces
                if t.get("api_bank_level") == level and t.get("api_bank_track") == track]

    dims = {
        "L1_call": _mean(pick(1, "api")), "L1_response": _mean(pick(1, "response")),
        "L2_call": _mean(pick(2, "api")), "L2_response": _mean(pick(2, "response")),
        "L3_call": _mean(pick(3, "api")), "L3_response": _mean(pick(3, "response")),
    }
    n_l3 = 50  # canonical fixed denominator (50 L3 dialogues)
    dims["L3_sample_success"] = round(100.0 * (n_l3 - len(l3_errored)) / n_l3, 2)

    call_levels = [dims["L1_call"], dims["L2_call"], dims["L3_call"]]
    headline = _mean_pct([v for v in call_levels if v is not None])

    res["accuracy"] = headline if headline is not None else res.get("accuracy", 0.0)
    res["dimension_scores"] = dims
    res["correct_answers"] = sum(1 for t in traces if t.get("api_bank_track") == "api" and t.get("is_correct"))
    res["total_questions"] = len(traces)
    res["metric"] = ("API-Bank headline = mean execution Correctness over L1/L2/L3 (call track); "
                     "response ROUGE-L + L3 sample-success in dimension_scores. No LLM judge.")
    res["leaderboard_comparable"] = False
    return res


def _mean_pct(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 2) if vals else None
