# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: t_eval
# Description: T-Eval (Shanghai AI Lab 6 Separable Sub-Skills for Tool-Augmented LLMs)

"""gbench native built-in runner for t_eval (Tool Use & Function Calling).

Canonical T-Eval (Chen et al., ACL 2024; arXiv:2312.14033; github.com/open-compass/T-Eval;
HF `lovesnowbest/T-Eval`) measures SIX separable sub-skills over 8 English data files. Each
sample is a pre-baked single-turn context; the model emits ONE completion scored against
ground_truth by the authors' own deterministic evaluators (NO LLM judge). The upstream
evaluators are vendored verbatim in `_teval_vendor/` and driven per sample here; the six
dimensions are combined exactly as upstream `convert_results.py`:

  Instruct  = mean([(json_format+json_args_em)/2, (string_format+string_args_em)/2])
  Plan      = mean([plan_str.f1, plan_json.f1])                (bertscore graph match + LIS)
  Reason    = mean([reason_str.thought, rru_json.thought])     (sentence-embedding cosine)
  Retrieve  = mean([retrieve_str.name, rru_json.name])
  Understand= mean([understand_str.args, rru_json.args])
  Review    = review_str.review_quality
  Overall   = mean of the six  (headline `accuracy`)

Plan and Reason need the `all-mpnet-base-v2` sentence-transformers model (downloaded once,
~420MB; shared across evaluators; runs on GPU if available). No API key, no judge, no server.

Sampling: temperature defaults to 0.0 (greedy, matching upstream do_sample=False) for
no-think runs and 1.0 with `--thinking`; the measurement is in base.DEFAULT_TEMPERATURE.
Override for a whole run with `--temperature`, or for this suite with
`GBENCH_T_EVAL_TEMPERATURE`.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import run_eval_suite
from .sampling import stratified_sample
from ._teval_vendor import (InstructEvaluator, PlanningEvaluator,
                            ReasonRetrieveUnderstandEvaluator, ReviewEvaluator)
from ._teval_vendor import planning_evaluator as _pe_mod
from ._teval_vendor import reason_retrieve_understand_evaluator as _rru_mod

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/t_eval.md"

_HF_REPO = "lovesnowbest/T-Eval"
_BERT_MODEL = "all-mpnet-base-v2"

# (dimension key, data file stem). All 8 English files count toward the headline.
_FILES: List[Tuple[str, str]] = [
    ("instruct", "instruct_v2"),
    ("plan_str", "plan_str_v2"),
    ("plan_json", "plan_json_v2"),
    ("reason_str", "reason_str_v2"),
    ("retrieve_str", "retrieve_str_v2"),
    ("understand_str", "understand_str_v2"),
    ("rru_json", "reason_retrieve_understand_json_v2"),
    ("review_str", "review_str_v2"),
]

# The vendored evaluators load SentenceTransformer in __init__; neuter that so construction
# is cheap, and inject a single shared model into the evaluators that actually need it.
_pe_mod.SentenceTransformer = lambda *a, **k: None
_rru_mod.SentenceTransformer = lambda *a, **k: None

_SHARED_ST = None


def _shared_st():
    """Lazily load one all-mpnet-base-v2 shared by Plan + Reason scorers.

    Prefers GPU but falls back to CPU when the GPU has no free VRAM - a full vLLM server owns all
    of it (gpu-memory-utilization 0.95), so `torch.cuda.is_available()` is True yet the load OOMs.
    The embedder scores identically on either device (only slower on CPU), so this never changes the
    measurement. Force a device with GBENCH_T_EVAL_DEVICE=cpu|cuda."""
    global _SHARED_ST
    if _SHARED_ST is None:
        import os
        from sentence_transformers import SentenceTransformer
        forced = os.environ.get("GBENCH_T_EVAL_DEVICE", "").strip().lower()
        if forced in ("cpu", "cuda"):
            device = forced
        else:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        logger.info("t_eval: loading %s on %s (one-time)", _BERT_MODEL, device)
        try:
            _SHARED_ST = SentenceTransformer(_BERT_MODEL, device=device)
        except Exception as e:  # CUDA OOM etc.: the served model owns the VRAM -> use CPU
            if device == "cuda" and forced != "cuda":
                logger.warning("t_eval: GPU load of %s failed (%s: %s) - the served model likely "
                               "owns all VRAM; falling back to CPU (identical scores, slower).",
                               _BERT_MODEL, type(e).__name__, str(e)[:120])
                _SHARED_ST = SentenceTransformer(_BERT_MODEL, device="cpu")
            else:
                raise
    return _SHARED_ST


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant", "function": "user"}


def _to_chat(origin_prompt: Any) -> List[Dict[str, str]]:
    """Map a T-Eval origin_prompt to a chat message list: `function`->`user` (upstream renders
    them identically), then merge consecutive same-role turns so the served chat template does
    not reject adjacent same-role messages."""
    if not isinstance(origin_prompt, list):
        return [{"role": "user", "content": str(origin_prompt)}]
    msgs: List[Dict[str, str]] = []
    for turn in origin_prompt:
        if not isinstance(turn, dict):
            continue
        role = _ROLE_MAP.get(turn.get("role", "user"), "user")
        content = turn.get("content", "") or ""
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] = (msgs[-1]["content"] + "\n\n" + content).strip()
        else:
            msgs.append({"role": role, "content": content})
    return msgs or [{"role": "user", "content": ""}]


def _load_file_samples(
    dim: str, stem: str, limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load one T-Eval file; carry the WHOLE sample (minus origin_prompt) in the gold JSON so
    the vendored evaluator gets the exact keys it needs (template / meta / meta_data / gt)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(_HF_REPO, filename=f"data/{stem}.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    if not rows:
        raise RuntimeError(f"t_eval: {stem}.json returned empty rows")
    rows = stratified_sample(rows, limit, None, seed=f"t_eval:{dim}")

    samples = []
    for row in rows:
        msgs = _to_chat(row.get("origin_prompt"))
        sample = {k: v for k, v in row.items() if k != "origin_prompt"}
        gold = json.dumps({"dimension": dim, "sample": sample})
        samples.append((msgs, gold, {"category": f"t_eval::{dim}"}))
    return samples


# --------------------------------------------------------------------------- #
# Scoring - drive each vendored evaluator per sample, return the file's aggregate.
# --------------------------------------------------------------------------- #
def _datum(trace: Dict[str, Any]) -> Dict[str, Any]:
    try:
        blob = json.loads(trace.get("gold_answer") or "{}")
    except Exception:
        blob = {}
    d = dict(blob.get("sample") or {})
    d["prediction"] = trace.get("response_text") or ""
    return d


def _score_file(dim: str, traces: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics for one file, storing each per-sample metric back on its trace."""
    if dim == "instruct":
        ev = InstructEvaluator(dataset_path=None)
        results = []
        for t in traces:
            m = ev._evaluate(ev._process_response(_datum(t)))
            t["t_eval_metric"] = m
            results.append(m)
        agg = ev._post_process(results)

    elif dim in ("plan_str", "plan_json"):
        ev = PlanningEvaluator(dataset_path=None)
        ev.sentence_model = _shared_st()
        results = []
        for t in traces:
            ds, _err = ev._process_response(_datum(t))
            m = ev._evaluate(ds)
            t["t_eval_metric"] = m
            results.append(m)
        agg = ev._post_process(results)

    elif dim in ("reason_str", "retrieve_str", "understand_str", "rru_json"):
        eval_type = {"reason_str": "reason", "retrieve_str": "retrieve",
                     "understand_str": "understand", "rru_json": "reason"}[dim]
        prompt_type = "json" if dim == "rru_json" else "str"
        ev = ReasonRetrieveUnderstandEvaluator(
            dataset_path=None, default_prompt_type=prompt_type, eval_type=eval_type)
        if dim in ("reason_str", "rru_json"):  # thought cosine needs the embedder
            ev.sentence_model = _shared_st()
        data_samples = []
        for t in traces:
            ds, _err = ev._process_response(_datum(t))
            data_samples.append(ev._evaluate(ds))  # main class scores in _post_process
        agg = ev._post_process(data_samples)

    elif dim == "review_str":
        ev = ReviewEvaluator(dataset_path=None)
        results = []
        for t in traces:
            m = ev._evaluate(ev._process_response(_datum(t)))
            t["t_eval_metric"] = m
            results.append(m)
        agg = ev._post_process(results)
    else:
        raise ValueError(f"t_eval: unknown dimension {dim}")

    return {k: float(v) for k, v in dict(agg).items()}


def _f(d: Dict[str, float], k: str) -> float:
    try:
        return float(d.get(k, 0.0))
    except Exception:
        return 0.0


def _safe_mean(vals: List[float]) -> float:
    vs = [v for v in vals if v == v]  # drop NaN (empty response_format group at tiny --eval-limit)
    return float(np.mean(vs)) if vs else 0.0


def run_t_eval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run all 6 T-Eval dimensions (8 files) and report the canonical Overall as accuracy."""
    import importlib.util as _ilu
    from .swebench_common import infra_required
    if _ilu.find_spec("sentence_transformers") is None:
        raise infra_required(
            "t_eval",
            "requires sentence-transformers (+torch) for the canonical BERTScore semantic "
            "matching used by the Plan/Reason/Retrieve/Understand scorers",
            "docs/evals/t_eval.md",
        )

    limit = kwargs.get("limit")
    max_out = kwargs.get("max_output_tokens", 2048)

    file_metrics: Dict[str, Dict[str, float]] = {}
    all_traces: List[Dict[str, Any]] = []
    base_res: Optional[Dict[str, Any]] = None
    statuses: List[str] = []
    total_n = 0

    # --eval-limit is a TOTAL budget across the 8 required T-Eval dimensions, not a per-file
    # cap (the headline averages every dimension, so we can't drop any). Distribute it so the
    # total stays ~limit while each dimension keeps >=1 sample: ceil(limit / n_files) per file.
    _n_files = len(_FILES)
    _per_file = None if not limit else max(1, (int(limit) + _n_files - 1) // _n_files)

    for dim, stem in _FILES:
        samples = _load_file_samples(dim, stem, _per_file)
        logger.info("t_eval: [%s] %d samples", dim, len(samples))
        res = run_eval_suite(
            eval_name="t_eval",
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=samples,
            eval_fn=lambda pred, gold: False,  # placeholder; real metric computed below
            thinking=enable_thinking,
            extra_payload=kwargs.get("extra_payload"),
            limit=None,  # already limited per file in the loader
            max_output_tokens=max_out,
        )
        if base_res is None:
            base_res = res
        statuses.append(res.get("status", "success"))
        traces = res.get("sample_traces", []) or []
        for t in traces:
            t["t_eval_dimension"] = dim
        total_n += len(traces)
        all_traces.extend(traces)
        file_metrics[dim] = _score_file(dim, traces)
        logger.info("t_eval: [%s] metrics=%s", dim, file_metrics[dim])

    fr = file_metrics
    instruct = _safe_mean([
        (_f(fr["instruct"], "json_format_metric") + _f(fr["instruct"], "json_args_em_metric")) / 2,
        (_f(fr["instruct"], "string_format_metric") + _f(fr["instruct"], "string_args_em_metric")) / 2,
    ])
    plan = _safe_mean([_f(fr["plan_str"], "f1_score"), _f(fr["plan_json"], "f1_score")])
    reason = _safe_mean([_f(fr["reason_str"], "thought"), _f(fr["rru_json"], "thought")])
    retrieve = _safe_mean([_f(fr["retrieve_str"], "name"), _f(fr["rru_json"], "name")])
    understand = _safe_mean([_f(fr["understand_str"], "args"), _f(fr["rru_json"], "args")])
    review = _f(fr["review_str"], "review_quality")
    overall = _safe_mean([instruct, plan, reason, retrieve, understand, review])

    dimension_scores = {
        "instruct": round(instruct * 100, 2), "plan": round(plan * 100, 2),
        "reason": round(reason * 100, 2), "retrieve": round(retrieve * 100, 2),
        "understand": round(understand * 100, 2), "review": round(review * 100, 2),
    }

    result = dict(base_res or {})
    result["benchmark_type"] = "eval"
    result["eval_name"] = "t_eval"
    result["model_name"] = model_name
    result["status"] = "success" if all(s == "success" for s in statuses) else "completed_with_errors"
    result["accuracy"] = round(overall * 100, 2)
    result["total_questions"] = total_n
    result["correct_answers"] = round(overall * total_n)
    result["dimension_scores"] = dimension_scores
    result["t_eval_file_metrics"] = file_metrics
    result["metric"] = ("T-Eval Overall = mean of 6 dimensions (Instruct/Plan/Reason/Retrieve/"
                        "Understand/Review), str & json weighted equally; canonical convert_results")
    result["sample_traces"] = all_traces
    return result
