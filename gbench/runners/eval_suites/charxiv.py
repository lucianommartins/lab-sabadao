# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: charxiv
# Description: CharXiv (Princeton Complex Academic Chart Reasoning & Numerical Understanding)

"""gbench native built-in runner for charxiv (Multimodal & Vision).

Canonical CharXiv (Wang et al., NeurIPS 2024 D&B; arXiv:2406.18521;
github.com/princeton-nlp/CharXiv; HF princeton-nlp/CharXiv) has TWO tracks over the
1000-chart `validation` split:

  * Descriptive - 4 questions per chart (one of 19 templates each), 4000 total.
  * Reasoning   - 1 open-ended question per chart, 1000 total.

Both are single-shot multimodal VQA graded by an LLM judge against a per-question rubric.
This runner fans each chart out into its 5 canonical questions, sends the exact upstream
prompt templates (vendored in `charxiv_constants.py`), and grades with the exact upstream
descriptive/reasoning rubrics via the gbench judge cascade. It reports the descriptive and
reasoning accuracies SEPARATELY (the paper reports no combined figure; the leaderboard is
sorted by Reasoning, so that is the headline `accuracy`).

Judge backend: canonical CharXiv pins gpt-4o-2024-05-13; gbench grades with its standard
Gemini judge cascade - the reference grader gbench uses across all judged suites, by
convention (a grader choice, not a fidelity defect). The prompts, rubrics and JSON score
contract are ported verbatim. Numbers are therefore CharXiv-protocol-faithful, graded by
gbench's cascade rather than the paper's GPT-4o, i.e. a gbench-internal number rather than a
like-for-like leaderboard entry (leaderboard_comparable=False); label accordingly when
comparing to the paper's GPT-4o leaderboard.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with `--temperature`, or for this
suite alone with `GBENCH_CHARXIV_TEMPERATURE`, which takes precedence over both. LLM-judge
grading is pinned at 0.0 and is not affected by either.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import (run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL,
                   judge_generate_cascade, judge_config)
from .sampling import stratified_sample
from .dataset_utils import extract_lossless_image_b64
from .charxiv_constants import (DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_PREFIX,
                                DESCRIPTIVE_GRADING_QMAP, DESCRIPTIVE_GRADING_ICL,
                                REASONING_RESP_INST, REASONING_GRADING_PREFIX,
                                REASONING_GRADING_INST)

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"


# --------------------------------------------------------------------------- #
# Canonical helpers, ported verbatim from CharXiv src/descriptive_utils.py and
# src/reasoning_utils.py (behavior preserved exactly).
# --------------------------------------------------------------------------- #
def get_rubric(qid: int) -> str:
    """The in-context grading rubric for a descriptive template id (upstream get_rubric)."""
    if qid in (1,):
        return DESCRIPTIVE_GRADING_ICL['title']
    if qid in (2, 3, 4, 5, 6, 7):
        return DESCRIPTIVE_GRADING_ICL['ocr']
    if qid in (8, 9, 10, 12, 14, 15, 17, 19):
        return DESCRIPTIVE_GRADING_ICL['quant']
    if qid in (11,):
        return DESCRIPTIVE_GRADING_ICL['bool']
    if qid in (13,):
        return DESCRIPTIVE_GRADING_ICL['enum']
    if qid in (16,):
        return DESCRIPTIVE_GRADING_ICL['trend']
    if qid in (18,):
        return DESCRIPTIVE_GRADING_ICL['layout']
    raise ValueError(f"CharXiv: no rubric for descriptive qid {qid}")


def descriptive_query_helper(qid: int, subplot_loc: Any) -> str:
    """Build the descriptive question from template id + subplot location (upstream)."""
    if qid in (18, 19):
        # layout / #subplots questions take no subplot prefix
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            prefix = "For the current plot, "
        else:
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"CharXiv: invalid subplot_loc: {subplot_loc!r}")
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def get_number_instruction(answer: str) -> str:
    """Decimal/integer format instruction for reasoning inst_category 4 (upstream)."""
    base = str(answer).split('.')
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    if whole is not None and decimal is None:
        return "* Your final answer must be an exact integer."
    if whole is not None and decimal is not None:
        return f"* Your final answer must be a number with {len(decimal)} decimal places."
    raise ValueError(f"CharXiv: invalid answer for number instruction: {answer!r}")


def _mk_sample(b64: Optional[str], question: str, gold_blob: Dict[str, Any],
               track: str) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """One (messages, gold, meta) sample. Scoring metadata rides in the gold JSON, NOT the
    meta dict - base.run_eval_suite merges the meta dict (minus `category`) into the request
    payload, so anything but `category` there would leak into the model request."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": question}]
    if b64:
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return ([{"role": "user", "content": content}], json.dumps(gold_blob), {"category": track})


def _load_charxiv_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load CharXiv validation and fan each chart into its 5 canonical questions."""
    rows = []
    try:
        from datasets import load_dataset, Image
        ds = load_dataset("princeton-nlp/CharXiv", split="validation")
        try:
            ds = ds.cast_column("image", Image(decode=False))
        except Exception:
            pass
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for charxiv: {e}")
        raise RuntimeError(f"Could not load dataset for charxiv: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for charxiv returned empty rows")

    # Limit at the CHART level (stratified by subject), then fan out - so --eval-limit N
    # yields N charts x 5 questions, not a truncated question list.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="charxiv")

    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    for item in rows:
        b64 = extract_lossless_image_b64(item.get("image"))

        loc = item.get("subplot_loc")
        if loc in (None, "", []):
            loc = [item.get("subplot_row"), item.get("subplot_col")]

        # Descriptive: 4 questions/chart (template ids live in descriptive_q1..q4).
        for i in range(1, 5):
            qraw = item.get(f"descriptive_q{i}")
            if qraw is None:
                continue
            qid = int(qraw)
            gold = "" if item.get(f"descriptive_a{i}") is None else str(item.get(f"descriptive_a{i}"))
            question = descriptive_query_helper(qid, loc)
            samples.append(_mk_sample(b64, question,
                                      {"track": "descriptive", "qid": qid, "answer": gold},
                                      "descriptive"))

        # Reasoning: 1 open-ended question/chart.
        rq = str(item.get("reasoning_q") or "").strip()
        ra = str(item.get("reasoning_a") or "").strip()
        ic_raw = item.get("reasoning_a_type")
        if rq and ic_raw is not None:
            ic = int(ic_raw)
            if ic == 4:
                question = REASONING_RESP_INST[ic].format(rq, get_number_instruction(ra))
            else:
                question = REASONING_RESP_INST[ic].format(rq)
            samples.append(_mk_sample(b64, question,
                                      {"track": "reasoning", "inst_category": ic,
                                       "raw_question": rq, "answer": ra},
                                      "reasoning"))

    if not samples:
        raise RuntimeError("CharXiv produced no questions")
    logger.info("Loaded %d charxiv questions (%d charts x descriptive+reasoning).",
                len(samples), len(rows))
    return samples


# --------------------------------------------------------------------------- #
# Grading - build the exact upstream descriptive/reasoning grading prompts.
# --------------------------------------------------------------------------- #
def _build_descriptive_grading_prompt(qid: int, resp: str, answer: str) -> str:
    prefix = (DESCRIPTIVE_GRADING_PREFIX
              .replace("<|NUM_TRIPLETS|>", "1")
              .replace("<|OVERARCHING_QUESTION|>", DESCRIPTIVE_GRADING_QMAP[qid])
              .replace("<|JSON_KEYS|>", str(["extract_answer_T1", "score_T1"])))
    body = "T1:\nResponse 1: {}\nGround Truth 1: {}\n\n".format(resp, answer)
    return prefix + get_rubric(qid) + body


def _build_reasoning_grading_prompt(inst_category: int, raw_question: str,
                                    answer: str, resp: str) -> str:
    return (REASONING_GRADING_PREFIX
            + REASONING_GRADING_INST[inst_category]
            .replace("<|question|>", raw_question)
            .replace("<|ground_truth|>", answer)
            .replace("<|response|>", resp))


def _parse_judge_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text.strip(), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _score_from(data: Dict[str, Any], key: str) -> int:
    """Canonical: a score not in {0,1} (parse/format failure) counts as 0, in-denominator."""
    try:
        s = int(data[key])
    except Exception:
        return -1
    return s if s in (0, 1) else -1


def _eval_charxiv(response_text: str, gold_target: str) -> bool:
    """No-key/phase-1 fallback: relaxed match on the answer inside the gold blob.

    Canonical CharXiv is judge-graded; `run_charxiv` hard-requires GEMINI_API_KEY, so this
    only stands in for the (skipped) no-judge path and the pre-judge phase-1 pass.
    """
    from .vqa_common import eval_relaxed, extract_short_answer
    if not response_text or not gold_target:
        return False
    try:
        answer = str(json.loads(gold_target).get("answer") or "")
    except Exception:
        answer = str(gold_target)
    return eval_relaxed(extract_short_answer(response_text), answer)


async def _async_judge_charxiv(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 32,
) -> None:
    """Canonical descriptive + reasoning grading via the gbench judge cascade (JSON scores)."""
    import asyncio
    from tqdm import tqdm

    have_key = bool(os.environ.get("GEMINI_API_KEY"))
    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = judge_config(response_mime_type="application/json")

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        try:
            blob = json.loads(trace.get("gold_answer") or "{}")
        except Exception:
            blob = {}
        track = blob.get("track") or trace.get("category")
        answer = str(blob.get("answer") or "")
        resp = trace.get("response_text")

        if not resp:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            trace["status"] = "OK"
            pbar.update(1)
            return

        if not have_key:
            # Dead path (gemini_required_skip gates first), kept defensively.
            trace["is_correct"] = _eval_charxiv(resp, trace.get("gold_answer"))
            trace["scoring_mode"] = "judge_fallback"
            trace["status"] = "OK"
            pbar.update(1)
            return

        if track == "descriptive":
            prompt = _build_descriptive_grading_prompt(int(blob["qid"]), resp, answer)
            score_key = "score_T1"
            ans_key = "extract_answer_T1"
        else:
            prompt = _build_reasoning_grading_prompt(
                int(blob["inst_category"]), str(blob.get("raw_question") or ""), answer, resp)
            score_key = "score"
            ans_key = "extracted_answer"

        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt, config=cfg)

        if text is None:
            # Judge cascade exhausted (infra outage): excluded from the denominator.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return

        data = _parse_judge_json(text)
        score = _score_from(data, score_key) if data else -1
        if score < 0:
            trace["is_correct"] = False
            trace["judge_grade"] = "INVALID"
            trace["status"] = "OK"
            pbar.update(1)
            return
        trace["is_correct"] = (score == 1)
        trace["judge_grade"] = "CORRECT" if score == 1 else "INCORRECT"
        if isinstance(data, dict) and ans_key in data:
            trace["extracted_answer"] = data.get(ans_key)
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [CHARXIV]") as pbar:
        await asyncio.gather(*[_judge_single(t, pbar) for t in sample_traces])


def run_charxiv(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native CharXiv (descriptive + reasoning tracks, canonical rubrics)."""
    skip = gemini_required_skip("charxiv", model_name)
    if skip is not None:
        return skip
    samples = _load_charxiv_samples(limit=limit)
    res = run_eval_suite(
        eval_name="charxiv",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_charxiv,
        async_eval_fn=_async_judge_charxiv,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,  # already limited at chart level in the loader; do not re-cut the fan-out
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )

    # Canonical reporting: descriptive & reasoning accuracy SEPARATELY (no combined figure).
    # Category == track, so exclude only true judge outages from each denominator.
    traces = res.get("sample_traces", []) or []

    def _track_acc(name: str) -> Optional[float]:
        scored = [t for t in traces
                  if t.get("category") == name and t.get("judge_grade") != "JUDGE_OUTAGE"]
        if not scored:
            return None
        return round(100.0 * sum(1 for t in scored if t.get("is_correct")) / len(scored), 2)

    d_acc = _track_acc("descriptive")
    r_acc = _track_acc("reasoning")
    res["charxiv_descriptive_accuracy"] = d_acc
    res["charxiv_reasoning_accuracy"] = r_acc
    # Headline = Reasoning (paper's flagship + leaderboard sort key); fall back to descriptive.
    res["accuracy"] = r_acc if r_acc is not None else (d_acc if d_acc is not None else res.get("accuracy", 0.0))
    res["metric"] = ("CharXiv reasoning accuracy (headline) + descriptive accuracy; canonical "
                     "per-qid / per-inst_category rubrics graded by gbench's standard Gemini judge "
                     "cascade (the paper's grader is GPT-4o; gbench-internal, not a like-for-like "
                     "leaderboard entry)")
    res["leaderboard_comparable"] = False
    return res
