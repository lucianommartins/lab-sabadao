# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: beam_128k
# Description: BEAM (Beyond a Million Tokens: Long-Term Memory in LLMs) - 400 conversations at the 100K context band (the repo ships 100K/500K/1M; 100K is the closest available to a 128K window)

"""gbench native built-in runner for beam_128k (Long Context & Retrieval).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BEAM_128K_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict, judge_generate_cascade
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"


def _load_beam_128k_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BEAM-128K benchmark dataset directly from HF Hub ('Mohammadta/BEAM')."""
    samples = []
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        parquet_file = hf_hub_download(repo_id="Mohammadta/BEAM", filename="data/100K-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            conv_id = str(row.get("conversation_id", ""))
            chat = row.get("chat", [])
            chat_list = list(chat) if hasattr(chat, "__iter__") else []

            # Format conversation transcript
            conv_turns = []
            for turn in chat_list:
                if isinstance(turn, dict):
                    role = turn.get("role", "user").capitalize()
                    content = turn.get("content", "").strip()
                    conv_turns.append(f"[{role}]: {content}")
                else:
                    conv_turns.append(str(turn))

            history_text = "\n\n".join(conv_turns)

            raw_pq = row.get("probing_questions", {})
            if isinstance(raw_pq, str):
                try:
                    pq_dict = ast.literal_eval(raw_pq)
                except Exception:
                    pq_dict = json.loads(raw_pq)
            else:
                pq_dict = raw_pq

            if isinstance(pq_dict, dict):
                for memory_ability, q_items in pq_dict.items():
                    if isinstance(q_items, list):
                        for q_item in q_items:
                            if isinstance(q_item, dict):
                                question = q_item.get("question", "").strip()
                                ideal_resp = q_item.get("ideal_response", "") or q_item.get("answer", "")
                                # BEAM scores against a per-question RUBRIC (a list of nuggets),
                                # not a single ideal string. Carry both so the judge can score
                                # nugget coverage; keep the ideal for the no-key fallback.
                                rubric = _parse_rubric(q_item.get("rubric"))
                                gold_payload = json.dumps({"ideal": str(ideal_resp),
                                                           "rubric": rubric})
                                if question:
                                    prompt = (
                                        f"[Conversation History]\n{history_text}\n\n"
                                        f"[Memory Probing Question]\n{question}\n\n"
                                        "Answer the probing question based strictly on the conversation history above:"
                                    )
                                    messages = [{"role": "user", "content": prompt}]
                                    samples.append(
                                        (
                                            messages,
                                            gold_payload,
                                            {
                                                "category": memory_ability,
                                                "conversation_id": conv_id,
                                                "difficulty": q_item.get("difficulty", "medium"),
                                            },
                                        )
                                    )
    except Exception as e:
        logger.error(f"Failed to load dataset for beam_128k: {e}")
        raise RuntimeError(f"Could not load dataset for beam_128k: {e}") from e

    if not samples:
        raise RuntimeError("Dataset for beam_128k returned empty rows")

    # Stratified, not a contiguous head (audit RC-1). `samples` holds built
    # (messages, gold, meta) tuples, not raw dict rows, so the key must read meta - `r.get`
    # raised "'tuple' object has no attribute 'get'" and killed the whole suite.
    samples = stratified_sample(
        samples, limit,
        lambda s: (s[2] or {}).get("memory_ability") if len(s) > 2 and isinstance(s[2], dict) else None,
        seed="beam_128k")

    logger.info(f"Loaded {len(samples)} beam_128k samples.")
    return samples


def _parse_rubric(raw: Any) -> List[str]:
    """BEAM's `rubric` ships as a (stringified) list of nugget criteria."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    for loader in (ast.literal_eval, json.loads):
        try:
            val = loader(text)
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            continue
    return [text]


def _beam_gold(gold_target: Any) -> Tuple[str, List[str]]:
    """(ideal_response, rubric_nuggets) from the JSON gold payload."""
    try:
        obj = json.loads(gold_target)
        if isinstance(obj, dict):
            return str(obj.get("ideal") or ""), _parse_rubric(obj.get("rubric"))
    except Exception:
        pass
    return str(gold_target or ""), []


def _eval_beam_128k(response_text: str, gold_target: str) -> bool:
    """No-key deterministic fallback (labelled judge_fallback); NOT the canonical nugget metric."""
    if not response_text:
        return False
    resp = response_text.strip()
    ideal, _rubric = _beam_gold(gold_target)
    gold = ideal.strip()
    if not gold:
        return False
    if gold.lower() in resp.lower():
        return True
    if "no information" in gold.lower() or "not mentioned" in gold.lower():
        if any(w in resp.lower() for w in ["not mentioned", "no information", "not found", "cannot determine"]):
            return True
    return False


async def _async_judge_beam_128k(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical Meta BEAM 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for BEAM.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_beam_128k(cleaned, gold)
            trace["scoring_mode"] = "judge_fallback"
            trace["status"] = "OK"
        return

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if not resp_text:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[-1].get("content", "") if messages else ""
        ideal, rubric = _beam_gold(trace.get("gold_answer"))
        if not rubric:
            rubric = [ideal] if ideal else []
        rubric_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(rubric)) or "(none)"

        prompt = (
            "You are grading a model's answer to a long-context memory probing question "
            "against a RUBRIC of required facts (nuggets). Score EACH rubric item as met "
            "(1.0), partially met (0.5), or not met (0.0) by the model's response, then "
            "report the AVERAGE across items.\n\n"
            f"Question:\n{question}\n\n"
            f"Ideal Answer:\n{ideal}\n\n"
            f"Rubric items:\n{rubric_text}\n\n"
            f"Model Response:\n{resp_text}\n\n"
            "Output exactly:\nCoverage: <a number from 0.00 to 1.00>"
        )

        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return
        m = re.search(r"coverage\s*[:=]?\s*([01](?:\.\d+)?|0?\.\d+)", text, re.IGNORECASE)
        if not m:
            trace["judge_grade"] = "JUDGE_OUTAGE"     # unparseable -> excluded, not 0
            trace["status"] = "OK"
            pbar.update(1)
            return
        coverage = max(0.0, min(1.0, float(m.group(1))))
        trace["beam_score"] = round(coverage, 4)
        trace["is_correct"] = coverage >= 0.5          # secondary pass_rate only
        trace["judge_grade"] = f"COVERAGE_{coverage:.2f}"
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [BEAM_128K]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_beam_128k(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native beam_128k evaluation suite."""
    from .metrics import finalize_mean_metric
    skip = gemini_required_skip("beam_128k", model_name)
    if skip is not None:
        return skip
    samples = _load_beam_128k_samples(limit=limit)
    result = run_eval_suite(
        eval_name="beam_128k",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_beam_128k,
        async_eval_fn=_async_judge_beam_128k,
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # Canonical BEAM headline = mean per-rubric nugget coverage (0/0.5/1 per nugget), not a
    # binary pass rate. (event_ordering's specialised Kendall-tau-b x F1 remains a refinement;
    # rubric coverage still scores those rows.)
    return finalize_mean_metric(
        result,
        metric_name="mean rubric-nugget coverage (canonical BEAM); event_ordering Kendall-tau pending",
        score_key="beam_score",
        secondary_key="pass_rate_at_0.5",
    )
