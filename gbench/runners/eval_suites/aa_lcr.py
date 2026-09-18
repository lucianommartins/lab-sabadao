# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: aa_lcr
# Description: AA-LCR (Artificial Analysis Long Context Reasoning Benchmark - 100 multi-document reasoning questions)

"""gbench native built-in runner for aa_lcr (Long Context & Retrieval).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_AA_LCR_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import io
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict, judge_generate_cascade
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"


def _load_aa_lcr_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load AA-LCR benchmark dataset directly from HF Hub ('ArtificialAnalysis/AA-LCR')."""
    samples = []
    try:
        from huggingface_hub import hf_hub_download
        csv_file = hf_hub_download(repo_id="ArtificialAnalysis/AA-LCR", filename="AA-LCR_Dataset.csv", repo_type="dataset")
        zip_path = hf_hub_download(repo_id="ArtificialAnalysis/AA-LCR", filename="extracted_text/AA-LCR_extracted-text.zip", repo_type="dataset")

        docs_cache = {}
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                base_name = os.path.basename(name)
                if base_name and not name.endswith("/"):
                    docs_cache[base_name] = zf.read(name).decode("utf-8", errors="ignore")

        rows = []
        with open(csv_file, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                rows.append(r)

        # Stratified, not a contiguous head (audit RC-1).
        rows = stratified_sample(rows, limit, lambda r: (r or {}).get("document_category"), seed="aa_lcr")

        for item in rows:
            category = item.get("document_category", "general")
            q_id = item.get("question_id", "")
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            file_names = [f.strip() for f in str(item.get("data_source_filenames", "")).split(";") if f.strip()]

            doc_texts = []
            for fn in file_names:
                if fn in docs_cache:
                    doc_texts.append(f"--- Document: {fn} ---\n{docs_cache[fn]}")

            context = "\n\n".join(doc_texts)
            prompt = (
                f"[Supporting Documents]\n{context}\n\n"
                f"[Question]\n{question}\n\n"
                "Synthesize the answer based solely on the provided documents. State the final answer clearly and concisely."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, answer, {"category": category, "question_id": q_id}))
    except Exception as e:
        logger.error(f"Failed to load dataset for aa_lcr: {e}")
        raise RuntimeError(f"Could not load dataset for aa_lcr: {e}") from e

    if not samples:
        raise RuntimeError("Dataset for aa_lcr returned empty rows")

    logger.info(f"Loaded {len(samples)} aa_lcr samples.")
    return samples


def _eval_aa_lcr(response_text: str, gold_target: str) -> bool:
    """Fallback local evaluation if GEMINI_API_KEY is not available (unit tests/offline)."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    if gold.lower() in resp.lower():
        return True
    return False


async def _async_judge_aa_lcr(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical ArtificialAnalysis AA-LCR 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for AA-LCR.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_aa_lcr(cleaned, gold)
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
        question = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")

        # The question itself has to be in the prompt: without it the judge is comparing
        # two strings with no idea what was asked, so it cannot tell a differently-worded
        # correct answer from a wrong one (beam_128k already passes the question).
        prompt = (
            "You are an expert evaluator for the Artificial Analysis Long-Context Reasoning (AA-LCR) benchmark.\n"
            "Evaluate whether the candidate model response accurately answers the question based on the ground truth answer.\n\n"
            f"Question:\n{question[-4000:]}\n\n"
            f"Ground Truth Answer:\n{gold}\n\n"
            f"Candidate Model Response:\n{resp_text}\n\n"
            "Output exactly:\n"
            "Grade: CORRECT / INCORRECT"
        )

        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from base's pass/fail
            # accuracy denominator, NOT scored INCORRECT.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return

        grade_str = text.strip().upper()
        is_corr = parse_grade_verdict(grade_str)
        trace["is_correct"] = is_corr
        trace["judge_grade"] = grade_str
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [AA_LCR]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_aa_lcr(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native aa_lcr evaluation suite."""
    skip = gemini_required_skip("aa_lcr", model_name)
    if skip is not None:
        return skip
    samples = _load_aa_lcr_samples(limit=limit)
    return run_eval_suite(
        eval_name="aa_lcr",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_aa_lcr,
        async_eval_fn=_async_judge_aa_lcr,
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
