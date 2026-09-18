# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: healthbench
# Description: HealthBench (OpenAI) - clinical conversations graded against physician rubrics

"""gbench native built-in runner for healthbench (Medical & Health).

Canonical HealthBench (openai/healthbench): each conversation ships a physician
rubric of weighted criteria. A grader model judges which criteria the response
meets; the per-conversation score is clip(sum of met points / sum of positive
points, 0, 1). Requires GEMINI_API_KEY (skips cleanly without it); no heuristic
auto-pass fallback.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_HEALTHBENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, strip_thinking_tags, DEFAULT_JUDGE_MODEL, judge_config, judge_generate_cascade
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Medical & Health"
_PASS_THRESHOLD = 0.5  # per-conversation rubric score counted as a "pass" for accuracy


def _load_healthbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load HealthBench conversations + weighted rubrics; raises on load/schema failure."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="openai/healthbench",
            filename="2025-05-07-06-14-12_oss_eval.jsonl",
            repo_type="dataset",
        )
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load dataset for healthbench: {e}")
        raise RuntimeError(f"Could not load dataset for healthbench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for healthbench returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("example_tags"), seed="healthbench")

    samples = []
    for item in rows:
        raw_prompt = item.get("prompt")
        if isinstance(raw_prompt, list):
            messages = raw_prompt
        elif raw_prompt:
            messages = [{"role": "user", "content": str(raw_prompt)}]
        else:
            raise RuntimeError("healthbench: unexpected schema (missing 'prompt')")
        rubrics = item.get("rubrics") or []
        parsed = [{"criterion": r.get("criterion", ""), "points": r.get("points", 0)}
                  for r in rubrics if isinstance(r, dict) and r.get("criterion")]
        if not parsed:
            raise RuntimeError(
                "healthbench: unexpected schema (missing 'rubrics'); refusing to fabricate"
            )
        tags = item.get("example_tags") or []
        cat = tags[0] if tags else "clinical"
        samples.append((messages, json.dumps(parsed), {"category": cat}))

    logger.info(f"Loaded {len(samples)} healthbench samples.")
    return samples


def _make_scorer(judge_model: str = DEFAULT_JUDGE_MODEL, judge_concurrency: int = 32):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        from tqdm import tqdm
        import asyncio

        semaphore = asyncio.Semaphore(judge_concurrency)

        async def _judge(trace: Dict[str, Any], pbar) -> None:
            resp = strip_thinking_tags(trace.get("response_text") or "")
            try:
                rubric = json.loads(trace.get("gold_answer") or "[]")
            except Exception:
                rubric = []
            if not resp or not rubric:
                trace["is_correct"] = False
                trace["healthbench_score"] = 0.0
                trace["status"] = "OK"
                pbar.update(1)
                return

            messages = trace.get("messages", [])
            convo = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                if isinstance(m, dict)
            )
            criteria_block = "\n".join(
                f"{i}. {c['criterion']}" for i, c in enumerate(rubric)
            )
            prompt = (
                "You are a physician grading an AI assistant's response to a health "
                "conversation against a rubric. For each numbered criterion, decide "
                "whether the assistant's response satisfies it.\n\n"
                f"[Conversation]\n{convo}\n\n[Assistant Response]\n{resp}\n\n"
                f"[Rubric criteria]\n{criteria_block}\n\n"
                "Return ONLY a JSON object of the form {\"met\": [true, false, ...]} "
                "with exactly one boolean per criterion, in order."
            )

            met: List[bool] = []
            async with semaphore:
                text, _judge_used = await judge_generate_cascade(prompt)
            if text is None:
                # Judge cascade exhausted (infra outage): excluded from the mean below,
                # NOT scored 0. base.run_eval_suite also drops it from pass/fail accuracy.
                trace["judge_grade"] = "JUDGE_OUTAGE"
                trace["status"] = "OK"
                pbar.update(1)
                return
            try:
                m = re.search(r"\{[\s\S]*\}", text)
                met = json.loads(m.group(0)).get("met", []) if m else []
            except Exception:
                met = []

            total_pos = sum(c["points"] for c in rubric if c["points"] > 0) or 1
            achieved = 0
            for i, c in enumerate(rubric):
                if i < len(met) and met[i]:
                    achieved += c["points"]
            score = max(0.0, min(1.0, achieved / total_pos))
            trace["healthbench_score"] = round(score, 4)
            trace["is_correct"] = bool(score >= _PASS_THRESHOLD)
            trace["status"] = "OK"
            pbar.update(1)

        with tqdm(total=len(sample_traces), desc="Judging [HEALTHBENCH]") as pbar:
            await asyncio.gather(*[_judge(t, pbar) for t in sample_traces])
    return _score


def run_healthbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run HealthBench rubric grading (skips without GEMINI_API_KEY)."""
    skip = gemini_required_skip("healthbench", model_name)
    if skip is not None:
        return skip
    samples = _load_healthbench_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="healthbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # HealthBench's canonical score is the MEAN weighted rubric score, not the share of
    # conversations clearing an arbitrary 0.5. The pass rate was the headline `accuracy`
    # and the real metric only a secondary field, so a model averaging 0.49 everywhere and
    # one scoring 0 everywhere both reported 0%.
    # Judge outages carry no healthbench_score and must NOT be averaged in as 0.0 - that
    # would let an infra outage depress the canonical mean. They are excluded here (and from
    # base's pass/fail accuracy) and surfaced via judge_outages instead.
    traces = result.get("sample_traces", [])
    scores = [t["healthbench_score"] for t in traces
              if t.get("judge_grade") != "JUDGE_OUTAGE" and "healthbench_score" in t]
    result["pass_rate_at_0.5"] = result.get("accuracy")
    result["metric"] = "mean weighted rubric score (canonical HealthBench)"
    if scores:
        mean_score = sum(scores) / len(scores)
        result["healthbench_mean_rubric_score"] = round(mean_score, 4)
        result["accuracy"] = round(mean_score * 100.0, 2)
    return result
