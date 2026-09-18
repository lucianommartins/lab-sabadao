# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: browsecomp
# Description: OpenAI BrowseComp (Agentic Web Search & Deep Research Fact Retrieval)

r"""gbench native built-in runner for browsecomp (Agentic & Web Research).

The smolagents/browse_comp dataset stores the 'problem' and 'answer' fields
XOR-encrypted with a per-row 'canary' password (the canonical OpenAI
simple-evals scheme). We decrypt them at load time and grade with the canonical
BrowseComp LLM grader.

What is canonical here, verified against `openai/simple-evals@main/browsecomp_eval.py`
on 2026-08-18:

* `QUERY_TEMPLATE` is byte-identical to upstream - the question followed by the
  Explanation / Exact Answer / Confidence format block, as a single user message. There is
  no extra system prompt, header or footer.
* `GRADER_TEMPLATE` matches upstream apart from writing `0%` where upstream writes the
  escaped `0|\%|`; semantically identical.
* `n_repeats` is 1 upstream, and 1 here.

* **Tools / protocol.** Upstream declares NO tools and runs a single turn - browsing-capable
  models brought their own browser, others answered closed-book from parametric memory. gbench
  matches this exactly: no tools, single turn, no injected search. (An earlier gbench build
  injected a Gemini-grounded `web_search` + tool loop because a non-browsing model scored 0/20;
  that was a deliberate deviation and made the suite non-comparable. It has been REMOVED - the
  canonical protocol is closed-book, and a low closed-book score is the correct result for a
  non-browsing model, not a bug. Upstream itself reports 0.6% for GPT-4o without browsing.)

What DEVIATES, deliberately:

* **Grader model.** Upstream uses `gpt-4.1`; gbench grades with its standard Gemini cascade
  (a gbench convention), pinned at temperature 0. gbench treats its Gemini cascade as the
  reference grader across suites, so this is a grader choice, not a fidelity defect.
* **Sampling.** Upstream's `ChatCompletionSampler` defaults to temperature 0.5; gbench uses
  the model's own shipped default (1.0) unless overridden.

Because gbench grades with its own cascade and samples at the model default, a run here is a
gbench-internal number rather than a like-for-like `gpt-4.1`-graded leaderboard entry, so it is
reported with `leaderboard_comparable=False`, consistent with the other Gemini-graded suites.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BROWSECOMP_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.

A note on the expected score: this is closed-book (no browsing), and BrowseComp is built to
defeat exactly that - upstream reports 0.6% for GPT-4o without browsing against 51.5% for Deep
Research. A low number here is the model's genuine closed-book fact-retrieval score, not a
harness bug.
"""

import base64
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, judge_config, judge_generate_cascade
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Web Research"

# Canonical OpenAI simple-evals BrowseComp templates.
QUERY_TEMPLATE = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the canary password (SHA-256 keystream)."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64 XOR-ciphertext using the canary-derived keystream."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _load_browsecomp_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BrowseComp from HF Hub (smolagents/browse_comp) and canary-decrypt each row.

    Raises on load/schema/decrypt failure (no fabricated fallback).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("smolagents/browse_comp", split="test")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for browsecomp: {e}")
        raise RuntimeError(f"Could not load dataset for browsecomp: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for browsecomp returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("problem_topic"), seed="browsecomp")

    samples = []
    for item in rows:
        enc_problem = item.get("problem")
        enc_answer = item.get("answer")
        canary = item.get("canary")
        if not enc_problem or not enc_answer or not canary:
            raise RuntimeError(
                "browsecomp: unexpected dataset schema "
                "(missing 'problem'/'answer'/'canary'); refusing to fabricate sample data"
            )
        try:
            question = _decrypt(enc_problem, canary)
            gold = _decrypt(enc_answer, canary)
        except Exception as e:
            raise RuntimeError(
                f"browsecomp: failed to canary-decrypt a row: {e}"
            ) from e
        topic = str(item.get("problem_topic") or "General")
        messages = [{"role": "user", "content": QUERY_TEMPLATE.format(question=question)}]
        samples.append((messages, gold, {"category": topic}))

    logger.info(f"Loaded {len(samples)} browsecomp samples (canary-decrypted).")
    return samples


async def _async_judge_browsecomp(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical BrowseComp LLM grader (correct: yes/no). Requires GEMINI_API_KEY."""
    from tqdm import tqdm
    import asyncio

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if not resp_text:
            # The judge is NOT called - there is nothing to grade. Named for what actually
            # happened: a bare "FAILED" reads like a judge error, and was misread as one.
            # The sample is already counted under empty_responses / failed_requests /
            # discarded_outputs; it must not also inflate judge_failures.
            trace["is_correct"] = False
            trace["judge_grade"] = "not_judged_empty_response"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")
        prompt = GRADER_TEMPLATE.format(
            question=question, response=resp_text, correct_answer=gold
        )

        async with semaphore:
            grade_str, _judge_used = await judge_generate_cascade(prompt)
        if grade_str is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy, NOT scored
            # wrong. base.run_eval_suite drops JUDGE_OUTAGE from the pass/fail denominator
            # and reports it separately, so an outage can never masquerade as a low score.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return

        match = re.search(r"correct:\s*(yes|no)", grade_str, re.IGNORECASE)
        trace["is_correct"] = bool(match and match.group(1).lower() == "yes")
        # A parseable "yes"/"no" is a real grade. If the judge replied but the reply had no
        # `correct:` line it stays unparseable => wrong (JUDGE_FAILED), the suite's existing
        # behavior; a true outage (no reply at all) was handled above as JUDGE_OUTAGE.
        if match:
            trace["judge_grade"] = match.group(0)
            trace["status"] = "OK"
        else:
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["status"] = "FAILED"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [BROWSECOMP]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_browsecomp(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute browsecomp native built-in evaluation suite."""
    skip = gemini_required_skip("browsecomp", model_name)
    if skip:
        return skip
    samples = _load_browsecomp_samples(limit=kwargs.get("limit"))
    # Canonical BrowseComp is CLOSED-BOOK: upstream (openai/simple-evals) declares no tools and
    # ran a single turn per question - browsing-capable models brought their own browser, others
    # answered from parametric memory. gbench matches that protocol exactly (no injected search,
    # no tool loop). BrowseComp is BUILT to defeat a non-browsing model - upstream reports 0.6%
    # for GPT-4o without browsing vs 51.5% for Deep Research - so a low number here is a faithful
    # result, not a harness bug: it is the closed-book score of the model under test.

    result = run_eval_suite(
        eval_name="browsecomp",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_browsecomp,
        thinking=enable_thinking,
        extra_payload=dict(kwargs.get("extra_payload") or {}),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # Protocol now matches upstream (closed-book, single turn, byte-identical templates). gbench
    # grades with its standard Gemini cascade by convention (upstream uses gpt-4.1) and samples at
    # the model's shipped default rather than upstream's fixed 0.5, so a run here is a gbench-internal
    # number rather than a like-for-like gpt-4.1-graded leaderboard entry - reported as
    # leaderboard_comparable False, consistent with the other Gemini-graded suites (e.g. charxiv).
    result["leaderboard_comparable"] = False
    return result
