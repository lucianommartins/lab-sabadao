# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: loft_x_arxiv
# Description: LOFT long-context retrieval (SciFact scientific-paper passages)

"""gbench native built-in runner for loft_x_arxiv (Long Context & Retrieval).

Canonical LOFT (Long Context Frontiers, arXiv:2406.13121) in-context retrieval.
LOFT has no arXiv task; SciFact (scientific-paper title+abstract passages, claim
-> verifying passage) is the faithful scientific-retrieval analog. The whole
corpus is placed in the prompt and the model must return the gold passage ID;
scored by canonical Recall@1. Distributed as a public GCS zip (no HF mirror).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_LOFT_X_ARXIV_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import json
import logging
import os
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"

_SCIFACT_URL = "https://storage.googleapis.com/loft-bench/retrieval/scifact.zip"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gbench", "loft")

_CORPUS_INSTRUCTION = (
    "You will be given a list of passages. You need to read carefully and "
    "understand all of them. Then you will be given a claim, and your goal is to "
    "find all passages from the list that can help verify the claim as true of "
    "false. Print out the ID and TITLE of each passage."
)
_FORMATTING_INSTRUCTION = (
    "Your final answer should be a list of IDs, in the following format:\n"
    "Final Answer: [id1, id2, ...]\n"
    "If there is only one ID, it should be in the format:\n"
    "Final Answer: [id1]\n\n"
    "If there is no perfect answer output the closest one. Do not give an empty "
    "final answer."
)
_CORPUS_FORMAT = "ID: {pid} | TITLE: {title} | CONTENT: {passage} | END ID: {pid}"


def _ensure_scifact(context_length: str) -> str:
    """Download+unzip the LOFT SciFact bundle; return path to scifact/<len>/. Raises on failure."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(_CACHE_DIR, "scifact.zip")
    target_dir = os.path.join(_CACHE_DIR, "scifact", context_length)
    if not os.path.isdir(target_dir):
        if not os.path.exists(zip_path):
            logger.info(f"Downloading LOFT SciFact from {_SCIFACT_URL}")
            urllib.request.urlretrieve(_SCIFACT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_CACHE_DIR)
    if not os.path.isdir(target_dir):
        raise RuntimeError(f"loft_x_arxiv: expected LOFT dir missing: {target_dir}")
    return target_dir


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_loft_x_arxiv_samples(
    limit: Optional[int] = None,
    context_length: str = "128k",
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load LOFT SciFact (corpus-in-prompt) samples; raises on load/schema failure."""
    try:
        base = _ensure_scifact(context_length)
        corpus = _read_jsonl(os.path.join(base, "corpus.jsonl"))
        test_queries = _read_jsonl(os.path.join(base, "test_queries.jsonl"))
        few_shot = _read_jsonl(os.path.join(base, "few_shot_queries.jsonl"))
    except Exception as e:
        logger.error(f"Failed to load dataset for loft_x_arxiv: {e}")
        raise RuntimeError(f"Could not load dataset for loft_x_arxiv: {e}") from e

    if not corpus or not test_queries:
        raise RuntimeError("loft_x_arxiv returned empty corpus/queries")

    # Build the shared long context once (corpus is identical for every query).
    corpus_lines = []
    for doc in corpus:
        pid, title, passage = doc.get("pid"), doc.get("title_text"), doc.get("passage_text")
        if pid is None or title is None or passage is None:
            raise RuntimeError("loft_x_arxiv: unexpected corpus schema (pid/title_text/passage_text)")
        corpus_lines.append(_CORPUS_FORMAT.format(pid=pid, title=title, passage=passage))
    corpus_block = "\n".join(corpus_lines)

    fewshot_block = ""
    for fs in few_shot:
        q = fs.get("query_text")
        ans = fs.get("answers") or []
        if not q or not ans:
            continue
        gold_pid = str(ans[0][0])
        fewshot_block += f"claim: {q}\nFinal Answer: [{gold_pid}]\n\n"

    context = (
        f"{_CORPUS_INSTRUCTION}\n\n{_FORMATTING_INSTRUCTION}\n\n"
        f"{corpus_block}\n\n{fewshot_block}"
    )

    samples = []
    for q in test_queries:
        qid = q.get("qid")
        query_text = q.get("query_text")
        answers = q.get("answers")
        if not query_text or not answers:
            raise RuntimeError("loft_x_arxiv: unexpected query schema (query_text/answers)")
        gold_pids = [str(pair[0]) for pair in answers]
        prompt = (
            f"{context}"
            f"claim: {query_text}\n"
            "Which passages from the list above can help verify the claim as true "
            "or false? Then format the IDs into a list."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((
            messages, gold_pids,
            {"category": "loft_retrieval_scifact", "qid": qid, "context_length": context_length},
        ))

    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="loft_x_arxiv")
    logger.info(f"Loaded {len(samples)} loft_x_arxiv (SciFact {context_length}) samples.")
    return samples


def _extract_prediction(text: str) -> Optional[List[Any]]:
    """LOFT utils.extract_prediction: first line with [...] -> ast.literal_eval list."""
    if not text:
        return None
    t = text.replace("*", "").replace("`", "")
    for line in t.splitlines():
        if "[" in line and "]" in line:
            frag = line[line.find("["): line.rfind("]") + 1]
            try:
                val = ast.literal_eval(frag)
            except Exception:
                continue
            if isinstance(val, list):
                return val
    if "[" in t and "]" in t:
        frag = t[t.find("["): t.rfind("]") + 1]
        try:
            val = ast.literal_eval(frag)
            if isinstance(val, list):
                return val
        except Exception:
            return None
    return None


def _eval_loft_x_arxiv(response_text: str, gold_pids: Any) -> bool:
    """Canonical LOFT Recall@1: first predicted ID must be in the gold pid set."""
    parsed = _extract_prediction(response_text)
    if not parsed:
        return False
    ids = [str(x) for x in parsed]
    gold = {str(g) for g in gold_pids}
    return bool(ids) and ids[0] in gold


def run_loft_x_arxiv(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native LOFT SciFact long-context retrieval evaluation suite."""
    samples = _load_loft_x_arxiv_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="loft_x_arxiv",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_loft_x_arxiv,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # sovereign; base.py defaults when unset
    )
    # LOFT has no arXiv retrieval task; gbench runs the LOFT SciFact analog as a faithful stand-in,
    # so the number is not comparable to a "LOFT arXiv" leaderboard entry (there is none).
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "LOFT SciFact retrieval stand-in (LOFT has no arXiv task); a substitute task, not a "
        "canonical LOFT-arXiv number")
    return result
