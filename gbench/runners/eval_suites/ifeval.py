# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: ifeval
# Description: Google IFEval - verifiable instruction-following constraints (strict prompt-level)

"""gbench native built-in runner for ifeval (Instruction Following).

Canonical IFEval (google/IFEval): each prompt carries an `instruction_id_list`
plus per-instruction `kwargs`. Scoring is a programmatic checker per instruction
type; this suite reports prompt-level STRICT accuracy (a sample is correct iff the
response satisfies EVERY instruction). Deterministic, stdlib-only (langdetect used
for the rare language constraint if installed; otherwise a script heuristic).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_IFEVAL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import re
import string
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Instruction Following"

_CONSTRAINED = {"my answer is yes.", "my answer is no.", "my answer is maybe."}


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _paragraphs(text: str) -> List[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _cmp(count: int, relation: Optional[str], target: int) -> bool:
    relation = (relation or "").lower()
    if "at least" in relation or "more than or equal" in relation:
        return count >= target
    if "at most" in relation or "less than or equal" in relation:
        return count <= target
    if "less than" in relation:
        return count < target
    if "more than" in relation or "greater than" in relation:
        return count > target
    return count == target


def _detect_language(text: str, lang: str) -> bool:
    try:
        from langdetect import detect
        return detect(text) == lang
    except Exception:
        # Heuristic fallback: English ~ ASCII letters; else require non-ASCII content.
        ascii_letters = sum(c in string.ascii_letters for c in text)
        total_letters = sum(c.isalpha() for c in text) or 1
        is_ascii = (ascii_letters / total_letters) > 0.9
        return is_ascii if lang == "en" else not is_ascii


def _check(instruction_id: str, kw: Dict[str, Any], resp: str, prompt: str) -> bool:
    """Return True iff `resp` satisfies the single IFEval instruction."""
    kw = kw or {}
    low = resp.lower()
    fam = instruction_id

    if fam == "keywords:existence":
        return all(str(k).lower() in low for k in kw.get("keywords", []))
    if fam == "keywords:frequency":
        kwd = str(kw.get("keyword", ""))
        n = len(re.findall(r"\b" + re.escape(kwd) + r"\b", resp, re.IGNORECASE))
        return _cmp(n, kw.get("relation"), int(kw.get("frequency", 0)))
    if fam == "keywords:forbidden_words":
        return not any(str(w).lower() in low for w in kw.get("forbidden_words", []))
    if fam == "keywords:letter_frequency":
        letter = str(kw.get("letter", "")).lower()
        n = low.count(letter)
        return _cmp(n, kw.get("let_relation"), int(kw.get("let_frequency", 0)))
    if fam == "language:response_language":
        return _detect_language(resp, str(kw.get("language", "en")))
    if fam == "length_constraints:number_sentences":
        return _cmp(len(_sentences(resp)), kw.get("relation"), int(kw.get("num_sentences", 0)))
    if fam == "length_constraints:number_paragraphs":
        return len(_paragraphs(resp)) == int(kw.get("num_paragraphs", 0))
    if fam == "length_constraints:number_words":
        return _cmp(len(_words(resp)), kw.get("relation"), int(kw.get("num_words", 0)))
    if fam == "length_constraints:nth_paragraph_first_word":
        paras = _paragraphs(resp)
        n = int(kw.get("num_paragraphs", 0))
        nth = int(kw.get("nth_paragraph", 1))
        first = str(kw.get("first_word", "")).lower()
        if len(paras) != n or nth < 1 or nth > len(paras):
            return False
        pw = _words(paras[nth - 1])
        return bool(pw) and pw[0].lower() == first
    if fam == "detectable_content:number_placeholders":
        return len(re.findall(r"\[.+?\]", resp)) >= int(kw.get("num_placeholders", 0))
    if fam == "detectable_content:postscript":
        marker = str(kw.get("postscript_marker", "P.S.")).lower()
        return marker.lower() in low
    if fam == "detectable_format:number_bullet_lists":
        bullets = re.findall(r"^\s*[\*\-]\s+\S", resp, re.MULTILINE)
        return len(bullets) == int(kw.get("num_bullets", 0))
    if fam == "detectable_format:constrained_response":
        return resp.strip().lower() in _CONSTRAINED
    if fam == "detectable_format:number_highlighted_sections":
        # Canonical google-research instruction_following_eval: count single-*
        # and double-** highlighted sections with two separate regexes and sum
        # them, dropping empty matches. A single regex miscounts adjacent bold
        # (e.g. `**a** **b**` -> a spurious `* *` match), over-crediting.
        num_highlights = 0
        for h in re.findall(r"\*[^\n\*]*\*", resp):
            if h.strip("*").strip():
                num_highlights += 1
        for h in re.findall(r"\*\*[^\n\*]*\*\*", resp):
            if h.strip("*").strip():
                num_highlights += 1
        return num_highlights >= int(kw.get("num_highlights", 0))
    if fam == "detectable_format:multiple_sections":
        spliter = str(kw.get("section_spliter", "Section"))
        n = len(re.findall(r"\b" + re.escape(spliter) + r"\b", resp, re.IGNORECASE))
        return n >= int(kw.get("num_sections", 0))
    if fam == "detectable_format:json_format":
        t = re.sub(r"```(?:json)?", "", resp).replace("```", "").strip()
        try:
            json.loads(t)
            return True
        except Exception:
            return False
    if fam == "detectable_format:title":
        return bool(re.search(r"<<[^\n]+>>", resp))
    if fam == "combination:two_responses":
        parts = [p for p in resp.split("******") if p.strip()]
        return len(parts) == 2
    if fam == "combination:repeat_prompt":
        rep = str(kw.get("prompt_to_repeat", "")).strip()
        return resp.strip().startswith(rep) and len(resp.strip()) > len(rep)
    if fam == "startend:end_checker":
        return resp.strip().lower().endswith(str(kw.get("end_phrase", "")).strip().lower())
    if fam == "startend:quotation":
        s = resp.strip()
        return len(s) >= 2 and s.startswith('"') and s.endswith('"')
    if fam == "change_case:capital_word_frequency":
        caps = [w for w in _words(resp) if w.isupper() and len(w) > 1]
        return _cmp(len(caps), kw.get("capital_relation"), int(kw.get("capital_frequency", 0)))
    if fam == "change_case:english_capital":
        return resp.upper() == resp and any(c.isalpha() for c in resp)
    if fam == "change_case:english_lowercase":
        return resp.lower() == resp and any(c.isalpha() for c in resp)
    if fam == "punctuation:no_comma":
        return "," not in resp
    # Unknown instruction id: cannot verify -> do not silently pass.
    logger.warning(f"ifeval: unhandled instruction id '{instruction_id}'")
    return False


def _load_ifeval_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load IFEval prompts + instruction constraints from HF Hub; raises on load/schema failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for ifeval: {e}")
        raise RuntimeError(f"Could not load dataset for ifeval: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for ifeval returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="ifeval")

    samples = []
    for item in rows:
        prompt = item.get("prompt")
        rule_ids = item.get("instruction_id_list") or []
        kwargs_list = item.get("kwargs") or []
        if not prompt or not rule_ids:
            raise RuntimeError(
                "ifeval: unexpected schema (missing 'prompt'/'instruction_id_list'); "
                "refusing to fabricate sample data"
            )
        gold = json.dumps({"ids": list(rule_ids), "kwargs": list(kwargs_list), "prompt": prompt})
        cat = rule_ids[0].split(":")[0] if rule_ids else "general"
        messages = [{"role": "user", "content": str(prompt)}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} ifeval samples.")
    return samples


def _eval_ifeval(response_text: str, gold_target: str) -> bool:
    """Prompt-level STRICT accuracy: the response must satisfy EVERY instruction."""
    if not response_text:
        return False
    try:
        data = json.loads(gold_target)
    except Exception:
        return False
    ids = data.get("ids", [])
    kwargs_list = data.get("kwargs", [])
    prompt = data.get("prompt", "")
    if not ids:
        return False
    resp = response_text.strip()
    for i, rid in enumerate(ids):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        if not _check(rid, kw, resp, prompt):
            return False
    return True


def run_ifeval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute ifeval native built-in evaluation suite (prompt-level strict)."""
    samples = _load_ifeval_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="ifeval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_ifeval,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
