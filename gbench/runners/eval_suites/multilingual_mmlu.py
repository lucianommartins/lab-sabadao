# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Multilingual MMLU (14-Language Reasoning and Knowledge) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MULTILINGUAL_MMLU_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D"]

# Canonical multilingual MMLU = OpenAI's MMMLU: human professional translations of the MMLU
# test set into 14 languages, the set Gemma tech reports quote. The previous source
# (alexandrainst/m_mmlu) is Okapi's ChatGPT-machine-translated MMLU (CC BY-NC) and is not
# comparable to published multilingual-MMLU numbers. MMMLU config codes are LANG_REGION;
# columns are 'Question', 'A'..'D', 'Answer' (letter A-D), 'Subject'. The 'default' config
# is excluded (it is not a language split). Use --eval-limit to cap; rows are interleaved
# across languages so a limit still spans languages.
MMMLU_CONFIGS = [
    "AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN", "ID_ID",
    "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE", "YO_NG", "ZH_CN",
]


def _load_multilingual_mmlu_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load Multilingual MMLU from HF Hub (openai/MMMLU, per-language configs).

    Raises on load/schema failure (no fabricated fallback, no silent fallback to
    English cais/mmlu).
    """
    from datasets import load_dataset

    per_lang: List[Tuple[str, List[Dict[str, Any]]]] = []
    for cfg in MMMLU_CONFIGS:
        lang = cfg.split("_")[0].lower()          # AR_XY -> ar, PT_BR -> pt
        try:
            ds = load_dataset("openai/MMMLU", cfg, split="test")
        except Exception as e:
            logger.error(f"Failed to load MMMLU language '{cfg}': {e}")
            raise RuntimeError(
                f"Could not load multilingual_mmlu language '{cfg}': {e}"
            ) from e
        per_lang.append((lang, list(ds)))

    # Round-robin interleave so a --eval-limit still spans multiple languages.
    lang_rows: List[Tuple[str, Dict[str, Any]]] = []
    max_len = max((len(rows) for _, rows in per_lang), default=0)
    for i in range(max_len):
        for lang, rows in per_lang:
            if i < len(rows):
                lang_rows.append((lang, rows[i]))

    if not lang_rows:
        raise RuntimeError("Dataset for multilingual_mmlu returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    lang_rows = stratified_sample(lang_rows, limit, lambda r: r[0], seed="multilingual_mmlu")
    logger.info(f"Loaded {len(lang_rows)} Multilingual MMLU samples from HF Hub.")

    samples = []
    for lang, item in lang_rows:
        q_text = item.get("Question")
        opts = [item.get(c) for c in ("A", "B", "C", "D")]
        ans = item.get("Answer")
        if (not q_text or any(o is None for o in opts)
                or not (isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS)):
            raise RuntimeError(
                "multilingual_mmlu: unexpected dataset schema "
                "(need 'Question', 'A'..'D', 'Answer' letter); "
                "refusing to fabricate sample data"
            )
        gold_letter = ans.strip().upper()
        options_str = "\n".join(
            f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(opts)
        )
        prompt = f"Question:\n{q_text}\n\nOptions:\n{options_str}\n\n"
        if enable_thinking:
            prompt += "Let's think step by step and output the correct option letter in the format: 'Answer: (X)'."
        else:
            prompt += "Output only the correct option letter in the format: 'Answer: (X)'."

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": lang}))

    return samples


#: Answer-word anchors across the MMMLU languages (+ Latin fallbacks). The options are
#: always labelled with the Latin letters A-D, so only the anchor words are localized.
_ANSWER_WORDS = (
    r"answer|choice|option|resposta|réponse|reponse|antwort|risposta|jawaban|respuesta|"
    r"答案|正确答案|正解|答え|정답|उत्तर|الإجابة|الجواب|jibu|ìdáhùn|đáp\s*án"
)


def _extract_mc_letter(resp: str) -> Optional[str]:
    """The model's chosen option letter (A-D), robust to non-Latin scripts.

    The prior extractor used Latin-only anchor words and a ``\\b([A-D])\\b`` fallback; in
    CJK/Arabic/Hindi text there is no word boundary between a script character and the Latin
    letter, so non-Latin languages were systematically under-scored. Here: boxed -> a
    parenthesised letter (incl. full-width) -> a localized answer-word anchor -> the LAST
    Latin A-D not embedded in a Latin word (so a letter next to non-Latin text still counts).
    """
    up = resp.strip()
    boxed = re.findall(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}", up)
    if boxed:
        return boxed[-1].upper()
    paren = re.findall(r"[\(（]\s*([A-Da-d])\s*[\)）]", up)
    if paren:
        return paren[-1].upper()
    anchored = re.findall(
        rf"(?:{_ANSWER_WORDS})\s*(?:is|:|：|=|は|为|是)?\s*\(?([A-Da-d])\)?",
        up, re.IGNORECASE)
    if anchored:
        return anchored[-1].upper()
    # A Latin A-D not flanked by other Latin letters: matches "答案:A", "الإجابة A", a bare
    # "C", etc., but not the "A" inside an English word. Last occurrence = final answer.
    loose = re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", up)
    if loose:
        return loose[-1].upper()
    return None


def _eval_multilingual_mmlu(response_text: str, gold_letter: str) -> bool:
    """Extract predicted option letter and compare with gold answer."""
    if not response_text or not gold_letter:
        return False
    picked = _extract_mc_letter(response_text)
    return bool(picked) and picked == gold_letter.strip().upper()


def run_multilingual_mmlu(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Multilingual MMLU evaluation suite."""
    samples = _load_multilingual_mmlu_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    return run_eval_suite(
        eval_name="multilingual_mmlu",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_multilingual_mmlu,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096 if enable_thinking else 512),
        temperature=kwargs.get("temperature"),
    )
