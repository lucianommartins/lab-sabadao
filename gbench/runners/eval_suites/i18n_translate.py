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

"""Native Multilingual Translation Quality (WMT) evaluation suite.

Headline = CORPUS chrF (the metric WMT reports; see chrf_corpus) over the into-English pairs
de/zh/ru/cs -> en on wmt/wmt19 validation (newstest2018). Scope + split + subsampling make it
`leaderboard_comparable=False`.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_I18N_TRANSLATE_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .swebench_common import infra_required

_DOCS_URL = "docs/evals/i18n_translate.md"

logger = logging.getLogger(__name__)

LANG_PAIRS = [
    ("de-en", "German", "English", "de", "en"),
    ("zh-en", "Chinese", "English", "zh", "en"),
    ("ru-en", "Russian", "English", "ru", "en"),
    ("cs-en", "Czech", "English", "cs", "en"),
]


def _load_i18n_translate_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load translation evaluation pairs from canonical WMT dataset ('wmt/wmt19')."""
    from datasets import load_dataset

    samples = []
    per_pair = (limit // len(LANG_PAIRS)) + 1 if limit is not None else 25

    for pair_name, src_lang, tgt_lang, src_code, tgt_code in LANG_PAIRS:
        # A pair that fails to load is a missing prerequisite, not something to silently drop:
        # swallowing it shrinks the eval to an unannounced subset (or to nothing) while still
        # reporting a number. Hard-error instead (audit 2nd-pass).
        try:
            # wmt/wmt19 exposes only train/validation; `validation` is newstest2018. There is no
            # newstest2019 test split in this dataset, which is disclosed via leaderboard_comparable.
            ds = load_dataset("wmt/wmt19", pair_name, split="validation", streaming=True)
            count = 0
            for item in ds:
                tr = item.get("translation", {})
                src_text = tr.get(src_code, "").strip()
                tgt_text = tr.get(tgt_code, "").strip()
                if not src_text or not tgt_text or len(src_text) < 10:
                    continue

                prompt = f"Translate the following text from {src_lang} into {tgt_lang}. Output only the translation:\n\n{src_text}"
                messages = [{"role": "user", "content": prompt}]
                samples.append((messages, tgt_text, {"category": pair_name}))
                count += 1
                if count >= per_pair:
                    break
        except Exception as e:
            raise infra_required(
                "i18n_translate",
                f"could not load WMT pair {pair_name} from wmt/wmt19 ({e})",
                _DOCS_URL) from e

    if not samples:
        raise infra_required(
            "i18n_translate",
            "wmt/wmt19 yielded no usable sentence pairs across " + ", ".join(p[0] for p in LANG_PAIRS),
            _DOCS_URL)

    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="i18n_translate")

    logger.info(f"Loaded {len(samples)} multilingual translation samples from WMT.")
    return samples


def _eval_i18n_translate(response_text: str, gold_text: str) -> bool:
    """Secondary pass flag: sentence chrF >= 40 against the gold reference.

    chrF is case-sensitive by construction (it credits partial word/morphology matches), so
    neither side is lowercased - lowercasing would inflate the score and diverge from the
    canonical metric. Only enclosing quotes the model may add are stripped.
    """
    if not response_text or not gold_text:
        return False

    pred = re.sub(r'^["\'`](.*)["\'`]$', r'\1', response_text.strip(), flags=re.S).strip()
    gold = gold_text.strip()

    return chrf_score(pred, gold) >= 40.0


def chrf_score(pred: str, gold: str) -> float:
    """chrF (character n-gram F-score, 0-100) - a canonical MT metric.

    Replaces a set-based word F1 with a 0.45 threshold. That measure was order- and
    repetition-insensitive - a bag of the right words in any order scored the same as a
    correct translation, and a translation repeating one word scored the same as one
    using it once - and it discarded morphology entirely, which matters most for exactly
    the languages here. chrF is character-level, so it credits partial word matches and is
    the standard choice when COMET is unavailable.

    `sacrebleu` is used when installed; the fallback is an equivalent local implementation
    (character 1-6 grams, beta=2) so the suite does not silently change metric.
    """
    pred, gold = str(pred or "").strip(), str(gold or "").strip()
    if not pred or not gold:
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.sentence_chrf(pred, [gold]).score)
    except ImportError:
        pass

    from collections import Counter
    beta, total_p, total_r, orders = 2.0, 0.0, 0.0, 0
    p_chars = re.sub(r"\s+", "", pred)
    g_chars = re.sub(r"\s+", "", gold)
    for n in range(1, 7):
        p_ngrams = Counter(p_chars[i:i + n] for i in range(len(p_chars) - n + 1))
        g_ngrams = Counter(g_chars[i:i + n] for i in range(len(g_chars) - n + 1))
        if not p_ngrams or not g_ngrams:
            continue
        overlap = sum((p_ngrams & g_ngrams).values())
        total_p += overlap / sum(p_ngrams.values())
        total_r += overlap / sum(g_ngrams.values())
        orders += 1
    if not orders:
        return 0.0
    precision, recall = total_p / orders, total_r / orders
    if precision + recall == 0:
        return 0.0
    return 100.0 * (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def chrf_corpus(hyps: List[str], refs: List[str]) -> float:
    """CORPUS-level chrF (0-100) - the metric WMT actually reports.

    chrF is a corpus statistic: character n-gram match/total counts are POOLED across every
    sentence before precision/recall/F-beta is taken - not averaged per sentence. A macro
    (mean-of-sentence) chrF over-weights short sentences and is not comparable to the WMT
    number. An empty/failed hypothesis is kept in the pool (it contributes ref n-grams with no
    matches, penalising recall) rather than dropped, so a model that fails on hard sentences
    cannot inflate the score. `sacrebleu.corpus_chrf` is used when installed; the fallback pools
    the same character 1-6 grams (beta=2) locally so the metric does not silently change.
    """
    pairs = [(str(h or ""), str(r or "")) for h, r in zip(hyps, refs)]
    if not pairs:
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.corpus_chrf([h for h, _ in pairs],
                                           [[r for _, r in pairs]]).score)
    except ImportError:
        pass

    from collections import Counter
    beta = 2.0
    order_stats = {n: [0, 0, 0] for n in range(1, 7)}  # n -> [overlap, pred_total, ref_total]
    for hyp, ref in pairs:
        p_chars = re.sub(r"\s+", "", hyp)
        g_chars = re.sub(r"\s+", "", ref)
        for n in range(1, 7):
            p_ngrams = Counter(p_chars[i:i + n] for i in range(len(p_chars) - n + 1))
            g_ngrams = Counter(g_chars[i:i + n] for i in range(len(g_chars) - n + 1))
            order_stats[n][0] += sum((p_ngrams & g_ngrams).values())
            order_stats[n][1] += sum(p_ngrams.values())
            order_stats[n][2] += sum(g_ngrams.values())
    total_p = total_r = 0.0
    orders = 0
    for n in range(1, 7):
        overlap, ptot, rtot = order_stats[n]
        if ptot == 0 or rtot == 0:
            continue
        total_p += overlap / ptot
        total_r += overlap / rtot
        orders += 1
    if not orders:
        return 0.0
    precision, recall = total_p / orders, total_r / orders
    if precision + recall == 0:
        return 0.0
    return 100.0 * (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def run_i18n_translate(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native multilingual translation quality evaluation suite.

    Translation quality is continuous, so the headline is the **corpus chrF** (the metric WMT
    reports; see chrf_corpus). The pass rate at the sentence chrF>=40 threshold is kept as a
    secondary. Scope: WMT19 into-English only (de/zh/ru/cs -> en), so this does not measure
    generation *into* those languages, and it runs the validation split (newstest2018) because
    HF `wmt/wmt19` exposes no newstest2019 test split - hence leaderboard_comparable=False.
    """
    samples = _load_i18n_translate_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="i18n_translate",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_i18n_translate,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    hyps: List[str] = []
    refs: List[str] = []
    for trace in result.get("sample_traces", []):
        # A failed/empty generation stays in the corpus as an empty hypothesis (penalises
        # recall) rather than being dropped, so failures cannot inflate the score.
        raw = trace.get("response_text") or ""
        hyp = re.sub(r'^["\'`](.*)["\'`]$', r'\1', raw.strip(), flags=re.S).strip()
        gold = str(trace.get("gold_answer") or "")
        trace["chrf"] = round(chrf_score(hyp, gold), 2)   # sentence-level, for inspection only
        hyps.append(hyp)
        refs.append(gold)
    result["pass_rate_chrf40"] = result.get("accuracy")
    result["metric"] = ("corpus chrF (into-English only: de/zh/ru/cs -> en; WMT19 validation "
                        "= newstest2018)")
    if hyps:
        result["accuracy"] = round(chrf_corpus(hyps, refs), 2)
        result["corpus_chrf"] = result["accuracy"]
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "into-English-only subset (de/zh/ru/cs->en; WMT is multi-pair and bidirectional), scored "
        "on wmt/wmt19 validation (newstest2018) because the dataset exposes no newstest2019 test "
        "split, and sub-sampled per pair")
    return result
