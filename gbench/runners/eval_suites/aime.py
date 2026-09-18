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

"""Native AIME Olympiad Competition Math evaluation suite.

Two datasets, split by contamination window:

* **Headline = post-cutoff AIME 2025** (`yentinglin/aime_2025`, 30 problems, Feb/Mar 2025).
  This is the contamination-clean reasoning number and drives `accuracy`.
* **Reference = pre-cutoff AIME 2022-2024** (`AI-MO/aimo-validation-aime`, 90 problems),
  kept by default (`GBENCH_AIME_INCLUDE_PRECUTOFF=0` to drop). These problems AND their full
  worked solutions live on artofproblemsolving.com (the dataset's own `url` field) and
  pre-date the training cutoff, so a high score is indistinguishable from recall. It is
  reported as `accuracy_pre_cutoff_reference`, NOT the headline.

Both share one schema; each sample's category is its contest year, so `category_accuracy`
shows every year and the runner separates the clean headline from the contaminated
reference. Override the sources with `GBENCH_AIME_POSTCUTOFF_DATASET` /
`GBENCH_AIME_PRECUTOFF_DATASET`.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_AIME_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .extraction_common import final_number

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/aime.md"

def _aime_year(row: Any) -> str:
    """Contest year from the AoPS wiki url, e.g. `.../2022_AIME_I_Problems/Problem_1`."""
    m = re.search(r"/(20\d\d)_AIME", str((row or {}).get("url") or ""))
    return m.group(1) if m else "unknown"


#: Pre-cutoff reference set: AIME 2022/2023/2024 (90 problems). These problems AND their
#: full worked solutions live on artofproblemsolving.com (the dataset's own `url` field) and
#: predate the model's training cutoff, so a high score here cannot be distinguished from
#: recall of memorized solutions. Kept as a REFERENCE, not the headline.
_AIME_PRECUTOFF_DATASET = os.environ.get("GBENCH_AIME_PRECUTOFF_DATASET",
                                         "AI-MO/aimo-validation-aime")
#: Post-cutoff set: AIME 2025 (30 problems, Feb/Mar 2025). Same schema
#: (id/problem/answer/solution/url/year). This is the contamination-clean headline.
_AIME_POSTCUTOFF_DATASET = os.environ.get("GBENCH_AIME_POSTCUTOFF_DATASET",
                                          "yentinglin/aime_2025")
#: Include the pre-cutoff reference set alongside the post-cutoff headline (default on).
_AIME_INCLUDE_PRECUTOFF = os.environ.get("GBENCH_AIME_INCLUDE_PRECUTOFF", "1") != "0"


def _load_aime_samples(enable_thinking: bool = False, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load AIME samples: the post-cutoff 2025 set (clean headline) plus, by default, the
    pre-cutoff 2022-2024 set as a contamination reference.

    Both sets share the same schema. Each sample's category is its contest YEAR (parsed from
    the AoPS `url`), so `category_accuracy` reports every year separately and the runner can
    split the clean (2025) headline from the contaminated (2022-2024) reference.
    """
    from datasets import load_dataset
    rows: List[Dict[str, Any]] = []
    try:
        post = list(load_dataset(_AIME_POSTCUTOFF_DATASET, split="train"))
        rows.extend(post)
        logger.info("Loaded %d post-cutoff AIME 2025 problems ('%s').",
                    len(post), _AIME_POSTCUTOFF_DATASET)
    except Exception as e:
        raise RuntimeError(f"aime: could not load post-cutoff set "
                           f"'{_AIME_POSTCUTOFF_DATASET}': {e}") from e
    if _AIME_INCLUDE_PRECUTOFF:
        try:
            pre = list(load_dataset(_AIME_PRECUTOFF_DATASET, split="train"))
            rows.extend(pre)
            logger.info("Loaded %d pre-cutoff AIME 2022-2024 reference problems ('%s').",
                        len(pre), _AIME_PRECUTOFF_DATASET)
        except Exception as e:
            logger.warning("aime: pre-cutoff reference set unavailable (%s); "
                           "running post-cutoff only. See %s", e, DOCS_URL)
    # Stratify by YEAR so a --eval-limit keeps the year mix (incl. 2025) rather than a head.
    ds = limit_dataset(rows, limit, _aime_year, seed="aime")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} AIME samples total (post-cutoff headline + "
                f"pre-cutoff reference).")

    samples = []
    for item in raw_samples:
        q_text = item.get("problem") or item.get("question", "")
        gold_ans = str(item["answer"]).strip()

        if enable_thinking:
            prompt = (
                f"Problem: {q_text}\n\n"
                "Solve the math problem step by step. Write your final answer as an integer between 000 and 999 "
                "at the end in the format: 'The final answer is X'."
            )
        else:
            prompt = (
                f"Problem: {q_text}\n\n"
                "Solve the math problem. Write your final answer as an integer between 000 and 999 "
                "at the end in the format: 'The final answer is X'."
            )
        messages = [{"role": "user", "content": prompt}]
        # Category = contest YEAR, so `category_accuracy` reports AIME 2022 / 2023 / 2024
        # separately. Published AIME figures are almost always a single year (usually
        # AIME 2024, 30 problems); a blended 90-problem number is not comparable to them
        # unless the per-year split is visible. `item.get("category")` was always absent,
        # so every row was labelled "math" and the breakdown carried no information.
        samples.append((messages, gold_ans, {"category": f"aime_{_aime_year(item)}"}))
    return samples


def _eval_aime(response_text: str, gold_answer: str) -> bool:
    """Check the model's stated integer against the gold AIME answer (0-999).

    Shared extraction (audit CC7): \\boxed{} first, then any answer anchor, then the last
    number. The previous version matched one literal phrasing and otherwise took the last
    1-4 digit token, so `\\boxed{042}` was missed and a trailing year or step number in the
    working could be read as the answer.
    """
    pred = final_number(response_text)
    if pred is None:
        return False
    try:
        return int(float(pred)) == int(float(gold_answer))
    except ValueError:
        return False


def run_aime(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native AIME: post-cutoff AIME 2025 headline + pre-cutoff 2022-2024 reference."""
    samples = _load_aime_samples(enable_thinking, limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="aime",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_aime,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )

    # Split the clean (post-cutoff) headline from the contaminated (pre-cutoff) reference.
    # AIME 2022-2024 problems + solutions are on AoPS and predate the training cutoff, so a
    # high score there can be recall rather than reasoning; AIME 2025 is post-cutoff and is
    # the number to trust. Both are reported; the headline `accuracy` is the clean one.
    cat = result.get("category_accuracy", {}) or {}

    def _agg(years):
        c = sum(cat[y]["correct"] for y in years if y in cat)
        t = sum(cat[y]["total"] for y in years if y in cat)
        return (round(100.0 * c / t, 2), c, t) if t else (None, 0, 0)

    post_years = [y for y in cat if y >= "aime_2025"]
    pre_years = [y for y in cat if y < "aime_2025"]
    post_acc, post_c, post_t = _agg(post_years)
    pre_acc, pre_c, pre_t = _agg(pre_years)

    result["accuracy_all_years"] = result.get("accuracy")
    if post_t:
        result["accuracy"] = post_acc                       # headline = post-cutoff, clean
        result["accuracy_post_cutoff"] = post_acc
        result["post_cutoff_n"] = post_t
        result["post_cutoff_years"] = sorted(post_years)
    if pre_t:
        result["accuracy_pre_cutoff_reference"] = pre_acc
        result["pre_cutoff_reference_n"] = pre_t
    result["metric"] = ("AIME 2025 pass@1 exact-match (post-cutoff headline); AIME "
                        "2022-2024 kept as a CONTAMINATED reference, not comparable")
    result["contamination_note"] = (
        "Headline `accuracy` is the post-cutoff AIME 2025 set. "
        "`accuracy_pre_cutoff_reference` (2022-2024) shares the AoPS solutions the model was "
        "trained on and reads high from recall; do not quote it as a reasoning measure.")

    # @k block (only present with --attempt-count>1): base computed avg@k/pass@k/pass^k over
    # ALL traces, folding the contaminated 2022-2024 reference into the headline @k so it
    # silently disagreed with the post-cutoff `accuracy` above. Recompute the headline @k over
    # post-cutoff traces only; keep the all-years and pre-cutoff figures as references.
    attempts_block = result.get("attempts")
    if isinstance(attempts_block, dict) and attempts_block.get("attempts_per_sample", 1) > 1:
        def _atk(traces):
            by_src: Dict[Any, List[bool]] = {}
            for t in traces:
                key = t.get("source_sample_idx", t.get("sample_idx"))
                by_src.setdefault(key, []).append(bool(t.get("is_correct")))
            n = len(by_src)
            if not n:
                return None
            gens = sum(len(v) for v in by_src.values())
            any_ok = sum(1 for v in by_src.values() if any(v))
            all_ok = sum(1 for v in by_src.values() if v and all(v))
            correct = sum(sum(1 for x in v if x) for v in by_src.values())
            return {
                "samples": n,
                "generations": gens,
                "avg_at_k": round(correct / gens * 100.0, 2) if gens else 0.0,
                "pass_at_k": round(any_ok / n * 100.0, 2),
                "pass_hat_k": round(all_ok / n * 100.0, 2),
                "unstable_samples": sum(1 for v in by_src.values() if any(v) and not all(v)),
            }
        traces = result.get("sample_traces", []) or []
        post_atk = _atk([t for t in traces if str(t.get("category") or "") >= "aime_2025"])
        pre_atk = _atk([t for t in traces if str(t.get("category") or "") < "aime_2025"])
        if post_atk:
            result["attempts_all_years"] = attempts_block
            merged = dict(attempts_block)
            merged.update(post_atk)          # headline @k = post-cutoff, clean
            result["attempts"] = merged
            if pre_atk:
                result["attempts_pre_cutoff_reference"] = pre_atk
    # Headline is over the post-cutoff AIME 2025 single-year subset; the reason reflects whether
    # this was a single-attempt pass@1 or a multi-sample @k run. Always non-comparable (single-year
    # subset), reported conservatively.
    _attempts = result.get("attempts")
    _multi = isinstance(_attempts, dict) and (_attempts.get("attempts_per_sample", 1) or 1) > 1
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "post-cutoff AIME 2025 single-year subset scored avg@k over --attempt-count samples; not a "
        "full published-shape AIME leaderboard entry"
        if _multi else
        "single-attempt pass@1 greedy over the post-cutoff AIME 2025 single-year subset; published "
        "AIME figures are usually avg@k over multiple samples (use --attempt-count>1 for @k)")
    return result
