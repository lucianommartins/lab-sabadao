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
"""Regression locks for the WS10 fast-verify fixes (2 bugs + 7 suspicious).

Covered here:
* scoring_mode label is TRUTHFUL - execution suites (bigcodebench/multipl_e/lcb/ojbench/
  spider2/cyberseceval) are "execution", coco_caption is "deterministic", judges stay "judge".
* an unmeasurable / judge-outage sample is excluded from the numerator AND every per-category
  total (the gdpval "excluded Film task shown as a category 0/1 failure" contradiction).
* TextVQA headline is the canonical MEAN soft accuracy, not the >=0.5 binary pass-rate.
"""

from gbench.runners.eval_suites.base import (
    classify_scoring_mode, tally_batch_scores,
)


# --------------------------------------------------------------------------- #
# scoring_mode label                                                          #
# --------------------------------------------------------------------------- #
def test_declared_execution_beats_the_async_judge_heuristic():
    # bigcodebench/multipl_e batch through async_eval_fn but RUN TESTS, not a judge.
    assert classify_scoring_mode(False, "execution", True) == "execution"


def test_declared_deterministic_for_corpus_metric():
    # coco_caption batches a deterministic corpus CIDEr metric through async_eval_fn.
    assert classify_scoring_mode(False, "deterministic", True) == "deterministic"


def test_undeclared_async_scorer_is_still_a_judge():
    assert classify_scoring_mode(False, None, True) == "judge"


def test_plain_eval_fn_is_deterministic():
    assert classify_scoring_mode(False, None, False) == "deterministic"


def test_fallback_overrides_even_a_declared_mode():
    # A no-judge substring fallback is a lower bound and must win over any declaration.
    assert classify_scoring_mode(True, "execution", True) == "judge_fallback"


# --------------------------------------------------------------------------- #
# unscored samples excluded from numerator AND category totals                #
# --------------------------------------------------------------------------- #
def test_excluded_and_outage_samples_leave_no_category_footprint():
    traces = [
        {"category": "film", "is_correct": True},
        {"category": "film", "is_correct": False},
        # gdpval marks an all-file-property task unmeasurable for a text endpoint:
        {"category": "film", "is_correct": False, "scoring_excluded": True},
        # a judge outage is infra, not a wrong answer:
        {"category": "law", "is_correct": False, "judge_grade": "JUDGE_OUTAGE"},
        {"category": "law", "is_correct": True},
    ]
    correct, cats = tally_batch_scores(traces)
    assert correct == 2
    # film: only the 2 measurable tasks count (1/2), NOT 1/3 with the excluded one as a failure.
    assert cats["film"] == {"correct": 1, "total": 2}
    # law: the outage is out of the denominator too (1/1, not 1/2).
    assert cats["law"] == {"correct": 1, "total": 1}


def test_excluded_sample_never_counts_as_correct():
    traces = [{"category": "x", "is_correct": True, "scoring_excluded": True}]
    correct, cats = tally_batch_scores(traces)
    assert correct == 0
    assert "x" not in cats


# --------------------------------------------------------------------------- #
# TextVQA canonical mean soft accuracy                                        #
# --------------------------------------------------------------------------- #
def test_textvqa_headline_is_mean_soft_not_binary_passrate():
    from gbench.runners.eval_suites.textvqa import _promote_soft_headline
    from gbench.runners.eval_suites.vqa_common import vqa_accuracy, extract_short_answer, eval_vqa
    # One exact match (soft 1.0, binary pass) and two that match a SINGLE annotator each:
    # soft = 1/3 = 0.333 apiece but binary FAIL (<0.5). Binary pass-rate drops that partial
    # credit (1/3 = 33.33); the canonical soft mean keeps it.
    golds_exact = ["cat"] * 10
    golds_one = ["dog"] + ["cat"] * 9      # response "dog" matches exactly 1 annotator
    traces = [
        {"response_text": "cat", "gold_answer": golds_exact},
        {"response_text": "dog", "gold_answer": golds_one},
        {"response_text": "dog", "gold_answer": golds_one},
    ]
    # Binary form the base eval_fn would have produced:
    binary = round(sum(1 for t in traces
                       if eval_vqa(extract_short_answer(t["response_text"]), t["gold_answer"]))
                   / len(traces) * 100.0, 2)
    soft = round(sum(vqa_accuracy(extract_short_answer(t["response_text"]), t["gold_answer"])
                     for t in traces) / len(traces) * 100.0, 2)
    result = {"accuracy": binary, "sample_traces": [dict(t) for t in traces]}
    out = _promote_soft_headline(result)
    assert out["binary_pass_rate"] == binary          # binary form preserved, clearly named
    assert out["accuracy"] == soft                    # headline promoted to canonical soft mean
    assert out["accuracy"] > out["binary_pass_rate"]  # soft keeps the partial credit binary drops
    assert out["sample_traces"][0]["vqa_soft_score"] == 1.0   # per-trace soft recorded
