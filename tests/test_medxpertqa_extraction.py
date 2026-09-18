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
"""Regression: medxpertqa answer extraction must anchor to the FINAL 'Answer: (X)', last match.

WS10 scorecard audit (2026-09-12): the old `re.search` (FIRST match) on a loose regex
`(?:answer|correct option|choice)...([A-H])` grabbed stray letters from the reasoning - e.g. the
'C' in '**Answer Choices**' - and coincidental matches == gold produced FALSE POSITIVES that
inflated the score (a real 0/3 reported as 33.33%)."""

from gbench.runners.eval_suites.medxpertqa import _eval_medxpertqa


def test_answer_choices_header_does_not_false_match():
    # The real final answer is (A); an earlier '**Answer Choices**' must NOT be read as 'C'.
    resp = ("**Evaluation of the Answer Choices**\n"
            "Option C looks plausible but...\nAfter analysis, Answer: (A)")
    assert _eval_medxpertqa(resp, "A") is True     # picks the final (A)
    assert _eval_medxpertqa(resp, "C") is False    # NOT fooled by 'Answer Choices' / 'Option C'


def test_last_answer_wins_over_earlier_mentions():
    resp = "Initially I thought Answer: (B), but reconsidering, Answer: (D)."
    assert _eval_medxpertqa(resp, "D") is True
    assert _eval_medxpertqa(resp, "B") is False


def test_plain_final_answer():
    assert _eval_medxpertqa("Answer: (E)", "E") is True
    assert _eval_medxpertqa("The answer is F", "F") is True


def test_genuine_wrong_answer_stays_wrong():
    assert _eval_medxpertqa("Answer: (A)", "D") is False


def test_empty_or_missing():
    assert _eval_medxpertqa("", "A") is False
    assert _eval_medxpertqa("Answer: (A)", "") is False
