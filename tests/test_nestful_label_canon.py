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
"""Regression: NESTFUL full/partial/slot match must be label-spelling agnostic. WS10 audit: the gold
is internally inconsistent ('$var_1' in most traces, '$var1' in others) while the prompt MANDATES
'$var_1', so a compliant, functionally-correct answer literal-mismatched the gold and was a false
full-match negative. Canonicalizing '$var_N' <-> '$varN' on both sides fixes it; a real mismatch
still differs."""

from gbench.runners.eval_suites.nestful import _ibm_api_with_args, _ibm_canon_label


def test_canon_drops_underscore():
    assert _ibm_canon_label("$var_1.output_0$") == "$var1.output_0$"
    assert _ibm_canon_label("$var_12.result$") == "$var12.result$"
    assert _ibm_canon_label("plain") == "plain"


def test_prompt_compliant_answer_full_matches_inconsistent_gold():
    gold = [{"name": "f", "label": "$var1", "arguments": {"n": 5}},
            {"name": "g", "label": "$var2", "arguments": {"data": "$var1.output_0$"}}]
    pred = [{"name": "f", "label": "$var_1", "arguments": {"n": 5}},
            {"name": "g", "label": "$var_2", "arguments": {"data": "$var_1.output_0$"}}]
    assert _ibm_api_with_args(gold, fix_dollar=False) == _ibm_api_with_args(pred, fix_dollar=True)


def test_true_negative_preserved():
    gold = [{"name": "f", "label": "$var1", "arguments": {"n": 5}}]
    wrong = [{"name": "f", "label": "$var_1", "arguments": {"n": 99}}]
    assert _ibm_api_with_args(gold, fix_dollar=False) != _ibm_api_with_args(wrong, fix_dollar=True)
