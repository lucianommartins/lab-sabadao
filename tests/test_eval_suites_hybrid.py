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

"""Unit tests for the new hybrid evaluation suites in gbench."""

import pytest
from gbench.runners.eval_suites.amc_aime import _eval_amc_aime, _load_amc_aime_samples
from gbench.runners.eval_suites.bundled_detection import _eval_bundled_detection, _compute_iou, _load_bundled_detection_samples
from gbench.runners.eval_suites.loft_x_arxiv import _eval_loft_x_arxiv, _load_loft_x_arxiv_samples
from gbench.runners.eval_suites.i18n_translate import _eval_i18n_translate, _load_i18n_translate_samples
from gbench.runners.eval_suites.causalbench import _eval_causalbench, _load_causalbench_samples
from gbench.runners.eval_suites.lmsys_noncoding_hard import _eval_lmsys_noncoding_hard, _load_lmsys_noncoding_hard_samples
from gbench.runners.eval_suites.tau2 import _eval_tau2, _load_tau2_samples
from gbench.runners.eval_suites.tau3 import _eval_tau3, _load_tau3_samples


def test_amc_aime():
    assert _eval_amc_aime("The final answer is 142.", "142") is True
    assert _eval_amc_aime("Final Answer: 054", "54") is True
    assert _eval_amc_aime("The final answer is 100", "99") is False
    samples = _load_amc_aime_samples(limit=4)
    assert len(samples) > 0


def test_bundled_detection():
    # Test IoU calculation
    box1 = [100.0, 100.0, 300.0, 300.0]
    box2 = [100.0, 100.0, 300.0, 300.0]
    assert _compute_iou(box1, box2) == 1.0

    gold_objects = [{"label": "cat", "box_2d": [100.0, 100.0, 300.0, 300.0]}]
    pred_json = '[{"label": "cat", "box_2d": [110.0, 110.0, 290.0, 290.0]}]'
    assert _eval_bundled_detection(pred_json, gold_objects) is True

    bad_pred = '[{"label": "dog", "box_2d": [600.0, 600.0, 900.0, 900.0]}]'
    assert _eval_bundled_detection(bad_pred, gold_objects) is False


def test_loft_x_arxiv():
    # Canonical LOFT retrieval: gold is a passage-ID set; Recall@1 on the first
    # predicted ID (extracted from a "Final Answer: [id]" list).
    gold = ["31715818"]
    assert _eval_loft_x_arxiv("Final Answer: [31715818]", gold) is True
    assert _eval_loft_x_arxiv("The passage is ID 31715818.\nFinal Answer: [31715818]", gold) is True
    assert _eval_loft_x_arxiv("Final Answer: [99999]", gold) is False
    assert _eval_loft_x_arxiv("no list here", gold) is False


def test_i18n_translate():
    gold = "Munich 1856: Four maps that will change your view of the city"
    pred = "Munich 1856: Four maps that change your perspective of the city"
    assert _eval_i18n_translate(pred, gold) is True
    bad_pred = "This is a completely unrelated sentence about cooking."
    assert _eval_i18n_translate(bad_pred, gold) is False


def test_causalbench():
    assert _eval_causalbench("Yes, the hypothesis is valid.", "Yes") is True
    assert _eval_causalbench("No, the relation does not hold.", "No") is True
    assert _eval_causalbench("Yes", "No") is False
    samples = _load_causalbench_samples(limit=2)
    assert len(samples) > 0


def test_lmsys_noncoding_hard():
    checklist = ["list all tube stations in alphabetical order", "numbered list format"]
    good_resp = "Here is a numbered list of all tube stations in alphabetical order:\n1. Acton Central\n2. Acton Town\n3. Aldgate"
    assert _eval_lmsys_noncoding_hard(good_resp, checklist) is True
    short_resp = "no"
    assert _eval_lmsys_noncoding_hard(short_resp, checklist) is False


def test_tau2():
    gold = {"tool": "search_flights", "args": {"origin": "AUS", "destination": "SEA", "date": "2026-10-14"}}
    good_call = '{"name": "search_flights", "arguments": {"origin": "AUS", "destination": "SEA", "date": "2026-10-14"}}'
    assert _eval_tau2(good_call, gold) is True
    bad_call = '{"name": "cancel_flight", "arguments": {"confirmation_code": "123"}}'
    assert _eval_tau2(bad_call, gold) is False
    samples = _load_tau2_samples(limit=2)
    assert len(samples) > 0


def test_tau3():
    gold = {"tool": "lookup_billing_charges", "args": {"phone_number": "555-0144"}}
    good_call = '{"name": "lookup_billing_charges", "arguments": {"phone_number": "555-0144"}}'
    assert _eval_tau3(good_call, gold) is True
    bad_call = '{"name": "cancel_account", "arguments": {"account_id": "123"}}'
    assert _eval_tau3(bad_call, gold) is False
    samples = _load_tau3_samples(limit=2)
    assert len(samples) > 0
