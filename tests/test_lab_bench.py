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

"""lab_bench regression tests: no answer leak + robust letter extraction."""

from gbench.runners.eval_suites.lab_bench import _eval_lab_bench, _build_mcq_prompt


def test_extractor_tolerates_wrappers_around_letter():
    # The WS10 smoke produced "Final Answer: <C>" and the old regex scored it wrong
    # (the '<' blocked [A-Za-z]) -> a whole suite of correct answers read as 0/5.
    assert _eval_lab_bench("reasoning...\n\nFinal Answer: <C>", "C")
    assert _eval_lab_bench("Final Answer: C", "c")
    assert _eval_lab_bench("blah\nFinal Answer: **B**", "B")
    assert _eval_lab_bench("Final Answer: (A)", "A")
    assert _eval_lab_bench('Final Answer: "D"', "D")


def test_extractor_rejects_wrong_or_missing_answer():
    assert not _eval_lab_bench("Final Answer: <C>", "A")     # wrong letter
    assert not _eval_lab_bench("I cannot decide.", "A")       # no answer line
    assert not _eval_lab_bench("", "A")                       # empty response


def test_prompt_does_not_leak_gold_letter():
    # The concluding instruction is format-only and must never name the correct
    # letter via the "Final Answer:" cue, regardless of where the shuffle places it.
    p1, g1 = _build_mcq_prompt("proto", "q1?", "aaa ideal", ["mmm", "zzz"])
    p2, g2 = _build_mcq_prompt("proto", "q2?", "zzz ideal", ["aaa", "mmm"])
    for p, g in ((p1, g1), (p2, g2)):
        assert f"Final Answer: <{g}>" not in p
        assert f"Final Answer: {g}" not in p
    assert p1.splitlines()[-1] == p2.splitlines()[-1]         # answer-independent footer


def test_gold_letter_uses_deterministic_shuffle_not_alphabetical():
    # Options are shuffled DETERMINISTICALLY by question (reproducible), not sorted,
    # so the correct answer is not pinned to the alphabetical slot (positional bias).
    _, g_a = _build_mcq_prompt("p", "same-q", "aaa", ["mmm", "zzz"])
    _, g_a2 = _build_mcq_prompt("p", "same-q", "aaa", ["mmm", "zzz"])
    assert g_a == g_a2 and g_a in ("A", "B", "C")             # reproducible + valid letter
    # Across different questions the gold lands in more than one slot (bias removed);
    # under a plain alphabetical sort "aaa" would always be "A".
    letters = {_build_mcq_prompt("p", f"q{i}", "aaa", ["mmm", "zzz"])[1] for i in range(8)}
    assert len(letters) > 1
