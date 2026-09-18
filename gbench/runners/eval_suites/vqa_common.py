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

"""Canonical scoring metrics for the visual-question-answering suites.

These suites were all scored with the same bidirectional substring test
(``gold in pred or pred in gold``), which is far more lenient than any of their real
metrics: a verbose answer that merely mentions the gold passes, and a one-character
prediction contained in the gold passes too. This module implements what each benchmark
actually uses:

* **ANLS** (DocVQA, InfographicVQA) - average normalized Levenshtein similarity, correct
  when >= 0.5, which tolerates OCR-level typos but not different answers.
* **VQA accuracy** (TextVQA) - ``min(#matching annotators / 3, 1)`` over the 10 human
  answers, after the standard VQA normalization.
* **Relaxed accuracy** (ChartQA, CharXiv) - numeric answers within 5%, otherwise exact
  match on the normalized string.
"""

from __future__ import annotations

import re
from typing import Any, List

_ARTICLES = {"a", "an", "the"}
_NUM_WORDS = {
    "none": "0",  # canonical VQA manualMap includes none->0
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_PUNCT = re.compile(r"[^\w\s.%-]")
_WS = re.compile(r"\s+")


def normalize_answer(text: Any) -> str:
    """Standard VQA normalization: casefold, strip punctuation/articles, unify numbers."""
    s = str(text or "").strip().lower()
    s = s.replace("\n", " ")
    s = _PUNCT.sub("", s)
    tokens = [t for t in _WS.split(s) if t and t not in _ARTICLES]
    tokens = [_NUM_WORDS.get(t, t) for t in tokens]
    out = " ".join(tokens).strip()
    # trailing units/percent kept, but a bare trailing period is noise
    return out.rstrip(".").strip()


def as_gold_list(gold: Any) -> List[str]:
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold] if gold.strip() else []
    if isinstance(gold, (list, tuple, set)):
        return [str(g) for g in gold if str(g).strip()]
    return [str(gold)]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_numeric(text: str) -> bool:
    """True when the whole answer is a number, allowing a currency symbol or %/unit suffix.

    ANLS normalization is now light (it keeps punctuation), so the numeric-equality override
    - which stops "200.00" from matching "100.50" by edit distance - must still recognise
    "$200.00" / "50%" as numeric rather than treating the symbol as a letter.
    """
    return bool(re.fullmatch(r"[-+]?[$€£¥]?\s*\d[\d,]*\.?\d*\s*%?", (text or "").strip()))


def _first_number(text: str):
    m = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(m[0]) if m else None

def _anls_normalize(text: Any) -> str:
    """Canonical ANLS normalization (DocVQA / ST-VQA): lowercase + whitespace-collapse ONLY.

    The full VQA ``normalize_answer`` strips articles, punctuation and maps number words,
    which is the wrong (over-aggressive) preprocessing for the edit-distance-based ANLS and
    made it diverge from published DocVQA/InfographicVQA numbers.
    """
    return _WS.sub(" ", str(text or "").replace("\n", " ").strip().lower()).strip()


def anls_score(pred: Any, golds: Any, threshold: float = 0.5) -> float:
    """ANLS over all reference answers (DocVQA / InfographicVQA)."""
    p = _anls_normalize(pred)
    best = 0.0
    for g in as_gold_list(golds):
        gn = _anls_normalize(g)
        if not gn and not p:
            return 1.0
        denom = max(len(p), len(gn))
        if denom == 0:
            continue
        # Canonical DocVQA/ST-VQA ANLS = 1 - normalized Levenshtein, thresholded at
        # 0.5, with NO numeric special case (kept faithful to the published metric).
        sim = 1.0 - (_levenshtein(p, gn) / denom)
        best = max(best, sim)
    return best if best >= threshold else 0.0


def eval_anls(pred: Any, golds: Any, threshold: float = 0.5) -> bool:
    """Binary form of the ANLS metric."""
    return anls_score(pred, golds, threshold) > 0.0


def vqa_accuracy(pred: Any, golds: Any) -> float:
    """Canonical VQA accuracy (Antol et al.; TextVQA/VQAv2 as in mmf / lmms-eval):
    the 10-way leave-one-out average of min(#matches-among-the-other-9 / 3, 1). With a
    full 10-annotator ensemble this is the published metric; with fewer annotations it
    reduces to the simple min(#matches / 3, 1) soft-accuracy form."""
    p = normalize_answer(pred)
    if not p:
        return 0.0
    gl = [normalize_answer(g) for g in as_gold_list(golds)]
    n = len(gl)
    if n < 10:
        matches = sum(1 for g in gl if g == p)
        return min(matches / 3.0, 1.0)
    accs = []
    for i in range(n):
        others = gl[:i] + gl[i + 1:]
        m = sum(1 for g in others if g == p)
        accs.append(min(m / 3.0, 1.0))
    return sum(accs) / n


def eval_vqa(pred: Any, golds: Any, threshold: float = 0.5) -> bool:
    """Binary form of VQA accuracy (>=2 of the annotators agree by default)."""
    golds_l = as_gold_list(golds)
    if len(golds_l) <= 2:
        # not an annotator ensemble -> exact match on the normalized answer
        p = normalize_answer(pred)
        return bool(p) and any(normalize_answer(g) == p for g in golds_l)
    return vqa_accuracy(pred, golds) >= threshold




def eval_relaxed(pred: Any, golds: Any, tolerance: float = 0.05) -> bool:
    """Relaxed accuracy (ChartQA / CharXiv): numeric within `tolerance`, else exact match."""
    p = normalize_answer(pred)
    if not p:
        return False
    pnum = _first_number(p)
    for g in as_gold_list(golds):
        gn = normalize_answer(g)
        if gn and gn == p:
            return True
        gnum = _first_number(gn)
        if pnum is not None and gnum is not None:
            if abs(pnum - gnum) <= tolerance * abs(gnum) or abs(pnum - gnum) < 1e-6:
                return True
    return False


def extract_short_answer(text: Any) -> str:
    """Pull the stated answer out of a verbose response before metric comparison.

    Models often answer "The total is 42." - the metrics expect "42". Prefer an explicit
    Answer:/Final Answer: span, else the last non-empty line.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    m = re.findall(r"(?i)(?:final\s+answer|answer)\s*[:=]\s*(.+)", s)
    if m:
        return m[-1].strip().split("\n")[0].strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    last = lines[-1] if lines else s
    # Strip a short label prefix ("Final value: 42.5%", "Total: 7"). This is extraction,
    # not leniency: the value still has to match, a wrong value still fails.
    label = re.match(r"^[A-Za-z][A-Za-z ]{0,24}[:=]\s*(\S.*)$", last)
    if label:
        return label.group(1).strip()
    return last
