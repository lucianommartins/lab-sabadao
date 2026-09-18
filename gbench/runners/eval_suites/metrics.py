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

"""Shared trace-walk finalizers for canonical (non-binary) eval metrics.

``base.run_eval_suite`` scores every sample to a single boolean and reports
``accuracy = correct / scored``. Many benchmarks are NOT pass-rates: DocVQA is mean-ANLS,
Seal-Tools is micro tool/parameter F1, HealthBench is a mean weighted-rubric score. The
established pattern (ruler.py, healthbench.py, mrcr.py, omnidocbench.py, i18n_translate.py,
...) is to run the suite for the binary pass-rate + traces, then walk ``result['sample_traces']``
and OVERWRITE ``result['accuracy']`` with the canonical metric, keeping the binary number as a
secondary field. That override is copy-pasted boilerplate with one subtle, easily-missed
invariant: judge outages carry no score and must be excluded from the mean (an infra outage
must not depress the metric). These helpers own that boilerplate and that invariant once.

Return the SAME ``result`` dict (mutated) so a suite ends with ``return finalize_mean_metric(...)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional


def scorable_traces(result: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Traces that count toward a metric: a real response, not a judge outage.

    Mirrors base's ``scored_count`` denominator (base.py:1341) - JUDGE_OUTAGE is a distinct
    outcome excluded from numerator AND denominator, and a ``None`` response is a failed
    request, not a zero.
    """
    for t in result.get("sample_traces", []) or []:
        if t.get("response_text") is None:
            continue
        if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE":
            continue
        yield t


def finalize_mean_metric(
    result: Dict[str, Any],
    *,
    metric_name: str,
    score_key: Optional[str] = None,
    scorer: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
    stash_key: Optional[str] = None,
    secondary_key: str = "pass_rate",
    scale: float = 100.0,
    failure_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Overwrite ``result['accuracy']`` with the MEAN of a per-sample float metric.

    Provide EXACTLY ONE of:
      * ``score_key`` - each scorable trace already carries this float (e.g. a judge wrote
        ``trace['healthbench_score']``); or
      * ``scorer`` - ``scorer(trace) -> Optional[float]`` recomputes the score from the
        trace (e.g. ``anls_score(trace['response_text'], trace['gold_answer'])``); ``None``
        skips the sample. The value is stashed on the trace under ``stash_key`` (default
        ``score_key`` or ``'score'``) so per-sample scores survive into the report.

    ``failure_score``: when ``None`` (default) a sample with no response is EXCLUDED from
    the mean (matching healthbench/ruler). Set a float (e.g. ``0.0``) to COUNT a
    failed/empty-response sample at that score instead of dropping it - use it when the
    canonical metric treats a non-answer as wrong (MRCR, ANLS) rather than as an outage, so
    a model that times out on the hard items cannot inflate its mean by having them dropped.
    ``JUDGE_OUTAGE`` is always excluded regardless.

    The prior binary ``accuracy`` is preserved under ``secondary_key`` (default
    ``'pass_rate'``); ``result['metric']`` records what ``accuracy`` now means.
    """
    if (score_key is None) == (scorer is None):
        raise ValueError("finalize_mean_metric: pass exactly one of score_key / scorer")
    stash = stash_key or score_key or "score"

    scores: List[float] = []
    for t in result.get("sample_traces", []) or []:
        if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE":
            continue
        if t.get("response_text") is None:            # failed / empty request
            if failure_score is not None:
                scores.append(float(failure_score))
            continue
        if score_key is not None:
            if score_key not in t:
                if failure_score is not None:
                    scores.append(float(failure_score))
                continue
            raw = t[score_key]
        else:
            raw = scorer(t)  # type: ignore[misc]
            if raw is None:
                if failure_score is not None:
                    scores.append(float(failure_score))
                continue
            t[stash] = round(float(raw), 6)
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            if failure_score is not None:
                scores.append(float(failure_score))

    result.setdefault(secondary_key, result.get("accuracy"))
    result["metric"] = metric_name
    if scores:
        result["accuracy"] = round(sum(scores) / len(scores) * scale, 2)
        result["scored_samples"] = len(scores)
    return result


def pooled_prf(result: Dict[str, Any], count_key: str) -> Dict[str, float]:
    """Corpus-pooled (micro) precision/recall/F1 from per-sample ``(tp, fp, fn)`` counts.

    Each scorable trace carries ``trace[count_key]`` = ``(tp, fp, fn)`` or a dict with those
    keys (produce them with ``fc_common.prf_counts``). P/R/F1 pool counts across the whole
    run - they are NOT averageable per sample, so this must not be a mean of per-sample
    ratios. Returns the prf dict; the caller decides which figure becomes the headline.
    """
    tp = fp = fn = 0
    for t in scorable_traces(result):
        c = t.get(count_key)
        if c is None:
            continue
        if isinstance(c, dict):
            tp += int(c.get("tp", 0)); fp += int(c.get("fp", 0)); fn += int(c.get("fn", 0))
        else:
            a, b, d = c
            tp += int(a); fp += int(b); fn += int(d)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def finalize_f1_metric(
    result: Dict[str, Any],
    *,
    count_key: str,
    metric_name: str,
    secondary_key: str = "pass_rate",
    scale: float = 100.0,
) -> Dict[str, Any]:
    """Overwrite ``result['accuracy']`` with a corpus-pooled micro-F1 headline.

    Convenience wrapper over :func:`pooled_prf`: also records precision/recall as percentages.
    """
    prf = pooled_prf(result, count_key)
    result.setdefault(secondary_key, result.get("accuracy"))
    result["metric"] = metric_name
    result["precision"] = round(prf["precision"] * scale, 2)
    result["recall"] = round(prf["recall"] * scale, 2)
    result["f1"] = round(prf["f1"] * scale, 2)
    if (prf["tp"] + prf["fp"] + prf["fn"]) > 0:
        result["accuracy"] = round(prf["f1"] * scale, 2)
    return result
