#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Validate the gdpval rubric judge against GDPval's own gold deliverables.

The question underneath any gdpval score is "is the judge any good?". This answers it
without a model in the loop: GDPval ships the reference artifact for each task
(`deliverable_files`), so we can render the GOLD answer to text and score it against its
OWN rubric. A judge that cannot recognise the reference answer as good is broken, and the
model's number is meaningless until that is ruled out.

Interpreting the output:

  gold ~0.8-1.0   judge is calibrated; model scores are meaningful
  gold ~= model   judge cannot separate a reference answer from a model's prose - the
                  score is measuring the judge, not the model
  gold LOW        either the judge is too strict, or the text rendering of the gold file
                  loses what the rubric grades (a chart, a formula, layout) - in which case
                  those criteria should be marked "na", not failed

The gold is a *file* rendered to text, so it is handicapped exactly where a text endpoint
is: expect it to lose the file-property criteria, which is why `na` handling matters.

Usage:
    GEMINI_API_KEY=... python scripts/gdpval_validate_judge.py [--tasks 5]
"""

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gbench.runners.eval_suites import gdpval as G  # noqa: E402
from gbench.runners.eval_suites.base import DEFAULT_JUDGE_MODEL, judge_config  # noqa: E402


def render_gold(rel_paths):
    """Render a task's gold deliverable files to text, the same way inputs are rendered."""
    chunks, notes = [], []
    for rel in rel_paths or []:
        ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
        if ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "mp4", "psd", "step"):
            notes.append(f"[gold '{os.path.basename(rel)}' is a {ext}; not rendered to text]")
            continue
        text, note = G.render_attachment(rel)
        (chunks if text else notes).append(text or note)
    return "\n\n".join(chunks), notes


async def score(client, criteria, response_text):
    listing = "\n".join(f'{i}. [{c.get("score")} pts] {c.get("criterion")}'
                        for i, c in enumerate(criteria))
    prompt = G._JUDGE_PROMPT.format(criteria=listing, response=response_text[:60000])
    res = await client.aio.models.generate_content(
        model=DEFAULT_JUDGE_MODEL, contents=prompt, config=judge_config())
    m = re.search(r"\{[\s\S]*\}", res.text or "")
    verdicts = json.loads(m.group(0)) if m else {}
    earned = possible = 0.0
    na = 0
    for i, c in enumerate(criteria):
        pts = float(c.get("score") or 0)
        v = str(verdicts.get(str(i), "")).strip().lower()
        if v == "na":
            na += 1
            continue
        if pts > 0:
            possible += pts
        if v == "yes":
            earned += pts
    return (earned / possible if possible else 0.0), na, len(criteria)


async def main(n):
    from datasets import load_dataset
    from google import genai
    ds = load_dataset(G._REPO, split="train")
    rows = [r for r in ds if r["deliverable_files"]][:n]
    client = genai.Client()

    print(f"{'task':<10}{'occupation':<32}{'crit':>5}{'na':>4}{'GOLD':>8}   rendered from")
    print("-" * 96)
    scores = []
    for r in rows:
        criteria = json.loads(r["rubric_json"])
        gold_text, notes = render_gold(r["deliverable_files"])
        if not gold_text.strip():
            print(f"{r['task_id'][:8]:<10}{r['occupation'][:30]:<32}"
                  f"{len(criteria):>5}{'-':>4}{'n/a':>8}   {'; '.join(notes)[:40]}")
            continue
        frac, na, tot = await score(client, criteria, gold_text)
        scores.append(frac)
        src = ", ".join(os.path.basename(f) for f in r["deliverable_files"])[:38]
        print(f"{r['task_id'][:8]:<10}{r['occupation'][:30]:<32}{tot:>5}{na:>4}{frac:>8.3f}   {src}")

    if scores:
        mean = sum(scores) / len(scores)
        print("-" * 96)
        print(f"\nMEAN GOLD SCORE: {mean:.3f}  over {len(scores)} reference deliverables")
        print("\nRead against the model's mean_rubric_fraction from your run:")
        print("  gold >> model  -> judge is calibrated, the model's score is meaningful")
        print("  gold ~= model  -> the judge cannot tell a reference answer from prose")
        print("  gold low       -> judge too strict, or text rendering loses what is graded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    a = ap.parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is required (the judge is a Gemini call).")
    asyncio.run(main(a.tasks))
