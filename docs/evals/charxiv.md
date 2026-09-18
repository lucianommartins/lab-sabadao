# charxiv

Canonical CharXiv (Wang et al., NeurIPS 2024 D&B; arXiv:2406.18521;
[princeton-nlp/CharXiv](https://github.com/princeton-nlp/CharXiv);
HF `princeton-nlp/CharXiv`): realistic academic-chart understanding over the 1000-chart
`validation` split, in **two tracks**:

- **Descriptive**: 4 questions per chart (one of 19 element-reading templates each), 4000 total.
- **Reasoning**: 1 open-ended synthesis question per chart, 1000 total.

Each chart is fanned out into its 5 canonical questions and asked as single-shot multimodal
VQA (image + the exact upstream prompt template).

## Scoring

**LLM-judge against the exact upstream rubrics**, not string matching. The descriptive and
reasoning grading prompts (per-template / per-inst_category) are vendored verbatim from the
CharXiv repo in `gbench/runners/eval_suites/charxiv_constants.py`; the judge returns a JSON
`score ∈ {0,1}` per question. An unparseable/invalid grade counts as 0 (kept in the
denominator, as upstream does); a judge-cascade outage is excluded from the denominator.

Two numbers are reported **separately** (the paper reports no combined figure; its
leaderboard is sorted by Reasoning, so that is the headline `accuracy`):

- `charxiv_reasoning_accuracy`, the headline.
- `charxiv_descriptive_accuracy`.

**Judge backend.** Canonical CharXiv pins `gpt-4o-2024-05-13`; gbench grades with its standard
Gemini judge cascade - the reference grader it uses across all judged suites, by convention (a
grader choice, not a fidelity defect). The prompts, rubrics and JSON contract are identical. The
numbers are CharXiv-protocol-faithful, graded by gbench's cascade rather than the paper's GPT-4o,
so they are a gbench-internal number rather than a like-for-like leaderboard entry
(`leaderboard_comparable=false`); label accordingly against the paper's GPT-4o leaderboard.

## Requirements

- A **vision-language** (multimodal) model endpoint (observations are chart images).
- **`GEMINI_API_KEY`**: required (the judge grades every question; without it the suite
  **hard-errors** via `infra_required`, never skipping and never scoring on fabricated data).
  No other API keys.
- `datasets` (a core gbench dependency) to fetch `princeton-nlp/CharXiv` `validation` from HF
  Hub (needs network on first run; cached thereafter).
- No new pip packages: the grading rubrics are vendored in-tree.
- **Sampling:** `GBENCH_CHARXIV_TEMPERATURE` overrides the temperature for this suite; else the
  run default (0.0 greedy / 1.0 with `--thinking`). The judge is pinned at temperature 0.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals charxiv --eval-limit 20
```
`--eval-limit N` samples **N charts** (stratified by subject) and runs all 5 questions each.
