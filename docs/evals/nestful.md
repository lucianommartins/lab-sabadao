# nestful

Canonical NESTFUL (`ibm-research/nestful`, IBM): nested function calling with dependent
parameters: the output of one call feeds the arguments of the next (a call DAG). The model is
asked to compose a **sequence of nested calls** to the provided tools, referencing earlier
outputs as `$<label>.<field>$`.

## Scoring

gbench reports the canonical metric set **exactly as IBM-research/NESTFUL `src/scorer.py`
(`calculate_scores`) + `src/utils.py` compute them**, the same numbers the published NESTFUL
leaderboard uses (gbench parses the model output into call dicts, then applies scorer.py's math
verbatim; it does **not** re-derive its own metric):

- **Full Match Accuracy** (headline `accuracy`, `full_match_accuracy`) and **Partial Match
  Accuracy** (`partial_match_accuracy`): per example, sklearn `accuracy_score` over the
  name-aligned `f_name(sorted args)` strings (scorer.py `post_process_api_with_args`); Full Match
  is the fraction of examples whose Partial Match is exactly 1.0.
- **F1 Intent** (`f1_intent`) and **F1 Slot** (`f1_slot`): sklearn **macro** F1 via
  `MultiLabelBinarizer` (fit on gold) over per-example API-name lists (intent) and per-(example,api)
  `arg = val` lists (slot). This is scorer.py's `compute_score_sklearn`; it is **not** the
  micro/corpus-pooled F1 gbench reported before 2026-09 (that did not match the leaderboard).
- **Win Rate** (execution accuracy, `win_rate`): the predicted sequence is **actually executed**
  against IBM-research/NESTFUL's reference function library, and the final value is compared to
  `gold_answer` (floats rounded to the gold's precision). This is NESTFUL's distinguishing "does it
  actually work" metric and is a faithful port of `scorer.py` (`calculate_ans` /
  `calculate_win_score`), including its 10-second execution timeout.

## Requirements

1. A running OpenAI-compatible `/v1` endpoint.
2. **Network + the `datasets` client**: the benchmark data auto-loads from HF Hub
   (`ibm-research/nestful`, `split="train"`) on first run and is cached locally thereafter.
3. **IBM-research/NESTFUL's reference functions** for the executable Win Rate. Point
   `GBENCH_NESTFUL_FUNC_DIR` at the checkout's `data_v2/executable_functions` directory (it holds
   `basic_functions.py`, `func_file_map.json`, and ~4348 `py_code_file_*.py`):
   ```bash
   git clone https://github.com/IBM/NESTFUL.git
   export GBENCH_NESTFUL_FUNC_DIR=$PWD/NESTFUL/data_v2/executable_functions
   ```
   This is **required**: the Win Rate is a canonical NESTFUL metric and gbench hard-errors
   (never skips or reports a partial) if the directory is missing.

> **Note: code execution.** The Win Rate executes NESTFUL's reference function implementations
> (the benchmark's own code, chosen by function *name*; the model cannot inject code). Each
> sequence is capped at 10s. This is exactly what upstream's scorer does.

## Run

```bash
export GBENCH_NESTFUL_FUNC_DIR=/path/to/NESTFUL/data_v2/executable_functions
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals nestful --eval-limit 20
```

The result carries `full_match_accuracy`, `partial_match_accuracy`, `f1_intent`, `f1_slot` and
`win_rate` (headline `accuracy` = `full_match_accuracy`).

**Sampling:** `GBENCH_NESTFUL_TEMPERATURE` overrides the temperature for this suite; else the run
default (0.0 greedy / 1.0 with `--thinking`).
