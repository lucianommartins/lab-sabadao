# Evaluation plugins

`gbench` discovers custom evaluation suites dynamically from a plugin directory, allowing new suites to be added modularly without modifying the core codebase:

```bash
gbench --evals-only --evals all \
       --eval-plugins-dir /path/to/plugins \
       --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model>
```

`--evals all` includes every discovered plugin (or use `--evals plugins` to run only plugins, or specify plugin names individually). Each plugin module exposes `run_<plugin_name>(model_name, base_url, concurrency, enable_thinking=False, **kwargs)` and returns the standard `gbench` result dictionary.

## Start here: a worked example

`gbench` ships a minimal, dependency-free example under [`examples/custom_evals/`](../../examples/custom_evals/):

- **[`custom_qa_eval.py`](../../examples/custom_evals/custom_qa_eval.py)** is a complete plugin. It defines
  `run_custom_qa(model_name, base_url, concurrency, enable_thinking=False, **kwargs)` (the loader
  registers any `run_<name>` function), builds its samples, and delegates to the shared
  `run_eval_suite` helper with a scoring function. Copy it and adapt the loader + scorer:
  ```bash
  gbench --evals-only --eval-plugins-dir examples/custom_evals --evals custom_qa \
         --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model>
  ```
- **[`sample_benchmark.jsonl`](../../examples/custom_evals/sample_benchmark.jsonl)** is a dataset for
  the **built-in `custom_jsonl`** runner, which is the zero-code form of the same idea: point it at a
  JSONL of your own prompts and it scores each row deterministically. Use this when your questions
  have a checkable answer and you do not need custom loading, an LLM judge, or a bespoke metric:
  ```bash
  gbench --evals-only --eval-custom-jsonl examples/custom_evals/sample_benchmark.jsonl \
         --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model>
  ```
  Row format: a prompt (`prompt` / `question` / `input` / `messages`), a gold (`gold` / `answer` /
  `gold_answer` / `target`), an optional `eval_type` (`contains` [default], `exact`, `numeric`,
  `multiple_choice`), and an optional `category`.

**Which to use:** reach for the built-in `custom_jsonl` (a data file, no code) when the four
`eval_type` string/number checks are enough; write a plugin like `custom_qa_eval.py` (code) when you
need custom sample loading, a judge, or a metric the built-in scorer does not cover. `custom_jsonl`
is itself structured exactly like a plugin, so it doubles as a second reference implementation
([`gbench/runners/eval_suites/custom_jsonl.py`](../../gbench/runners/eval_suites/custom_jsonl.py)).

## Datasets
A plugin reads `"$GBENCH_DATA_DIR"/<plugin_name>.jsonl` (default `$HOME/gbench-data` or `./data`). Supported shapes:

| Shape | Recognised by | Notes |
|---|---|---|
| Chat messages | `messages: [...]` | OpenAI chat format |
| Prompt | `prompt` (+ optional `instruction_id_list` / `test_suite`) | IFEval and sandbox-executed variants |
| Structured textproto | `content` | Roles parsed from message blocks |
| CSV | Header row | Case-insensitive column mapping |

## Scoring
Scoring is deterministic where possible (sandbox execution, IFEval rules, function calling AST match); otherwise an **LLM judge** is used with defined evaluation schemas:

- **`violation`**: Safety/policy rubrics. Correct = *no* violation, so reported accuracy equals `1 - violation_rate`.
- **`binary`**: Correct = the rubric is satisfied.
- **`sxs`**: Side-by-side comparison against a baseline response; correct = candidate wins.

Suites whose expected output is a set of tool calls are scored **structurally** with an exact FunctionCall-AST match (`fc_common.score_exact_call_set`): matching tool name, argument keys, equivalent values, and parameter types.

## Prompt Diversity
Plugin results report prompt diversity statistics:

| Field | Meaning |
|---|---|
| `distinct_prompts` / `effective_n` | Number of distinct user prompts evaluated |
| `modal_prompt_share` | Fraction of rows sharing the single most common prompt |
| `low_diversity` | `true` when ≥10 rows contain fewer than 20 distinct prompts |

If a dataset replicates identical prompts across candidate versions, `effective_n` reflects the unique question count and a warning is surfaced in the summary report.

## Skip Handling
Missing prerequisites produce explicit skips (`skipped_result`) rather than failing or reporting synthetic scores:

| Skip Reason | Remedy |
|---|---|
| Dataset file not found / empty | Stage `<plugin_name>.jsonl` in `$GBENCH_DATA_DIR` |
| Judged suite without `GEMINI_API_KEY` | Set `GEMINI_API_KEY` in the environment |
| Missing compiler/sandbox toolchain | Install required runtime toolchain (see [`docs/evals/toolchains.md`](toolchains.md)) |

## Environment Variables
- `GBENCH_DATA_DIR`: Custom dataset directory path.
- `GEMINI_API_KEY`: API key for LLM judge evaluation.
- `GBENCH_JUDGE_MODEL`: Model override for the judge (default: `gemini-3.6-flash`).
- `GBENCH_RUBRICS_DIR`: Directory path containing custom rubric definitions.
