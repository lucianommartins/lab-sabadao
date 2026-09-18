# custom_jsonl

Run your own evaluation from a JSONL file (no plugin, no code).

> **Note:** `custom_jsonl` is *not* part of `--evals all` (it needs a file argument). Pass
> it explicitly with `--eval-custom-jsonl`, or see it listed via `gbench --list evals`.

## Usage
```bash
gbench --evals-only \
       --eval-custom-jsonl /path/to/my_eval.jsonl \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it
```

## File format
One JSON object per line. Either a chat-style `messages` list or a bare `prompt`:

```jsonl
{"prompt": "What is the capital of France?", "answer": "Paris", "category": "geography"}
{"messages": [{"role": "user", "content": "2+2?"}], "answer": "4"}
```

| Field | Required | Meaning |
|---|---|---|
| `prompt` *or* `messages` | yes | the request sent to the model |
| `answer` / `gold` | no | expected answer; without it the sample cannot be scored |
| `category` | no | groups rows in `category_accuracy` |
| `eval_type` | no | how the row is scored (default `contains`) |

## Scoring
Deterministic, using the strategy the row declares in `eval_type`:

| `eval_type` | Passes when |
|---|---|
| `contains` *(default)* | the response equals or contains the gold; falls back to the letter/numeric checks |
| `exact` | the whole response equals the gold (case-insensitive), nothing else |
| `numeric` | some number in the response equals the gold within 1e-4 |
| `multiple_choice` | the model's *stated* choice (boxed / after an answer anchor / last standalone letter) is the gold letter |

Rows with no gold cannot be scored and are reported as incorrect rather than silently
passed. Supply a gold for every row you want measured.

## Run knobs
`--eval-limit` caps the number of rows · `--batch-sizes` sets concurrency ·
`--eval-thinking` enables the model's reasoning channel (raise `--max-output-tokens` with
it; the default budget can truncate long reasoning and yield empty responses).
