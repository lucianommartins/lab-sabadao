# simpleqa setup

Canonical SimpleQA (`basicv8vc/SimpleQA`, OpenAI): short-form factuality with
abstention. Each question has a single verifiable short answer; the model's response
is graded by an **LLM judge** into `correct` / `incorrect` / `not_attempted`
(abstention is not penalized as wrong), and the reported accuracy is the canonical
`correct / (correct + incorrect + not_attempted)` grade. Substring matching cannot
score this; the judge is required.

## Requirements
- **`GEMINI_API_KEY`**: the grader model. The suite **hard-errors** (`infra_required`)
  if it is unset (it will not silently fall back to a heuristic).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals simpleqa --eval-limit 20
```
`--max-output-tokens` ≥ 512. Answers are short-form; no thinking budget required.
