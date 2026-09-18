# beam_128k setup

Canonical BEAM (`Mohammadta/BEAM`): a long-context benchmark exercising retrieval and
reasoning. Open-form answers are graded by an **LLM judge** against the gold answer.

> **Scope:** despite the name, gbench loads the dataset's **100K** bucket, not 128k, and
> that bucket holds ~20 conversations. Treat the number as 100K-context, small-n.

## Requirements
- **`GEMINI_API_KEY`**: the judge model. The suite **hard-errors** (infra_required) if it is unset.
- A serving endpoint with a 128k context window (run at the model's native context
  length; short `max_model_len` will truncate the inputs).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals beam_128k --eval-limit 20
```
`--max-output-tokens` ≥ 4096.
