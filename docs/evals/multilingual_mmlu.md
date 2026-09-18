# multilingual_mmlu

Canonical Multilingual MMLU (`alexandrainst/m_mmlu`): MMLU translated across 14+
languages, testing knowledge and reasoning cross-lingually. **Scoring: extract the
predicted option letter and match gold.** `category_accuracy` breaks down by language.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multilingual_mmlu --eval-limit 20
```
