# mrcr

Canonical MRCR (`openai/mrcr`): Multi-Round Co-reference Resolution, a needle-in-a-
haystack long-context retrieval where the target is one of several near-identical
"needles" planted in a long multi-turn context. **Scoring: the canonical MRCR metric.** The response must START with the row's
`random_string_to_prepend` canary (0 otherwise), the canary is stripped from both
sides, and the score is the `difflib.SequenceMatcher` ratio. The headline `accuracy`
is the **mean ratio**; `exact_match_rate` reports verbatim reproductions.

## Requirements
- A serving endpoint supporting the required **long context** (≥131k for the 131k
  configuration; a short `max_model_len` truncates the haystack and invalidates the
  metric).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mrcr --eval-limit 20
```
