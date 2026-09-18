# ifeval

Canonical IFEval (`google/IFEval`): verifiable instruction following. Each prompt
carries an `instruction_id_list` plus per-instruction `kwargs` (e.g. word counts,
required keywords, formats, casing, no-commas). **Scoring is programmatic**: a
registry of ~25 checkers verifies each instruction, and gbench reports the canonical
**strict prompt-level accuracy**: the response must satisfy *every* instruction in
the prompt. No LLM judge.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ifeval --eval-limit 20
```
