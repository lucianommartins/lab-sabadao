# putnam setup

Canonical PutnamBench (`amitayusht/PutnamBench`): collegiate William Lowell Putnam
Mathematical Competition problems evaluated as **informal** proofs. The model writes a
natural-language solution/proof.

**Scoring** splits on what the reference actually contains:
- a reference stating a *single* closed-form value (one `$...$` span, or a bare value)
  is compared deterministically. The answer expression is extracted from both sides
  and must match after LaTeX normalization. An explicitly anchored answer
  (`\boxed{}` / `Final Answer:`) is held to exact equality; an unanchored one is
  searched for the value with word boundaries, so `2` is not found inside `12`.
- everything else (a prose reference such as "The limit does not exist.", a case split,
  or a pure-proof problem with no reference) goes to the **LLM proof judge**, which
  receives the reference when the dataset has one. For the
machine-checked Lean 4 track, see [putnam_formal](putnam_formal.md).

## Requirements
- **`GEMINI_API_KEY`**, the proof-grading judge, which is **required** (only ~130 of 271 golds
  are deterministically scorable single closed-forms; the rest are judge territory). The suite
  **hard-errors** (`infra_required` → `status:"error"`, never skips, never scores proofs as wrong)
  if it is unset. Provide it via the env var or `--gemini-api-key`.
- **`google-genai`**, the judge client (in `gbench[evals]`). `pip install 'gbench[evals]'`.
- **Network / HF-Hub** access to `amitayusht/PutnamBench` (downloaded once, cached).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals putnam --eval-thinking --eval-limit 10
```
`--eval-thinking` recommended (multi-step proofs). `--max-output-tokens` ≥ 8192.

## Notes / caveats
- **Sampling:** `GBENCH_PUTNAM_TEMPERATURE` overrides the temperature for this suite (else the run
  default: 0.0 greedy / 1.0 with `--thinking`); it takes precedence over `--temperature`. The proof
  judge is pinned at temperature 0.0.
- **`leaderboard_comparable` is always `false`**. This is the **informal** PutnamBench track
  (answer string-match + LLM proof judge), a gbench protocol. PutnamBench's canonical metric is
  **formal (Lean 4) proof verification**; for that, use the [putnam_formal](putnam_formal.md) suite.
