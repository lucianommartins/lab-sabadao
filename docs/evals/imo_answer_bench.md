# imo_answer_bench

Canonical **IMO-AnswerBench** (`OpenEvals/IMO-AnswerBench`, Google DeepMind IMO-Bench,
CC-BY-4.0): 400 olympiad problems with a short final answer, evenly split across
**Algebra / Combinatorics / Geometry / Number theory** (100 each). `category_accuracy`
breaks down by those four.

**Scoring: IMO-Bench's AnswerAutoGrader, semantic equivalence, no partial credit.**
The model's stated answer (`\boxed{}` → `Final Answer:` → last line) is compared to the
reference in two stages:

1. normalized textual equality (presentation-only LaTeX is stripped, so `$3$`, `3` and
   `\(3\)` agree). This is **equality**, not containment: a response listing several
   candidate answers does not pass because the right one is among them;
2. anything that does not match textually goes to an LLM equivalence judge, which accepts
   algebraically identical expressions and reordered sets, and rejects partial answers,
   missing cases and spurious extra cases.

## Requirements
- A running OpenAI-compatible `/v1` endpoint.
- **`GEMINI_API_KEY`** for stage 2. Without it the suite still runs, but every answer that
  is not a textual match is counted incorrect and the score is an explicit lower bound
  (logged as such, with the affected count).

> **Dataset change:** this suite previously loaded `AI-MO/NuminaMath-CoT` and kept rows
> whose source looked like an olympiad and whose text mentioned "IMO". NuminaMath-CoT is a
> ~860k-row chain-of-thought *training* corpus: that subset was not the benchmark, its
> size depended on a substring match, and any model trained on NuminaMath had already seen
> the items. Numbers from before this change are not comparable with the ones after it.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals imo_answer_bench --eval-thinking --eval-limit 20
```
`--eval-thinking` recommended; raise `--max-output-tokens` with it (the default thinking
budget truncates long derivations and yields empty responses, which the run now reports as
`empty_responses`).
