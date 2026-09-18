# putnam_formal setup

Canonical PutnamBench formal track (`amitayusht/PutnamBench`): Putnam competition
problems evaluated as **machine-checked Lean 4 proofs**. The model emits a Lean 4
proof term, which is compiled and verified inside a Docker sandbox. A problem is
correct iff the Lean kernel accepts the proof (no heuristic or LLM judge is involved).
For the informal, LLM-judged track, see [putnam](putnam.md). SANDBOX_EVAL.

## Requirements
- **Docker** (a reachable daemon), and a **Lean image that includes Mathlib**. Every PutnamBench
  statement `import`s Mathlib (`Finset.Icc`, `Tendsto`, `Polynomial`, `𝓝`, ...), so the stock
  `leanprovercommunity/lean4:latest` (toolchain only, no Mathlib) makes **every** proof fail to
  elaborate regardless of the model. The suite runs a Mathlib probe at startup and **hard-errors**
  (`infra_required`) if the image has no Mathlib, rather than report a structural 0%.
- Build the bundled Mathlib image once (bakes Mathlib's prebuilt olean cache; several GB):
  ```bash
  docker build -t gbench-putnam-formal -f docker/putnam_formal.Dockerfile docker
  export GBENCH_PUTNAM_FORMAL_LEAN_IMAGE=gbench-putnam-formal
  ```
  `GBENCH_PUTNAM_FORMAL_LEAN_IMAGE` (legacy alias `PUTNAM_LEAN_IMAGE`) selects the image; point it at
  any Mathlib-provisioned image or a prebuilt lake project with a Mathlib cache. Pin an `@sha256`
  digest for a publishable, reproducible number (a floating tag can rename/deprecate lemmas).
- No `GEMINI_API_KEY` needed; verification is done by the Lean kernel, not a judge.

## Run
```bash
export GBENCH_PUTNAM_FORMAL_LEAN_IMAGE=gbench-putnam-formal
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals putnam_formal \
       --sandboxes 8 --eval-limit 10
```
`--sandboxes` bounds concurrent Lean verification containers. `--max-output-tokens` should be
generous for full proof terms. `GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S` (default 300) caps each
compile; `import Mathlib` elaboration alone can take tens of seconds.
