# complexfuncbench

Canonical **ComplexFuncBench** (`THUDM/ComplexFuncBench`, arXiv:2501.10132): 1000
multi-STEP, multi-constraint tool-calling episodes over the Booking.com API surface.
Each sample is an agentic loop: the model is driven through a conversation while the
harness feeds back the **recorded** observations for the calls it gets right (the
"Golden Function Call List Updating" strategy), advancing the golden call chain as the
model consumes it. This is **not** a single-turn answer-match; gbench runs the real
loop in-process and scores it with **ComplexEval**.

## What is measured

| metric | meaning |
| --- | --- |
| **Success Rate** (headline `accuracy`) | fraction of samples where the model completes the whole golden call chain and then stops at the right moment |
| **Call Accuracy** | correct calls / total golden calls, pooled |
| **Completeness** (0-2) | a judge scores whether the final natural-language answer covers every part of the request |
| **Correctness** (0-2) | a judge scores whether the final answer is consistent with the observations |

A predicted call matches a golden call through the upstream 4-tier cascade (ported
verbatim from `utils/compare_method.py`): **(1)** rule-based exact match → **(2)** a
`value_checker` over each function's critical parameters (`utils/exact_match_values.json`,
vendored under `gbench/runners/eval_suites/_complexfuncbench_data/`) → **(3)** an OPTIONAL
live-RapidAPI response tie-breaker → **(4)** an LLM equivalence judge. When a step emits
several PARALLEL calls, they are aligned to the golden calls with **bge-large-en-v1.5**
embeddings + a max-weight assignment before comparison.

### Judge: gbench's standard Gemini cascade

Upstream uses **GPT-4o** in two judge roles (the call-equivalence judge and the response
completeness/correctness judge). gbench grades both roles with its **standard Gemini cascade**
(`base.judge_generate_cascade`) - the reference grader it uses across every judged suite, by
convention (a grader choice, not a fidelity defect). Upstream prompts are ported verbatim; only
the grader model differs from upstream. Because gbench grades with its own cascade where the
published leaderboard is GPT-4o-graded, a run here is a gbench-internal number rather than a
like-for-like leaderboard entry, so **`leaderboard_comparable` is always False**.

## Requirements

- **A tool-calling `/v1` endpoint.** The served model must support native tool calls
  (return `message.tool_calls`). For vLLM that means the server was started with
  `--enable-auto-tool-choice` and a matching `--tool-call-parser`. A model that instead
  narrates the call in text will finish the first turn as a "final answer" and score as a
  legitimate failure (not a fabricated number).
- **`GEMINI_API_KEY`**: powers both judge roles. **Hard-errors** (`infra_required`, never
  skips) if unset.
- **The dataset** `THUDM/ComplexFuncBench/ComplexFuncBench.jsonl` (~40 MB, 1000 rows),
  fetched via `huggingface_hub` at run time (needs network/HF access, or a pre-populated HF
  cache). Hard-errors if unreachable.
- **An embedding backend for parallel-call alignment.** gbench uses, in order:
  `FlagEmbedding` → `sentence-transformers` → **`transformers.AutoModel`** (the fallback,
  reproduced faithfully as CLS pooling + L2 normalisation, identical to bge's
  `FlagModel.encode`). The `transformers` path needs **no extra pip install** in the
  canonical env (torch + transformers are already present; `bge-large-en-v1.5` ships
  `tokenizer.json`, so no `sentencepiece`/`scipy` is required; the max-weight assignment is
  pure-Python). On first use it downloads `BAAI/bge-large-en-v1.5` (~1.3 GB) into the HF
  cache. The single-call-per-step case (the majority) is embedding-free (a 1×1 assignment
  is trivial), so the model is only pulled when a step has multiple non-exact parallel calls.
- **`RAPID_API_KEY` (OPTIONAL).** Only enables the tier-3 response tie-breaker (live
  Booking.com RapidAPI). Without it, the (rare) response-only matches fall through to the
  Gemini equivalence judge, a documented graceful degradation, **not** a skip. `tool_info.json`
  is vendored alongside the exact-match asset for this path.

## Environment knobs

| var | default | effect |
| --- | --- | --- |
| `GBENCH_COMPLEXFUNCBENCH_TEMPERATURE` | (run default) | per-suite temperature override |
| `GBENCH_COMPLEXFUNCBENCH_MAX_TOKENS` | `2048` | model `max_tokens` per turn (upstream protocol) |
| `GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS` | `40` | safety cap on loop rounds (longest golden chain ≈19) |
| `GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S` | `600` | per-request HTTP timeout |
| `GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | embedding model id |
| `GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE` | `cpu` | embedder device (CPU by default so it never contends with the served model's GPU) |

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 with `--thinking`
(`base.DEFAULT_TEMPERATURE`); override with `--temperature` or the per-suite var above. The
Gemini judges are pinned at 0.0.

## Run

```bash
export GEMINI_API_KEY=...            # the two judge roles
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals complexfuncbench \
       --batch-sizes 8 --eval-limit 20
```

`--eval-limit` takes a stratified sample across the five domains (Car-Rental / Hotels /
Attraction / Flights / Cross); a **full, greedy** run over all 1000 samples reports the
per-domain Success Rate with upstream's normalisation (150 per domain, Cross=400). The
result payload carries `success_rate`, `call_accuracy`, `completeness`, `correctness`,
`per_domain_success_rate`, `embedding_backend`, and the temperature provenance.
`leaderboard_comparable` is always False (Gemini judge ≠ the canonical GPT-4o).
