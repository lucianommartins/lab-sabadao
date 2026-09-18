# browsecomp

Canonical BrowseComp (`smolagents/browse_comp`, OpenAI): hard web-browsing/research
questions. The dataset stores `problem` and `answer` XOR-encrypted with a per-row
`canary` password; gbench decrypts them canonically before use. **Scoring: the
canonical BrowseComp LLM grader** decides correct (yes/no) against the decrypted gold
answer. The prompt (`QUERY_TEMPLATE`), grader (`GRADER_TEMPLATE`) and `n_repeats=1` are
byte-for-byte upstream (`openai/simple-evals`).

> **Closed book, and that is the canonical protocol.** Upstream BrowseComp declares **no
> tools** and runs **one turn**: browsing-capable models bring their own browser, everything
> else answers from parametric memory. gbench matches that exactly: no injected search, no
> tool loop. BrowseComp is *built* to defeat a non-browsing model (upstream reports **0.6%**
> for GPT-4o without browsing vs 51.5% for Deep Research), so a low score here is the model's
> genuine closed-book fact-retrieval result, **not** a harness bug. An earlier gbench build
> injected a Gemini-grounded `web_search`; that deviated from the protocol and has been removed.
>
> `leaderboard_comparable` is `False` for one remaining reason only: the grader is gbench's
> shared **Gemini** judge (temp 0), not upstream's `gpt-4.1`. The protocol itself is canonical.

## Requirements
- **`GEMINI_API_KEY`**: the canonical grader model is required (the eval calls the
  judge to decide correctness). The suite **hard-errors** (never skips) if it is unset.
  Set it before running.
- **Sampling:** `GBENCH_BROWSECOMP_TEMPERATURE` overrides the temperature for this suite; else
  the run default (0.0 greedy / 1.0 with `--thinking`). The judge is pinned at temperature 0.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals browsecomp --eval-limit 20
```
