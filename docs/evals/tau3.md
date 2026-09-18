# tau3 setup (τ³-bench `banking_knowledge`)

> **Before a large sweep:** tau2/tau3 stage a ~700-file retrieval knowledge base into
> `/tmp/agentic_search_*` per simulation and can leak them (filling the tmpfs inode table,
> which crashes later evals with a misleading `No space left on device`). Raise the `/tmp`
> inode cap and clean stale dirs first. See [large-sweep-setup.md](large-sweep-setup.md).


**What tau3 is.** τ³-bench ("tau three") is the current generation of the Sierra tau-bench
line (`sierra-research/tau2-bench`, the repo now *is* τ³-bench). Its defining new **text**
domain is **`banking_knowledge`**: a retrieval-augmented (RAG) customer-service domain where
the agent answers banking tasks over a ~700-document knowledge base using retrieval tools
(BM25 + dense embeddings + shell/grep). gbench's `tau3` runs this domain.

> **Why not telecom?** telecom is a **τ²** domain and is already covered by the
> [tau2](tau2.md) suite (airline + retail + telecom). Running it again under "tau3" was a
> mislabel and a duplicate, so tau3 now runs the genuinely-new τ³ domain instead.

The τ-line lineage: τ-bench (airline, retail) → τ²-bench (+ telecom, dual-control) →
τ³-bench (+ `banking_knowledge`, + voice full-duplex, + 75+ task fixes).

## It cannot be scored single-turn: default is a hard error
banking_knowledge is a multi-turn **environment** benchmark with a retrieval backend; a plain
single-turn `/v1` endpoint cannot measure it (the agent must retrieve over the KB and act on
environment state the model never sees). By default `tau3` **hard-errors** (`infra_required`)
with that reason instead of a misleading ~0%. (The `_eval_tau3` / `_load_tau3_samples`
helpers are a legacy telecom-derived single-turn tool-call diagnostic kept only for tests;
they are unrelated to the banking_knowledge scoring path.)

## Full environment (canonical): `TAU2_ENV_RUN=1`
gbench drives the tau2-bench **Python API** (`run_domain`, domain `banking_knowledge`),
routing the agent to your endpoint per-call. On top of the shared tau2 requirements
(simulator importable, `GEMINI_API_KEY` for the user simulator + nl-assertion judge, see
[tau2.md](tau2.md)), banking_knowledge needs a **retrieval backend**:

### Requirements
- **`rank-bm25`**, the BM25 retriever used by the default/`alltools` configs:
  ```bash
  pip install rank-bm25
  ```
- **Agentic-shell sandbox binaries** (system packages). `banking_knowledge`'s retrieval tools run in
  tau2's agentic-shell sandbox, whose constructor hard-errors "required binaries are not installed"
  if any of `srt`, `rg`, `bwrap`, `socat` is missing (`tau2/knowledge/sandbox_manager.py`
  `_REQUIRED_BINARIES_LINUX`). On Debian/Ubuntu:
  ```bash
  sudo apt install ripgrep socat bubblewrap python3-pysrt
  # rg <- ripgrep, socat <- socat, bwrap <- bubblewrap, srt <- python3-pysrt
  ```
  Verify with `which srt rg bwrap socat` (all four must resolve). This affects `tau3` (and any tau2
  run of the `banking_knowledge` domain); the retail/airline/telecom tau2 domains do NOT need it.
- **An embeddings backend for dense retrieval.** By default gbench uses **Gemini embeddings**
  via Google's **OpenAI-compatible** endpoint, reusing your existing `GEMINI_API_KEY`, so no
  OpenAI/OpenRouter key is required. gbench registers an `alltools-gemini` retrieval variant
  (BM25 + dense `gemini-embedding-2` + shell) and points the OpenAI SDK embedder at Gemini.
- If the simulator isn't importable, `rank-bm25` is missing, or (for the default path) no
  embeddings key is available, tau3 **hard-errors** (`infra_required`); it never emits a misleading score.

### Retrieval knobs (env)
- **`TAU2_RETRIEVAL_CONFIG`**, override the retrieval backend with any stock tau2-bench
  variant: `alltools` (canonical, needs your own OpenAI/OpenRouter creds), `bm25`, `bm25_grep`,
  `grep_only`, `no_knowledge`, `openai_embeddings*`, `qwen_embeddings*`, … When set, gbench
  uses it verbatim and does **not** wire Gemini.
- **`TAU2_EMBED_MODEL`** (default `gemini-embedding-2`), the Gemini embedding model for the
  default path (e.g. `gemini-embedding-001`). The KB docs and queries are embedded with the
  same model; the two Gemini embedding spaces are **incompatible**, so don't mix models across
  a cached run.
- **`TAU2_EMBED_BASE_URL`** (default Google's OpenAI-compat endpoint) and
  **`TAU2_EMBED_API_KEY`** (default `GEMINI_API_KEY`).

All the `TAU2_*` sampling / trials / progress / trace knobs from [tau2.md](tau2.md) apply
here too (`TAU2_NUM_TRIALS`, `TAU2_TEMPERATURE`, `TAU2_SAVE_TRACES`, `TAU2_PROGRESS_SECS`, …).

## Comparability caveats
- **Embedder ≠ leaderboard.** The τ³-Banking leaderboard's `alltools` uses a specific embedder
  (OpenAI `text-embedding-3-large` or Qwen). The default Gemini embedder is a legitimate,
  reproducible variant but **not byte-identical** to the board; treat leaderboard comparison
  with care (switch to the exact embedder via `TAU2_RETRIEVAL_CONFIG=alltools` + OpenAI creds
  if you need parity).
- **v1.0.1 grading.** tau2-bench v1.0.1 changed `banking_knowledge` grading; results from
  `< 1.0.1` and `>= 1.0.1` are **not comparable**.
- banking is the hardest τ³ domain; even frontier models score low, so a modest number is
  expected, not a bug.

## Run
```bash
# default (no TAU2_ENV_RUN): hard-errors (infra_required) with an explanation, never a misleading 0%
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals tau3
# full environment (the only scored path): rank-bm25 + Gemini embeddings via your GEMINI key
pip install rank-bm25
export GEMINI_API_KEY="..."
TAU2_ENV_RUN=1 gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals tau3 --sandboxes 8
```
`accuracy` is the canonical tau2 **mean reward**; `correct_answers` is the perfect-task
(reward ≥ 1) count; `category_accuracy` reports the `banking_knowledge` domain; a `tau2_report`
field carries mean reward / perfect tasks / infra errors.
