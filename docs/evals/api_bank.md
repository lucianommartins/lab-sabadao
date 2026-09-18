# api_bank

Canonical API-Bank (Li et al., EMNLP 2023; arXiv:2304.08244; code
[AlibabaResearch/DAMO-ConvAI/api-bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank);
dataset `liminghao1630/API-Bank`). A **runnable** tool-use benchmark: a simulated pool of
~73 Python APIs backed by a local fake DB. Three progressively harder ability levels, two
tracks each:

- **Level-1 Call**: all API descriptions given, emit the right call.
- **Level-2 Retrieve+Call**: first `ToolSearcher`-retrieve the API, then call it.
- **Level-3 Plan+Retrieve+Call**: batch-inference steps over unknown tools.

## Scoring (execution-based, no LLM judge)

- **Correctness (Accuracy)**: the predicted call is **executed against the simulated
  backend** and compared to the gold result via each API's `check_api_call_correctness`
  (name + executed `output`/`exception`), exactly as upstream `evaluator.py`; parameter
  values that execute to the same result are correct, not lenient string matching. L1/L2
  reuse the upstream `Evaluator.evaluate` verbatim (fresh `ToolManager` per call); L3 uses
  `lv3_apis` + `level-3.json` gold and a shared `ToolManager`.
- **Response quality**: ROUGE-L F over the AI-turn generations.
- **Level-3 sample success rate**: `(50 − #dialogues-with-any-errored-step)/50`.

Reported: headline `accuracy` = mean execution Correctness over L1/L2/L3 (call track);
`dimension_scores` carries all six numbers (`L{1,2,3}_call`, `L{1,2,3}_response`) plus
`L3_sample_success`. `leaderboard_comparable=false` (the message roles are normalised for a
chat template (API-result turns folded to the user side), a minor deviation from the
OpenAI-role original).

## Requirements

- An OpenAI-compatible `/v1` endpoint (single-turn per step; **no** GEMINI/LLM judge).
- **The API-Bank source checkout**, pointed at by **`GBENCH_APIBANK_DIR`** (execution
  backend + L1/L2 raw conversations). The suite **hard-errors** with the clone command if it
  is absent:
  ```bash
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/AlibabaResearch/DAMO-ConvAI /path/apibank
  (cd /path/apibank && git sparse-checkout set api-bank)
  export GBENCH_APIBANK_DIR=/path/apibank/api-bank
  ```
- pip: the **checkout's own runtime deps** (NOT part of `gbench[evals]`; install into the
  same environment gbench runs in): `rouge` (ROUGE-L scoring in the checkout's `evaluator.py`),
  `sentence-transformers` (L2 ToolSearcher uses `paraphrase-MiniLM-L3-v2`, ~60MB, downloaded
  once), `nltk` (the checkout's `apis/search_engine.py`), and `googletrans` (the Translate API
  a few L3 gold steps invoke):
  ```bash
  pip install rouge sentence-transformers nltk googletrans
  ```
  Use the environment's current `sentence-transformers`. Do **not** downpin to the checkout's
  `requirements.txt==2.2.2`; gbench's ToolSearcher path works with the modern release.
- L3 data auto-downloads from HF (`liminghao1630/API-Bank`: `level-3-batch-inf*.json`,
  `level-3.json`). Needs network + the `datasets` client on first run.
- A handful of gold steps (Translate/Dictionary, ~1-2%) hit live-network simulated APIs.

## Run
```bash
export GBENCH_APIBANK_DIR=/path/apibank/api-bank
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals api_bank --eval-limit 20
```

## Notes / caveats

- **Sampling:** `GBENCH_API_BANK_TEMPERATURE` overrides the temperature for this suite (note the
  underscore: the eval name `api_bank` slugs to `API_BANK`); else the run default (0.0 greedy /
  1.0 with `--thinking`). No LLM judge: scoring is execution-based + ROUGE-L.
