# gbench environment knob reference

Every gbench evaluation knob is an environment variable named `GBENCH_<SUITE>_<KNOB>` (for a per
suite knob) or `GBENCH_<AREA>_<KNOB>` (for a cross cutting one). This page is the consolidated list.
Per suite provisioning details also live in each `docs/evals/<suite>.md`.

Command line flags (`--temperature`, `--max-output-tokens`, `--eval-limit`, `--sandboxes`,
`--thinking`, `--eval-n-shot`, ...) are documented by `gbench --help`; this page covers the
environment knobs, which tune behavior that has no dedicated flag.

## Convention and precedence

- Canonical form is `GBENCH_<SUITE>_<KNOB>`, upper snake case, where `<SUITE>` is the registered
  suite name. Example: `GBENCH_SCICODE_EVAL_TIMEOUT_S`.
- Some suites shipped older bare prefixed names (for example `TAU2_NUM_TRIALS`,
  `SCICODE_EVAL_TIMEOUT_S`, `SPIDER2_REPO_DIR`, `PUTNAM_LEAN_IMAGE`). Those still work as
  deprecated aliases: the canonical `GBENCH_` name takes precedence, and using a bare name logs a one
  time warning naming its replacement. New scripts should use the canonical names.
- Temperature precedence (highest first): `GBENCH_<SUITE>_TEMPERATURE`, then `--temperature`, then the
  suite default (think aware: 0.0 without `--thinking`, 1.0 with it). See
  `resolve_temperature` in `gbench/runners/eval_suites/base.py`.
- Standard third party variables are read directly and are not gbench knobs: `GEMINI_API_KEY`
  (and `GEMINI_API_KEYS`, `GEMINI_OPENAI_BASE_URL`), `OPENAI_API_KEY`, `HF_TOKEN` /
  `HUGGING_FACE_HUB_TOKEN`, `BRAVE_API_KEY`, `SERPAPI_API_KEY`, `RAPID_API_KEY`, `XDG_CACHE_HOME`.

## Cross cutting knobs

### Serving and run infrastructure
- `GBENCH_SERVER_PORT` (default 8000): port the local vLLM server binds, and the port every eval
  targets for a local run.
- `GBENCH_SANDBOX`: sandbox backend selector for containerized evals.
- `GBENCH_RESULTS_DIR`: base directory for result artifacts (set by the run manager).
- `GBENCH_ENV`: deployment environment label used by the quality pillar.
- `GBENCH_EVAL_PLUGINS_PATH`: path searched for eval plugins (same as `--eval-plugins-dir`).
- `GBENCH_SWE_OMP_THREADS`: OpenMP thread cap applied inside the swebench family harnesses.
- `GBENCH_PREREQS_DIR` (default `~/.cache/gbench/gbench-prereqs`): shared root for suites that need a
  local checkout / data dir. When a suite's own dir var is unset, it falls back to
  `$GBENCH_PREREQS_DIR/<subdir>` **if that path exists** (else it hard-errors as before - the fallback
  never masks a missing prereq). Provision the tree once and set this, instead of exporting a per-suite
  var each time. Subdir per suite: `codeforces` (GBENCH_CODEFORCES_GENERATED_TESTS),
  `PurpleLlama/CodeShield` (cyberseceval), `NESTFUL/data_v2/executable_functions` (nestful),
  `OJBench_testdata` (ojbench), `RULER` (ruler), `Spider2/spider2-lite[/evaluation_suite/gold |
  /resource/databases/spider2-localdb]` (spider2), `SWE-bench_Pro-os` (swe_bench_pro),
  `SWELancer-Benchmark` (swe_lancer), `WildClawBench` (wildclawbench), `aider-bench` (aider_polyglot),
  `apibank/api-bank` (api_bank). An explicit per-suite var always overrides the fallback.
- `SHAREGPT_PATH`: path to the ShareGPT dataset for the `--campaign chat-like` serving workload; if
  unset and the default cache copy is missing, it is downloaded just-in-time.
- `GBENCH_GCP_AUTH` (default unset): opt in to automatic GCP identity-token auth for a remote
  `*.run.app` / `*.googleapis.com` endpoint (metadata server, then a `gcloud auth
  print-identity-token` fallback). Unset, gbench makes no metadata-server or `gcloud` call - pass
  `VLLM_API_KEY` instead.

Stress pillar knobs (all four resolve **flag > `GBENCH_STRESS_*` env > default**):
- `GBENCH_STRESS_CLIENT_PROCS` (flag `--stress-client-procs`; default `min(8, cpu_count-1)`): OS
  processes for the multi-process load generator, which removes the single-asyncio-client CPU
  confound at the knee. **Auto-capped to `cpu_count-1`**, so a large value is safe on small hosts
  (it won't spawn more procs than cores). Raise it alongside `GBENCH_STRESS_MAX_QPS` so the
  client can actually offer the higher load; if it can't, those points are flagged `client_bound`
  and excluded rather than reported as a server capacity result.
- `GBENCH_STRESS_REPS` (flag `--stress-reps`; default 3): full QPS sweeps per point. The knee is
  reported as mean + bootstrap CI over reps, since a single sweep varies run-to-run.
- `GBENCH_STRESS_MAX_QPS` (flag `--stress-max-qps`; default 512): safety cap on the open-loop
  arrival-rate sweep. Raise it when a fast model passes the SLO at 512 req/s (the knee is otherwise
  censored at 512 rather than measured). Safe to raise as a default because the `client_bound` guard
  and the `--stress-client-procs` auto-cap protect weak clients and small hosts.
- `GBENCH_STRESS_MAX_PROMPTS` (flag `--stress-max-prompts`; default 1200): cap on requests per rate
  point. Raise it so high-QPS points span a real steady-state window (at 1000 req/s, 1200 prompts is
  ~1.2s of arrivals; 8000 is ~8s).

### Decoding and scoring (apply to every native suite, defined in base.py)
These change headline numbers and their comparability, so record them when you set them.
- `GBENCH_REPETITION_PENALTY`, `GBENCH_FREQUENCY_PENALTY`, `GBENCH_PRESENCE_PENALTY`: decoding
  penalties added to every request. Unset by default (no penalty).
- `GBENCH_IMAGE_SOFT_TOKENS` (default 1120): per image soft token budget used to estimate multimodal
  prompt length for context clamping. See also the roadmap note on per model derivation.
- `GBENCH_MIN_DECODE_TOK_S`: minimum decode tokens per second before a request is treated as stalled.
- `GBENCH_REQUEST_TIMEOUT_S`: per request timeout for the eval client.
- `GBENCH_NON_CONVERGENT_RATIO`: threshold for flagging a non convergent generation.
- `GBENCH_RETRY_DISCARDED_FINAL`: whether to retry a discarded final turn.
- `GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS`: per turn output ceiling inside tool calling loops.
- `GBENCH_<SUITE>_TEMPERATURE`: per suite temperature override for any suite (highest precedence).

### Judge cascade (any suite that uses the Gemini cascade judge)
- `GBENCH_JUDGE_MODELS`: comma separated judge model cascade (tried in order).
- `GBENCH_JUDGE_MODEL`: single judge model (shorthand when the cascade is one model).
- `GBENCH_JUDGE_CASCADE_ROUNDS`: retry rounds across the cascade before a judge outage.
- `GBENCH_JUDGE_BACKOFF`: backoff seconds between judge retries.

### Web search cascade (browsecomp, frames, simpleqa, gaia, deepsearch_qa, ...)
- `GBENCH_SEARCH_MODELS` / `GBENCH_SEARCH_MODEL`: grounding model cascade / single model.
- `GBENCH_SEARCH_MAX_RESULTS`: results returned per query.
- `GBENCH_SEARCH_ATTEMPTS`, `GBENCH_SEARCH_BACKOFF`, `GBENCH_SEARCH_CASCADE_ROUNDS`: retry shape.
- `GBENCH_SEARCH_MAX_RPS`, `GBENCH_SEARCH_MAX_BURST`: client side rate limiting.

## Per suite knobs

Each suite reads only the knobs listed under it. Names are canonical; where a bare legacy alias
exists it is noted. Full provisioning (checkouts, images, keys) is in each `docs/evals/<suite>.md`.

### Pillar 1: general knowledge and reasoning
- aime: `GBENCH_AIME_INCLUDE_PRECUTOFF`, `GBENCH_AIME_PRECUTOFF_DATASET`,
  `GBENCH_AIME_POSTCUTOFF_DATASET` (contamination split control).
- cyberseceval: `GBENCH_CYBERSECEVAL_ICD` (path to the CodeShield Insecure Code Detector checkout;
  legacy alias `CYBERSECEVAL_ICD_RULES` derives it from the rules dir).
- culer: `GBENCH_CULER_ALL_DOMAINS` (legacy alias `CULER_ALL_DOMAINS`; score all domains vs the
  default subset).

### Pillar 2: mathematics and proofs
- putnam_formal: `GBENCH_PUTNAM_FORMAL_LEAN_IMAGE`, `GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S` (legacy
  aliases `PUTNAM_LEAN_IMAGE`, `PUTNAM_COMPILE_TIMEOUT_S`; the Lean/Mathlib toolchain image and the
  per proof compile timeout, both of which affect the verified score).

### Pillar 3: coding and software engineering
- scicode: `GBENCH_SCICODE_TEST_DATA`, `GBENCH_SCICODE_TEST_DATA_REPO`, `GBENCH_SCICODE_EVAL_TIMEOUT_S`,
  `GBENCH_SCICODE_HTTP_TIMEOUT_S`, `GBENCH_SCICODE_THREADS`, `GBENCH_SCICODE_WITH_BACKGROUND` (the
  first four have legacy bare `SCICODE_` aliases).
- lcb: `GBENCH_LCB_VERSION`, `GBENCH_LCB_MIN_DATE`, `GBENCH_LCB_INCLUDE_TEST_GEN` (release tag,
  contamination cutoff, and whether to include the test generation split; legacy alias
  `LCB_INCLUDE_TEST_GEN`).
- ojbench: `GBENCH_OJBENCH_TESTDATA` (legacy alias `OJBENCH_TESTDATA`; path to the online judge test
  data, bind-mounted read-only into the judge container), `GBENCH_OJBENCH_IMAGE` (default
  `gbench-ojbench`; the locally-built judge image carrying ojbench + dmoj + PyPy3 + g++ on Py3.11),
  `GBENCH_OJBENCH_JUDGE_TIMEOUT_S` (host-side cap on the whole judge container; default
  `max(1200, 180*N)`), and `OJBENCH_GET_TIMEOUT_S` (in-image; default 120s inter-result poll for the
  dead-worker break that stops a crashed worker from hanging the judge loop).
- codeforces: `GBENCH_CODEFORCES_SPLIT`, `GBENCH_CODEFORCES_CONFIG`,
  `GBENCH_CODEFORCES_GENERATED_TESTS`, `GBENCH_CODEFORCES_MAX_GENERATED_TESTS`,
  `GBENCH_CODEFORCES_TIME_MULTIPLIER`.
- aider_polyglot: `GBENCH_AIDER_IMAGE`, `GBENCH_AIDER_MODEL`, `GBENCH_AIDER_EDIT_FORMAT`,
  `GBENCH_AIDER_LANGUAGES`, `GBENCH_AIDER_BENCHMARK_DIR`, `GBENCH_AIDER_EXTRA_ARGS`.
- multipl_e: `GBENCH_MULTIPL_E_IMAGE`, `GBENCH_MULTIPL_E_LANGS`, `GBENCH_MULTIPL_E_SAMPLES`.
- swe_bench_pro: `GBENCH_SWE_BENCH_PRO_HARNESS_DIR`, `GBENCH_SWE_BENCH_PRO_RUN`,
  `GBENCH_SWE_BENCH_PRO_RAW_SAMPLE` (legacy bare `SWE_BENCH_PRO_` aliases), plus the separate
  `GBENCH_SWEBENCH_PRO_AGENTIC` toggle.
- swe_lancer: `GBENCH_SWE_LANCER_HARNESS_DIR`, `GBENCH_SWE_LANCER_EVAL_CMD`, `GBENCH_SWE_LANCER_RUN`
  (legacy bare `SWELANCER_` aliases).
- multi_swe_bench: `GBENCH_MULTI_SWE_REPO_DIR`.
- terminal_bench: `GBENCH_TB_TIMEOUT_MULTIPLIER`, `GBENCH_TB_TURN_MAX_TOKENS`,
  `GBENCH_TB_CONTAINER_THREADS`, `GBENCH_TB_PARSER`, `GBENCH_TB_KEEP_JOBS_DIR`,
  `GBENCH_TB_SALVAGE_PATCH`.
- spider2: `GBENCH_SPIDER2_REPO_DIR`, `GBENCH_SPIDER2_GOLD_DIR`, `GBENCH_SPIDER2_LOCALDB_DIR` (legacy
  bare `SPIDER2_` aliases).

### Pillar 4: long context and retrieval
- ruler: `GBENCH_RULER_DIR`, `GBENCH_RULER_TOKENIZER`, `GBENCH_RULER_LENGTHS`,
  `GBENCH_RULER_NUM_SAMPLES` (the NVIDIA RULER checkout, tokenizer, context bands, and per band count).

### Pillar 5: tool use and agentic workflows
- api_bank: `GBENCH_APIBANK_DIR`.
- bfcl_v4_agentic: `GBENCH_BFCL_PROJECT_ROOT`, `GBENCH_BFCL_SEARCH_BACKEND`,
  `GBENCH_BFCL_SEARCH_GEMINI_MODEL`, `GBENCH_BFCL_TIMEOUT_S`, `GBENCH_BFCL_REUSE_GENERATIONS` (legacy
  bare `BFCL_` aliases).
- nestful: `GBENCH_NESTFUL_FUNC_DIR`.
- mcp_atlas: `GBENCH_MCP_ATLAS_JUDGE_MODEL`.
- mcp_bench: `GBENCH_MCP_BENCH_IMAGE`, `GBENCH_MCP_BENCH_TIMEOUT_S`, `GBENCH_MCP_BENCH_NO_SUBSET`,
  `GBENCH_MCP_BENCH_DISTRACTION`, `GBENCH_MCP_BENCH_DISABLE_STABILITY`,
  `GBENCH_MCP_BENCH_ASSUME_NETWORK`.
- toolbench: `GBENCH_TOOLBENCH_IMAGE`, `GBENCH_TOOLBENCH_METHOD`, `GBENCH_TOOLBENCH_GROUPS`,
  `GBENCH_TOOLBENCH_REFERENCE`, `GBENCH_TOOLBENCH_REFERENCE_DIR`, `GBENCH_TOOLBENCH_EVAL_TIMES`,
  `GBENCH_TOOLBENCH_SIMULATOR_BASE`, `GBENCH_TOOLBENCH_SIMULATOR_MODEL`,
  `GBENCH_TOOLBENCH_SIMULATOR_KEY`, `GBENCH_TOOLBENCH_MAX_QUERY_COUNT` (per-task DFS search budget,
  canonical 200; distinct from `--eval-limit`, which caps the number of TASKS per group).
- complexfuncbench: `GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS`, `GBENCH_COMPLEXFUNCBENCH_MAX_TOKENS`,
  `GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S`, `GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL`,
  `GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE`.
- skillsbench: `GBENCH_SKILLSBENCH_IMAGE`, `GBENCH_SKILLSBENCH_AGENT`, `GBENCH_SKILLSBENCH_TASKS`,
  `GBENCH_SKILLSBENCH_CONDITIONS`, `GBENCH_SKILLSBENCH_TIMEOUT_S`, `GBENCH_SKILLSBENCH_API_KEY`,
  `GBENCH_SKILLSBENCH_TASK_HOST`, `GBENCH_SKILLSBENCH_TASK_ENDPOINT`.
- agent_dojo: `GBENCH_AGENTDOJO_ATTACK`, `GBENCH_AGENTDOJO_VERSION`, `GBENCH_AGENTDOJO_MODEL`,
  `GBENCH_AGENTDOJO_MODEL_LABEL`.
- tau2 and tau3 (shared, canonical prefix `GBENCH_TAU2_`, legacy bare `TAU2_`):
  `GBENCH_TAU2_NUM_TRIALS`, `GBENCH_TAU2_MAX_STEPS`, `GBENCH_TAU2_MAX_ERRORS`, `GBENCH_TAU2_SEED`,
  `GBENCH_TAU2_TOP_P`, `GBENCH_TAU2_TOP_K`, `GBENCH_TAU2_USER_TEMPERATURE`, `GBENCH_TAU2_USER_LLM`,
  `GBENCH_TAU2_EVAL_LLM`, `GBENCH_TAU2_BENCH_SRC`, `GBENCH_TAU2_ENV_RUN`, `GBENCH_TAU2_SAVE_TRACES`,
  `GBENCH_TAU2_PROGRESS_SECS`, `GBENCH_TAU2_VERBOSE`, `GBENCH_TAU2_LLM_BACKOFF`,
  `GBENCH_TAU2_RETRIEVAL_CONFIG`, `GBENCH_TAU2_EMBED_MODEL`, `GBENCH_TAU2_EMBED_BASE_URL`,
  `GBENCH_TAU2_EMBED_API_KEY`, `GBENCH_TAU2_EMBED_BATCH`, `GBENCH_TAU_SANDBOX_MAX_AGE_H`. The
  assistant temperature follows the standard precedence via `GBENCH_TAU2_TEMPERATURE` /
  `GBENCH_TAU3_TEMPERATURE`.
- wildclawbench: `GBENCH_WILDCLAWBENCH_IMAGE`, `GBENCH_WILDCLAWBENCH_HOST_DIR`,
  `GBENCH_WILDCLAWBENCH_TASK_IMAGE`, `GBENCH_WILDCLAWBENCH_CATEGORIES`, `GBENCH_WILDCLAWBENCH_TASK`,
  `GBENCH_WILDCLAWBENCH_THINKING`, `GBENCH_WILDCLAWBENCH_PARALLEL`,
  `GBENCH_WILDCLAWBENCH_TASK_NETWORK`, `GBENCH_WILDCLAWBENCH_TASK_HOST`,
  `GBENCH_WILDCLAWBENCH_TASK_ENDPOINT`, `GBENCH_WILDCLAWBENCH_JUDGE_PORT`,
  `GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT`, `GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S` (legacy alias
  `WILDCLAW_JUDGE_REQUEST_TIMEOUT_S`), `GBENCH_WILDCLAWBENCH_MODEL_API_KEY`,
  `GBENCH_WILDCLAWBENCH_TIMEOUT_S`, `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION`.

### Pillar 6: multimodal vision and grounding
- screenspot: `GBENCH_SCREENSPOT_DATA_DIR` (legacy alias `SCREENSPOT_PRO_DATA_DIR`; an optional local
  ScreenSpot-Pro data dir. When absent, the canonical ScreenSpot-v2 dataset is scored and the result
  records which dataset was used).
- omnidocbench: `GBENCH_OMNIDOCBENCH_IMAGE`.
- ui_control_osworld: `GBENCH_UI_CONTROL_OSWORLD_IMAGE`, `GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE`,
  `GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S`, `GBENCH_OSWORLD_VM_DIR`, `GBENCH_OSWORLD_DOMAIN`,
  `GBENCH_OSWORLD_LIMIT`, `GBENCH_OSWORLD_MAX_STEPS`, `GBENCH_OSWORLD_OBS_TYPE`,
  `GBENCH_OSWORLD_TEST_META`, `GBENCH_OSWORLD_MODEL_ENDPOINT`, `GBENCH_OSWORLD_MODEL_API_KEY`.

### Cross pillar
- gaia2: `GBENCH_GAIA2_IMAGE`, `GBENCH_GAIA2_MODE`, `GBENCH_GAIA2_CONFIG`, `GBENCH_GAIA2_LIMIT`,
  `GBENCH_GAIA2_NUM_RUNS`, `GBENCH_GAIA2_MAX_CONCURRENT`, `GBENCH_GAIA2_SCENARIO_TIMEOUT`,
  `GBENCH_GAIA2_TIMEOUT_S`, `GBENCH_GAIA2_JUDGE_MODEL`, `GBENCH_GAIA2_JUDGE_PORT`,
  `GBENCH_GAIA2_JUDGE_ENDPOINT`, `GBENCH_GAIA2_MODEL_ENDPOINT`, `GBENCH_GAIA2_MODEL_API_KEY`,
  `GBENCH_GAIA2_SKIP_KEY_VALIDATION`.
- gdpval: `GBENCH_GDPVAL_MAX_FILE_CHARS`, `GBENCH_GDPVAL_MAX_PDF_PAGES`, `GBENCH_GDPVAL_MAX_JUDGE_CHARS`, `GBENCH_GDPVAL_MAX_RUBRIC_CHARS`.

Suites not listed here read no suite specific environment knob beyond the cross cutting ones (they are
configured entirely by the command line flags and their pinned dataset).
