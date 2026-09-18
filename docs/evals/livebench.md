# livebench

Canonical LiveBench (White et al.; [livebench.ai](https://livebench.ai);
[github.com/LiveBench/LiveBench](https://github.com/LiveBench/LiveBench)), a monthly-refreshed,
contamination-limited benchmark. gbench runs its **6 core categories** (`coding`,
`data_analysis`, `instruction_following`, `language`, `math`, `reasoning`), each scored by
**LiveBench's own ground-truth judges** (no LLM judge), with **real code execution** for the
coding category.

The 7th category, `agentic_coding`, is **excluded** (LiveBench's default benchmark set already
excludes it; it needs a separate ~150GB Multi-SWE-Bench container harness).

## Why it's containerized

LiveBench's package (`latex2sympy2`, `spacy`, `litellm`, …) and its coding-execution
environment (`code_runner/requirements_eval.txt`: `tensorflow`, `numba`, `opencv`,
`scikit-image`, …) would churn a tightly-pinned torch/vLLM serving venv. So gbench does **not**
install LiveBench into your serving env; it runs LiveBench's own pipeline inside an isolated
`gbench-livebench` Docker image, and only reads back the scores. gbench hard-errors (never
silently skips) if Docker or the image is missing.

## Setup (step by step)

### 1. Get the LiveBench checkout (needs git-lfs)

LiveBench LFS-tracks its `*.json` data, so cloning needs **git-lfs**:

```bash
# git-lfs - apt (system) OR pip:
sudo apt-get install -y git-lfs        # Debian/Ubuntu
#   or:  pip install git-lfs
git lfs install

git clone https://github.com/LiveBench/LiveBench
cd LiveBench && git lfs pull           # pull the LFS data
```

### 2. Build the `gbench-livebench` image

The build context **must be your LiveBench checkout** (so its LFS-pulled data is included):

```bash
docker build -t gbench-livebench \
    -f docker/livebench.Dockerfile /path/to/LiveBench
```

This installs LiveBench + its coding-execution requirements (heavy, `tensorflow` etc.) and
downloads the NLTK data. It's a large image; the first build is slow. If a package in
`requirements_eval.txt` needs an extra system lib, add it to the `apt-get` line in the
Dockerfile and rebuild. Override the image name with `GBENCH_LIVEBENCH_IMAGE` if you tag it
differently.

### 3. Run

```bash
# All 6 core categories (pin a release where coding is still active):
export GBENCH_LIVEBENCH_RELEASE=2025-04-02
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals livebench \
       --suite-timeout 36000                 # LiveBench is long; raise the timeout
```
(Omit `GBENCH_LIVEBENCH_RELEASE` to score the latest release's active categories. `coding`
will be empty; see "Releases" above.)

Internally the suite runs LiveBench with `--mode single` (no tmux) and scores with
`--ignore-missing-judgments`; you don't set those. Quick smoke test:
`GBENCH_LIVEBENCH_CATEGORIES="reasoning math" GBENCH_LIVEBENCH_EXTRA_ARGS="--question-end 2"`.

What happens: gbench `docker run`s the image (`--network host`, so the container reaches your
`127.0.0.1` endpoint) and executes LiveBench's own `run_livebench.py`: it downloads the
questions, generates answers against your endpoint (via `--api-base` + `--use-litellm`), grades
them with LiveBench's per-category judges, then `show_livebench_result.py` writes
`all_groups.csv`. gbench reads that and reports **`accuracy` = the `average` over the 6
categories**, with each category in `dimension_scores`.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_LIVEBENCH_IMAGE` | image tag (default `gbench-livebench`) |
| `GBENCH_LIVEBENCH_MODEL` | the model id LiveBench/LiteLLM uses (default = the gbench model name; a plain vLLM OpenAI endpoint works as-is (validated) but set e.g. `openai/…` if your LiteLLM setup needs a provider prefix) |
| `GBENCH_LIVEBENCH_RELEASE` | **pin a LiveBench release date** (default = latest). See "Releases" below; this matters. |
| `GBENCH_LIVEBENCH_CATEGORIES` | restrict to a subset of the 6 (space/comma list; default all 6). Handy for validation. |
| `GBENCH_LIVEBENCH_EXTRA_ARGS` | extra flags appended to `run_livebench.py` (e.g. `--question-end 5`, `--max-tokens 4096`) |

## Releases & the `coding` category (important)

LiveBench is **monthly-refreshed**: categories/questions are added and **deprecated** over
time, and each run is defined against one **release date**. Critically, the non-agentic
**`coding` category was deprecated after `2025-04-02`** (superseded by `agentic_coding`, which
this suite excludes). So:

- **Default (latest release):** the active core categories are whatever the latest release
  still has; `coding` will be **empty** and reported under `categories_missing`, so you get a
  ~5-category average and `status: completed_with_errors`.
- **For a complete, reproducible 6-category number:** pin a release where all 6 are active,
  e.g. `GBENCH_LIVEBENCH_RELEASE=2025-04-02`. `leaderboard_comparable` is set when it's the
  **full** question set (no `--eval-limit`) for a **pinned** release with all default
  categories active. It is **not** gated on greedy/`--thinking`; LiveBench's leaderboard
  includes reasoning models, so a `--thinking` run is still a faithful measurement.

`--eval-limit N` is honored (mapped to LiveBench's `--question-end N`, i.e. the first N
questions per task) for quick subset runs; such a run is a subset, so it isn't
`leaderboard_comparable`.

The result records `livebench_release` and `categories_missing` so the coverage is explicit.
If **no** category is active for the chosen release, the suite hard-errors (nothing to report).

## Requirements

- **Docker** + the **`gbench-livebench` image** (built above). Hard-errors with build
  instructions if either is missing.
- **git-lfs** (apt `git-lfs` or pip `git-lfs`) to check out LiveBench's LFS `*.json` data.
- A reachable OpenAI-compatible **endpoint** for the model under test (gbench's normal target;
  reached from the container via `--network host`).
- No API key, no LLM judge (LiveBench scores against ground truth).

## Notes / caveats

- **Runtime:** the full 6-category run generates + grades thousands of questions; it is slow.
  Raise `--suite-timeout`.
- **Endpoint routing:** LiveBench reaches the model through LiteLLM; if your served model needs
  a provider prefix, set `GBENCH_LIVEBENCH_MODEL` accordingly.
- `leaderboard_comparable` is set when all 6 default categories score on a pinned release over
  the full question set (no `--eval-limit`). It is **not** gated on greedy decoding: LiveBench's
  leaderboard includes reasoning/thinking models, so a `--thinking` run is still a faithful
  measurement (see the runner's `leaderboard_comparable` gate: `release and default_cats and not
  missing and not limit`).
- **Sampling:** `GBENCH_LIVEBENCH_TEMPERATURE` overrides the temperature for this suite; else the
  run default (0.0 greedy / 1.0 with `--thinking`), passed to LiveBench via `--force-temperature`.
