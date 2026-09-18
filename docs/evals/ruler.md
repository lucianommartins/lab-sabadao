# ruler

Canonical RULER (Hsieh et al., NVIDIA, COLM 2024; arXiv:2404.06654;
[NVIDIA/RULER](https://github.com/NVIDIA/RULER), Apache-2.0). A tokenizer-defined
long-context benchmark: **13 tasks × 6 context bands** (4k/8k/16k/32k/64k/128k), 500
samples each, with haystacks padded to exact token counts under the **target model's own
tokenizer**.

## How it runs (per-tokenizer data generation, cached)

RULER is a data-*generation* benchmark, not a fixed dataset. This suite drives NVIDIA's own
`scripts/data/prepare.py` (from your checkout) to generate the per-`--tokenizer` data on the
first run, **caches it to `~/.cache/gbench/ruler/<tokenizer>/<band>/`**, and reuses it on
every later run. Pay the generation cost once (the 64k/128k bands are slow); reuse forever
after (until you change the tokenizer, bands, or sample count).

- Tokenizer = your `--tokenizer` value (falls back to the served model; override with
  `GBENCH_RULER_TOKENIZER`).
- Bands via `GBENCH_RULER_LENGTHS` (default `4096 8192 16384 32768 65536 131072`);
  samples/cell via `GBENCH_RULER_NUM_SAMPLES` (default 500).

## Scoring

Canonical string match (from RULER `scripts/eval/synthetic/constants.py`), after
`postprocess_pred`: **string_match_all** (niah / vt / cwe / fwe, fraction of references
present) and **string_match_part** (qa_1 / qa_2, any reference present). Greedy, with small
per-task generation caps (niah 128 / vt 30 / cwe 120 / fwe 50 / qa 32); `--thinking` is
non-canonical here (the caps truncate reasoning).

Headline `accuracy` = **RULER Avg** = mean over bands of (mean of the 13 task scores).
Also reported: `ruler_length_scores`, `ruler_task_scores`, `ruler_cell_scores`,
`effective_length` (largest band scoring ≥ 85.6), `wavg_inc`/`wavg_dec`, `length_bands`.

## Requirements

- **A serving endpoint whose context window covers the largest band** (≥ ~140k for the 128k
  band). The suite reads the server's `max_model_len` (`GET /v1/models`) and **hard-errors**
  (does not silently skip) if it's too small: serve with e.g. vLLM `--max-model-len 140000`,
  or restrict `GBENCH_RULER_LENGTHS`.
- **The NVIDIA/RULER checkout**, at `GBENCH_RULER_DIR`. Because RULER LFS-tracks its `*.json`
  data, cloning it needs **git-lfs**, and its corpora are fetched by its own scripts:
  ```bash
  # git-lfs - apt (system) OR pip:
  sudo apt-get install -y git-lfs        # Debian/Ubuntu
  #   or:  pip install git-lfs
  git lfs install

  git clone https://github.com/NVIDIA/RULER
  cd RULER && git lfs pull               # fetch english_words.json (and other *.json blobs)
  # corpora for niah + qa tasks:
  python scripts/data/synthetic/json/download_paulgraham_essay.py
  bash   scripts/data/synthetic/json/download_qa_dataset.sh
  export GBENCH_RULER_DIR=$PWD
  ```
- pip: `wonderwords`, `nltk`, `html2text`, `tenacity`. These are **RULER's own data-generation
  deps** (from the RULER checkout's `requirements.txt`), NOT part of `gbench[evals]`; install them
  into the same environment gbench runs in:
  ```bash
  pip install wonderwords nltk html2text tenacity
  ```
  Plus **git-lfs** (apt `git-lfs` or pip `git-lfs`) to check out the LFS data.
- No LLM judge, no API key.
- **Sampling:** `GBENCH_RULER_TEMPERATURE` overrides the temperature for this suite; else the run
  default (0.0 greedy / 1.0 with `--thinking`). `leaderboard_comparable` requires greedy, the full
  13 tasks × exact 6 canonical bands (4k-128k), and `GBENCH_RULER_NUM_SAMPLES=500` (the default).

## Run
```bash
export GBENCH_RULER_DIR=/path/to/RULER
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals ruler --eval-limit 20
```
`--eval-limit N` stratifies N samples across the `(task@band)` cells (still generates the
full per-band data first, so the first run is not faster; it just scores a subset).
