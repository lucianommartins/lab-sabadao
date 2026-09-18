# aider_polyglot

Canonical Aider Polyglot ([Aider-AI/aider](https://github.com/Aider-AI/aider) `benchmark/`):
225 Exercism exercises across 6 languages (C++, Go, Java, JavaScript, Python, Rust). The model
edits the stub in its **native edit format** (diff / SEARCH-REPLACE for capable models); aider
applies the edit, runs the exercise's hidden unit tests, and on failure feeds the errors back for
a second try. **Headline: pass@2.**

## Why it's containerized

Installing aider into the serving env would **break the pinned graph** (it downgrades `openai`
and moves `numpy`/`huggingface_hub`/`pydantic`/`pillow`), and the exercises need 6 language
toolchains. So gbench does **not** install aider into the gbench serving environment; it delegates to aider's **own
benchmark harness inside aider's own benchmark image** (`aider-benchmark`, built from the
checkout's `benchmark/Dockerfile`, which bundles python3.11 + openjdk-21 + go + rust + node +
gcc), and only reads back the per-exercise `.aider.results.json`. gbench hard-errors (never
silently skips) if Docker, the image, or the exercises are missing.

## Setup (step by step)

### 1. Get the aider checkout (already at `gbench-prereqs/aider`) and build its image

```bash
# context MUST be the aider checkout. Use gbench's Dockerfile (= aider's own benchmark/Dockerfile,
# with all 6 toolchains + openjdk-21 + aider, PLUS a setuptools-scm version pin so the editable
# install of the version-metadata-less checkout succeeds).
docker build -t aider-benchmark \
    -f docker/aider_polyglot.Dockerfile \
    $GBENCH_PREREQS_DIR/aider
```
This is a large image (all toolchains); the first build is slow. Override the tag with
`GBENCH_AIDER_IMAGE` if you build it under a different name.

> Why gbench's Dockerfile and not aider's directly: the `gbench-prereqs/aider` checkout has no
> git version metadata, so aider's own `benchmark/Dockerfile` fails at
> `uv pip install -e /aider[dev]` with *"setuptools-scm was unable to detect version"*. gbench's
> copy adds `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT` to fix exactly that; everything else
> (openjdk-21, go, rust, node, gcc) is identical to aider's.

### 2. Get the polyglot exercises

`benchmark.py` expects them at `$AIDER_BENCHMARK_DIR/polyglot-benchmark`:

```bash
mkdir -p $GBENCH_PREREQS_DIR/aider-bench
git clone https://github.com/Aider-AI/polyglot-benchmark \
    $GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark
export GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench
```
This directory is mounted into the container as `/benchmarks` and also receives the run's
timestamped results directory.

### 3. Run

```bash
export GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals aider_polyglot \
       --suite-timeout 36000          # the full 225-exercise pass@2 run is long
```

Internally gbench `docker run`s the image (`--network host`, so the container reaches your
`127.0.0.1` endpoint), points aider at it via litellm (`OPENAI_API_BASE` + `--model openai/<id>`),
and runs `benchmark.py --tries 2 --exercises-dir polyglot-benchmark --new` across the 6
languages. It reads the resulting `.aider.results.json` files and reports **`accuracy` = pass@2**
(with `pass_rate_1` and per-language `category_accuracy`).

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_AIDER_IMAGE` | image tag (default `aider-benchmark`) |
| `GBENCH_AIDER_BENCHMARK_DIR` | host dir with `polyglot-benchmark/` (mounted to `/benchmarks`), **required** |
| `GBENCH_AIDER_MODEL` | the model id litellm uses (default `openai/<gbench model name>`; litellm routes `openai/*` to `OPENAI_API_BASE`) |
| `GBENCH_AIDER_EDIT_FORMAT` | force an edit format (default: aider's native choice for the model) |
| `GBENCH_AIDER_LANGUAGES` | restrict to a subset of the 6 (comma/space list) |
| `GBENCH_AIDER_EXTRA_ARGS` | extra flags appended to `benchmark.py` |

## Notes / caveats

- **`leaderboard_comparable`** is set only for the **full** 225-exercise set (no `--eval-limit`)
  across all 6 languages with aider's **native** edit format. `--eval-limit N` maps to
  `--num-tests N` (a subset → not comparable).
- **Runtime:** the full pass@2 run generates + compiles + tests thousands of files. It is slow.
  Raise `--suite-timeout`.
- **Sampling:** aider's benchmark controls the model's temperature itself (0 for reproducibility);
  gbench does not inject one. There is no LLM judge; grading is the exercises' unit tests.
