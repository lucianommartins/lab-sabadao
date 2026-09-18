# codeforces

Competitive programming from [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces),
following the protocol that dataset's card prescribes, the only reproducible open
Codeforces protocol there is.

> **This is not a Codeforces Elo.** Frontier labs report a Codeforces **Elo rating or
> percentile**, obtained by simulating real contests on the live platform. That is not
> reproducible offline. This is a pass-rate over a held-out split, and
> `leaderboard_comparable` is always `false`.

## Protocol

| | |
| --- | --- |
| config | `verifiable`: `executable` **and** (`official_tests_complete` **or** has `generated_tests`) |
| split | `test`: "problems from late 2024 and early 2025" (the card asks you not to train on these) |
| **N** | **422** |
| prompt | the dataset's own `verifiable-prompts`, Python variant |
| tests | `official_tests` + the separately-downloaded `generated_tests/` shards |
| checker | `generated_checker`, where present |
| limits | per-problem `time_limit` × `GBENCH_CODEFORCES_TIME_MULTIPLIER` (default 3, since CPython is far slower than the C++ the limits assume) |

Accepted iff **every** test case passes. A problem with no usable tests is reported
incorrect, never auto-passed.

## Generated test cases: required for a rigorous number

`official_tests` alone covers a **median of 9.7%** of each problem's real test suite
(mean 21.5%; `official_tests_complete` is true for only 21 of 422). The platform truncates
visible tests to ~400 characters, so those cases "can be solved with an easy brute force
solution". The full generated set lives in the same repo but is **not** part of the dataset
load (~110 GB):

```bash
hf download open-r1/codeforces --repo-type dataset \
   --include 'generated_tests/*.parquet' --max-workers 8 \
   --local-dir $GBENCH_PREREQS_DIR/codeforces
```

`huggingface-cli` is deprecated; the current CLI is `hf`. Then point the suite at it, in
fish:

```fish
set -gx GBENCH_CODEFORCES_GENERATED_TESTS $GBENCH_PREREQS_DIR/codeforces
```

or bash:

```bash
export GBENCH_CODEFORCES_GENERATED_TESTS=$GBENCH_PREREQS_DIR/codeforces
```

The suite reads `<DIR>/generated_tests/test_cases_<contest>.parquet`, and also accepts the
shards sitting directly in `<DIR>`. Only the shards for contests in the split are read, so
a **partial mirror is fine**: you do not need all 1,705 files to run a subset, but it must
cover the contests in your split.

The mirror is **required**: `official_tests` alone (~9.7% coverage) is a near-meaningless
partial, so the suite **hard-errors** (`infra_required`) if `GBENCH_CODEFORCES_GENERATED_TESTS`
is unset or the directory is missing. It never silently runs a reduced-coverage number.

The canonical prompts (`verifiable-prompts`) are likewise **required** and load from the same
dataset repo (needs network on first load). A load failure hard-errors rather than silently
substituting a built-in prompt (prompt wording changes the measured quantity).

## Checkers are not optional

**~30% of Codeforces problems accept multiple valid outputs** and need a checker program to
grade. Exact stdout match marks correct solutions wrong. Where `generated_checker` is present,
gbench runs it exactly as open-r1's protocol prescribes (verified against real checker sources):

- argv is `checker.py <input> <reference/jury answer> <submission/model output>`;
- the checker **prints its verdict to STDOUT** (`0` = wrong, its max (`1` or `100`) = correct)
  and **always exits 0**. gbench reads the score from stdout, **not** the return code. (Reading
  the return code was a bug that auto-passed every checker-graded problem regardless of output.)
- "full credit" is **calibrated** by also scoring the reference answer as if it were the
  submission (it is correct by construction), so 1-scale and 100-scale checkers both work with
  no hardcoded maximum.

Where no checker is present, comparison is token-wise (the platform's `wcmp`), because the
dataset carries `\r\n` line endings verbatim. Each checker runs in the bubblewrap jail.

> **Memory limit:** each problem's `memory_limit` is recorded but **not enforced** (reliable
> RLIMIT_AS enforcement on CPython causes false MLEs from interpreter overhead). Like the
> CPython time multiplier, this is a documented deviation from platform judging.

## What changed on 2026-08-20, and why

The suite previously loaded `open-r1/codeforces-cots` **train** and graded against the
statement `examples`. Four separate problems, all measured:

- **`-cots` is a chain-of-thought distillation set with 5 generations per problem.** The
  "500 samples" were **100 distinct problems counted five times each**.
- **A hardcoded 40/30/30 CF/ICPC/IOI quota**, invented here and called "balanced". IOI is
  85 of 10,024 problems (**0.85%**), so the quota over-represented it ~35×. 46% of IOI
  problems ship no `input_format` at all (they are grader-based and cannot run as
  stdin/stdout), which is what produced IOI 17.33% against CF 68% / ICPC 76%.
- **Grading on statement examples** (mean 1.95 cases; 205 of 500 rows had exactly one).
- **Exact stdout match**, which fails the ~30% of problems with multiple valid answers.

The split is also temporal (2024-2025) where `-cots` starts at 2010, so the new number has
far better contamination properties.

## Environment

| var | default | effect |
| --- | --- | --- |
| `GBENCH_CODEFORCES_GENERATED_TESTS` | *(unset)* | directory holding the `generated_tests/` mirror |
| `GBENCH_CODEFORCES_MAX_GENERATED_TESTS` | `0` | cap on generated cases per problem (`0` = no cap, run all, the canonical default); they are ordered hardest-first |
| `GBENCH_CODEFORCES_TIME_MULTIPLIER` | `3.0` | multiple of the declared `time_limit` |
| `GBENCH_CODEFORCES_CONFIG` / `_SPLIT` | `verifiable` / `test` | override the subset |
| `GBENCH_CODEFORCES_TEMPERATURE` | *(run default)* | per-suite sampling override; else 0.0 greedy (no-think) / 1.0 (`--thinking`) |
| `GBENCH_SANDBOX` | `required` | must isolate: with `required` (default) the suite **hard-errors** if bubblewrap is unavailable/blocked (it runs model-written code); set `none` to run UNSANDBOXED on purpose |

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals codeforces --sandboxes 8
```
Generated code runs in a bubblewrap jail (read-only root, no network, strict timeout);
`--sandboxes` bounds concurrency.
