# swe_bench_multilingual setup

Canonical SWE-bench Multilingual (`SWE-bench/SWE-bench_Multilingual`, 300 issues
across 9 non-Python languages) scored by execution-based **resolved rate** via the
**vanilla** swebench Docker harness (namespace `swebench`). The model emits one
unified-diff patch per issue; the harness applies it and runs
`FAIL_TO_PASS`/`PASS_TO_PASS` inside the prebuilt per-instance image.

## Requirements
- **Docker** (per-instance images pulled from DockerHub `swebench/...`).
- `swebench`: in the `gbench[evals]` extra, **pinned `>=4.1,<5`**. This dataset needs the **vanilla
  upstream** harness; it was validated with **swebench 4.1.0** (the 4.0.3 SWE-bench-Live *fork* fails
  this dataset). The pin excludes v5.0.0+, which removed `swebench.harness.test_spec.test_spec` and
  changed the TestSpec schema (`make_test_spec` raises `KeyError: 'image'`), so v5 cannot build this
  suite's specs. If you already have v5 installed, downgrade: `pip install 'swebench>=4.1,<5'`.
  `datasets` is a base install.
- Host Node/toolchains are **not** needed; all language tests run inside the images.

If `swebench`/`datasets`/Docker (or the dataset TestSpec) are missing the suite **hard-errors**
(`infra_required` → `status:"error"`; it never skips and never reports a fabricated 0%). A run
where the harness could evaluate only some instances is recorded with `partial_evaluation` and
`leaderboard_comparable: false`.

## Sampling & comparability

`GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE` overrides the temperature for this suite (else the run
default: 0.0 greedy / 1.0 with `--thinking`); it takes precedence over `--temperature`. The judge
does not apply here (execution-based resolved rate). `leaderboard_comparable` is **always `false`**:
gbench generates a single-turn patch with no repository access, which the shared SWE-bench finalizer
always records as a canonical deviation; `--eval-limit`, `--thinking`, or a partial harness run add
further reasons. The reason string is always recorded alongside the flag.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_bench_multilingual \
       --sandboxes 8 --eval-limit 20
```
First run pulls sizable images; use `--eval-limit` and `--sandboxes` to bound it
(`--sandboxes` → harness `--max_workers`). Patches truncate if `--max-output-tokens`
is small; keep it ≥ 8192.
