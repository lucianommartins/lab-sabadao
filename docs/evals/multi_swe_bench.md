# multi_swe_bench setup

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.


Canonical Multi-SWE-bench (`ByteDance-Seed/Multi-SWE-bench`, 7 languages) scored by
execution-based **resolved rate**: the model's unified diff is applied and the
task's tests are run inside its Docker image via the project's **own harness**
(`multi_swe_bench.harness.run_evaluation`). Vanilla `swebench` cannot score it: the
prediction schema (`{org, repo, number, fix_patch}`) and per-language test fields
differ.

The dataset is loaded via `list_repo_files` + per-file `hf_hub_download` (the repo's
`load_dataset` path is broken); the exact rows that were run are re-materialized into
a subset `dataset.jsonl` the harness consumes, so gold/tests always match the
predictions.

## Requirements
- **Docker** (the harness clones each repo and builds/pulls per-instance images).
- **`multi-swe-bench`**: `pip install multi-swe-bench` (also available in
  `gbench[evals]`).

Hard-errors (`infra_required`, never skips, never a fabricated number) and names the
missing piece if Docker is unreachable or the package is absent.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multi_swe_bench \
       --sandboxes 4 --eval-limit 5
```
gbench writes `preds.jsonl` + `config.json`, runs
`python -m multi_swe_bench.harness.run_evaluation --config …`, and reads
`resolved_ids` from `final_report.json` (matched by `instance_id`, with a
repo+number fallback). `--sandboxes` maps to the harness `max_workers`;
`category_accuracy` reports per language. The result carries a
`multi_swe_bench_report` (total / resolved / error / empty-patch counts). The model
must return a single unified git diff.
