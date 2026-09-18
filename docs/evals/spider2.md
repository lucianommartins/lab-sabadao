# spider2 setup

Canonical Spider 2.0-lite (`xlang-ai/Spider2`, `spider2-lite`) enterprise text-to-SQL scored by
**execution accuracy**: the model's SQL is run against the target SQLite database and its result set
is compared to the gold with the official `evaluate.py` semantics, column-subset containment (every
required gold column must be matched by *some* predicted column, per `condition_cols`), numeric
tolerance (`abs_tol=1e-2`), NA-normalized (`None`/`NaN`/`NaT` → 0), and order-insensitive row
comparison; multi-gold instances pass if **any** accepted gold variant (`{id}.csv`, `{id}_a.csv`, …)
matches.

The prompt is the canonical baseline input: the **full database schema** (every table's
`CREATE TABLE` DDL, introspected from the `.sqlite`) + the instance's **external-knowledge document**
when it has one + the question. (A bare question with no schema is not the Spider2-lite task; the
model cannot know the table/column names.)

Only the **`local*` (SQLite) subset, 135 instances, runs offline.** The 180 BigQuery + 207
Snowflake + 25 BigQuery-GA instances of the 547-instance set need live cloud accounts/credentials
and are **excluded** (not scored 0). Because it is a subset, `leaderboard_comparable` is always
`false`. The reported `total_questions` is the scorable local subset; check the run log line
`Loaded N spider2 local(SQLite) samples`.

## Requirements

- `pandas` + `sqlite3`: `pandas` is a base gbench dependency; `sqlite3` ships with Python. No cloud
  SDKs are needed for the local subset. No network at run time (all data is local, see below).
- **The Spider2 repo checkout**: set **`SPIDER2_REPO_DIR`** to the `spider2-lite/` directory of an
  `xlang-ai/Spider2` clone. Question/db/external-knowledge **metadata is read from the repo**
  (`spider2-lite.jsonl`), NOT from the HF `xlangai/spider2-lite` snapshot, which is stale and drifts
  from the current gold (different questions/DB casing for some ids). External-knowledge documents
  are read from `<repo>/resource/documents/*.md`.
  ```bash
  git clone https://github.com/xlang-ai/Spider2
  export SPIDER2_REPO_DIR=$PWD/Spider2/spider2-lite
  ```
  If `SPIDER2_REPO_DIR` is unset it is derived from `SPIDER2_GOLD_DIR` (two levels up), so a standard
  checkout only needs `SPIDER2_GOLD_DIR` set.
- **Gold**: set `SPIDER2_GOLD_DIR` to `<repo>/spider2-lite/evaluation_suite/gold` (must contain
  `spider2lite_eval.jsonl` and `exec_result/`).
- **Local DBs**: set `SPIDER2_LOCALDB_DIR` to the directory of `*.sqlite` files. These are **not in
  the git repo**; download the `spider2-localdb` archive linked from
  `spider2-lite/evaluation_suite/README.md` (a Google Drive archive) and unzip it. Each instance's
  `db` maps 1:1 to `{db}.sqlite`. Concretely (Google Drive file id from the upstream README):
  ```bash
  pip install -U gdown
  gdown 1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG -O /tmp/spider2-localdb.zip
  mkdir -p "$SPIDER2_REPO_DIR/resource/databases/spider2-localdb"
  unzip -q /tmp/spider2-localdb.zip -d "$SPIDER2_REPO_DIR/resource/databases/spider2-localdb"
  # point at whichever dir directly contains the *.sqlite files (flatten one level if the zip nested them)
  export SPIDER2_LOCALDB_DIR="$SPIDER2_REPO_DIR/resource/databases/spider2-localdb"
  ```

The suite **hard-errors** (`infra_required`, never skips, never a fabricated 0%) if pandas, the repo
metadata, the gold dir, or the local DB dir is missing, **and** if the checkout is incomplete (any
referenced `{db}.sqlite` or external-knowledge `.md` is absent, it lists them rather than silently
scoring those instances wrong). Each SQLite DB is copied into an in-memory connection before the
query runs, so the on-disk files are never mutated.

## Run
```bash
export SPIDER2_REPO_DIR=/data/Spider2/spider2-lite
export SPIDER2_GOLD_DIR=$SPIDER2_REPO_DIR/evaluation_suite/gold
export SPIDER2_LOCALDB_DIR=/data/spider2/localdb
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals spider2 \
       --sandboxes 8 --eval-limit 20
```
The model must return SQL in a `sql` fenced block.

## Notes / caveats

- **Sampling:** `GBENCH_SPIDER2_TEMPERATURE` overrides the temperature for this suite (else the run
  default: 0.0 greedy / 1.0 with `--thinking`); it takes precedence over `--temperature`.
- **Concurrency:** `--sandboxes` bounds concurrent **model generation** (HTTP requests). SQL grading
  runs concurrently over all traces (each query capped at **120 s**); a query that times out counts
  as a wrong answer, while a grading-harness crash is recorded as `SCORING_ERROR`. It still counts
  as incorrect in accuracy (as swebench does for per-instance harness errors) but its count is
  surfaced in `scoring_errors` so it is not mistaken for a genuine model miss.
- **`leaderboard_comparable` is always `false`**: the SQLite-only subset is not the full published
  Spider2-lite set (which includes BigQuery + Snowflake).
