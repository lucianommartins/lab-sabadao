# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: spider2
# Description: Spider 2.0-lite - enterprise text-to-SQL, execution accuracy (SQLite subset)

"""gbench native built-in runner for spider2 (Data & SQL Engineering).

Canonical Spider 2.0-lite (xlang-ai/Spider2) execution accuracy: run the model's SQL
against the target SQLite database and compare the result set to the gold (column-subset
containment, numeric tolerance, NA-normalized, order-insensitive, multi-gold), per the
official evaluation_suite/evaluate.py. The prompt carries the full DB schema + any
external-knowledge document (the canonical baseline input). Only the 135 local* (SQLite)
instances of the 547-instance set run offline; the BigQuery + Snowflake instances need cloud
accounts and are excluded, so leaderboard_comparable is always False. Metadata is read from the
repo checkout (GBENCH_SPIDER2_REPO_DIR), not the stale HF snapshot; gold + local DBs are not on HF -
point GBENCH_SPIDER2_GOLD_DIR / GBENCH_SPIDER2_LOCALDB_DIR at them (see docs; the bare SPIDER2_*
names still work as deprecated aliases). Hard-errors (infra_required,
never skips) if any prerequisite - or the referenced dbs/knowledge docs - are absent.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SPIDER2_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import glob
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, suite_env
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Data & SQL Engineering"
DOCS_URL = "docs/evals/spider2.md"
_ABS_TOL = 1e-2


def extract_sql_query(text: str) -> str:
    """Extract SQL from a ```sql fenced block, else the raw text (canonical)."""
    m = re.search(r"```sql\n(.*?)\n```", text or "", re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()


def _norm(v: Any, pd) -> Any:
    """Upstream evaluate.py normalize(): any NA (None / NaN / NaT / pd.NA) -> 0."""
    try:
        if pd.isna(v):
            return 0
    except (TypeError, ValueError):
        pass
    return v


def _vectors_match(gold_vec: List[Any], pred_vec: List[Any], ignore_order: bool = True) -> bool:
    """Faithful port of upstream evaluate.py vectors_match: NATIVE-type numeric compare (not
    "any float-parseable string is numeric") and NATIVE inequality (not str(x)!=str(y)), so e.g.
    gold '007' vs pred '7' is a mismatch exactly as upstream scores it."""
    import pandas as pd
    if len(gold_vec) != len(pred_vec):
        return False
    a = [_norm(x, pd) for x in gold_vec]
    b = [_norm(x, pd) for x in pred_vec]
    if ignore_order:
        a = sorted(a, key=lambda z: str(z))
        b = sorted(b, key=lambda z: str(z))
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not math.isclose(float(x), float(y), abs_tol=_ABS_TOL):
                return False
        elif x != y:
            return False
    return True


def _columns(df) -> List[List[Any]]:
    return [list(df.iloc[:, i]) for i in range(df.shape[1])]


def compare_pandas_table(pred_df, gold_df, condition_cols=None, ignore_order=True) -> int:
    """1 iff every required gold column is matched by SOME predicted column (subset containment)."""
    pred_cols = _columns(pred_df)
    gold_cols = _columns(gold_df)
    idxs = condition_cols if condition_cols else list(range(len(gold_cols)))
    for gi in idxs:
        if gi >= len(gold_cols):
            return 0
        if not any(_vectors_match(gold_cols[gi], pv, ignore_order) for pv in pred_cols):
            return 0
    return 1


def _score_against_golds(pred_df, gold_paths: List[str], condition_cols, ignore_order: bool) -> int:
    """Multi-gold: pass if pred matches ANY accepted gold variant."""
    import pandas as pd
    # condition_cols may be a flat list (single) or list-of-lists (per gold variant)
    nested = bool(condition_cols) and all(isinstance(c, list) for c in condition_cols)
    for i, gp in enumerate(gold_paths):
        try:
            gold_df = pd.read_csv(gp)
        except Exception:
            continue
        cc = condition_cols[i] if nested and i < len(condition_cols) else (None if nested else condition_cols)
        if compare_pandas_table(pred_df, gold_df, cc, ignore_order) == 1:
            return 1
    return 0


def _get_sqlite_result(db_path: str, sql: str):
    """Run SQL against an in-memory copy of the SQLite DB -> DataFrame (canonical)."""
    import sqlite3
    import pandas as pd
    src = sqlite3.connect(db_path)
    mem = sqlite3.connect(":memory:")
    try:
        src.backup(mem)
    finally:
        src.close()
    try:
        return pd.read_sql_query(sql, mem)
    finally:
        mem.close()


def _gold_dir() -> Optional[str]:
    return prereqs_path("Spider2/spider2-lite/evaluation_suite/gold",
                        suite_env("GBENCH_SPIDER2_GOLD_DIR", "SPIDER2_GOLD_DIR"))


def _localdb_dir() -> Optional[str]:
    return prereqs_path("Spider2/spider2-lite/resource/databases/spider2-localdb",
                        suite_env("GBENCH_SPIDER2_LOCALDB_DIR", "SPIDER2_LOCALDB_DIR"))


def _repo_dir() -> Optional[str]:
    """The `spider2-lite/` dir of an xlang-ai/Spider2 checkout.

    Canonical scoring (upstream evaluate.py) sources the question/db metadata from the SAME repo
    checkout as the gold - NOT from HF `xlangai/spider2-lite`, which is a stale snapshot whose
    questions/dbs drift from the current gold. Prefer GBENCH_SPIDER2_REPO_DIR; else derive it from
    GBENCH_SPIDER2_GOLD_DIR (which is <repo>/evaluation_suite/gold, so the repo is two levels up).
    """
    explicit = prereqs_path("Spider2/spider2-lite",
                            suite_env("GBENCH_SPIDER2_REPO_DIR", "SPIDER2_REPO_DIR"))
    if explicit:
        return explicit
    gd = _gold_dir()
    if gd:
        return os.path.dirname(os.path.dirname(os.path.abspath(gd)))  # gold -> evaluation_suite -> spider2-lite
    return None


def _metadata_path() -> Optional[str]:
    repo = _repo_dir()
    return os.path.join(repo, "spider2-lite.jsonl") if repo else None


def _documents_dir() -> Optional[str]:
    repo = _repo_dir()
    return os.path.join(repo, "resource", "documents") if repo else None


#: {db: "CREATE TABLE ..." joined} - introspected once per db from its .sqlite (self-contained;
#: matches the CREATE TABLE DDL the canonical DAIL-SQL baseline feeds, without the repo's
#: schema-dir casing quirks since the .sqlite filename is `{db}.sqlite` per evaluate.py).
_SCHEMA_CACHE: Dict[str, str] = {}


def _schema_for_db(localdb_dir: str, db: str) -> Optional[str]:
    """Full DDL (all CREATE TABLE statements) for a local SQLite db, or None if the file is absent."""
    if db in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db]
    import sqlite3
    db_path = os.path.join(localdb_dir, f"{db}.sqlite")
    if not os.path.isfile(db_path):
        return None
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "ORDER BY name").fetchall()
    finally:
        con.close()
    schema = "\n\n".join(r[0].strip() for r in rows if r and r[0])
    _SCHEMA_CACHE[db] = schema
    return schema


def _resolve_gold_paths(gold_dir: str, iid: str) -> List[str]:
    """Gold CSV(s) for an instance: `{iid}.csv` plus single-letter variants `{iid}_a.csv` ...

    Mirrors upstream resolve_gold_paths' `^{id}(_[a-z])?\\.csv$` - many instances ship several
    accepted gold answers as suffixed files and a prediction passes if it matches ANY of them.
    """
    exec_dir = os.path.join(gold_dir, "exec_result")
    if not os.path.isdir(exec_dir):
        return []
    pat = re.compile(rf"^{re.escape(iid)}(_[a-z])?\.csv$")
    return sorted(os.path.join(exec_dir, n) for n in os.listdir(exec_dir) if pat.match(n))


def check_spider2_prerequisites() -> Tuple[bool, str]:
    """pandas + repo metadata + local gold dir + local SQLite DB dir (SQLite subset runs offline)."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False, "Python package 'pandas' is not installed."
    gd, ld = _gold_dir(), _localdb_dir()
    if not gd or not os.path.isfile(os.path.join(gd, "spider2lite_eval.jsonl")) \
            or not os.path.isdir(os.path.join(gd, "exec_result")):
        return False, ("Spider2 gold not found: set GBENCH_SPIDER2_GOLD_DIR to a checkout of "
                       "xlang-ai/Spider2/spider2-lite/evaluation_suite/gold "
                       "(needs spider2lite_eval.jsonl + exec_result/).")
    meta = _metadata_path()
    if not meta or not os.path.isfile(meta):
        return False, ("Spider2 metadata (spider2-lite.jsonl) not found: set GBENCH_SPIDER2_REPO_DIR to the "
                       "spider2-lite/ dir of an xlang-ai/Spider2 checkout (or leave it unset to "
                       "derive it from GBENCH_SPIDER2_GOLD_DIR). Canonical scoring reads questions/dbs from "
                       "the repo, NOT the stale HF snapshot.")
    if not ld or not glob.glob(os.path.join(ld, "*.sqlite")):
        return False, ("Spider2 local SQLite DBs not found: set GBENCH_SPIDER2_LOCALDB_DIR to the "
                       "unzipped spider2-localdb (*.sqlite) directory.")
    return True, ""


def _load_gold_config(gold_dir: str) -> Dict[str, Dict[str, Any]]:
    cfg = {}
    with open(os.path.join(gold_dir, "spider2lite_eval.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cfg[row["instance_id"]] = row
    return cfg


def _load_spider2_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Spider2-lite local* (SQLite) instances joined with local gold; raises on failure.

    Canonical: metadata (question/db/external_knowledge) comes from the REPO checkout (same source
    as the gold), the prompt carries the full DB schema + any external-knowledge document, and the
    scored set is the 135 local SQLite instances. A local db whose .sqlite (or a referenced
    external-knowledge doc) is missing is an INFRA gap - hard-error listing them, never silently
    score those instances wrong.
    """
    gold_dir = _gold_dir()
    localdb_dir = _localdb_dir()
    docs_dir = _documents_dir()
    meta_path = _metadata_path()
    try:
        rows = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to load spider2 metadata from {meta_path!r}: {e}")
        raise RuntimeError(f"Could not load spider2 metadata ({meta_path!r}): {e}") from e

    gold_cfg = _load_gold_config(gold_dir)
    samples = []
    missing_db: List[str] = []
    missing_doc: List[str] = []
    missing_gold: List[str] = []
    for item in rows:
        iid = item.get("instance_id")
        if not iid or not iid.startswith("local"):
            continue  # offline SQLite subset only
        g = gold_cfg.get(iid)
        if g is None:
            continue  # not in the official gold config -> not an officially-scored instance
        # An officially-scored instance whose exec_result CSV is absent is an INCOMPLETE checkout,
        # not a "skip this one": fold it into the hard-error below instead of silently dropping it.
        gold_paths = _resolve_gold_paths(gold_dir, iid)
        if not gold_paths:
            missing_gold.append(iid)
            continue
        db = item.get("db")
        question = item.get("question")
        if not db or not question:
            raise RuntimeError("spider2: unexpected schema (db/question); refusing to fabricate")

        # Full DB schema (canonical baselines feed every table's DDL). A missing .sqlite is infra.
        schema = _schema_for_db(localdb_dir, db)
        if schema is None:
            missing_db.append(f"{iid}({db}.sqlite)")
            continue

        # external_knowledge is a single .md filename (or null) resolved from resource/documents/.
        ek_name = item.get("external_knowledge")
        ek_text = ""
        if ek_name:
            ek_path = os.path.join(docs_dir, ek_name) if docs_dir else None
            if ek_path and os.path.isfile(ek_path):
                with open(ek_path, encoding="utf-8") as f:
                    ek_text = f.read().strip()
            else:
                missing_doc.append(f"{iid}({ek_name})")
                continue

        parts = ["/* Given the following database schema: */", schema]
        if ek_text:
            parts += ["", "/* External knowledge (read carefully): */", ek_text]
        parts += [
            "",
            f"Database: {db}",
            "",
            "Question:",
            question,
            "",
            "Write a single complete SQLite SQL query that answers the question. "
            "Return only the SQL inside a ```sql ... ``` code block.",
        ]
        samples.append(([{"role": "user", "content": "\n".join(parts)}], iid, {
            "category": "sqlite", "db": db,
            "condition_cols": g.get("condition_cols") or [],
            "ignore_order": bool(g.get("ignore_order", True)),
            "gold_paths": gold_paths,
        }))

    # No-partial: an incomplete checkout (missing local db, external-knowledge doc, or the gold CSV
    # of an officially-scored instance) must hard-error, not silently drop or mis-score instances
    # (the old code let a missing db be graded as a wrong answer and a missing gold CSV be dropped).
    if missing_db or missing_doc or missing_gold:
        bits = []
        if missing_db:
            bits.append(f"{len(missing_db)} local db file(s) absent from GBENCH_SPIDER2_LOCALDB_DIR: "
                        + ", ".join(missing_db[:10]) + (" ..." if len(missing_db) > 10 else ""))
        if missing_doc:
            bits.append(f"{len(missing_doc)} external-knowledge doc(s) absent from "
                        f"{docs_dir!r}: " + ", ".join(missing_doc[:10])
                        + (" ..." if len(missing_doc) > 10 else ""))
        if missing_gold:
            bits.append(f"{len(missing_gold)} gold exec_result CSV(s) absent for scored "
                        f"instance(s): " + ", ".join(missing_gold[:10])
                        + (" ..." if len(missing_gold) > 10 else ""))
        from .swebench_common import infra_required
        raise infra_required("spider2",
                             "the Spider2 checkout is incomplete - " + "; ".join(bits), DOCS_URL)

    if not samples:
        raise RuntimeError("spider2: no scorable local SQLite instances found")
    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="spider2")
    # Canonical local set: 135 SQLite instances (of the repo's 547; the 180 BigQuery + 207
    # Snowflake + 25 BigQuery-GA instances need cloud creds and are excluded).
    logger.info("Loaded %d spider2 local(SQLite) samples (of 135 local in the 547-instance "
                "spider2-lite set; BigQuery/Snowflake need cloud creds).", len(samples))
    return samples


def _make_scorer(localdb_dir: str):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio

        def _grade(tr: Dict[str, Any]) -> bool:
            extra = tr.get("extra_payload") or {}
            db = extra.get("db")
            sql = extract_sql_query(tr.get("response_text") or "")
            if not sql or not db:
                return False
            db_path = os.path.join(localdb_dir, f"{db}.sqlite")
            if not os.path.isfile(db_path):
                return False
            try:
                pred_df = _get_sqlite_result(db_path, sql)
            except Exception:
                return False
            if pred_df is None:
                return False
            # An empty result set can be the RIGHT answer ("which orders shipped late?"
            # -> none did). Rejecting every empty prediction made those questions
            # impossible to answer correctly; let the gold comparison decide instead.
            return _score_against_golds(
                pred_df, extra.get("gold_paths") or [],
                extra.get("condition_cols") or [], bool(extra.get("ignore_order", True))
            ) == 1

        async def _one(tr):
            try:
                tr["is_correct"] = await asyncio.wait_for(asyncio.to_thread(_grade, tr), timeout=120)
                tr["status"] = "OK"
            except asyncio.TimeoutError:
                # A query that will not finish in 120s fails execution accuracy (no result set) -
                # that IS a wrong answer, so is_correct=False, but note why.
                tr["is_correct"] = False
                tr["status"] = "OK"
                tr["scoring_note"] = "query timed out (>120s)"
            except Exception as e:
                # A grading-harness crash (e.g. reading a gold CSV) is not a clean model verdict.
                # It still counts as is_correct=False in the accuracy denominator (as swebench does
                # for per-instance harness errors), but is marked SCORING_ERROR and surfaced via
                # result['scoring_errors'] so a reader can see how many verdicts were harness crashes
                # rather than genuine wrong answers.
                tr["is_correct"] = False
                tr["status"] = "SCORING_ERROR"
                tr["scoring_error"] = f"{type(e).__name__}: {e}"

        await asyncio.gather(*[_one(t) for t in sample_traces])
    return _score


def run_spider2(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Spider2-lite SQLite execution accuracy (hard-errors if gold/repo/DBs absent)."""
    ok, reason = check_spider2_prerequisites()
    if not ok:
        # No-skip policy: missing prereqs HARD-ERROR (never a fabricated 0%/skip row).
        raise infra_required("spider2", reason, DOCS_URL)
    samples = _load_spider2_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="spider2",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(_localdb_dir()),
        declared_scoring_mode="execution",  # SQL execution + result comparison, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
        temperature=kwargs.get("temperature"),
    )
    # gbench runs ONLY the offline local* (SQLite) subset of Spider2-lite (the BigQuery + Snowflake
    # instances need live cloud accounts and are excluded), so the number is NEVER comparable to the
    # published Spider2-lite leaderboard, which is over the full set. Record that unconditionally.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "SQLite-only subset of Spider2-lite (BigQuery + Snowflake instances excluded); not the "
        "full published leaderboard set")
    # Surface grading-harness crashes (SCORING_ERROR): they count as incorrect in accuracy (like
    # swebench's per-instance errors) but are reported here so they are not mistaken for model 0s.
    scoring_errors = sum(1 for t in result.get("sample_traces", [])
                         if t.get("status") == "SCORING_ERROR")
    if scoring_errors:
        result["scoring_errors"] = scoring_errors
    return result
