# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native RULER long-context evaluation suite (Hsieh et al., NVIDIA, COLM 2024).

Canonical RULER is a tokenizer-defined DATA-GENERATION benchmark: 13 tasks x 6 context
bands (4k/8k/16k/32k/64k/128k), 500 samples each, with haystacks padded to exact token
counts under the TARGET model's own tokenizer. This runner drives NVIDIA/RULER's own
`prepare.py` (from a checkout at GBENCH_RULER_DIR) to generate the per-tokenizer data once
(cached to disk, reused on later runs), sends each prompt single-shot, and scores with the
canonical metrics: string_match_all (niah/vt/cwe/fwe, fraction of references present) and
string_match_part (qa, any reference present), after RULER's postprocess_pred. The headline
`accuracy` is the canonical Avg = mean over bands of (mean of the 13 task scores); per-band,
per-task, per-cell scores + effective_length + wAvg are also reported.

Requires a serving endpoint whose context window covers the largest band (>=~140k for 128k);
the suite hard-errors (does not silently skip) if the server reports an insufficient
max_model_len. Generation is greedy with small per-task token caps; `--thinking` is
non-canonical here (the short caps truncate reasoning).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_RULER_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import run_eval_suite
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path
from .mrcr import _get_server_max_model_len

logger = logging.getLogger(__name__)

PILLAR = "Long Context"
DOCS_URL = "docs/evals/ruler.md"

_ENV = "GBENCH_RULER_DIR"

# The 13 headline tasks (RULER scripts/synthetic.yaml).
_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]
_DEFAULT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
# Canonical per-base-type generation caps (RULER scripts/data/synthetic/constants.py).
_GEN_CAP = {"niah": 128, "variable_tracking": 30,
            "common_words_extraction": 120, "freq_words_extraction": 50, "qa": 32}
_EFF_THRESHOLD = 85.6  # RULER "effective length": largest band scoring >= 85.6 (0.856)


def _base_type(task: str) -> str:
    if task.startswith("niah"):
        return "niah"
    return {"vt": "variable_tracking", "cwe": "common_words_extraction",
            "fwe": "freq_words_extraction", "qa_1": "qa", "qa_2": "qa"}[task]


def _band_label(n: int) -> str:
    return f"{n // 1024}k"


# --- canonical scoring (ported from RULER scripts/eval/synthetic/constants.py + evaluate.py) ---
def _postprocess_pred(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"[\x00-\x1f]", "\n", text).strip()


def _string_match_all(pred: str, refs: List[str]) -> float:
    """Fraction of reference strings (case-insensitive) present in the prediction."""
    if not refs:
        return 0.0
    p = pred.lower()
    return sum(1.0 if str(r).lower() in p else 0.0 for r in refs) / len(refs)


def _string_match_part(pred: str, refs: List[str]) -> float:
    """1.0 if ANY reference string is present, else 0.0."""
    if not refs:
        return 0.0
    p = pred.lower()
    return max(1.0 if str(r).lower() in p else 0.0 for r in refs)


def _item_score(task: str, pred: str, refs: List[str]) -> float:
    fn = _string_match_part if _base_type(task) == "qa" else _string_match_all
    return fn(_postprocess_pred(pred), refs)


def _eval_ruler(pred: str, gold: str) -> bool:
    """Phase-1 pass/fail (all references matched); the reported metric is the mean score."""
    try:
        blob = json.loads(gold)
    except Exception:
        return False
    return _item_score(blob.get("task", ""), pred or "", blob.get("outputs") or []) >= 0.999


# --- config / prerequisites ------------------------------------------------- #
def _ruler_dir() -> str:
    d = prereqs_path("RULER", (os.environ.get(_ENV) or "").strip()) or ""
    hint = (f"Set {_ENV} to an NVIDIA/RULER checkout:\n"
            "  git clone https://github.com/NVIDIA/RULER   (needs git-lfs for its *.json data)\n"
            "  run its scripts/data/synthetic/json/{download_paulgraham_essay.py,download_qa_dataset.sh}\n"
            f"  export {_ENV}=<path>/RULER")
    if not d or not os.path.isfile(os.path.join(d, "scripts", "data", "prepare.py")):
        raise infra_required("ruler", f"RULER checkout not found ({_ENV}={d!r}). {hint}", DOCS_URL)
    return d


def _lengths() -> List[int]:
    raw = (os.environ.get("GBENCH_RULER_LENGTHS") or "").strip()
    if raw:
        return [int(x) for x in raw.replace(",", " ").split()]
    return list(_DEFAULT_LENGTHS)


def _require_context(base_url: str, max_band: int) -> None:
    need = max_band + max(_GEN_CAP.values())  # prompt band + generation headroom
    max_len = _get_server_max_model_len(base_url)
    if max_len is None:
        logger.warning("ruler: could not read the server's max_model_len; ensure it covers >= %d "
                       "tokens for the %s band.", need, _band_label(max_band))
        return
    logger.info("ruler: server max_model_len=%d (need >= %d for %s band)", max_len, need, _band_label(max_band))
    if max_len < need:
        raise infra_required(
            "ruler",
            f"served max_model_len ({max_len}) is below the {_band_label(max_band)} band "
            f"(needs >= {need}). Serve the model with a larger context (e.g. vLLM "
            f"--max-model-len {need}) or restrict bands via GBENCH_RULER_LENGTHS.",
            DOCS_URL)


# --- per-tokenizer data generation (RULER prepare.py; cached + reused) ------- #
def _prepare_task(ruler_dir: str, tokenizer: str, task: str, band: int,
                  num_samples: int) -> List[Dict[str, Any]]:
    tok_slug = re.sub(r"[^A-Za-z0-9._-]", "_", tokenizer)
    cache = Path.home() / ".cache" / "gbench" / "ruler" / tok_slug / str(band)
    out = cache / task / "validation.jsonl"

    if out.is_file():  # reuse: RULER writes exactly num_samples lines when complete
        with open(out, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) == num_samples:
            return rows

    cache.mkdir(parents=True, exist_ok=True)
    data_dir = os.path.join(ruler_dir, "scripts", "data")
    # RULER's prepare.py hardcodes a bare `python` for its nested generator subprocess, so put
    # THIS interpreter's dir first on PATH (it has wonderwords/nltk/etc.).
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    cmd = [sys.executable, "prepare.py",
           "--save_dir", str(cache), "--benchmark", "synthetic", "--task", task,
           "--tokenizer_path", tokenizer, "--tokenizer_type", "hf",
           "--max_seq_length", str(band), "--num_samples", str(num_samples),
           "--model_template_type", "base"]
    logger.info("ruler: generating %s @ %s (%d samples) under %s", task, _band_label(band), num_samples, tokenizer)
    proc = subprocess.run(cmd, cwd=data_dir, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"ruler: prepare.py failed for {task}@{_band_label(band)} "
            f"(rc={proc.returncode}): {(proc.stderr or proc.stdout)[-600:]}")
    with open(out, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ruler(model_name: str, base_url: str, concurrency: int,
              enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run the full RULER matrix (per-tokenizer generated), canonical string-match scoring."""
    ruler_dir = _ruler_dir()
    tokenizer = kwargs.get("tokenizer") or os.environ.get("GBENCH_RULER_TOKENIZER") or model_name
    lengths = _lengths()
    num_samples = int(os.environ.get("GBENCH_RULER_NUM_SAMPLES", "500"))
    _require_context(base_url, max(lengths))

    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    for band in lengths:
        for task in _TASKS:
            rows = _prepare_task(ruler_dir, tokenizer, task, band, num_samples)
            cap = _GEN_CAP[_base_type(task)]
            for r in rows:
                gold = json.dumps({"outputs": r.get("outputs") or [], "task": task})
                # RULER's canonical short caps (qa=32/niah=128) are correct for NO-think greedy
                # decoding, but under --thinking they truncate the reasoning before any answer is
                # emitted (see the module header) -> empty response -> a fake 0.0. Apply the short
                # cap ONLY when not thinking; under --thinking omit it so base.py's sovereign,
                # thinking-aware budget (or the operator's --max-output-tokens) governs and the
                # reasoning + the short answer both fit.
                meta = {"category": f"{task}@{_band_label(band)}",
                        **({"max_tokens": cap} if not enable_thinking else {})}
                samples.append(([{"role": "user", "content": r["input"]}], gold, meta))

    if kwargs.get("limit"):
        samples = stratified_sample(samples, kwargs["limit"],
                                    key_fn=lambda s: (s[2] or {}).get("category"), seed="ruler")

    res = run_eval_suite(
        eval_name="ruler",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_ruler,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,  # already fully materialised (and stratified) above
        max_output_tokens=max(_GEN_CAP.values()),  # per-task caps ride in each sample's meta
        temperature=kwargs.get("temperature"),
    )

    # --- canonical aggregation (nested mean), via trace-walk ---
    cells: Dict[Tuple[str, str], List[float]] = {}
    for t in res.get("sample_traces", []) or []:
        try:
            blob = json.loads(t.get("gold_answer") or "{}")
        except Exception:
            continue
        task = blob.get("task", "")
        s = _item_score(task, t.get("response_text") or "", blob.get("outputs") or [])
        t["ruler_score"] = round(s, 4)
        band = (t.get("category") or "@").split("@")[-1]
        cells.setdefault((task, band), []).append(s)

    cell_mean = {k: sum(v) / len(v) for k, v in cells.items() if v}
    by_band: Dict[str, List[float]] = {}
    by_task: Dict[str, List[float]] = {}
    for (task, band), m in cell_mean.items():
        by_band.setdefault(band, []).append(m)
        by_task.setdefault(task, []).append(m)
    length_score = {b: round(100 * sum(v) / len(v), 2) for b, v in by_band.items()}
    task_score = {t: round(100 * sum(v) / len(v), 2) for t, v in by_task.items()}

    present = [_band_label(b) for b in lengths if _band_label(b) in length_score]
    sl = [length_score[b] for b in present]
    headline = round(sum(sl) / len(sl), 2) if sl else 0.0

    eff = None
    for b in lengths:
        if length_score.get(_band_label(b), -1) >= _EFF_THRESHOLD:
            eff = b
    n = len(sl)
    wavg_inc = round(sum(s * (i + 1) for i, s in enumerate(sl)) / (n * (n + 1) / 2), 2) if n else 0.0
    wavg_dec = round(sum(s * (n - i) for i, s in enumerate(sl)) / (n * (n + 1) / 2), 2) if n else 0.0

    res["accuracy"] = headline
    res["metric"] = ("RULER Avg = mean over length bands of (mean of the 13 task scores); "
                     "string_match_all (niah/vt/cwe/fwe) + string_match_part (qa)")
    res["ruler_length_scores"] = length_score
    res["ruler_task_scores"] = task_score
    res["ruler_cell_scores"] = {f"{t}@{b}": round(100 * m, 2) for (t, b), m in cell_mean.items()}
    res["length_bands"] = present
    res["effective_length"] = eff
    res["wavg_inc"] = wavg_inc
    res["wavg_dec"] = wavg_dec
    # Fully canonical only for the complete greedy matrix: all 13 tasks x the EXACT 6 canonical
    # bands (4k..128k) at 500 samples each, no --thinking/--eval-limit. A custom GBENCH_RULER_LENGTHS
    # (e.g. 6 small bands) or a reduced GBENCH_RULER_NUM_SAMPLES is a valid run but NOT comparable
    # to the published RULER leaderboard, so it must not carry the flag.
    _canonical_bands = {_band_label(b) for b in _DEFAULT_LENGTHS}
    res["leaderboard_comparable"] = bool(
        not kwargs.get("limit") and not enable_thinking
        and set(present) == _canonical_bands and len(by_task) == len(_TASKS)
        and num_samples == 500)
    return res
