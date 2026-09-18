# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: multipl_e
# Description: MultiPL-E - execution-based pass@1 across languages (HumanEval family)

"""gbench native built-in runner for multipl_e (Polyglot Coding & Software).

Canonical MultiPL-E (nuprl/MultiPL-E): translate-and-execute HumanEval across all target
languages. For each problem the program is `prompt + completion + "\\n" + tests`, compiled/run
per language; canonical metric = unbiased pass@1 (= avg over samples of per-sample OK).

Execution is delegated to the OFFICIAL MultiPL-E evaluator, built LOCALLY (never pulled) from
pinned upstream source into the image `gbench-multipl-e` - it ships all 24 language toolchains and
the canonical per-language status rule (OK/SyntaxError/Exception/Timeout). See docs/evals/multipl_e.md
for the one-time `docker build`. The suite HARD-ERRORS (infra_required, never skips, never a
fabricated 0%) if Docker or the image is unavailable.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's shipped
`generation_config.json`) with `--thinking`; override for a whole run with `--temperature`, or for
this suite alone with `GBENCH_MULTIPL_E_TEMPERATURE`. The published MultiPL-E leaderboard is n-sample
pass@1 at temperature 0.2 - set `GBENCH_MULTIPL_E_SAMPLES` (completions/problem, default 1) and
temperature 0.2 for a leaderboard-comparable run; the default single greedy completion is a valid
but higher-variance pass@1 and is reported as leaderboard_comparable=False.
"""

import glob
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .sampling import stratified_sample
from .base import run_eval_suite, strip_thinking_tags
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/multipl_e.md"

#: The canonical HumanEval MultiPL-E language configs (nuprl/MultiPL-E `humaneval-<lang>`), all
#: provided by the official evaluator image. MultiPL-E TRANSLATES Python's HumanEval into other
#: languages, so there is no `humaneval-py` config (requesting it raises BuilderConfig-not-found);
#: Python is intentionally absent. Ada (`adb`) ships in a SEPARATE upstream dockerfile and is
#: excluded from the default image; add it via GBENCH_MULTIPL_E_LANGS if the image includes it.
_CANONICAL_LANGS = [
    "cpp", "cs", "d", "go", "java", "js", "jl", "lua", "php", "pl", "r", "rb",
    "rkt", "rs", "scala", "sh", "swift", "ts", "clj", "dart", "elixir", "hs", "ml",
]

_IMAGE_DEFAULT = "gbench-multipl-e"


def _image() -> str:
    return os.environ.get("GBENCH_MULTIPL_E_IMAGE", _IMAGE_DEFAULT)


def _langs() -> List[str]:
    raw = (os.environ.get("GBENCH_MULTIPL_E_LANGS") or "").replace(",", " ").split()
    return raw or list(_CANONICAL_LANGS)


def _samples_per_problem() -> int:
    try:
        return max(1, int(os.environ.get("GBENCH_MULTIPL_E_SAMPLES", "1")))
    except ValueError:
        return 1


def check_multipl_e_prerequisites() -> Tuple[bool, str]:
    """`datasets` + Docker daemon + the locally-built MultiPL-E evaluator image."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed (pip install 'gbench[evals]')."
    image = _image()
    build = (f"Build the evaluator LOCALLY from gbench's own Dockerfile (never pulled; it clones "
             f"pinned MultiPL-E + bundles all 24 toolchains):\n"
             f"  docker build -t {image} -f docker/multipl_e.Dockerfile docker\n"
             f"See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    return True, ""


def _stop_at_stop_token(text: str, stop_tokens: List[str]) -> str:
    idxs = [text.find(t) for t in (stop_tokens or []) if t and text.find(t) != -1]
    return text[:min(idxs)] if idxs else text


def _extract_code(response_text: str) -> str:
    """Largest fenced code block, else the raw text."""
    blocks = re.findall(r"```[A-Za-z0-9+#._-]*\n([\s\S]*?)```", response_text or "")
    return max(blocks, key=len) if blocks else (response_text or "")


def _declaration_lines(prompt: str) -> List[str]:
    """Lines of the prompt that declare the target function (signature the model repeats)."""
    out = []
    for line in (prompt or "").splitlines():
        s = line.strip()
        if "(" in s and (s.endswith(":") or s.endswith("{") or s.endswith(")")):
            out.append(s)
    return out


def _find_decl_ws_insensitive(code: str, decl: str) -> int:
    """Index in `code` where declaration `decl` starts, ignoring whitespace differences.

    A chat model reformats the signature ('foo(nums) {' vs the prompt's 'foo(nums){'), so an exact
    `code.find(decl)` missed it, the dedup fell through, and the signature was emitted TWICE (a
    compile error scored as a wrong answer). Match on the whitespace-stripped text and map back to
    the original index."""
    i = code.find(decl)
    if i != -1:
        return i
    target = "".join(decl.split())
    if not target:
        return -1
    stripped, idxmap = [], []
    for k, ch in enumerate(code):
        if not ch.isspace():
            stripped.append(ch)
            idxmap.append(k)
    j = "".join(stripped).find(target)
    return idxmap[j] if j != -1 else -1


def _balance_against_tests(code: str, tests: str) -> str:
    """Make the final program `code + "\\n" + tests` brace-balanced by stripping the model's EXCESS
    trailing closers. DATA-DRIVEN (uses the actual test suite), so it is language-agnostic:
      - JS/TS/etc. whose tests are SELF-CONTAINED (start `const assert = ...`, close nothing): a
        chat model's complete self-closed function already balances -> nothing stripped.
      - Scala/Java/etc. whose tests CLOSE the prompt's object/def scope: the model's extra closers
        are stripped so the tests close exactly those scopes.
    The earlier heuristic stripped to the PROMPT's open-brace count uniformly, which corrupted the
    self-contained-test languages (JS -> 'Unexpected end of input', a false 0). No-op for indentation
    languages (no braces). Never ADDS braces: an under-closed program is a genuine model error."""
    if "{" not in code and "{" not in tests:
        return code
    def net(s: str) -> int:
        return s.count("{") - s.count("}")
    body = code.rstrip("\n")
    # program over-closes (more } than {) -> drop the model's trailing } until code+tests balances
    while net(body) + net(tests) < 0:
        stripped = body.rstrip()
        if stripped.endswith("}"):
            body = stripped[:-1].rstrip("\n")
        else:
            break
    return body


def _assemble_code(response_text: str, prompt: str, stop_tokens: List[str]) -> str:
    """The runnable code (prompt preamble + model's continuation), WITHOUT the test suite.

    Canonical MultiPL-E runs `prompt + completion + "\\n" + tests`; the container appends the
    tests, so this returns just `prompt + completion`. A chat model asked to complete the function
    answers with the whole definition, so a shared declaration line is enough to recognise it and
    avoid emitting the signature twice (a compile error scored as a wrong answer). Signature
    reformatting is handled by a whitespace-insensitive dedup; over-closed braces are reconciled
    LATER, data-driven, by _balance_against_tests (which needs the actual test suite).
    """
    code = _extract_code(strip_thinking_tags(response_text or ""))
    ps = prompt.strip()
    if ps and ps in code:  # model re-emitted the prompt verbatim -> keep the continuation
        return prompt + _stop_at_stop_token(code.split(ps, 1)[1], stop_tokens)
    for decl in reversed(_declaration_lines(prompt)):
        index = _find_decl_ws_insensitive(code, decl)
        if index == -1:
            continue
        preamble = prompt.split(decl, 1)[0] if decl in prompt else ""
        return preamble + _stop_at_stop_token(code[index:], stop_tokens)
    return prompt + _stop_at_stop_token(code, stop_tokens)


def _load_multipl_e_samples(
    langs: Optional[List[str]] = None,
    benchmark: str = "humaneval",
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load MultiPL-E (HumanEval family) for ALL requested languages; raises on failure."""
    langs = langs or _langs()
    from datasets import load_dataset
    samples = []
    for lang in langs:
        cfg = f"{benchmark}-{lang}"
        try:
            ds = load_dataset("nuprl/MultiPL-E", cfg, split="test")
        except Exception as e:
            raise RuntimeError(f"Could not load nuprl/MultiPL-E '{cfg}': {e}") from e
        for item in ds:
            name = item.get("name")
            prompt = item.get("prompt")
            tests = item.get("tests")
            if not name or not prompt or not tests:
                raise RuntimeError(
                    f"multipl_e: unexpected schema for {cfg} (name/prompt/tests); "
                    "refusing to fabricate sample data")
            gold = json.dumps({
                "name": name, "language": lang, "prompt": prompt,
                "tests": tests, "stop_tokens": item.get("stop_tokens") or [],
            })
            content = (
                f"Complete the following {lang} function. Return ONLY the completed "
                f"function in a single fenced code block, no explanation.\n\n"
                f"```{lang}\n{prompt}\n```")
            samples.append(([{"role": "user", "content": content}], gold,
                            {"category": lang, "name": name}))

    if limit is not None and limit > 0:
        # Stratify by language: the loop concatenates one language at a time, so a head is ONE
        # language; `--eval-limit 20` must sample ACROSS languages, not measure just the first.
        samples = stratified_sample(
            samples, limit,
            lambda x: (x[2] or {}).get("category") if len(x) > 2 and isinstance(x[2], dict) else None,
            seed="multipl_e")
    logger.info("Loaded %d multipl_e samples (langs: %s).", len(samples), langs)
    return samples


def _make_scorer(image: str, workers: int, metrics: Dict[str, Any]):
    """Delegate execution to the official MultiPL-E evaluator container (one input file per trace,
    1:1 result mapping so no completion-ordering assumptions), then set is_correct per trace."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_mpe_")
        os.chmod(workdir, 0o777)   # the evaluator may run as a non-root uid in the image
        by_uid: Dict[str, Dict[str, Any]] = {}
        try:
            for i, tr in enumerate(sample_traces):
                try:
                    g = json.loads(tr.get("gold_answer") or "{}")
                except Exception:
                    g = {}
                code = _assemble_code(tr.get("response_text") or "", g.get("prompt", ""),
                                      g.get("stop_tokens") or [])
                # Reconcile a chat model's over-closed braces against the ACTUAL test suite (some
                # languages' tests close the prompt's scope, others are self-contained).
                code = _balance_against_tests(code, g.get("tests", ""))
                uid = f"p{i:06d}"
                by_uid[uid] = tr
                # prompt="" so the container assembles `code + "\n" + tests`; `code` already carries
                # the prompt preamble + the model's continuation.
                payload = {"name": uid, "language": g.get("language", ""), "prompt": "",
                           "completions": [code], "tests": g.get("tests", "")}
                with open(os.path.join(workdir, f"{uid}.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.chmod(os.path.join(workdir, f"{uid}.json"), 0o644)

            cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{workdir}:/out:rw",
                   image, "--dir", "/out", "--output-dir", "/out", "--recursive",
                   "--max-workers", str(max(1, workers))]
            proc = await asyncio.to_thread(
                lambda: subprocess.run(cmd, capture_output=True, text=True))

            parsed = 0
            # The evaluator writes `<name>.results.json` (uncompressed) or `.results.json.gz`
            # depending on version - read both (verified against the built image: it emits plain
            # `.results.json`).
            result_files = (glob.glob(os.path.join(workdir, "*.results.json.gz"))
                            + glob.glob(os.path.join(workdir, "*.results.json")))
            for p in result_files:
                try:
                    opener = gzip.open if p.endswith(".gz") else open
                    with opener(p, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                tr = by_uid.get(data.get("name"))
                if tr is None:
                    continue
                res = data.get("results") or []
                tr["is_correct"] = bool(res) and res[0].get("status") == "OK" \
                    and res[0].get("exit_code") == 0
                tr["status"] = "OK"
                # Trajectory completeness: record what the sandbox actually did so a failure can be
                # read back (SyntaxError vs assertion vs timeout) instead of just "is_correct=False".
                if res:
                    r0 = res[0]
                    tr["execution"] = {
                        "status": r0.get("status"),
                        "exit_code": r0.get("exit_code"),
                        "stdout_tail": (r0.get("stdout") or "")[-1200:],
                        "stderr_tail": (r0.get("stderr") or "")[-1200:],
                    }
                parsed += 1

            if parsed == 0:
                # CC6: the container ran but scored nothing - a harness failure, not "0% solved".
                logger.error("multipl_e: container produced no parsed results. stderr tail: %s",
                             (proc.stderr or "")[-800:])
                metrics["multipl_e_report"] = {
                    "error": "the MultiPL-E evaluator container produced no parsed results",
                    "stderr_tail": (proc.stderr or "")[-400:]}
            # Any trace the container did not score is a harness miss, not a model 0: mark it and
            # count it (surfaced on the result) rather than silently reading it as wrong.
            unscored = 0
            for tr in by_uid.values():
                if "is_correct" not in tr:
                    tr["is_correct"] = False
                    tr["status"] = "OK"
                    tr["scoring_note"] = "no container result"
                    unscored += 1
            if unscored:
                metrics["multipl_e_unscored"] = unscored
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return _score


def run_multipl_e(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical MultiPL-E execution pass@1 via the local evaluator container."""
    ok, reason = check_multipl_e_prerequisites()
    if not ok:
        raise infra_required("multipl_e", reason, DOCS_URL)
    langs = _langs()
    limit = kwargs.get("limit")
    samples = _load_multipl_e_samples(langs=langs, limit=limit)
    metrics: Dict[str, Any] = {}
    n = _samples_per_problem()
    result = run_eval_suite(
        eval_name="multipl_e",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(_image(), concurrency, metrics),
        declared_scoring_mode="execution",  # sandboxed test execution, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
        temperature=kwargs.get("temperature"),
        attempt_count=n,          # n completions/problem; framework avg@k == unbiased pass@1
        supports_attempts=True,   # scoring is per-completion, so @k is real
    )
    result.update(metrics)
    if isinstance(metrics.get("multipl_e_report"), dict) and metrics["multipl_e_report"].get("error"):
        result["status"] = "error"
        result["error"] = metrics["multipl_e_report"]["error"]
    result["languages_evaluated"] = list(langs)
    result["samples_per_problem"] = n
    # The published leaderboard is n>=20-sample pass@1 at temperature 0.2 over the FULL language
    # set. Anything short (a --eval-limit subset, a language subset, single/low-sample greedy, or a
    # non-0.2 temperature) is a valid pass@1 but not that number.
    temp = result.get("temperature")
    noncanon: List[str] = []
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if set(langs) != set(_CANONICAL_LANGS):
        noncanon.append("language subset")
    if n < 20:
        noncanon.append(f"{n} completion(s)/problem (leaderboard uses >=20)")
    if temp != 0.2:
        noncanon.append(f"temperature={temp} (leaderboard uses 0.2)")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result
