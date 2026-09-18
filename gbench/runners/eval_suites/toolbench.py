# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: toolbench
# Description: ToolBench / StableToolBench - multi-turn agentic RapidAPI tool use, ToolEval SoPR/SoWR

"""gbench native built-in runner for toolbench (Tool Use & Function Calling).

Canonical ToolBench is scored by StableToolBench (THUNLP-MT/StableToolBench, arXiv:2403.07714):
the model under test runs a MULTI-TURN DFSDT agentic loop, calling RapidAPI tools against a CACHED
API server (deterministic, LLM simulator for cache misses), finishing with a `Finish` call. gbench
DELEGATES the inference + answer-format conversion to StableToolBench's own pipeline (subprocess,
pointing inference at the gbench-served /v1), then scores the converted answer trees with:

  * SoPR (Solvable Pass Rate) - each answer judged Solved/Unsure/Unsolved (1/0.5/0), mean over
    queries, mean over evaluate_times. HEADLINE `accuracy`.
  * SoWR (Solvable Win Rate)  - preference vs the GPT-3.5-CoT reference (virtual_chatgpt_cot).

The judge is gbench's standard Gemini cascade (base.judge_generate_cascade) - the reference grader
gbench uses across every judged suite, by convention. ToolEval's exact prompts + verdict protocol are
ported verbatim below; only the grader model differs from upstream. Because gbench grades with its own
cascade where the published StableToolBench leaderboard uses gpt-4-turbo, a run here is a
gbench-internal number rather than a like-for-like leaderboard entry (leaderboard_comparable=False).

HARD-ERRORS (infra_required, never skips, never a fabricated number) if the StableToolBench checkout,
the cached /virtual server, GEMINI_API_KEY (the judge), or the solvable-query data is absent. Heavy
external provisioning - see docs/evals/toolbench.md. The delegation CLI is version-sensitive;
validate it against your checkout (all paths env-configurable).

Sampling: gbench does NOT pin a temperature for toolbench - StableToolBench's DFSDT loop owns
sampling and drives the model under its own protocol, so `--temperature` /
`GBENCH_TOOLBENCH_TEMPERATURE` are NOT applied here (a passed value would be a silent no-op, so
gbench does not record the illusion of control); a reasoning/thinking mode is likewise not forwarded
to the container. The Gemini judge is pinned at 0.0 (gbench convention), so evaluate_times defaults
to 1 (a deterministic judge gains nothing from repeats).

Sharding (`--shard I/N`): NOT applied for toolbench. Task selection is owned inside the external
`gbench-toolbench` image - the DFSDT driver reads whole per-group query files
(`/stb/solvable_queries/test_instruction/*.json`) *inside the container*; the entrypoint can cap the
FIRST N tasks of each group (TB_LIMIT, used by `--eval-limit`) but exposes no arbitrary shard/offset
API, so there is no host-visible sorted task list this runner could interleave. Forwarding
`GBENCH_SHARD` would therefore be a silent no-op that runs the FULL groups while fabricating the
illusion of a subset, so gbench deliberately does NOT forward it; instead the runner logs a one-time
WARNING when `GBENCH_SHARD` is set. To split toolbench work, partition `GBENCH_TOOLBENCH_GROUPS`
across runs (a coarse, group-level split) rather than `--shard`.
"""

import asyncio
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .swebench_common import infra_required
from .base import judge_generate_cascade, gemini_key_live_valid, free_port

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/toolbench.md"

_GROUPS = ["G1_instruction", "G1_category", "G1_tool",
           "G2_category", "G2_instruction", "G3_instruction"]
_METHOD_DEFAULT = "DFS_woFilter_w2"
_REFERENCE_DEFAULT = "virtual_chatgpt_cot"

# ToolEval prompts, ported VERBATIM from StableToolBench evaluators/tooleval_gpt-3.5-turbo_default/
# template.txt (the canonical SoPR/SoWR evaluator). Only an explicit output-anchor line is appended
# so the free-text Gemini judge is parseable (ToolEval forced an OpenAI function call; we parse the
# enum out of the text instead - see the _parse_* helpers).
_ANCHOR_STATUS = '\n\nEnd your reply with a single line exactly: `answer_status: <Solved|Unsolved|Unsure>`'
_ANCHOR_INDEX = '\n\nEnd your reply with a single line exactly: `index: <0|1>`'
_ANCHOR_TASK = '\n\nEnd your reply with a single line exactly: `task_status: <Solvable|Unsolvable|Unsure>`'

_CHECK_ANSWER_STATUS = (
    "Giving the query and answer, you need give `answer_status` of the answer by following rules:\n"
    "1. If the answer is a sorry message or not a positive/straight response for the given query, "
    "return \"Unsolved\".\n"
    "2. If the answer is a positive/straight response for the given query, you have to further check.\n"
    "2.1 If the answer is not sufficient to determine whether the solve the query or not, return "
    "\"Unsure\".\n"
    "2.2 If you are confident that the answer is sufficient to determine whether the solve the query "
    "or not, return \"Solved\" or \"Unsolved\".\n\n"
    "Query:\n{query}\nAnswer:\n{answer}\n\n"
    "Now give your reason in \"content\" and `answer_status` of JSON to `check_answer_status`." + _ANCHOR_STATUS)

_PARSE_ANSWER_STATUS = (
    "Giving the query and the correspond execution detail of an answer, you need give "
    "`answer_status` of the answer by following rules:\n"
    "1. If all 'tool' nodes' message indicate that there are errors happened, return \"Unsolved\"\n"
    "2. If you find the information in the \"final_answer\" is not true/valid according to the "
    "messages in 'tool' nodes, return \"Unsolved\"\n"
    "3. If you are unable to verify the authenticity and validity of the information, return "
    "\"Unsure\"\n"
    "4. If there are 'tool' node in the chain contains successful func calling and those calling "
    "indeed solve the query, return \"Solved\"\n\n"
    "Query:\n{query}\nAnswer:\n{answer}\n\n"
    "Now you are requested to give reason in \"content\" and `answer_status` of JSON to "
    "`parse_answer_status`." + _ANCHOR_STATUS)

_SELECT_BETTER_ANSWER = (
    "Query:\n{query}\n\nAnswer_0:\n{answer_0}\n\nAnswer_1:\n{answer_1}\n\n"
    "Given above query and answers in JSON format, you must follow the rules to select the "
    "relatively better answer and give the index of the answer **(0 for Answer_0, 1 for "
    "Answer_1)**:\n"
    "1. Compare the value of \"final_answer\" in following aspects:\n"
    "- Informative: whether it contains all necessary information to reply to the query.\n"
    "- Factuality: whether it accurately describes what has been done, and what failed in the end.\n"
    "- Reasoning: If answer does not solve the query, whether gives a detailed and accurate reason "
    "for failure.\n"
    "2. If you cannot determine yet, compare the value of \"answer_details\" in following aspects:\n"
    "- Tool calling costs: calculating the percentage of failed and replicated tools calling.\n"
    "- Running costs: calculating the total tokens T used in execution.\n"
    "- Milestone: calculating the milestone(fixed subtasks) reached in execution.\n"
    "- Exploration: whether tries potential useful tools in execution. Just count times of "
    "successful tool calling with different tools/arguments in execution.\n\n"
    "If you have made your decision, calling `select_better_answer`, else if you cannot determine, "
    "select a random answer." + _ANCHOR_INDEX)

_STATUS_ENUM = ("Solved", "Unsolved", "Unsure")


_IMAGE_DEFAULT = "gbench-toolbench"

# Guard so the "--shard not applied" advisory is logged at most once per process.
_SHARD_WARNED = False


def _warn_shard_not_applied() -> None:
    """Log a one-time WARNING if `--shard` (GBENCH_SHARD) is set: toolbench cannot honor it.

    Task selection lives entirely inside the external image (the DFSDT driver reads its query-id
    lists from `/stb/solvable_queries/test_query_ids/*.json` in the container), so there is no
    host-visible sorted task list to interleave and no shard/offset API on the entrypoint. gbench
    therefore does NOT forward GBENCH_SHARD - a forwarded no-op would run the full groups while
    pretending to be a subset (fabrication). See the module docstring / docs/evals/toolbench.md.
    """
    global _SHARD_WARNED
    if _SHARD_WARNED or not os.environ.get("GBENCH_SHARD", "").strip():
        return
    _SHARD_WARNED = True
    logger.warning(
        "toolbench: --shard %s is NOT applied for this suite - task selection is owned inside the "
        "external %r image (query-id lists live in the container, no host-visible task list to "
        "interleave and no shard/offset API on the entrypoint). GBENCH_SHARD is intentionally NOT "
        "forwarded (a no-op env would run the FULL groups while faking a subset). Split work via "
        "GBENCH_TOOLBENCH_GROUPS across runs instead.",
        os.environ.get("GBENCH_SHARD", "").strip(), _image())


def _image() -> str:
    return os.environ.get("GBENCH_TOOLBENCH_IMAGE", _IMAGE_DEFAULT)


def _groups() -> List[str]:
    raw = (os.environ.get("GBENCH_TOOLBENCH_GROUPS") or "").replace(",", " ").split()
    return raw or list(_GROUPS)


def _served_model_id(base_url: str, fallback: str) -> str:
    """The model id the endpoint actually serves. StableToolBench's OpenAI client sends the `model`
    field verbatim to vLLM, so it must be the EXACT served id (e.g. `google/gemma-4-26B-A4B-it`);
    gbench passes a stripped short name (`gemma-4-26B-A4B-it`) that vLLM 404s on ("model does not
    exist") -> the DFSDT loop retries to exhaustion and the container dies rc=1. Mirror
    aider_polyglot._served_model_id (same class of bug for its litellm client)."""
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:                                              # noqa: BLE001
        logger.warning("toolbench: could not read served model id (%s); using %r", e, fallback)
        return fallback


def check_toolbench_prerequisites() -> Tuple[bool, str]:
    """Docker + the locally-built `gbench-toolbench` image (bundles StableToolBench + cache +
    server) + GEMINI_API_KEY (the gbench Gemini cascade judge)."""
    import shutil
    image = _image()
    build = (f"Build the harness LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f docker/toolbench.Dockerfile docker\n"
             f"It bundles pinned StableToolBench + the 236MB cache + the cached /virtual server. "
             f"See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    if not os.environ.get("GEMINI_API_KEY"):
        return False, ("GEMINI_API_KEY is not set - it powers gbench's Gemini cascade judge that "
                       "scores ToolEval SoPR/SoWR here.")
    _ok, _why = gemini_key_live_valid(os.environ["GEMINI_API_KEY"])
    if not _ok:
        return False, (f"GEMINI_API_KEY was rejected by the judge endpoint ({_why}); a valid key is "
                       "required for the Gemini cascade judge (fail-fast before the container run).")
    return True, ""


# --------------------------------------------------------------------------- #
# Verdict parsing (free-text Gemini -> ToolEval's strict enums). Anchored line first, else the
# last enum word in the text; None when unparseable (treated as JUDGE_OUTAGE upstream).
# --------------------------------------------------------------------------- #
def _parse_enum(text: Optional[str], key: str, enum: Tuple[str, ...]) -> Optional[str]:
    if not text:
        return None
    low = {e.lower(): e for e in enum}
    m = re.search(rf"{key}\s*[:=]\s*[`\"']?([A-Za-z]+)", text, re.IGNORECASE)
    if m and m.group(1).lower() in low:
        return low[m.group(1).lower()]
    hits = [low[w.lower()] for w in re.findall(r"[A-Za-z]+", text) if w.lower() in low]
    return hits[-1] if hits else None


def _parse_answer_status(text: Optional[str]) -> Optional[str]:
    return _parse_enum(text, "answer_status", _STATUS_ENUM)


def _parse_index(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"index\s*[:=]\s*[`\"']?([01])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"[01]", text)
    return int(nums[-1]) if nums else None


def _final_step_has_finish(answer: Dict[str, Any]) -> bool:
    """Walk the answer_details chain; True iff the last tool node is a `Finish` call (ToolEval:
    the driver requires "'name': 'Finish'" in the final step, else auto-Unsolved)."""
    details = answer.get("answer_details") or []
    node = details[0] if details else None
    last_tool = None
    seen = 0
    while isinstance(node, dict) and seen < 1000:
        seen += 1
        if node.get("role") == "tool":
            last_tool = node.get("message")
        nxt = node.get("next") or []
        node = nxt[0] if nxt else None
    return "'name': 'Finish'" in str(last_tool) if last_tool is not None else False


def _process_answer_for_pref(answer: Dict[str, Any]) -> Dict[str, Any]:
    """ToolEval process_answer: truncate final_answer[:1000], drop `method`, keep answer_details."""
    a = dict(answer)
    a.pop("method", None)
    fa = a.get("final_answer")
    if isinstance(fa, str):
        a["final_answer"] = fa[:1000]
    return a


async def _ask(prompt: str, sem: asyncio.Semaphore) -> Optional[str]:
    async with sem:
        text, _used = await judge_generate_cascade(prompt)
    return text


async def _answer_status(query: str, answer: Dict[str, Any], sem: asyncio.Semaphore) -> Optional[str]:
    """ToolEval check_is_solved: deterministic pre-checks, then check_answer_status, escalating to
    parse_answer_status only on Unsure. Returns a status enum, or None on judge outage."""
    final_answer = str(answer.get("final_answer") or "")
    if final_answer == "" or "give_up_and_restart" in final_answer:
        return "Unsolved"
    if not _final_step_has_finish(answer):
        return "Unsolved"
    text = await _ask(_CHECK_ANSWER_STATUS.format(query=query, answer=final_answer), sem)
    status = _parse_answer_status(text)
    if status is None and text is None:
        return None  # JUDGE_OUTAGE
    if status == "Unsure":
        text2 = await _ask(
            _PARSE_ANSWER_STATUS.format(query=query, answer=json.dumps(answer)), sem)
        status2 = _parse_answer_status(text2)
        if status2 is not None:
            status = status2
    return status or "Unsolved"   # unparseable non-outage -> Unsolved (fail-closed, documented)


_STATUS_SCORE = {"Solved": 1.0, "Unsure": 0.5, "Unsolved": 0.0}


async def _select_better(query: str, ans0: Dict[str, Any], ans1: Dict[str, Any],
                         sem: asyncio.Semaphore) -> Optional[int]:
    prompt = _SELECT_BETTER_ANSWER.format(
        query=query,
        answer_0=json.dumps(_process_answer_for_pref(ans0)),
        answer_1=json.dumps(_process_answer_for_pref(ans1)))
    return _parse_index(await _ask(prompt, sem))


def _load_converted(path: str) -> Dict[str, Dict[str, Any]]:
    """Merge every {qid: {query, available_tools, answer}} JSON under a converted-answers dir."""
    merged: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(path):
        return merged
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, dict):
                for qid, v in data.items():
                    if isinstance(v, dict) and "answer" in v:
                        merged[str(qid)] = v
    return merged


async def _score_all(candidate: Dict[str, Dict[str, Any]],
                     reference: Dict[str, Dict[str, Any]],
                     evaluate_times: int, concurrency: int) -> Dict[str, Any]:
    """Compute SoPR (candidate) + SoWR (candidate vs reference) with the Gemini cascade."""
    sem = asyncio.Semaphore(max(1, concurrency))

    # --- SoPR: mean over queries of {1/0.5/0}, averaged over evaluate_times. JUDGE_OUTAGE excluded.
    async def _sopr_run() -> Optional[float]:
        async def _one(v):
            st = await _answer_status(v.get("query", ""), v.get("answer") or {}, sem)
            return None if st is None else _STATUS_SCORE.get(st, 0.0)
        scores = await asyncio.gather(*[_one(v) for v in candidate.values()])
        kept = [s for s in scores if s is not None]
        return (sum(kept) / len(kept)) if kept else None

    sopr_runs = [r for r in await asyncio.gather(*[_sopr_run() for _ in range(evaluate_times)])
                 if r is not None]
    sopr = (sum(sopr_runs) / len(sopr_runs)) if sopr_runs else None

    # --- SoWR: per round, per query, candidate vs reference. Round-majority -> strict wins / queries.
    sowr = None
    if reference:
        async def _status_map(pool: Dict[str, Dict[str, Any]], qids) -> Dict[str, Optional[str]]:
            res = await asyncio.gather(*[
                _answer_status(pool[q].get("query", ""), pool[q].get("answer") or {}, sem)
                for q in qids])
            return dict(zip(qids, res))

        async def _win_round() -> Optional[float]:
            qids = [q for q in candidate if q in reference]
            if not qids:
                return None
            cand_st = await _status_map(candidate, qids)
            ref_st = await _status_map(reference, qids)

            async def _winner(q) -> Optional[str]:
                cs, rs = cand_st.get(q), ref_st.get(q)
                # use_pass_rate shortcut (no LLM)
                if cs == "Solved" and rs == "Unsolved":
                    return "cand"
                if cs == "Unsolved" and rs == "Solved":
                    return "ref"
                # LLM preference (order-shuffled, index mapped back)
                pair = [("cand", candidate[q]["answer"]), ("ref", reference[q]["answer"])]
                random.shuffle(pair)
                # identical answers -> random pick (ToolEval check_identity_answers)
                if json.dumps(pair[0][1], sort_keys=True) == json.dumps(pair[1][1], sort_keys=True):
                    return pair[random.randint(0, 1)][0]
                idx = await _select_better(candidate[q].get("query", ""),
                                           pair[0][1], pair[1][1], sem)
                if idx not in (0, 1):
                    return None
                return pair[idx][0]

            winners = await asyncio.gather(*[_winner(q) for q in qids])
            return {q: w for q, w in zip(qids, winners)}

        rounds = [r for r in await asyncio.gather(*[_win_round() for _ in range(evaluate_times)])
                  if r]
        if rounds:
            qids = rounds[0].keys()
            wins = 0
            total = 0
            for q in qids:
                nc = sum(1 for rd in rounds if rd.get(q) == "cand")
                nr = sum(1 for rd in rounds if rd.get(q) == "ref")
                total += 1
                if nc > nr:
                    wins += 1
            sowr = (wins / total) if total else None

    return {"sopr": sopr, "sowr": sowr, "n_candidate": len(candidate), "n_reference": len(reference)}


def run_toolbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical ToolBench: StableToolBench DFSDT inference + gbench Gemini-judge SoPR/SoWR."""
    ok, reason = check_toolbench_prerequisites()
    if not ok:
        raise infra_required("toolbench", reason, DOCS_URL)

    # --shard (GBENCH_SHARD) cannot be honored: the external image owns task selection and exposes no
    # shard/offset API, so gbench does NOT forward it (a no-op env would fake a subset). Warn once.
    _warn_shard_not_applied()

    # StableToolBench's DFSDT loop owns sampling; gbench neither pins a temperature nor forwards a
    # reasoning mode to the container (a passed value would be a silent no-op). See the module note.
    limit = kwargs.get("limit")

    groups = _groups()
    method = os.environ.get("GBENCH_TOOLBENCH_METHOD", _METHOD_DEFAULT)
    reference = os.environ.get("GBENCH_TOOLBENCH_REFERENCE", _REFERENCE_DEFAULT)
    try:
        # gbench's judge is pinned at 0.0 (deterministic), so repeats add nothing by default.
        evaluate_times = max(1, int(os.environ.get("GBENCH_TOOLBENCH_EVAL_TIMES", "1")))
    except ValueError:
        evaluate_times = 1
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    # The container's OpenAI client sends `model` verbatim to vLLM, which serves the FULL id, not
    # gbench's stripped short name -> resolve the served id (else every call 404s -> rc=1).
    served_model = _served_model_id(endpoint, model_name)

    workdir = tempfile.mkdtemp(prefix="gbench_toolbench_")
    os.chmod(workdir, 0o777)   # the container writes /out as its own uid
    orch_name = "gbench_toolbench_" + os.path.basename(workdir)
    # The cached /virtual server binds a HOST port (the container runs --network host). A fixed 8080
    # collides when two toolbench runs overlap on the same machine, so pick a free host port and hand
    # it to the entrypoint (which points config.yml, the health probe, and SERVICE_URL at it).
    server_port = free_port(8080)

    def _reap():
        # docker --rm only fires on a clean container EXIT; on a subprocess timeout the client is
        # killed while the container keeps running (with its cached /virtual server bound on the host
        # under --network host), so reap the named container in finally + on timeout.
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    try:
        # Run the bundled StableToolBench container: it starts the cached /virtual server, runs the
        # DFSDT loop against the served model (--network host reaches the endpoint), converts the
        # answer trees, and writes candidate (+ reference) converted answers to the mounted /out.
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host", "-v", f"{workdir}:/out:rw",
               "-e", f"TB_MODEL={served_model}", "-e", f"TB_ENDPOINT={endpoint}",
               "-e", f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}",
               "-e", f"TB_GROUPS={' '.join(groups)}", "-e", f"TB_METHOD={method}",
               "-e", f"TB_SERVER_PORT={server_port}",
               "-e", f"TB_NUM_THREAD={max(1, concurrency)}"]
        for env_name, tb_name in (("GBENCH_TOOLBENCH_SIMULATOR_BASE", "TB_SIMULATOR_BASE"),
                                  ("GBENCH_TOOLBENCH_SIMULATOR_MODEL", "TB_SIMULATOR_MODEL"),
                                  ("GBENCH_TOOLBENCH_SIMULATOR_KEY", "TB_SIMULATOR_KEY"),
                                  ("GBENCH_TOOLBENCH_MAX_QUERY_COUNT", "TB_MAX_QUERY_COUNT")):
            v = os.environ.get(env_name)
            if v:
                cmd += ["-e", f"{tb_name}={v}"]
        # --eval-limit is a TOTAL budget across the groups. TB_LIMIT caps the NUMBER OF TASKS per
        # group (the entrypoint slices the query file to the first N) - NOT the per-task DFS search
        # budget (that is --max_query_count / TB_MAX_QUERY_COUNT, canonical 200). Distribute the
        # budget (ceil) across the groups to keep the total task count ~limit.
        if limit is not None:
            try:
                lim = int(limit)
            except (TypeError, ValueError):
                lim = None
            if lim and lim > 0:
                n_groups = max(1, len(groups))
                per_group = max(1, (lim + n_groups - 1) // n_groups)
                cmd += ["-e", f"TB_LIMIT={per_group}"]
                logger.info(
                    "toolbench: --eval-limit=%d -> TB_LIMIT=%d tasks/group across %d groups (~%d total). "
                    "Honored by an image built from the current docker/toolbench_entrypoint.sh "
                    "(slices the query file); the per-task DFS budget stays TB_MAX_QUERY_COUNT=200.",
                    lim, per_group, n_groups, per_group * n_groups)
        cmd.append(_image())
        logger.info("toolbench: docker run %s (groups=%s)", _image(), groups)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=16 * 60 * 60)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "toolbench", "the toolbench container exceeded its 16h timeout and was killed.",
                DOCS_URL) from exc
        if proc.returncode != 0:
            raise infra_required(
                "toolbench",
                f"the toolbench container failed (rc={proc.returncode}). "
                f"tail: {(proc.stderr or proc.stdout or '')[-600:]}", DOCS_URL)

        candidate = _load_converted(os.path.join(workdir, "candidate"))
        if not candidate:
            # fall back to the whole /out (excluding reference/) in case the entrypoint wrote flat
            candidate = {k: v for k, v in _load_converted(workdir).items()}
        if not candidate:
            raise infra_required(
                "toolbench", "the container produced no parseable converted answers under /out "
                "(a harness failure, not a 0%)", DOCS_URL)
        # Reference converted answers for SoWR: a host dir override, else what the image shipped.
        ref_host = os.environ.get("GBENCH_TOOLBENCH_REFERENCE_DIR")
        reference_answers = (_load_converted(ref_host) if ref_host
                             else _load_converted(os.path.join(workdir, "reference")))

        scores = asyncio.run(_score_all(candidate, reference_answers, evaluate_times, concurrency))
    finally:
        _reap()
        shutil.rmtree(workdir, ignore_errors=True)

    if scores.get("sopr") is None:
        raise infra_required(
            "toolbench", "the Gemini judge produced no SoPR verdict (all judge outages?)", DOCS_URL)

    sopr = round(scores["sopr"] * 100.0, 2)
    sowr = round(scores["sowr"] * 100.0, 2) if scores.get("sowr") is not None else None
    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "toolbench",
        "model_name": model_name,
        "status": "success",
        "accuracy": sopr,                 # headline = SoPR (%)
        "sopr": sopr,
        "sowr": sowr,
        "groups_evaluated": groups,
        "method": method,
        "reference_model": reference if sowr is not None else None,
        "judge": "gbench-gemini-cascade",
        "evaluate_times": evaluate_times,
        "total_questions": scores.get("n_candidate"),
        "sampling": ("StableToolBench's DFSDT loop owns sampling; gbench does not pin a temperature "
                     "(a passed --temperature / GBENCH_TOOLBENCH_TEMPERATURE would be a silent "
                     "no-op) and does not forward a reasoning/thinking mode to the container. The "
                     "Gemini judge cascade is pinned at 0.0."),
        "metric": ("StableToolBench SoPR (Solvable Pass Rate) headline + SoWR (win rate vs "
                   f"{reference}), scored by gbench's standard Gemini cascade judge (ToolEval prompts "
                   "ported verbatim; upstream's ToolEval grader is gpt-4-turbo)."),
        # A run here is a gbench-internal number, not a like-for-like StableToolBench leaderboard
        # entry: gbench grades with its standard Gemini cascade (a gbench convention) where the
        # published leaderboard's ToolEval judge is gpt-4-turbo.
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": (
            "graded by gbench's standard Gemini cascade (a gbench convention); the published "
            "StableToolBench leaderboard's ToolEval judge is gpt-4-turbo, so this is a "
            "gbench-internal number"),
    }
    if sowr is None:
        result["sowr_note"] = ("SoWR not computed - reference answers "
                               f"({reference}) not found; set GBENCH_TOOLBENCH_REFERENCE_DIR")
    return result
