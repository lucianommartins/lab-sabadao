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

"""Golden Set validation runner for the Gemma benchmark suite.

The Golden Set is a deterministic smoke test over basic capabilities.
Every case asserts one thing with a check that cannot be satisfied by
accident, and every case produces exactly one of three outcomes:

``passed``  the model answered correctly
``failed``  the model answered incorrectly
``error``   the harness never got an answer

Model failures and harness failures are deliberately never mixed. A
down endpoint must not look like a total regression.
"""

import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.config import BenchmarkConfig
from ..core.models import ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = f"http://127.0.0.1:{os.environ.get('GBENCH_SERVER_PORT', '8000')}/v1"
DEFAULT_TIMEOUT_S = 30
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0

# Bump when a change in this file could make the same answer score
# differently: the comparison rules, the tool-call matching, the
# normalisation applied before comparing, or what counts as a harness
# error rather than a failure. Do not bump for a docstring, a log line
# or a refactor that preserves verdicts.
#
# Declared rather than derived on purpose. The obvious alternative, a
# content hash of this module, moves the scaffold id on every unrelated
# edit and so breaks every historical series for changes that cannot
# affect a score. A version someone forgets to bump under-reports one
# change. A content hash over-reports every change, which trains people
# to ignore the id. `CONTRACT_VERSION` in core/scaffold.py is declared
# for the same reason.
SCORING_VERSION = 1

# Sampling is pinned so a golden run is reproducible. A case may
# override any of these through its own top-level "sampling" block, and
# whatever is actually used is echoed back in the task result.
DEFAULT_SAMPLING = {"temperature": 0.0, "max_tokens": 512}

# Status codes worth another attempt. Everything else is a real answer
# from the server and retrying it just wastes time.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"
# A case whose required modality the model architecturally lacks (verified from
# the model config, e.g. no audio tower) is NOT scored: it is neither a pass, a
# fail, nor a harness error. It is excluded from the denominator so a model is
# graded only on the modalities it actually has. This is capability-gating, not
# a lazy skip: a model that CAN do the modality but whose SERVER is
# misconfigured still errors (a real signal).
STATUS_NOT_APPLICABLE = "not_applicable"

# Every shipped case declares a category, but a case authored elsewhere
# and pointed at with --golden-tasks might not. Bucket those rather than
# letting a None key reach the summary.
UNCATEGORIZED = "uncategorized"

# Message fields that are part of the OpenAI chat schema but are not
# role or content. An assistant turn that made a tool call carries
# "tool_calls", and the "tool" turn answering it carries "tool_call_id"
# and "name". They have to survive into the request body untouched,
# because the server-side chat template is what serializes them and a
# case that never sends them cannot tell you whether that works.
PASSTHROUGH_MESSAGE_KEYS = ("tool_calls", "tool_call_id", "name")


class GoldenHarnessError(RuntimeError):
    """The harness could not obtain an answer from the model.

    Distinct from the model getting a case wrong. A connection failure,
    an auth failure or a malformed response says nothing about the
    model, so it must never be scored as one.
    """


def _gcloud_identity_token() -> Optional[str]:
    """Return a gcloud identity token, or None with a logged reason."""
    try:
        res = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as err:
        logger.warning(f"gcloud identity token lookup failed: {err}")
        return None

    if res.returncode != 0:
        detail = res.stderr.strip()[:200]
        logger.warning(
            "gcloud auth print-identity-token exited "
            f"{res.returncode}: {detail}"
        )
        return None

    return res.stdout.strip() or None


def _normalize(text: str, mode: str) -> str:
    """Normalize an answer before comparison.

    ``strip``       trim surrounding whitespace (default)
    ``whitespace``  additionally collapse internal whitespace runs
    ``lower``       collapse whitespace and lowercase
    ``code``        collapse whitespace and drop it around punctuation, so
                    ``s == s[::-1]`` and ``s==s[::-1]`` compare equal while
                    the boundary in ``return s`` is preserved
    """
    if mode == "whitespace":
        return " ".join(text.split())
    if mode == "lower":
        return " ".join(text.split()).lower()
    if mode == "code":
        collapsed = " ".join(text.split())
        return re.sub(r"\s*([^\w\s])\s*", r"\1", collapsed)
    return text.strip()


def _extract_answer(text: str, pattern: str) -> Optional[str]:
    """Return the LAST match of ``pattern`` in ``text``.

    Models routinely restate a value while reasoning, and the value they
    committed to is the final one. Taking the last match is what makes
    "I first computed 42, but the correct answer is 39" fail against a
    golden of 42, where a substring check would pass it.

    Returns the first capturing group if the pattern has one, otherwise
    the whole match. Returns None when the pattern never matches, which
    is itself a failure: the model did not answer in the required shape.
    """
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
    if not matches:
        return None
    last = matches[-1]
    return last.group(1) if last.groups() else last.group(0)


def _extract_json_value(text: str) -> Tuple[bool, Any]:
    """Pull the last top-level JSON value out of ``text``.

    Returns ``(found, value)``. Scanning to the last top-level value
    rather than the first means a model that shows a wrong object then
    corrects itself is judged on the correction, consistent with
    ``_extract_answer``. Nested objects are skipped because the decoder
    consumes the whole enclosing value in one step.
    """
    decoder = json.JSONDecoder()
    found = False
    value: Any = None
    idx = 0
    while idx < len(text):
        if text[idx] not in "{[":
            idx += 1
            continue
        try:
            decoded, end = decoder.raw_decode(text[idx:])
        except ValueError:
            idx += 1
            continue
        found, value = True, decoded
        idx += end
    return found, value


def _parse_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI ``message.tool_calls`` into name/arguments dicts.

    ``arguments`` arrives as a JSON string on the wire. Arguments that
    do not parse are kept under ``__unparsed__`` so the failure detail
    can show what the model actually emitted.
    """
    parsed = []
    for call in raw or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", {})
        if not isinstance(fn, dict):
            fn = {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"__unparsed__": args}
        if not isinstance(args, dict):
            args = {"__unparsed__": args}
        parsed.append({"name": fn.get("name", ""), "arguments": args})
    return parsed


def _argument_equal(got: Any, want: Any) -> bool:
    """Compare one tool-call argument value."""
    if isinstance(got, str) and isinstance(want, str):
        return _normalize(got, "lower") == _normalize(want, "lower")
    return got == want


def _tool_call_matches(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check one actual tool call against one expected tool call.

    Expected arguments must all be present and equal. Extra arguments
    the model volunteers (a unit, a format) are allowed, so a case does
    not have to enumerate every optional parameter in the schema.
    """
    if expected.get("name", "") != actual.get("name", ""):
        return False, f"name '{actual.get('name')}'"

    want_args = expected.get("arguments", {}) or {}
    got_args = actual.get("arguments", {}) or {}
    for key, want_val in want_args.items():
        if key not in got_args:
            return False, f"missing argument '{key}'"
        if not _argument_equal(got_args[key], want_val):
            return False, (
                f"argument '{key}' was {got_args[key]!r}, "
                f"expected {want_val!r}"
            )
    return True, "ok"


def _split_response(
    response: Union[str, Dict[str, Any], None]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Accept either a raw string or a parsed message dict."""
    if isinstance(response, dict):
        return response.get("content") or "", response.get("tool_calls") or []
    return response or "", []


class GoldenBenchmarkRunner:
    """Runner for Golden Set benchmark tasks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize Golden Set runner.

        Args:
            config: Benchmark configuration instance
        """
        self.config = config
        self.dataset_dir = Path(__file__).parent.parent / "golden_dataset"
        self._model_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load_tasks(
        self, selected_tasks: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load Golden Set task definitions from JSON files.

        Args:
            selected_tasks: Optional filter for specific task IDs or file
                names.

        Returns:
            List of task definition dictionaries, sorted by task id so a
            run is ordered the same way every time.
        """
        tasks: List[Dict[str, Any]] = []
        if not self.dataset_dir.exists():
            logger.error(
                f"Golden dataset directory {self.dataset_dir} does not "
                "exist. If gbench was installed as a wheel this means the "
                "dataset was not packaged."
            )
            return tasks

        for file_path in sorted(self.dataset_dir.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as err:
                logger.error(f"Failed to parse golden task {file_path}: {err}")
                continue

            task_id = data.get("id", file_path.stem)
            if selected_tasks and (
                task_id not in selected_tasks
                and file_path.name not in selected_tasks
            ):
                continue
            tasks.append(data)

        return sorted(tasks, key=lambda t: str(t.get("id", "")))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_task_response(
        self,
        task: Dict[str, Any],
        response: Union[str, Dict[str, Any], None],
    ) -> Dict[str, Any]:
        """Evaluate a model output against golden_truth expectations.

        Args:
            task: Task specification dictionary.
            response: Either the raw response text or the parsed message
                dict returned by :meth:`_query_remote_endpoint`.

        Returns:
            Dictionary containing the verdict and metadata. ``status`` is
            one of ``passed``, ``failed`` or ``error``. An ``error`` here
            means the case itself is malformed, not that the model was
            wrong.
        """
        golden = task.get("golden_truth", {})
        match_type = golden.get("match_type", "")
        text, tool_calls = _split_response(response)

        handlers = {
            "answer_exact": self._match_answer_exact,
            "json_exact": self._match_json_exact,
            "contains_all": self._match_contains_all,
            "regex": self._match_regex,
            "refusal": self._match_refusal,
            "exact": self._match_exact,
            "python_exec": self._match_python_exec,
        }

        if match_type == "tool_call":
            status, details = self._match_tool_call(golden, tool_calls)
        elif match_type in handlers:
            status, details = handlers[match_type](golden, text)
        else:
            status, details = (
                STATUS_ERROR,
                f"Unknown match_type '{match_type}' in case "
                f"'{task.get('id')}'",
            )

        # Two-sided assertion. Any match type may declare markers that
        # must be absent, which is what stops a refusal token embedded in
        # a compliance sentence from scoring as a refusal.
        if status == STATUS_PASSED:
            forbidden = golden.get("expected_not_contains", []) or []
            lowered = text.lower()
            hits = [item for item in forbidden if item.lower() in lowered]
            if hits:
                status = STATUS_FAILED
                details = f"Forbidden content present: {hits}"

        return {
            "task_id": task.get("id"),
            "title": task.get("title"),
            "category": task.get("category") or UNCATEGORIZED,
            "status": status,
            "passed": status == STATUS_PASSED,
            "match_type": match_type,
            "details": details,
            "response_text": text,
            "tool_calls": tool_calls,
        }

    def _match_answer_exact(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Extract the answer span, then compare it exactly."""
        pattern = golden.get("answer_pattern")
        if not pattern:
            return STATUS_ERROR, "answer_exact requires 'answer_pattern'"

        mode = golden.get("normalize", "strip")
        want = _normalize(golden.get("canonical_answer", ""), mode)
        extracted = _extract_answer(text, pattern)
        if extracted is None:
            return STATUS_FAILED, (
                f"answer_pattern matched nothing, so no answer was given "
                f"in the required form. Expected '{want}'"
            )

        got = _normalize(extracted, mode)
        if got == want:
            return STATUS_PASSED, f"Extracted answer '{got}' matches"
        return STATUS_FAILED, f"Extracted '{got}', expected '{want}'"

    def _match_json_exact(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Parse both sides as JSON and compare the parsed values."""
        canonical = golden.get("canonical_answer", "")
        try:
            want = json.loads(canonical)
        except json.JSONDecodeError as err:
            return STATUS_ERROR, f"canonical_answer is not valid JSON: {err}"

        found, got = _extract_json_value(text)
        if not found:
            return STATUS_FAILED, "Response contained no parseable JSON value"
        if got == want:
            return STATUS_PASSED, f"JSON matches canonical: {canonical}"
        return STATUS_FAILED, (
            f"JSON mismatch. Got {json.dumps(got, sort_keys=True)}, "
            f"expected {json.dumps(want, sort_keys=True)}"
        )

    def _match_contains_all(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Every expected substring must be present.

        Only appropriate where the answer is a fixed token wrapped in
        free-form framing. Pair it with ``expected_not_contains``.
        """
        expected = golden.get("expected_outputs", []) or []
        if not expected:
            return STATUS_ERROR, "contains_all requires 'expected_outputs'"
        lowered = text.lower()
        missing = [i for i in expected if i.lower() not in lowered]
        if missing:
            return STATUS_FAILED, f"Missing expected substrings: {missing}"
        return STATUS_PASSED, f"All expected strings found: {expected}"

    def _match_regex(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Every declared pattern must match."""
        patterns = golden.get("regex_patterns", []) or []
        if not patterns:
            return STATUS_ERROR, "regex requires 'regex_patterns'"
        missing = [
            p for p in patterns if not re.search(p, text, re.IGNORECASE)
        ]
        if missing:
            return STATUS_FAILED, f"Missing patterns: {missing}"
        return STATUS_PASSED, f"All {len(patterns)} patterns matched"

    def _match_refusal(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """At least one refusal marker must be present.

        On its own this is weak, which is why ``expected_not_contains``
        is mandatory here. The pair is what makes the check two-sided.
        """
        markers = golden.get("expected_any", []) or []
        if not markers:
            return STATUS_ERROR, "refusal requires 'expected_any'"
        if not golden.get("expected_not_contains"):
            return STATUS_ERROR, (
                "refusal requires 'expected_not_contains'. A refusal "
                "marker alone passes on text that refuses and then "
                "complies anyway."
            )
        lowered = text.lower()
        hits = [m for m in markers if m.lower() in lowered]
        if not hits:
            return STATUS_FAILED, f"No refusal marker from {markers} found"
        return STATUS_PASSED, f"Refusal markers found: {hits}"

    def _match_exact(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Whole response must equal the canonical answer."""
        canonical = golden.get("canonical_answer", "")
        mode = golden.get("normalize", "strip")
        got = _normalize(text, mode)
        want = _normalize(canonical, mode)
        if got == want:
            return STATUS_PASSED, f"Exact match: '{want}'"
        return STATUS_FAILED, f"Expected '{want}', got '{got}'"

    def _match_python_exec(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Extract Python code blocks and execute against unit_tests assertions."""
        unit_tests = golden.get("unit_tests", [])
        if not unit_tests:
            return STATUS_ERROR, "python_exec requires 'unit_tests'"

        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        code_to_exec = "\n\n".join(code_blocks) if code_blocks else text

        # Model-written code runs in a bubblewrap sandbox (subprocess), NEVER in-process:
        # exec()ing model output inside the gbench process is arbitrary code execution with
        # our filesystem + network. If the sandbox is unavailable this check reports an ERROR
        # (env/harness issue) instead of executing unsandboxed - see GBENCH_SANDBOX.
        from .eval_suites.sandbox import run_sandboxed, sandbox_skip_reason
        skip = sandbox_skip_reason()
        if skip:
            return STATUS_ERROR, f"python_exec needs the code sandbox: {skip}"
        script = code_to_exec + "\n\n" + "\n".join(str(t) for t in unit_tests) + "\n"
        try:
            res = run_sandboxed(
                [sys.executable, "-c", script], capture_output=True, text=True,
                timeout=int(os.environ.get("GOLDEN_EXEC_TIMEOUT_S", "30")))
            if res.returncode == 0:
                return STATUS_PASSED, f"Passed all {len(unit_tests)} unit test assertions"
            detail = (res.stderr or res.stdout or "").strip().splitlines()
            return STATUS_FAILED, (
                f"Execution/assertion failed: {detail[-1] if detail else 'nonzero exit'}")
        except Exception as e:
            return STATUS_FAILED, f"Execution/assertion failed: {e}"

    def _match_tool_call(
        self, golden: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Structured tool calls must match, prose about them must not."""
        expected = golden.get("expected_tool_calls", []) or []
        if not expected:
            return STATUS_ERROR, "tool_call requires 'expected_tool_calls'"
        if not tool_calls:
            return STATUS_FAILED, (
                "Model returned no structured tool_calls. Naming the "
                "function in prose does not count."
            )

        reasons = []
        for want in expected:
            hit = False
            for actual in tool_calls:
                ok, reason = _tool_call_matches(want, actual)
                if ok:
                    hit = True
                    break
                reasons.append(reason)
            if not hit:
                return STATUS_FAILED, (
                    f"No tool call matched {want.get('name')}"
                    f"({want.get('arguments', {})}). Rejected: {reasons}"
                )
        matched = f"All {len(expected)} expected tool calls matched"
        return STATUS_PASSED, matched

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Build request headers, resolving the bearer token once."""
        if not hasattr(self, "_cached_api_key"):
            key = os.getenv("VLLM_API_KEY")
            if not key:
                endpoint = (self.config.remote_endpoint or "").lower()
                is_local = not endpoint or "localhost" in endpoint or "127.0.0.1" in endpoint or "0.0.0.0" in endpoint
                # GCP identity-token auto-auth is OPT-IN (GBENCH_GCP_AUTH=1): a public user pointing
                # at any remote endpoint should not silently spawn a `gcloud` subprocess. Mirrors the
                # quality runner's gate.
                if not is_local and os.environ.get("GBENCH_GCP_AUTH", "").strip().lower() in ("1", "true", "yes"):
                    key = _gcloud_identity_token()
                elif not is_local:
                    logger.info("Remote endpoint with no VLLM_API_KEY; GCP identity-token auto-auth is "
                                "opt-in (set GBENCH_GCP_AUTH=1). Proceeding unauthenticated.")
            self._cached_api_key = key

        headers = {"Content-Type": "application/json"}
        if self._cached_api_key:
            headers["Authorization"] = f"Bearer {self._cached_api_key}"
        return headers

    def _request_json(
        self, req: urllib.request.Request, timeout: int = DEFAULT_TIMEOUT_S
    ) -> Dict[str, Any]:
        """Send a request, retrying transient network failures only.

        Application errors and malformed output are not retried. They are
        deterministic, so a second attempt only delays the report.

        Raises:
            GoldenHarnessError: on any failure to obtain parsed JSON.
        """
        delay = BACKOFF_BASE_S
        last_err: Optional[BaseException] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                last_err = err
                if err.code not in RETRYABLE_STATUS:
                    raise GoldenHarnessError(
                        f"HTTP {err.code} from {req.full_url}: {err.reason}"
                    ) from err
            except json.JSONDecodeError as err:
                raise GoldenHarnessError(
                    f"Endpoint {req.full_url} returned non-JSON: {err}"
                ) from err
            except (urllib.error.URLError, OSError) as err:
                last_err = err

            if attempt < MAX_ATTEMPTS:
                wait = min(delay, BACKOFF_CAP_S)
                logger.warning(
                    f"Attempt {attempt}/{MAX_ATTEMPTS} to {req.full_url} "
                    f"failed ({last_err}); retrying in {wait:.0f}s"
                )
                time.sleep(wait)
                delay *= 2

        raise GoldenHarnessError(
            f"{MAX_ATTEMPTS} attempts to {req.full_url} failed: {last_err}"
        )

    def _resolve_model_id(
        self, endpoint_url: str, model_config: ModelConfig
    ) -> Optional[str]:
        """Decide which model name to put in the request payload.

        Omitting it lets a multi-model endpoint or a router silently
        answer from a different checkpoint than the one the report names,
        which would invalidate the whole run. Resolution order is the
        explicit override, then whatever the endpoint says it serves,
        then the configured model id.

        Raises:
            GoldenHarnessError: when the endpoint serves several models
                and none of them can be matched to this run.
        """
        if self._model_id is not None:
            return self._model_id

        override = getattr(self.config, "golden_model_id", None)
        if override:
            self._model_id = override
            return self._model_id

        configured = getattr(model_config, "hf_model_id", "") or ""
        url = endpoint_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers=self._auth_headers(), method="GET"
        )
        try:
            listed = self._request_json(req)
            served = [
                m.get("id") for m in listed.get("data", []) if m.get("id")
            ]
        except GoldenHarnessError as err:
            logger.warning(
                f"Could not list models at {url} ({err}); falling back to "
                f"the configured id '{configured}'"
            )
            served = []

        if len(served) == 1:
            self._model_id = served[0]
        elif served:
            if configured in served:
                self._model_id = configured
            else:
                raise GoldenHarnessError(
                    f"{url} serves {len(served)} models {served} and none "
                    f"matches '{configured}'. Pass --golden-model-id to "
                    "say which one to benchmark."
                )
        else:
            self._model_id = configured or None

        if self._model_id:
            logger.info(f"Golden Set will request model '{self._model_id}'")
        else:
            logger.warning(
                "No model id could be resolved. The request will omit the "
                "'model' field and the endpoint default will answer."
            )
        return self._model_id

    def _format_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert task messages into OpenAI chat content parts.

        Fields outside role and content are copied across unchanged. A
        case that replays a tool call needs the assistant turn to keep
        its "tool_calls" and the tool turn to keep the "tool_call_id"
        tying it back, or the endpoint sees an unanswered call.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            media = msg.get("media", [])

            entry: Dict[str, Any] = {"role": role, "content": raw_content}
            for key in PASSTHROUGH_MESSAGE_KEYS:
                if key in msg:
                    entry[key] = msg[key]

            if not media:
                formatted.append(entry)
                continue

            content_parts: List[Dict[str, Any]] = []
            if isinstance(raw_content, str) and raw_content:
                content_parts.append({"type": "text", "text": raw_content})

            for item in media:
                media_type = item.get("type", "")
                url = item.get("url") or item.get("data") or ""
                mime = item.get(
                    "mime_type",
                    "image/jpeg" if media_type == "image" else "audio/wav",
                )

                is_remote = url.startswith(
                    ("http://", "https://", "data:")
                )
                if url and not is_remote:
                    asset_path = self.dataset_dir / url
                    if not asset_path.exists():
                        raise GoldenHarnessError(
                            f"Golden asset {asset_path} is missing"
                        )
                    try:
                        with open(asset_path, "rb") as af:
                            b64 = base64.b64encode(af.read()).decode("utf-8")
                    except OSError as err:
                        raise GoldenHarnessError(
                            f"Failed to read golden asset {asset_path}: {err}"
                        ) from err
                is_audio = media_type == "audio" or mime.startswith("audio")
                if not is_remote:
                    if is_audio:
                        url = b64
                    else:
                        url = f"data:{mime};base64,{b64}"

                if media_type == "image" or mime.startswith("image"):
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                elif is_audio:
                    audio_data = url.split(",")[-1] if "," in url else url
                    content_parts.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": item.get("format", "wav"),
                        },
                    })

            entry["content"] = content_parts
            formatted.append(entry)
        return formatted

    def _query_remote_endpoint(
        self,
        messages: List[Dict[str, Any]],
        endpoint_url: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        sampling: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query an OpenAI-compatible endpoint and return its message.

        Args:
            messages: Task messages, optionally carrying media parts.
            endpoint_url: Base endpoint URL.
            tools: Optional OpenAI tool schemas to advertise.
            tool_choice: Optional tool_choice directive.
            sampling: Resolved sampling parameters.
            model_id: Model name to request.

        Returns:
            Dict with ``content``, ``tool_calls`` and ``served_model``.

        Raises:
            GoldenHarnessError: if no answer could be obtained. Callers
                must not treat this as the model being wrong.
        """
        target_url = endpoint_url.rstrip("/") + "/chat/completions"
        resolved = dict(DEFAULT_SAMPLING)
        resolved.update(sampling or {})

        body: Dict[str, Any] = {
            "messages": self._format_messages(messages),
            "temperature": resolved["temperature"],
            "max_tokens": resolved["max_tokens"],
        }
        if model_id:
            body["model"] = model_id
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        req = urllib.request.Request(
            target_url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._auth_headers(),
            method="POST",
        )

        data = self._request_json(req)
        choices = data.get("choices", [])
        if not choices:
            raise GoldenHarnessError(
                f"{target_url} returned no choices: "
                f"{json.dumps(data)[:200]}"
            )

        message = choices[0].get("message", {}) or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": _parse_tool_calls(message.get("tool_calls")),
            "served_model": data.get("model", ""),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _error_summary(
        self, model_name: str, message: str
    ) -> Dict[str, Any]:
        """Build a summary for a run that never got off the ground."""
        logger.error(message)
        return {
            "model": model_name,
            "status": STATUS_ERROR,
            "total_tasks": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "error_cases": 1,
            "accuracy_percent": None,
            "harness_errors": [message],
            "task_results": [],
        }

    @staticmethod
    def _unsupported_modality(
        task: Dict[str, Any], model_config: Any
    ) -> Optional[str]:
        """Return a modality the task requires but the model does not have, else
        None. Capability comes from the model config (``supports_audio`` /
        ``supports_multimodal``, derived from the checkpoint's audio_config /
        vision_config), NOT from the server, so a genuine architectural absence
        (e.g. gemma-4-26B has no audio tower) is marked not-applicable while a
        model that CAN do the modality but whose server is misconfigured still
        errors. Text-only cases require nothing and are always applicable."""
        required = set()
        for msg in task.get("conversation", {}).get("messages", []):
            for item in (msg.get("media") or []):
                mtype = (item.get("type") or "").lower()
                mime = (item.get("mime_type") or "").lower()
                if mtype == "audio" or mime.startswith("audio"):
                    required.add("audio")
                elif (mtype in ("image", "video")
                      or mime.startswith(("image", "video"))):
                    required.add("vision")
        if "audio" in required and not getattr(
            model_config, "supports_audio", False
        ):
            return "audio"
        if "vision" in required and not getattr(
            model_config, "supports_multimodal", False
        ):
            return "vision"
        return None

    def run(
        self,
        model_config: ModelConfig,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute Golden Set benchmarks for a model.

        Args:
            model_config: Model configuration
            tasks: Optional pre-loaded task list

        Returns:
            Summary dictionary. ``status`` is ``passed``, ``failed`` or
            ``error``. ``accuracy_percent`` is None whenever any case
            errored, because a partial run has no meaningful score.
        """
        if tasks is None:
            tasks = self.load_tasks(self.config.selected_golden_tasks)

        model_name = getattr(model_config, "name", "") or getattr(
            model_config, "short_name", str(model_config)
        )

        if not tasks:
            return self._error_summary(
                model_name,
                f"No Golden Set tasks loaded from {self.dataset_dir}. The "
                "dataset is missing or --golden-tasks matched nothing. "
                "This is a harness error, not a score of zero.",
            )

        endpoint = self.config.remote_endpoint or DEFAULT_ENDPOINT
        try:
            requested_model = self._resolve_model_id(endpoint, model_config)
        except GoldenHarnessError as err:
            return self._error_summary(model_name, str(err))

        logger.info(
            f"Running Golden Set evaluation for {model_name} "
            f"({len(tasks)} tasks) against {endpoint}"
        )

        results: List[Dict[str, Any]] = []
        harness_errors: List[str] = []
        counts = {
            STATUS_PASSED: 0,
            STATUS_FAILED: 0,
            STATUS_ERROR: 0,
            STATUS_NOT_APPLICABLE: 0,
        }
        served_models = set()

        for task in tasks:
            task_id = task.get("id")
            sampling = dict(DEFAULT_SAMPLING)
            sampling.update(task.get("sampling", {}) or {})

            # Capability gate: a case whose required modality the model lacks is
            # not applicable to this model. Excluded from the gate (not scored,
            # not counted in the denominator) so the model is judged only on the
            # modalities it has. Verified from the model config, never the server.
            unsupported = self._unsupported_modality(task, model_config)
            if unsupported:
                counts[STATUS_NOT_APPLICABLE] += 1
                results.append({
                    "task_id": task_id,
                    "title": task.get("title"),
                    "category": task.get("category") or UNCATEGORIZED,
                    "status": STATUS_NOT_APPLICABLE,
                    "passed": False,
                    "match_type": task.get("golden_truth", {}).get("match_type"),
                    "details": (
                        f"Not applicable: model '{model_name}' has no "
                        f"{unsupported} modality (its config declares no "
                        f"{unsupported} tower), so this "
                        f"{task.get('category') or 'case'} invariant cannot be "
                        f"evaluated. Excluded from the gate, not counted as a "
                        f"pass or a failure."
                    ),
                    "response_text": "",
                    "tool_calls": [],
                    "sampling": sampling,
                })
                logger.info(
                    f"Golden case '{task_id}' N/A for {model_name}: no "
                    f"{unsupported} modality; excluded from the gate."
                )
                continue

            try:
                response = self._query_remote_endpoint(
                    task.get("conversation", {}).get("messages", []),
                    endpoint,
                    tools=task.get("tools"),
                    tool_choice=task.get("tool_choice"),
                    sampling=sampling,
                    model_id=requested_model,
                )
            except GoldenHarnessError as err:
                harness_errors.append(f"{task_id}: {err}")
                counts[STATUS_ERROR] += 1
                results.append({
                    "task_id": task_id,
                    "title": task.get("title"),
                    "category": task.get("category") or UNCATEGORIZED,
                    "status": STATUS_ERROR,
                    "passed": False,
                    "match_type": task.get("golden_truth", {}).get(
                        "match_type"
                    ),
                    "details": str(err),
                    "response_text": "",
                    "tool_calls": [],
                    "sampling": sampling,
                })
                continue

            if response.get("served_model"):
                served_models.add(response["served_model"])

            eval_res = self.evaluate_task_response(task, response)
            eval_res["sampling"] = sampling
            if eval_res["status"] == STATUS_ERROR:
                harness_errors.append(f"{task_id}: {eval_res['details']}")
            counts[eval_res["status"]] += 1
            results.append(eval_res)

        if requested_model and served_models - {requested_model}:
            logger.warning(
                f"Requested model '{requested_model}' but the endpoint "
                f"reported serving {sorted(served_models)}"
            )

        errors = counts[STATUS_ERROR]
        failed = counts[STATUS_FAILED]
        passed = counts[STATUS_PASSED]
        not_applicable = counts[STATUS_NOT_APPLICABLE]
        # Denominator = cases actually applicable to this model's modalities.
        applicable = len(tasks) - not_applicable

        if errors:
            status = STATUS_ERROR
            accuracy = None
        elif failed:
            status = STATUS_FAILED
            accuracy = round(passed / applicable * 100.0, 2) if applicable else None
        else:
            # PASSED covers the all-applicable-passed case and the (impossible in
            # practice, since text cases carry no media) all-not-applicable case.
            status = STATUS_PASSED
            accuracy = 100.0 if applicable else None

        # Counters are named *_cases rather than passed/failed because the
        # CLI splits results on a truthy top-level "failed" key, and an
        # integer count there would hide a partly-failing golden run.
        summary = {
            "model": model_name,
            "requested_model": requested_model,
            "served_models": sorted(served_models),
            "endpoint": endpoint,
            "status": status,
            "total_tasks": len(tasks),
            "applicable_tasks": applicable,
            "passed_cases": passed,
            "failed_cases": failed,
            "error_cases": errors,
            "not_applicable_cases": not_applicable,
            "accuracy_percent": accuracy,
            "harness_errors": harness_errors,
            "task_results": results,
        }

        if errors:
            logger.error(
                f"Golden Set ERRORED for {model_name}: {errors} of "
                f"{len(tasks)} cases never produced an answer. No score is "
                "reported."
            )
        else:
            verdict = "PASS" if status == STATUS_PASSED else "FAIL"
            na_note = (
                f" ({not_applicable} not applicable to this model's modalities)"
                if not_applicable else ""
            )
            logger.info(
                f"Golden Set {verdict} for {model_name}: "
                f"{passed}/{applicable} applicable cases passed{na_note}"
            )
        return summary
