#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# OpenAI-compatible cascade-judge PROXY for the gbench wildclawbench container. Runs INSIDE the
# gbench-wildclawbench orchestrator image and speaks the OpenAI Chat Completions wire protocol on
# an HTTP port. WildClawBench's per-task grading code (`automated_checks` -> `grade()`) constructs
# its OWN judge prompts and calls, VERBATIM, through the stock OpenAI SDK:
#
#     client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=os.environ["OPENROUTER_BASE_URL"])
#     client.chat.completions.create(model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"), ...)
#
# The gbench harness points OPENROUTER_BASE_URL at THIS proxy (via the docker bridge gateway, so it
# is reachable from the sibling task containers). The proxy preserves the upstream judge PROCEDURE
# and PROMPT completely - it just swaps the model call underneath: instead of OpenRouter's
# `openai/gpt-5.4`, it runs gbench's ESTABLISHED Gemini cascade (the same model list, rounds and
# backoff as base.judge_generate_cascade), reached through Gemini's OpenAI-compatible endpoint. The
# client's requested `model` is ignored; `temperature` is pinned to 0.0 (deterministic judging).
# Every other field the client sends (messages, max_tokens, response_format, ...) is forwarded
# unchanged, so structured-output requests and multimodal image content pass through verbatim.
#
# Because the judge model is Gemini (not the canonical openai/gpt-5.4 the WildClawBench leaderboard
# uses), a run scored this way is never leaderboard_comparable - by design, for consistency with
# every other gbench judged suite (toolbench, complexfuncbench, mcp_bench).
#
# Dependency-light on purpose: stdlib only (http.server + urllib), so it runs regardless of the
# image's `openai`/httpx versions. GEMINI_API_KEY stays in the orchestrator; it is never injected
# into the task containers.

import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("wildclawbench.cascade_judge")

# gbench's default judge cascade (kept in sync with base._DEFAULT_JUDGE_CASCADE / the mcp_bench
# port). Overridable by the SAME env knobs gbench uses so one setting moves every implementation.
_DEFAULT_JUDGE_CASCADE = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                          "gemini-3-flash-preview", "gemini-2.5-flash"]


def _cascade():
    """Mirror base.judge_cascade(): GBENCH_JUDGE_MODELS (csv) > GBENCH_JUDGE_MODEL (single) > default."""
    raw = (os.environ.get("GBENCH_JUDGE_MODELS") or "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    single = (os.environ.get("GBENCH_JUDGE_MODEL") or "").strip()
    if single:
        return [single]
    return list(_DEFAULT_JUDGE_CASCADE)


def _rounds():
    return max(1, int(os.environ.get("GBENCH_JUDGE_CASCADE_ROUNDS", "3")))


def _backoff():
    return float(os.environ.get("GBENCH_JUDGE_BACKOFF", "1.0"))


def _gemini_base():
    return os.environ.get(
        "GEMINI_OPENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")


def cascade_chat_completion(body):
    """Run one judge request through the Gemini cascade. `body` is the client's parsed request.

    Returns the raw parsed Gemini ChatCompletion dict of the first non-empty response. Raises
    RuntimeError on total outage (all models, all rounds), matching base.judge_generate_cascade.
    Structurally identical to the cascade: try every model each round; only back off (burst)
    BETWEEN rounds after a whole-cascade pass fails; temperature pinned 0.0.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set - the wildclawbench Gemini cascade judge "
                           "cannot run.")
    cascade, rounds, backoff = _cascade(), _rounds(), _backoff()
    # Per-request timeout must stay well under upstream's 120s docker-exec grading cap
    # (src/utils/grading.py wraps the WHOLE grade() - all its judge calls - in timeout=120), or a
    # single hung request kills grade() and scores the task 0 with no fallback recorded.
    req_timeout = float(os.environ.get("WILDCLAW_JUDGE_REQUEST_TIMEOUT_S", "40"))
    url = _gemini_base() + "/chat/completions"
    # Forward the client's request verbatim, overriding only the model (per cascade attempt) and
    # temperature (deterministic judging). Drop provider-specific / OpenRouter-only fields Gemini's
    # OpenAI-compat endpoint rejects (incl. `thinking`/`reasoning` from tasks' extra_body).
    _DROP = ("model", "temperature", "stream", "provider", "transforms", "route", "models",
             "thinking", "reasoning", "reasoning_effort")
    base_payload = {k: v for k, v in body.items() if k not in _DROP}
    base_payload["temperature"] = 0.0
    # Sanitize tools: Gemini's OpenAI-compat endpoint only knows standard `type:function` tools.
    # Tasks that pass OpenRouter's server-side plugin (e.g. {"type":"openrouter:web_search"}) would
    # 400 the whole call; drop non-function tools (and tool_choice if none survive) so the judge
    # still runs (that task's web-verification simply isn't reproducible on a non-web Gemini judge -
    # already leaderboard_comparable=False).
    tools = base_payload.get("tools")
    if isinstance(tools, list):
        kept = [t for t in tools if isinstance(t, dict) and t.get("type") == "function"]
        if kept:
            base_payload["tools"] = kept
        else:
            base_payload.pop("tools", None)
            base_payload.pop("tool_choice", None)
    last_err = None
    for rnd in range(rounds):
        for model in cascade:
            payload = dict(base_payload, model=model)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                choices = parsed.get("choices") or []
                content = (choices[0].get("message", {}).get("content")
                           if choices else None)
                if content is not None and str(content).strip() != "":
                    return parsed
                last_err = RuntimeError(f"empty content from {model}")
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8")[:300]
                except Exception:
                    pass
                last_err = RuntimeError(f"{model} HTTP {e.code}: {detail}")
                logger.debug("wildclawbench judge model %s failed: %s", model, last_err)
            except Exception as e:  # network/timeout/parse -> next model
                last_err = e
                logger.debug("wildclawbench judge model %s failed: %s", model, e)
        if rnd < rounds - 1:
            time.sleep(backoff * (2 ** rnd) + random.uniform(0, backoff))
    raise RuntimeError(f"JUDGE_OUTAGE: all Gemini cascade models failed ({last_err})")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default access logging
        pass

    def _send_json(self, code, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        # Lightweight health / models endpoints so clients that probe them succeed.
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(200, {"object": "list", "data": [
                {"id": "gbench-cascade", "object": "model", "owned_by": "gbench"}]})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as e:
            self._send_json(400, {"error": {"message": f"bad request: {e}"}})
            return
        try:
            parsed = cascade_chat_completion(body)
            self._send_json(200, parsed)
        except Exception as e:
            # Surface as 503 so the client's create() raises; the upstream grade() then retries and
            # (per its own design) may regex-fallback. The harness records that as a judge fallback.
            self._send_json(503, {"error": {"message": str(e), "type": "judge_outage"}})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    host = os.environ.get("WILDCLAW_JUDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("WILDCLAW_JUDGE_PORT", "18790"))
    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info("wildclawbench cascade judge proxy listening on %s:%d (cascade=%s, rounds=%d)",
                host, port, ",".join(_cascade()), _rounds())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
