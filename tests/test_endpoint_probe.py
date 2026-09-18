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

"""Multimodal capability probe regression tests.

The probe must decide capability by HOW the endpoint responds, not by whether it
responds within the timeout. This is what lets multimodal benchmarks run on slow
machines: a real MM model with a cold vision encoder can take far longer than the
probe timeout to produce its first token, so a generation timeout must read as
SUPPORTED (accepted the image payload), while a fast HTTP 4xx (text-only model
rejecting the payload) must read as UNSUPPORTED.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gbench.utils.endpoint import probe_multimodal_support


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _make_handler(status, delay=0.0, body=None):
    payload = json.dumps(body or {"choices": [{"message": {"content": "x"}}]}).encode()

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if delay:
                time.sleep(delay)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # silence the test server
            pass

    return _H


def test_slow_accept_is_supported():
    """A real MM model that accepts the payload but is slow to generate (cold
    vision load) must NOT be falsely skipped: a generation timeout => supported."""
    srv, port = _serve(_make_handler(200, delay=5.0))
    try:
        # timeout (2s) < server stall (5s): times out mid-generation.
        assert probe_multimodal_support(f"http://127.0.0.1:{port}/v1", "slow-mm", timeout=2) is True
    finally:
        srv.shutdown()


def test_fast_reject_is_unsupported():
    """A text-only model rejects an image payload fast with an HTTP 4xx."""
    srv, port = _serve(_make_handler(400, body={"error": "no image support"}))
    try:
        assert probe_multimodal_support(f"http://127.0.0.1:{port}/v1", "text-only", timeout=2) is False
    finally:
        srv.shutdown()


def test_server_error_is_unsupported():
    """A 5xx is a server failure, not a capability signal - don't force MM on it."""
    srv, port = _serve(_make_handler(500, body={"error": "boom"}))
    try:
        assert probe_multimodal_support(f"http://127.0.0.1:{port}/v1", "broken", timeout=2) is False
    finally:
        srv.shutdown()


def test_ok_is_supported():
    """A fast HTTP 200 is unambiguously supported."""
    srv, port = _serve(_make_handler(200))
    try:
        assert probe_multimodal_support(f"http://127.0.0.1:{port}/v1", "fast-mm", timeout=2) is True
    finally:
        srv.shutdown()


def test_unreachable_is_unsupported():
    """A connection failure can't confirm capability => unsupported."""
    assert probe_multimodal_support("http://127.0.0.1:1/v1", "dead", timeout=2) is False
