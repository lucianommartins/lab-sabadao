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

"""Endpoint verification utilities for checking liveness and multimodal support."""

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TINY_1X1_PNG_BASE64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def verify_endpoint_functional(base_url: str, timeout: int = 10) -> Tuple[bool, str, Optional[int]]:
    """Check if an OpenAI/vLLM endpoint is functional and answering requests.

    Args:
        base_url: Base URL of the endpoint (e.g. 'http://127.0.0.1:8000' or 'http://127.0.0.1:8000/v1')
        timeout: Timeout in seconds for the verification requests

    Returns:
        (is_functional, status_msg, max_model_len)
    """
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        models_url = f"{url}/models"
    else:
        models_url = f"{url}/v1/models"

    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                max_len = None
                if models and isinstance(models, list) and len(models) > 0 and "max_model_len" in models[0]:
                    max_len = int(models[0]["max_model_len"])
                return True, "Endpoint is functional and answering /v1/models requests", max_len
    except Exception as e1:
        # Fallback check /health
        try:
            health_url = f"{url}/health" if not url.endswith("/v1") else f"{url[:-3]}/health"
            with urllib.request.urlopen(health_url, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, "Endpoint is functional (answered /health)", None
        except Exception as e2:
            return False, f"Unreachable (/v1/models: {e1}; /health: {e2})", None
    return False, "Endpoint returned unexpected response format", None


def probe_multimodal_support(base_url: str, model_id: str, timeout: int = 10) -> bool:
    """Dynamically probe if an endpoint/model supports multimodal image inputs.

    Sends a 1x1 pixel Base64 PNG request to /v1/chat/completions with max_tokens=1.

    Capability is decided by HOW the endpoint responds, NOT by whether it responds
    within `timeout`. This is what lets MM work on slow machines: a genuinely
    text-only model REJECTS an image payload FAST with an HTTP 4xx (it validates
    the request before generating), whereas a real multimodal model ACCEPTS the
    payload and only then does the slow work (loading a cold vision encoder into
    VRAM, prefilling the image). On modest hardware that generation can take far
    longer than `timeout`, so a timeout here means "accepted, still generating" =
    SUPPORTED, not "unsupported". Decision table:
      - HTTP 200                          -> supported
      - HTTP 4xx (payload rejected)       -> unsupported
      - timeout waiting for generation    -> supported (accepted but slow)
      - HTTP 5xx / connection error       -> unsupported (can't confirm capability)

    Args:
        base_url: Base URL of the HTTP endpoint.
        model_id: Model ID or tag name.
        timeout: Seconds to wait for the first response before concluding the
            payload was accepted-but-slow (=> supported). A fast 4xx still returns
            unsupported regardless of this value.

    Returns:
        True if the endpoint accepts image payloads for this model, False otherwise.
    """
    url = base_url.rstrip("/")
    chat_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test"},
                    {"type": "image_url", "image_url": {"url": TINY_1X1_PNG_BASE64}},
                ],
            }
        ],
        "max_tokens": 1,
    }

    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        # The server actively REJECTED the image payload (e.g. 400/415/422 "this
        # model does not support images"). That is a definitive "unsupported". A
        # 5xx is a server-side failure, not a capability signal, so treat it the
        # same (don't hand image benchmarks to an endpoint we couldn't confirm).
        logger.info(f"Multimodal probe for {model_id} on {chat_url}: HTTP {e.code} "
                    f"-> unsupported")
        return False
    except (socket.timeout, TimeoutError) as e:
        # Timed out WAITING FOR GENERATION. A text-only model would have been
        # rejected fast (HTTPError above); reaching a generation timeout means the
        # payload was ACCEPTED and the model is producing tokens, just slowly (cold
        # vision encoder load on modest hardware). Treat as supported so MM is not
        # falsely skipped on slow machines.
        logger.info(f"Multimodal probe for {model_id} on {chat_url}: image payload "
                    f"accepted, still generating after {timeout}s (slow cold vision "
                    f"load) -> supported")
        return True
    except urllib.error.URLError as e:
        # URLError wraps either a socket timeout (accepted-but-slow => supported) or
        # a real transport failure (connection refused, DNS, etc. => can't confirm).
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            logger.info(f"Multimodal probe for {model_id} on {chat_url}: image "
                        f"payload accepted, still generating after {timeout}s (slow "
                        f"cold vision load) -> supported")
            return True
        logger.info(f"Multimodal probe for {model_id} on {chat_url} returned: {e} "
                    f"-> unsupported")
        return False
    except Exception as e:
        logger.info(f"Multimodal probe for {model_id} on {chat_url} returned: {e} "
                    f"-> unsupported")
        return False
