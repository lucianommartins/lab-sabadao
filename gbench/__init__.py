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

"""gbench - Open Model Performance Benchmark Suite.

A production-grade, modular benchmarking tool for comprehensive evaluation
of open LLMs via vLLM (HF/GGUF, text/embedding/multimodal).
"""

__version__ = "1.0.0"
__author__ = "Luciano Martins"

import os
import sys

# Process-wide Gemini API key multi-key support & auto-sanitization
_RAW_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
_GEMINI_KEYS = [k.strip() for k in _RAW_GEMINI_KEY.split(",") if k.strip()] if _RAW_GEMINI_KEY else []
if _GEMINI_KEYS:
    os.environ["GEMINI_API_KEYS"] = ",".join(_GEMINI_KEYS)
    # Ensure os.environ['GEMINI_API_KEY'] holds a single valid key for third-party libraries (LiteLLM, OpenAI SDK)
    os.environ["GEMINI_API_KEY"] = _GEMINI_KEYS[0]

# Auto-patch google.genai.Client globally so any eval or plugin can pass raw or multi-keys safely
try:
    from google import genai
    _orig_genai_client_init = genai.Client.__init__

    def _patched_genai_client_init(self, *args, **kwargs):
        if "api_key" in kwargs and kwargs["api_key"]:
            raw = str(kwargs["api_key"]).strip()
            if "," in raw:
                kwargs["api_key"] = raw.split(",")[0].strip()
        elif not kwargs.get("api_key"):
            raw = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).strip()
            if raw:
                kwargs["api_key"] = raw.split(",")[0].strip()
        _orig_genai_client_init(self, *args, **kwargs)

    genai.Client.__init__ = _patched_genai_client_init
except Exception:
    pass

