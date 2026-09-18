#!/usr/bin/env python3
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
"""gbench build-time patch for StableToolBench's chat_completion_request (chatgpt_function backbone).

Two upstream defects make the served-endpoint (non-GPT) path unusable, so the DFSDT loop never
produces an answer tree and the container dies rc=1:

  1. `if model.startswith("gpt"): ... else: raise NotImplementedError("Model not supported")` rejects
     any model whose id does not start with "gpt" (e.g. `google/gemma-4-26B-A4B-it`). gbench always
     targets an OpenAI-COMPATIBLE served /v1 endpoint (base_url is set), so route ANY model through
     the OpenAI client.
  2. `import pdb;  pdb.set_trace()` in the except handler drops a NON-INTERACTIVE container into pdb,
     which reads EOF -> BdbQuit -> the retry wrapper reports RetryError and the whole run dies.
     Remove it (keep the error-dict return).

Each replacement is asserted to match EXACTLY, so the build FAILS if upstream drifts rather than
silently shipping an unpatched image. Mirrors docker/ojbench_patch_judger.py.
"""

PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

# --- Patch 1: the model gate -> always use the OpenAI-compatible client. ---
GATE_OLD = (
    '        if model.startswith("gpt"):\n'
    "            client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)\n"
    "        else:\n"
    '            raise NotImplementedError("Model not supported")\n'
)
GATE_NEW = (
    "        # gbench: route ANY model through the OpenAI-compatible client (we always target a\n"
    '        # served /v1 endpoint via base_url); upstream hard-gated on model.startswith("gpt").\n'
    "        client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)\n"
)
assert GATE_OLD in src, "toolbench patch: model-gate block not found (upstream drifted?)"
src = src.replace(GATE_OLD, GATE_NEW, 1)

# --- Patch 2: remove the interactive pdb drop in the except handler. ---
PDB_OLD = '        import pdb;  pdb.set_trace()\n        return {"error": str(e), "total_tokens": 0}\n'
PDB_NEW = '        return {"error": str(e), "total_tokens": 0}\n'
assert PDB_OLD in src, "toolbench patch: pdb.set_trace block not found (upstream drifted?)"
src = src.replace(PDB_OLD, PDB_NEW, 1)

# Guard: no interactive pdb drop survives anywhere in the file.
assert "pdb.set_trace" not in src, "toolbench patch: a pdb.set_trace survived"

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("toolbench_patch_llm: patched", PATH)
