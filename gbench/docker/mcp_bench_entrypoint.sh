#!/usr/bin/env bash
# Entrypoint for the gbench-mcp-bench image. Wires the model-under-test at the gbench /v1
# endpoint (openai_compatible provider), writes any provided server API keys into the file the
# MCP-Bench loader reads, and runs the gbench launcher (which injects the Gemini cascade judge,
# filters tasks to the provisioned servers, and delegates to upstream main()).
#
# Required env (docker run -e): MCP_ENDPOINT (gbench /v1), MCP_MODEL (served model name),
#                               GEMINI_API_KEY (the cascade judge).
# Optional: GOOGLE_MAPS_API_KEY NCI_API_KEY HF_TOKEN NPS_API_KEY NASA_API_KEY,
#           MCP_DISTRACTION_COUNT, MCP_DISABLE_STABILITY, MCP_NO_SUBSET_FILTER.
set -uo pipefail

MCPBENCH_DIR="${MCPBENCH_DIR:-/app/mcp-bench}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
cd "$MCPBENCH_DIR"

# --- model-under-test: hijack the llama-3-1-8b openai_compatible slot -> gbench /v1 ---
: "${MCP_ENDPOINT:?MCP_ENDPOINT (gbench /v1 base url) is required}"
: "${MCP_MODEL:?MCP_MODEL (served model name) is required}"
export LLAMA_3_1_8B_API_KEY="${MCP_ENDPOINT_API_KEY:-dummy}"
export LLAMA_3_1_8B_BASE_URL="$MCP_ENDPOINT"
export LLAMA_3_1_8B_MODEL="$MCP_MODEL"
export MCP_MODEL_CONFIG_NAME="llama-3-1-8b"

# --- server API keys: the loader reads mcp_servers/api_key (KEY = VALUE lines) ---
KEYFILE="$MCPBENCH_DIR/mcp_servers/api_key"
: > "$KEYFILE"
for k in GOOGLE_MAPS_API_KEY NCI_API_KEY HF_TOKEN NPS_API_KEY NASA_API_KEY; do
  v="${!k:-}"
  if [ -n "$v" ]; then echo "$k = $v" >> "$KEYFILE"; fi
done

# --- judge (cascade) needs GEMINI_API_KEY; fail fast if absent ---
: "${GEMINI_API_KEY:?GEMINI_API_KEY is required (powers the gbench Gemini cascade judge)}"

exec python "$ADAPTER_DIR/mcp_bench_run.py"
