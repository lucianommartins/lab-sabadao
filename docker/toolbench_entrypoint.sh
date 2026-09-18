#!/usr/bin/env bash
# gbench toolbench container entrypoint: start the cached /virtual server, run StableToolBench's
# DFSDT inference against the model-under-test's served /v1 endpoint, convert the answer trees, and
# write them to /out. The SoPR/SoWR judge runs in gbench (Gemini cascade) after this exits.
#
# Required env: TB_MODEL (served model id), TB_ENDPOINT (served /v1 URL), GEMINI_API_KEY.
# Optional:     TB_GROUPS (default all 6), TB_METHOD (DFS_woFilter_w2), TB_NUM_THREAD (4),
#               TB_SIMULATOR_BASE / TB_SIMULATOR_MODEL / TB_SIMULATOR_KEY (cache-miss simulator).
set -uo pipefail

: "${TB_MODEL:?TB_MODEL (served model id) is required}"
: "${TB_ENDPOINT:?TB_ENDPOINT (served /v1 URL) is required}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY is required (cache-miss simulator)}"
TB_GROUPS="${TB_GROUPS:-G1_instruction G1_category G1_tool G2_category G2_instruction G3_instruction}"
TB_METHOD="${TB_METHOD:-DFS_woFilter_w2}"
TB_NUM_THREAD="${TB_NUM_THREAD:-4}"
# Per-task DFS SEARCH budget (canonical StableToolBench default 200). This is NOT a task count -
# TB_LIMIT (below) caps the number of test instances by slicing the query file.
TB_MAX_QUERY_COUNT="${TB_MAX_QUERY_COUNT:-200}"
TB_SIMULATOR_BASE="${TB_SIMULATOR_BASE:-https://generativelanguage.googleapis.com/v1beta/openai/}"
TB_SIMULATOR_MODEL="${TB_SIMULATOR_MODEL:-gemini-2.5-flash}"
TB_SIMULATOR_KEY="${TB_SIMULATOR_KEY:-$GEMINI_API_KEY}"
# Host port for the cached /virtual server. The runner picks a free host port (--network host) and
# passes it here so concurrent same-host runs do not collide on a fixed 8080.
TB_SERVER_PORT="${TB_SERVER_PORT:-8080}"

# Cached server config: point the miss-simulator at Gemini's OpenAI-compatible endpoint so the whole
# harness needs only GEMINI_API_KEY (no OpenAI). With the full cache, misses on the solvable set are
# rare, so the simulator is seldom hit.
cat > /stb/server/config.yml <<EOF
api_key: ${TB_SIMULATOR_KEY}
api_base: ${TB_SIMULATOR_BASE}
model: ${TB_SIMULATOR_MODEL}
temperature: 0
toolbench_url: http://127.0.0.1:0/rapidapi
tools_folder: "./tools"
cache_folder: "./tool_response_cache"
is_save: false
port: ${TB_SERVER_PORT}
log_file: "./server.log"
EOF

echo "[toolbench] starting cached /virtual server on port ${TB_SERVER_PORT}..."
( cd /stb/server && python main.py >/stb/server/server.stdout 2>&1 & )
for i in $(seq 1 90); do
  curl -sf "http://localhost:${TB_SERVER_PORT}/" >/dev/null 2>&1 && break
  # any HTTP reply (even 404) means it is up
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${TB_SERVER_PORT}/" 2>/dev/null || echo 000)
  [ "$code" != "000" ] && break
  sleep 2
done
echo "[toolbench] server probe done (http=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${TB_SERVER_PORT}/ 2>/dev/null))"

export OPENAI_API_BASE="$TB_ENDPOINT"
export OPENAI_BASE_URL="$TB_ENDPOINT"
export OPENAI_API_KEY="EMPTY"
export SERVICE_URL="http://localhost:${TB_SERVER_PORT}/virtual"
# /stb for `toolbench.*`; the inference dir for its repo-local `Tree` package (upstream relies on
# Python auto-adding the script's dir - make it explicit so imports resolve either way).
export PYTHONPATH=/stb:/stb/toolbench/inference

mkdir -p /work/answers /work/converted /out
rc=0
for g in $TB_GROUPS; do
  qfile="/stb/solvable_queries/test_instruction/${g}.json"
  if [ ! -f "$qfile" ]; then echo "[toolbench] MISSING query file: $qfile" >&2; rc=1; continue; fi
  # TB_LIMIT caps the NUMBER OF TASKS (test instances) per group by slicing the query file to the
  # first N. --max_query_count is the per-task DFS SEARCH budget, NOT a task count (it ran the full
  # 163-task group even at 1), so it cannot limit instances; slicing the JSON list is the only way.
  # Unset/blank TB_LIMIT -> run the full canonical group.
  runfile="$qfile"; nsel="full group"
  if [ -n "${TB_LIMIT:-}" ]; then
    runfile="/work/queries_${g}.json"; nsel="first ${TB_LIMIT} tasks"
    python - "$qfile" "$runfile" "$TB_LIMIT" <<'PYSLICE'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump(json.load(open(src))[:max(0, n)], open(dst, "w"))
PYSLICE
  fi
  echo "[toolbench] inference group ${g} (${nsel})..."
  # The pinned StableToolBench qa_pipeline has NO --test_ids arg (passing it is an argparse-fatal
  # "unrecognized arguments" -> rc=2). It also reads the served endpoint from --base_url (rapidapi*.py
  # has its OPENAI_API_BASE env read commented out and uses self.args.base_url, default
  # api.openai.com) -- so the endpoint MUST be passed as a flag, else every model call silently goes
  # to OpenAI instead of the gbench-served /v1.
  python /stb/toolbench/inference/qa_pipeline_multithread.py \
      --backbone_model chatgpt_function --chatgpt_model "$TB_MODEL" --openai_key EMPTY \
      --base_url "$TB_ENDPOINT" \
      --max_observation_length 1024 --method "$TB_METHOD" \
      --input_query_file "$runfile" --output_answer_file "/work/answers/${g}" \
      --tool_root_dir /stb/server/tools \
      --single_chain_max_step 50 --max_query_count "$TB_MAX_QUERY_COUNT" --num_thread "$TB_NUM_THREAD" \
      || rc=$?
done

echo "[toolbench] converting answer trees..."
# The pinned convert_to_answer_format.py takes FLAGS (--answer_dir/--method/--output), converts ONE
# answer dir at a time, and only reads files whose name contains the method. Inference wrote per-group
# dirs of {qid}_{method}.json, so convert per group into /work/converted/<group>.json. (Passing only
# --method, as before, is argparse-fatal: --answer_dir is required -> rc=2.)
for g in $TB_GROUPS; do
  [ -d "/work/answers/${g}" ] || continue
  python /stb/toolbench/tooleval/convert_to_answer_format.py \
      --answer_dir "/work/answers/${g}" --method "$TB_METHOD" \
      --output "/work/converted/${g}.json" || rc=$?
done

mkdir -p /out/candidate /out/reference
cp -r /work/converted/. /out/candidate/ 2>/dev/null || true
# Reference converted answers for SoWR, if the checkout ships them.
if [ -d /stb/data/model_predictions_converted ]; then
  cp -r /stb/data/model_predictions_converted/. /out/reference/ 2>/dev/null || true
fi
echo "[toolbench] done (rc=$rc)"
exit "$rc"
