#!/usr/bin/env bash
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

# ==============================================================================
# run_benchmarks.sh - Orchestrate all gbench benchmarks across 8× H100-80GB
#
# Usage:
#   nohup ./run_benchmarks.sh &> run_benchmarks.log &
#   tail -f run_benchmarks.log          # watch overall progress
#   tail -f logs/phase1_job_A.log       # watch a specific job
#
# Tiers (uniform GPU allocation within each tier):
#   Tier Small  (TP=1): edge/small/mid models - 4 parallel jobs (GPUs 0-3)
#   Tier Medium (TP=2): large-tier models - parallel jobs (GPUs 4-7 then 0-1)
#   Tier Large  (TP=8): xlarge-tier models - sequential (all 8 GPUs)
#   Tier XL     (TP=8): xxl-tier models - sequential (all 8 GPUs)
#
# Model assignments are defined in the MODELS section below.
# Edit those arrays to add/remove models from each phase.
# ==============================================================================
set -uo pipefail

# ── Model assignments per job ─────────────────────────────────────────────────
# Edit these arrays to control which models run in each phase/job.
# Use short_name values from the built-in registry, or HuggingFace model IDs
# (e.g. "Qwen/Qwen3-32B") for ad-hoc models - they auto-register at runtime.

# Phase 1a: TP=1 (4 jobs, GPUs 0-3) - ≤20B models
PHASE1A_JOB_A=(gemma-3-1b-it   gemma-4-E2B-it)
PHASE1A_JOB_B=(gemma-3-4b-it   gemma-4-E4B-it)
PHASE1A_JOB_C=(gemma-3-12b-it)

# Phase 1a: TP=2 (2 jobs, GPUs 4-7) - 21-80B models
PHASE1A_JOB_E=(gemma-3-27b-it    gemma-4-31B-it)
PHASE1A_JOB_F=(gemma-4-26B-A4B-it)

# Phase 1b: TP=2 - (unused, available for ad-hoc models)
PHASE1B_JOB_G=()

# Phase 2: TP=8 (sequential, GPUs 0-7) - >80B models
PHASE2_MODELS=()

# Phase 3: TP=8 (sequential, GPUs 0-7) - >160B models
PHASE3_MODELS=()

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${VENV:-${GBENCH_DIR}/.venv}"

TIMESTAMP="$(date +%m-%d-%H%M%S)"
RESULTS_DIR="${GBENCH_DIR}/results/${TIMESTAMP}"
LOG_DIR="${GBENCH_DIR}/logs/${TIMESTAMP}"

PRESET="default"
FORMAT="hf"
GPU_MEM_UTIL="0.90"

# Redirect caches away from / and /tmp (which may be small)
CACHE_DIR="${XDG_CACHE_HOME:-${GBENCH_DIR}/.cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_DIR}/flashinfer}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_DIR}/triton}"
export XDG_CACHE_HOME="${CACHE_DIR}"
export TMPDIR="${TMPDIR:-${CACHE_DIR}/tmp}"
mkdir -p "${FLASHINFER_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

# ── Setup ──────────────────────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# Activate venv if present
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

# Trap to report on exit
cleanup() {
    local exit_code=$?
    echo ""
    echo "========================================================================"
    if [ $exit_code -eq 0 ]; then
        echo "✅ ALL PHASES COMPLETE - $(date)"
    else
        echo "❌ SCRIPT EXITED WITH ERROR (code=$exit_code) - $(date)"
    fi
    echo "   Results: ${RESULTS_DIR}"
    echo "   Logs:    ${LOG_DIR}"
    echo "========================================================================"
}
trap cleanup EXIT

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── Helper: run a gbench job in the background ────────────────────────────────
# Usage: run_job <job_name> <gpus> <tp_size> <port> <model1> [model2] ...
run_job() {
    local job_name="$1"
    local gpus="$2"
    local tp_size="$3"
    local port="$4"
    shift 4
    local models=("$@")

    local job_log="${LOG_DIR}/${job_name}.log"

    log "  Starting ${job_name}: GPUs=${gpus} TP=${tp_size} port=${port} models=[${models[*]}]"

    (
        export CUDA_VISIBLE_DEVICES="${gpus}"
        export GBENCH_SERVER_PORT="${port}"
        gbench \
            --format "${FORMAT}" \
            --preset "${PRESET}" \
            --models "${models[@]}" \
            --tensor-parallel "${tp_size}" \
            --num-gpus "${tp_size}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --results-dir "${RESULTS_DIR}" \
            --num-iterations 2 \
            --no-skip-existing \
            2>&1
    ) > "${job_log}" 2>&1 &

    # Store PID in a variable named after the job
    eval "PID_${job_name}=$!"
    log "  ${job_name} PID=$! → ${job_log}"
}

# ── Helper: wait for a set of jobs, report status ─────────────────────────────
wait_for_jobs() {
    local all_ok=true

    for job_name in "$@"; do
        local pid_var="PID_${job_name}"
        local pid="${!pid_var}"
        local job_log="${LOG_DIR}/${job_name}.log"

        local exit_code=0
        wait "${pid}" 2>/dev/null || exit_code=$?

        if [ "${exit_code}" -eq 0 ]; then
            log "  ${job_name} (PID=${pid}) completed successfully"
        else
            log "  ${job_name} (PID=${pid}) failed with exit code ${exit_code}"
            log "     Last 5 lines: $(tail -5 "${job_log}" 2>/dev/null || echo '(no log)')"
            log "     Full log: ${job_log}"
            all_ok=false
        fi
    done

    if [ "${all_ok}" = false ]; then
        log "  Some jobs failed - continuing to next phase anyway"
    fi
}

# ==============================================================================
# PHASE 1a: TP=1 (GPUs 0-3) + TP=2 (GPUs 4-7)   [all run in parallel]
# ==============================================================================
log "========================================================================"
log "PHASE 1a: Small (TP=1) + Medium (TP=2) models"
log "  Results → ${RESULTS_DIR}"
log "  Logs    → ${LOG_DIR}"
log "========================================================================"

# TP=1 jobs - ≤20B models (ports 8000-8002)
PHASE1A_JOBS=()
[ ${#PHASE1A_JOB_A[@]} -gt 0 ] && { run_job "phase1a_tp1_A" "0" 1 8000 "${PHASE1A_JOB_A[@]}"; PHASE1A_JOBS+=(phase1a_tp1_A); }
[ ${#PHASE1A_JOB_B[@]} -gt 0 ] && { run_job "phase1a_tp1_B" "1" 1 8001 "${PHASE1A_JOB_B[@]}"; PHASE1A_JOBS+=(phase1a_tp1_B); }
[ ${#PHASE1A_JOB_C[@]} -gt 0 ] && { run_job "phase1a_tp1_C" "2" 1 8002 "${PHASE1A_JOB_C[@]}"; PHASE1A_JOBS+=(phase1a_tp1_C); }

# TP=2 jobs - 21-80B models (ports 8004-8005)
[ ${#PHASE1A_JOB_E[@]} -gt 0 ] && { run_job "phase1a_tp2_E" "4,5" 2 8004 "${PHASE1A_JOB_E[@]}"; PHASE1A_JOBS+=(phase1a_tp2_E); }
[ ${#PHASE1A_JOB_F[@]} -gt 0 ] && { run_job "phase1a_tp2_F" "6,7" 2 8005 "${PHASE1A_JOB_F[@]}"; PHASE1A_JOBS+=(phase1a_tp2_F); }

log ""
log "Phase 1a: ${#PHASE1A_JOBS[@]} jobs running in parallel... waiting for completion"
wait_for_jobs "${PHASE1A_JOBS[@]}"
log ""
log "Phase 1a complete - $(date)"

# GPU cooldown
log "Waiting 30s for GPU memory reclaim..."
sleep 30

# ==============================================================================
# PHASE 1b: TP=2 (remaining large-tier models using GPUs 0-1)
# ==============================================================================
log "========================================================================"
log "PHASE 1b: Remaining medium (TP=2) models"
log "========================================================================"

if [ ${#PHASE1B_JOB_G[@]} -gt 0 ]; then
    run_job "phase1b_tp2_G" "0,1" 2 8000  "${PHASE1B_JOB_G[@]}"
    log ""
    log "Phase 1b: 1 job running... waiting for completion"
    wait_for_jobs phase1b_tp2_G
    log ""
    log "Phase 1b complete - $(date)"
    log "Waiting 30s for GPU memory reclaim..."
    sleep 30
else
    log "Phase 1b: no models configured, skipping"
fi

# ==============================================================================
# PHASE 2: xlarge-tier models (TP=8) - sequential, full node
# ==============================================================================
log "========================================================================"
log "PHASE 2: xlarge-tier models (TP=8) - sequential, full node"
log "========================================================================"

phase2_idx=0
for model in "${PHASE2_MODELS[@]}"; do
    local_label="phase2_job_${phase2_idx}"
    run_job "${local_label}" "0,1,2,3,4,5,6,7" 8 8000  "${model}"
    log ""
    log "Phase 2 [${model}]: running... waiting for completion"
    wait_for_jobs "${local_label}"
    log "Phase 2 [${model}] complete - $(date)"

    log "Waiting 30s for GPU memory reclaim..."
    sleep 30
    phase2_idx=$((phase2_idx + 1))
done

# ==============================================================================
# PHASE 3: xxl-tier model (TP=8) - full node
# ==============================================================================
log "========================================================================"
log "PHASE 3: xxl-tier model (TP=8) - full node"
log "========================================================================"

phase3_idx=0
for model in "${PHASE3_MODELS[@]}"; do
    local_label="phase3_job_${phase3_idx}"
    run_job "${local_label}" "0,1,2,3,4,5,6,7" 8 8000  "${model}"
    log ""
    log "Phase 3 [${model}]: running... waiting for completion"
    wait_for_jobs "${local_label}"
    log "Phase 3 [${model}] complete - $(date)"
    phase3_idx=$((phase3_idx + 1))
done

# ==============================================================================
# SUMMARY
# ==============================================================================
log ""
log "========================================================================"
log "ALL BENCHMARKS COMPLETE"
log "========================================================================"
log "  Results: ${RESULTS_DIR}"
log "  Logs:    ${LOG_DIR}"
log ""
log "  Per-job logs:"
for f in "${LOG_DIR}"/*.log; do
    local_name="$(basename "$f")"
    # Count pass/fail from each log
    passed=$(grep -c "PASSED\|Results saved" "$f" 2>/dev/null || echo "?")
    failed=$(grep -c "FAILED\|Error" "$f" 2>/dev/null || echo "?")
    log "    ${local_name}: ~${passed} passed, ~${failed} issues"
done
log ""
log "  Total result files: $(find "${RESULTS_DIR}" -name '*.json' | wc -l)"
log "========================================================================"
