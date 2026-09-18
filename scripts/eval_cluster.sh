#!/usr/bin/env bash
# eval_cluster.sh: End-to-end distributed eval with cascading GPU failover and local async proxy
#
# Usage:
#   ./eval_cluster.sh MODEL EVALS --gpu GPU_TYPE [OPTIONS]
#
# Examples:
#   ./eval_cluster.sh google/gemma-4-26B-A4B-it "gpqa_diamond mmlu_pro aime" --gpu gb200
#   ./eval_cluster.sh google/gemma-4-26B-A4B-it "gpqa_diamond" --gpu h100 --tp 2 --dp 4
#   ./eval_cluster.sh google/gemma-4-26B-A4B-it "gpqa_diamond" --gpu gb200 --setup-only
#
# GPU types and optimal TP/DP for Gemma 4 26B-A4B (~52GB BF16):
#   gb200 (192GB, 4/node): TP=1 DP=4  nvidia.com/gpu=4
#   b200  (180GB, 8/node): TP=1 DP=8  nvidia.com/gpu=8
#   h100  (80GB,  8/node): TP=1 DP=8  nvidia.com/gpu=8
#   a100  (40GB,  8/node): TP=2 DP=4  nvidia.com/gpu=8
#   l4    (24GB,  8/node): TP=4 DP=2  nvidia.com/gpu=8
#
# All GPU resources are automatically released on exit (Ctrl+C safe).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="${PROXY_SCRIPT:-$SCRIPT_DIR/../../async_proxy.py}"
if [ ! -f "$PROXY_SCRIPT" ]; then
    PROXY_SCRIPT="$SCRIPT_DIR/../async_proxy.py"
fi
MANIFEST_PATH="${MANIFEST_PATH:-$SCRIPT_DIR/../../vllm-gke.yaml}"
if [ ! -f "$MANIFEST_PATH" ]; then
    MANIFEST_PATH="$SCRIPT_DIR/../vllm-gke.yaml"
fi

# --- Defaults ---
MODEL=""
EVAL_SUITE=""
GPU_TYPE=""
TP=""
DP=""
BATCH_SIZE=2048
THINKING=true
MAX_OUTPUT_TOKENS=65536
RESULTS_DIR=""
SANDBOXES=96
USER_NODES=""
USER_REGION=""
USER_TIMEOUT=""
USER_GPUS=""
SETUP_ONLY=false
CLI_POOLS=()

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --evals) EVAL_SUITE="$2"; shift 2;;
        --gpu) GPU_TYPE="$2"; shift 2;;
        --num-gpus|--gpus) USER_GPUS="$2"; shift 2;;
        --pool|--pools|--candidate-pools)
            IFS=',' read -ra POOL_ARR <<< "$2"
            for p in "${POOL_ARR[@]}"; do
                CLI_POOLS+=("$p")
            done
            shift 2;;
        --tp) TP="$2"; shift 2;;
        --dp) DP="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --sandboxes) SANDBOXES="$2"; shift 2;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --nodes) USER_NODES="$2"; shift 2;;
        --region) USER_REGION="$2"; shift 2;;
        --project) PROJECT_ID="$2"; shift 2;;
        --cluster) CLUSTER_NAME="$2"; shift 2;;
        --proxy-script) PROXY_SCRIPT="$2"; shift 2;;
        --manifest-path) MANIFEST_PATH="$2"; shift 2;;
        --timeout) USER_TIMEOUT="$2"; shift 2;;
        --no-thinking) THINKING=false; shift;;
        --setup-only) SETUP_ONLY=true; shift;;
        -*) echo "Unknown option: $1"; exit 1;;
        *) POSITIONAL_ARGS+=("$1"); shift;;
    esac
done

if [ ${#POSITIONAL_ARGS[@]} -ge 1 ]; then
    MODEL="${POSITIONAL_ARGS[0]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -ge 2 ]; then
    EVAL_SUITE="${POSITIONAL_ARGS[1]}"
fi

if [ -z "$MODEL" ]; then
    echo "ERROR: Model is required. Pass as first argument or via --model."
    echo "Example: ./eval_cluster.sh --model google/gemma-4-26B-A4B-it --gpu gb200"
    exit 1
fi

# Mandatory --pool validation
if [ ${#CLI_POOLS[@]} -eq 0 ]; then
    echo "ERROR: --pool is required. Specify the node pool(s) to use."
    echo "Example: ./eval_cluster.sh --model google/gemma-4-26B-A4B-it --gpu gb200 --num-gpus 4 --pool a4x-pool-us-east1-d"
    exit 1
fi
CANDIDATE_POOLS=("${CLI_POOLS[@]}")

# Mandatory --num-gpus validation
if [ -z "${USER_GPUS:-}" ]; then
    echo "ERROR: --num-gpus is required (e.g. 4 for gb200, 8 for a100/h100)."
    echo "Example: ./eval_cluster.sh --model google/gemma-4-26B-A4B-it --gpu a100 --num-gpus 8 --pool a2-spot-pool-multi"
    exit 1
fi
GPUS_PER_POD="$USER_GPUS"

# --- GPU hardware profiles ---
# Sets: DEFAULT_TP, DEFAULT_DP, TARGET_NODES
case "$GPU_TYPE" in
    gb200)
        DEFAULT_TP=1; DEFAULT_DP=4; TARGET_NODES=4
        ;;
    b200)
        DEFAULT_TP=1; DEFAULT_DP=8; TARGET_NODES=4
        ;;
    h100)
        DEFAULT_TP=1; DEFAULT_DP=8; TARGET_NODES=4
        ;;
    a100)
        DEFAULT_TP=2; DEFAULT_DP=4; TARGET_NODES=4
        ;;
    l4)
        DEFAULT_TP=4; DEFAULT_DP=2; TARGET_NODES=8
        ;;
    "")
        echo "ERROR: --gpu is required."
        echo ""
        echo "Available GPU types: gb200, b200, h100, a100, l4"
        echo "Example: ./eval_cluster.sh google/gemma-4-26B-A4B-it \"gpqa_diamond\" --gpu gb200 --pool a4x-pool-us-east1-d"
        exit 1
        ;;
    *)
        echo "ERROR: Unknown GPU type '$GPU_TYPE'. Choose: gb200, b200, h100, a100, l4"
        exit 1
        ;;
esac

TARGET_NODES="${USER_NODES:-$TARGET_NODES}"

TP="${TP:-$DEFAULT_TP}"
DP="${DP:-$DEFAULT_DP}"

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="results/eval_${GPU_TYPE}_$(date +%m%d_%H%M)"
fi

if [ -z "${PROJECT_ID:-}" ]; then
    echo "ERROR: --project is required."
    exit 1
fi
if [ -z "${CLUSTER_NAME:-}" ]; then
    echo "ERROR: --cluster is required."
    exit 1
fi
if [ -z "${USER_REGION:-}" ]; then
    echo "ERROR: --region is required."
    exit 1
fi
REGION="$USER_REGION"

echo "=========================================================="
echo "Distributed GPU Evaluation Sweep"
echo "=========================================================="
echo "  Cluster:     $CLUSTER_NAME ($REGION)"
echo "  Model:       $MODEL"
echo "  Evals:       $EVAL_SUITE"
echo "  GPU type:    $GPU_TYPE ($GPUS_PER_POD GPUs/pod)"
echo "  TP=$TP  DP=$DP  ($(( TP * DP )) GPUs, ${DP} engines/pod)"
echo "  Batch size:  $BATCH_SIZE"
echo "  Sandboxes:   $SANDBOXES"
echo "  Thinking:    $THINKING"
echo "  Max tokens:  $MAX_OUTPUT_TOKENS"
echo "  Results:     $RESULTS_DIR"
echo "  Setup only:  $SETUP_ONLY"
echo "=========================================================="

# Validate TP*DP == GPUS_PER_POD
if [ $((TP * DP)) -ne "$GPUS_PER_POD" ]; then
    echo "ERROR: TP($TP) * DP($DP) = $((TP * DP)), must equal $GPUS_PER_POD (GPUs per pod for $GPU_TYPE)."
    exit 1
fi

# --- Cleanup state ---
ACTIVE_POOL=""
TUNNEL_PIDS=()
PROXY_PID=""

cleanup() {
    echo ""
    echo "Cleaning up all resources..."

    # Kill proxy
    if [ -n "$PROXY_PID" ]; then
        kill "$PROXY_PID" 2>/dev/null || true
        echo "  Proxy (PID $PROXY_PID) killed"
    fi

    # Kill port-forward tunnels
    for pid in "${TUNNEL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -f 'kubectl.*port-forward' 2>/dev/null || true
    if [ ${#TUNNEL_PIDS[@]} -gt 0 ]; then
        echo "  ${#TUNNEL_PIDS[@]} port-forward tunnels killed"
    fi

    # Scale pods to 0
    kubectl scale deployment/vllm-deployment --replicas=0 2>/dev/null || true
    echo "  Deployment scaled to 0 replicas"

    # Scale node pool to 0
    if [ -n "$ACTIVE_POOL" ]; then
        gcloud container clusters resize "$CLUSTER_NAME" \
            --node-pool="$ACTIVE_POOL" \
            --num-nodes=0 \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --async --quiet 2>/dev/null || true
        echo "  Node pool '$ACTIVE_POOL' scaling to 0 (async)"
    fi

    echo "Cleanup complete. \$0 GPU billing."
}

# Only trap cleanup when NOT in setup-only mode
if [ "$SETUP_ONLY" = false ]; then
    trap cleanup EXIT
fi

# ============================================================
# 1. Ensure manifest is applied
# ============================================================
# Ensure hf-secret exists in cluster
if ! kubectl get secret hf-secret >/dev/null 2>&1 && [ -f "$HOME/.cache/huggingface/token" ]; then
    kubectl create secret generic hf-secret --from-literal=token="$(cat "$HOME/.cache/huggingface/token")" >/dev/null 2>&1 || true
fi

echo "Applying manifest: $MANIFEST_PATH"
kubectl apply -f "$MANIFEST_PATH"

# ============================================================
# 2. Configure GPU pool autoscaling
# ============================================================
echo ""
echo "Configuring GPU pool autoscaling for $GPU_TYPE..."

for POOL in "${CANDIDATE_POOLS[@]}"; do
    if ! gcloud container node-pools describe "$POOL" --cluster="$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo ">> Pool '$POOL' does not exist, skipping."
        continue
    fi

    echo ">> Configuring autoscaling on '$POOL' (total-max-nodes: $TARGET_NODES, location-policy: ANY)..."
    gcloud container node-pools update "$POOL" \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --enable-autoscaling \
        --total-min-nodes=0 \
        --total-max-nodes="$TARGET_NODES" \
        --location-policy=ANY \
        --quiet 2>&1 || true

    ACTIVE_POOL="$POOL"
    break
done

if [ -z "$ACTIVE_POOL" ]; then
    echo "ERROR: No valid GPU pool found for $GPU_TYPE."
    exit 1
fi

# ============================================================
# 3. Configure model, TP/DP, GPU count and scale deployment
# ============================================================
echo ""
echo "=========================================================="
echo "Configuring: $MODEL (TP=$TP, DP=$DP, $GPUS_PER_POD GPUs/pod) on $TARGET_NODES nodes"
echo "=========================================================="

# Patch GPU resource requests to match --num-gpus
kubectl patch deployment vllm-deployment --type='json' -p="[
  {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/limits/nvidia.com~1gpu\", \"value\": \"$GPUS_PER_POD\"},
  {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/requests/nvidia.com~1gpu\", \"value\": \"$GPUS_PER_POD\"}
]"

# Pin deployment scheduling strictly to the allocated GPU node pool
kubectl patch deployment vllm-deployment -p "{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"cloud.google.com/gke-nodepool\":\"$ACTIVE_POOL\"}}}}}"

kubectl set env deployment/vllm-deployment \
    MODEL_NAME="$MODEL" \
    TENSOR_PARALLEL_SIZE="$TP" \
    DATA_PARALLEL_SIZE="$DP"

kubectl scale deployment/vllm-deployment --replicas="$TARGET_NODES"

ROLLOUT_TIMEOUT="${USER_TIMEOUT:-$(( 600 + TARGET_NODES * 45 ))}"
kubectl patch deployment vllm-deployment -p "{\"spec\":{\"progressDeadlineSeconds\":$ROLLOUT_TIMEOUT}}" >/dev/null 2>&1 || true

echo "Waiting up to ${ROLLOUT_TIMEOUT}s for all $TARGET_NODES pods to load model and compile CUDA graphs..."
if ! kubectl rollout status deployment/vllm-deployment --timeout="${ROLLOUT_TIMEOUT}s"; then
    echo "ERROR: Deployment rollout failed."
    exit 1
fi

# ============================================================
# 4. Tunnel each pod + start async proxy
# ============================================================
echo ""
echo "Setting up local proxy tunnels..."

PODS=($(kubectl get pods -l app=vllm -o jsonpath='{.items[*].metadata.name}'))
BACKEND_PORTS=()
PORT=8001
TUNNELS_PER_POD="${DP:-4}"

for pod in "${PODS[@]}"; do
    for t in $(seq 1 "$TUNNELS_PER_POD"); do
        (while true; do
            kubectl port-forward "pod/$pod" "$PORT:8000" >/dev/null 2>&1
            sleep 3
        done) &
        TUNNEL_PIDS+=($!)
        BACKEND_PORTS+=($PORT)
        echo "  $pod -> 127.0.0.1:$PORT (tunnel $t/$TUNNELS_PER_POD)"
        ((PORT++))
        sleep 0.05
    done
done

# Wait for tunnels to establish (scales with tunnel count)
TUNNEL_WAIT=$(( 3 + TARGET_NODES / 5 ))
sleep "$TUNNEL_WAIT"

# Start async proxy
BACKENDS_CSV=$(IFS=,; echo "${BACKEND_PORTS[*]}")
source "${GBENCH_VENV:-.venv}/bin/activate"

python3 "$PROXY_SCRIPT" --port 8000 --backends "$BACKENDS_CSV" &
PROXY_PID=$!
sleep 2

echo "Proxy live: :8000 -> [$BACKENDS_CSV]"

# Verify connectivity (scales with node count)
HEALTH_ATTEMPTS=$(( 12 + TARGET_NODES * 2 ))
echo -n "Health check (up to $(( HEALTH_ATTEMPTS * 5 ))s)..."
for attempt in $(seq 1 "$HEALTH_ATTEMPTS"); do
    if curl -sf --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo " OK"
        break
    fi
    if [ "$attempt" -eq "$HEALTH_ATTEMPTS" ]; then
        echo " FAILED after $(( HEALTH_ATTEMPTS * 5 ))s"
        echo "ERROR: Cannot reach vLLM through proxy."
        exit 1
    fi
    sleep 5
    echo -n "."
done

# ============================================================
# 5. Setup-only mode: print status and exit (leave proxy running)
# ============================================================
if [ "$SETUP_ONLY" = true ]; then
    echo ""
    echo "=========================================================="
    echo "SETUP COMPLETE - proxy running on http://127.0.0.1:8000/v1"
    echo "=========================================================="
    echo ""
    echo "Pods:"
    kubectl get pods -l app=vllm -o wide
    echo ""
    echo "To run gbench manually:"
    echo "  gbench --evals-only --evals ... --remote-endpoint http://127.0.0.1:8000/v1 ..."
    echo ""
    echo "To tear down when done:"
    echo "  kill $PROXY_PID  # proxy"
    for pid in "${TUNNEL_PIDS[@]}"; do
        echo "  kill $pid  # tunnel"
    done
    echo "  kubectl scale deployment/vllm-deployment --replicas=0"
    echo "  gcloud container clusters resize $CLUSTER_NAME --node-pool=$ACTIVE_POOL --num-nodes=0 --region=$REGION --project=$PROJECT_ID --quiet"
    exit 0
fi

# ============================================================
# 6. Run gbench
# ============================================================
EVAL_SUITE="${EVAL_SUITE:-all}"
echo ""
echo "=========================================================="
echo "Launching gbench"
echo "=========================================================="

GBENCH_ARGS=(
    --evals-only
    --evals $EVAL_SUITE
    --tokenizer "$MODEL"
    --batch-sizes "$BATCH_SIZE"
    --sandboxes "$SANDBOXES"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --remote-endpoint "http://127.0.0.1:8000/v1"
    --results-dir "$RESULTS_DIR"
)

if [ "$THINKING" = true ]; then
    GBENCH_ARGS+=(--thinking)
fi

echo "Command: gbench ${GBENCH_ARGS[*]}"
echo ""

gbench "${GBENCH_ARGS[@]}"

echo ""
echo "=========================================================="
echo "Evaluation complete!"
echo "Results: $RESULTS_DIR"
echo "=========================================================="
# Cleanup runs automatically via EXIT trap
