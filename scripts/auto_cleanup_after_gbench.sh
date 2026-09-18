#!/usr/bin/env bash
# auto_cleanup_after_gbench.sh: Watches running gbench and tears down cluster when done.

set -euo pipefail

LOG_FILE="/tmp/auto_cleanup.log"
exec >> "$LOG_FILE" 2>&1

usage() {
    echo "Usage: $0 --pid PID --cluster CLUSTER --node-pool POOL --region REGION --project PROJECT"
    exit 1
}

GBENCH_PID=""
CLUSTER_NAME=""
NODE_POOL=""
REGION=""
PROJECT_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) GBENCH_PID="$2"; shift 2;;
        --cluster) CLUSTER_NAME="$2"; shift 2;;
        --node-pool) NODE_POOL="$2"; shift 2;;
        --region) REGION="$2"; shift 2;;
        --project) PROJECT_ID="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown argument: $1"; usage;;
    esac
done

if [[ -z "$GBENCH_PID" || -z "$CLUSTER_NAME" || -z "$NODE_POOL" || -z "$REGION" || -z "$PROJECT_ID" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

echo "=========================================================="
echo "Starting gbench monitor at $(date)"
echo "Cluster:   $CLUSTER_NAME ($REGION) [Project: $PROJECT_ID]"
echo "Node Pool: $NODE_POOL"
echo "PID:       $GBENCH_PID"
echo "=========================================================="

# Loop until gbench PID and any gbench process is gone
while kill -0 "$GBENCH_PID" 2>/dev/null || pgrep -f 'gbench.*evals-only' >/dev/null 2>&1; do
    sleep 15
done

echo ""
echo "=========================================================="
echo "gbench execution has COMPLETED at $(date)"
echo "Waiting 15s for file writes to flush..."
echo "=========================================================="
sleep 15

echo "Executing cluster teardown..."

# 1. Kill proxy
echo ">> Killing async_proxy.py..."
pkill -f 'async_proxy.py' 2>/dev/null || true

# 2. Kill tunnels
echo ">> Killing kubectl port-forward tunnels..."
pkill -f 'kubectl.*port-forward' 2>/dev/null || true

# 3. Scale deployment to 0
echo ">> Scaling deployment/vllm-deployment to 0..."
kubectl scale deployment/vllm-deployment --replicas=0 2>/dev/null || true

# 4. Release GPU nodes
echo ">> Resizing GKE node pool $NODE_POOL to 0..."
gcloud container clusters resize "$CLUSTER_NAME" \
  --node-pool="$NODE_POOL" \
  --num-nodes=0 \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --quiet

echo ""
echo "=========================================================="
echo "ALL RESOURCES CLEANED UP AT $(date). \$0 GPU BILLING."
echo "=========================================================="
