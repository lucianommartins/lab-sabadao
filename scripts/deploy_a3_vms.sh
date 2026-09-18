#!/usr/bin/env bash
# deploy_a3_vms.sh: Fast multi-region probe & launch for 4x A3 (32x H100) vLLM cluster
set -euo pipefail

usage() {
    echo "Usage: $0 --project PROJECT --model MODEL --vm-prefix VM_PREFIX"
    exit 1
}

PROJECT_ID=""
MODEL=""
VM_PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT_ID="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --vm-prefix) VM_PREFIX="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

if [[ -z "$PROJECT_ID" || -z "$MODEL" || -z "$VM_PREFIX" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Candidate zones (adjust to your project's regions / GPU quota)
CANDIDATE_ZONES=(
    "us-east4-b"
    "us-east4-a"
    "us-east4-c"
    "europe-west4-b"
    "europe-west4-c"
    "us-east7-b"
    "us-central1-b"
    "us-central1-a"
)

echo "=========================================================="
echo "Fast Multi-Zone A3 (32x H100) Cluster Provisioning"
echo "Project:   $PROJECT_ID"
echo "Model:     $MODEL"
echo "VM Prefix: $VM_PREFIX"
echo "Probing zones with available quota..."
echo "=========================================================="

ACTIVE_ZONE=""

for ZONE in "${CANDIDATE_ZONES[@]}"; do
    echo -n "Probing zone '$ZONE' for 4x a3-highgpu-8g machines... "
    
    # Attempt to create the 4 VMs in parallel
    FAILED=0
    for i in {1..4}; do
        gcloud compute instances create "${VM_PREFIX}-$i" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --machine-type=a3-highgpu-8g \
            --maintenance-policy=TERMINATE \
            --image-family=common-cu124-ubuntu-2204 \
            --image-project=deeplearning-platform-release \
            --boot-disk-size=500GB \
            --boot-disk-type=pd-ssd \
            --metadata=install-nvidia-driver=True \
            --quiet >/dev/null 2>&1 || { FAILED=1; break; }
    done

    if [ "$FAILED" -eq 0 ]; then
        echo "SUCCESS! 4x A3 instances created in $ZONE."
        ACTIVE_ZONE="$ZONE"
        break
    else
        echo "Out of stock / exhausted in $ZONE. Cleaning up and trying next zone..."
        for i in {1..4}; do
            gcloud compute instances delete "${VM_PREFIX}-$i" --zone="$ZONE" --project="$PROJECT_ID" --quiet >/dev/null 2>&1 || true
        done
    fi
done

if [ -z "$ACTIVE_ZONE" ]; then
    echo "Error: Could not allocate 4x A3 H100 machines across any candidate zones."
    exit 1
fi

echo ""
echo "=========================================================="
echo "Cluster allocated in: $ACTIVE_ZONE"
echo "Starting vLLM Docker containers across all 32 GPUs..."
echo "=========================================================="

# Fetch internal IPs
NODE_IPS=($(gcloud compute instances list --project="$PROJECT_ID" --filter="name ~ ${VM_PREFIX}" --format="value(networkInterfaces[0].networkIP)"))
echo "Node IPs: ${NODE_IPS[*]}"

# Launch vLLM container on each node
for i in {1..4}; do
    VM_NAME="${VM_PREFIX}-$i"
    echo "Launching vLLM on $VM_NAME..."
    gcloud compute ssh "$VM_NAME" --zone="$ACTIVE_ZONE" --project="$PROJECT_ID" --command="
        docker rm -f vllm 2>/dev/null || true
        docker run -d --name vllm --rm \
          --gpus all \
          --ipc=host \
          --network=host \
          -v /root/.cache/huggingface:/root/.cache/huggingface \
          -e HF_TOKEN=\"\${HF_TOKEN:-}\" \
          vllm/vllm-openai:nightly \
          \"$MODEL\" \
          --tensor-parallel-size 2 \
          --data-parallel-size 4 \
          --max-model-len 262144 \
          --gpu-memory-utilization 0.95 \
          --enable-chunked-prefill \
          --reasoning-parser gemma4 \
          --enable-auto-tool-choice \
          --tool-call-parser gemma4 \
          --trust-remote-code \
          --hf-overrides '{\"vision_config\": {\"default_output_length\": 1120}}' \
          --mm-processor-kwargs '{\"max_soft_tokens\": 1120}' \
          --limit-mm-per-prompt '{\"image\": 8}' \
          --port 8000
    " &
done
wait

echo "=========================================================="
echo "All 32x H100 GPUs are provisioning and serving $MODEL!"
echo "Primary Node IP: ${NODE_IPS[0]}"
echo "=========================================================="
