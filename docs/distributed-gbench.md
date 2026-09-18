<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 2D Cascading GPU & Multi-Zone Execution Plan: Scaling to 32-64 GPUs via On-Demand GKE

## 1. Problem Formulation & Objective

Evaluating open models (up to 31B parameters) across the 87 academic suites in `gbench` with Chain-of-Thought reasoning (`--thinking`) requires generating millions of output tokens. On a single node, evaluations bottleneck on serial queue depth:
- **`MMLU-Pro`** (12,032 questions): ~3.85 hours
- **`LiveCodeBench (LCB)`** (454 questions): ~2.28 hours
- **`Codeforces`** (422 questions): ~1.82 hours
- **Total Single-Instance Runtime**: **> 12 hours** per full evaluation sweep.

Using Google Cloud quota in project `YOUR_PROJECT_ID`, this plan establishes an automated **4-Tier, 13-Pool Multi-Zone On-Demand GKE cluster** with **zero idle GPU costs (`num-nodes=0`)** and **cascading failover**:
1. **Tier 0 (Blackwell GB200 / B200 - Unlimited & 256 Quota)**: `a4x-pool-us-central1-a` -> `a4x-pool-us-central1-b` -> `a4-pool-us-central1-b`
2. **Tier 1 (H100 - 4,800 Quota)**: `a3-pool-us-central1-a` -> `a3-pool-us-central1-b` -> `a3-pool-us-central1-c`
3. **Tier 2 (A100-40GB - 2,000 Quota)**: `a2-pool-us-central1-a` -> `a2-pool-us-central1-b` -> `a2-pool-us-central1-c` -> `a2-pool-us-central1-f`
4. **Tier 3 (L4 - 500 Quota)**: `l4-pool-us-central1-a` -> `l4-pool-us-central1-b` -> `l4-pool-us-central1-c` (Scales to 64 L4s / 32 workers)

When triggered, the cluster dynamically provisions **16 to 32 parallel workers (TP=2 x DP=2/4)** on the highest available GPU tier, finishes the full evaluation sweep in **< 45 minutes**, and immediately scales back to **0 nodes** when finished.

---

## 2. Complete 4-Tier, 13-Pool GPU Grid in `us-central1`

| Tier | Node Pool Name | Accelerator Hardware | Specs / Node | Storage Type Requirement | Quota in `us-central1` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 0** | **`a4x-pool-us-central1-a`** | **NVIDIA GB200** | 4x GB200 (192GB HBM3e) | `hyperdisk-balanced` | **Unlimited** |
| | **`a4x-pool-us-central1-b`** | **NVIDIA GB200** | 4x GB200 (192GB HBM3e) | `hyperdisk-balanced` | **Unlimited** |
| | **`a4-pool-us-central1-b`** | **NVIDIA B200** | 8x B200 (180GB HBM3e) | `hyperdisk-balanced` | **256 GPUs** |
| **Tier 1** | **`a3-pool-us-central1-a`** | **NVIDIA H100** | 8x H100 (80GB SXM5) | `pd-ssd` | **4,800 GPUs** |
| | **`a3-pool-us-central1-b`** | **NVIDIA H100** | 8x H100 (80GB SXM5) | `pd-ssd` | **4,800 GPUs** |
| | **`a3-pool-us-central1-c`** | **NVIDIA H100** | 8x H100 (80GB SXM5) | `pd-ssd` | **4,800 GPUs** |
| **Tier 2** | **`a2-pool-us-central1-a`** | **NVIDIA A100** | 8x A100 (40GB SXM4) | `pd-ssd` | **2,000 GPUs** |
| | **`a2-pool-us-central1-b`** | **NVIDIA A100** | 8x A100 (40GB SXM4) | `pd-ssd` | **2,000 GPUs** |
| | **`a2-pool-us-central1-c`** | **NVIDIA A100** | 8x A100 (40GB SXM4) | `pd-ssd` | **2,000 GPUs** |
| | **`a2-pool-us-central1-f`** | **NVIDIA A100** | 8x A100 (40GB SXM4) | `pd-ssd` | **2,000 GPUs** |
| **Tier 3** | **`l4-pool-us-central1-a`** | **NVIDIA L4** | 8x L4 (24GB Ada Lovelace) | `pd-ssd` | **500 GPUs** *(8 nodes / 64 L4s)* |
| | **`l4-pool-us-central1-b`** | **NVIDIA L4** | 8x L4 (24GB Ada Lovelace) | `pd-ssd` | **500 GPUs** *(8 nodes / 64 L4s)* |
| | **`l4-pool-us-central1-c`** | **NVIDIA L4** | 8x L4 (24GB Ada Lovelace) | `pd-ssd` | **500 GPUs** *(8 nodes / 64 L4s)* |

---

## 3. Step-by-Step Setup & Execution Guide

### Step 1: Provision Multi-Tier 0-Node Pools in `us-central1`

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export CLUSTER_NAME="gbench-eval-cluster"

# 1. Tier 0: Blackwell GB200 & B200 (Requires hyperdisk-balanced)
gcloud container node-pools create a4x-pool-us-central1-a \
    --project=$PROJECT_ID --cluster=$CLUSTER_NAME --region=$REGION \
    --node-locations=us-central1-a --machine-type=a4x-highgpu-4g \
    --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=latest \
    --num-nodes=0 --disk-type=hyperdisk-balanced --disk-size=500GB &

gcloud container node-pools create a4x-pool-us-central1-b \
    --project=$PROJECT_ID --cluster=$CLUSTER_NAME --region=$REGION \
    --node-locations=us-central1-b --machine-type=a4x-highgpu-4g \
    --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=latest \
    --num-nodes=0 --disk-type=hyperdisk-balanced --disk-size=500GB &

gcloud container node-pools create a4-pool-us-central1-b \
    --project=$PROJECT_ID --cluster=$CLUSTER_NAME --region=$REGION \
    --node-locations=us-central1-b --machine-type=a4-highgpu-8g \
    --accelerator=type=nvidia-b200,count=8,gpu-driver-version=latest \
    --num-nodes=0 --disk-type=hyperdisk-balanced --disk-size=500GB &

# 2. Tier 1: Hopper H100 (pd-ssd)
for ZONE in "us-central1-a" "us-central1-b" "us-central1-c"; do
  gcloud container node-pools create "a3-pool-${ZONE}" \
      --project=$PROJECT_ID --cluster=$CLUSTER_NAME --region=$REGION \
      --node-locations="$ZONE" --machine-type=a3-highgpu-8g \
      --accelerator=type=nvidia-h100-80gb,count=8,gpu-driver-version=latest \
      --num-nodes=0 --disk-size=500GB --disk-type=pd-ssd &
done

# 3. Tier 2: Ampere A100-40GB (pd-ssd)
for ZONE in "us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f"; do
  gcloud container node-pools create "a2-pool-${ZONE}" \
      --project=$PROJECT_ID --cluster=$CLUSTER_NAME --region=$REGION \
      --node-locations="$ZONE" --machine-type=a2-highgpu-8g \
      --accelerator=type=nvidia-tesla-a100,count=8,gpu-driver-version=latest \
      --num-nodes=0 --disk-size=500GB --disk-type=pd-ssd &
done

# 4. Tier 3: Ada Lovelace L4 (pd-ssd)
for ZONE in "us-central1-a" "us-central1-b" "us-central1-c"; do
  gcloud container node-pools create "l4-pool-${ZONE}" \
      --project=$PROJECT_ID --cluster=$CLUSTER_NAME --region=$REGION \
      --node-locations="$ZONE" --machine-type=g2-standard-96 \
      --accelerator=type=nvidia-l4,count=8,gpu-driver-version=latest \
      --num-nodes=0 --disk-size=500GB --disk-type=pd-ssd &
done

wait
echo "All 13 multi-tier, multi-zone pools created at 0 nodes ($0 cost)!"
```

---

### Step 2: Deploy the Kubernetes Manifest (`vllm-gke.yaml`)

Deploy [`vllm-gke.yaml`](../vllm-gke.yaml):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  annotations:
    networking.gke.io/load-balancer-type: "Internal"
spec:
  type: LoadBalancer
  selector:
    app: vllm
  ports:
  - name: http
    port: 8000
    targetPort: 8000

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 0 # Starts at 0 replicas ($0 GPU cost)
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      hostIPC: true # NVLink shared memory / multi-GPU IPC
      tolerations:
      - key: "nvidia.com/gpu" # Tolerates any NVIDIA GPU taint
        operator: "Exists"
        effect: "NoSchedule"
      - key: "kubernetes.io/arch" # Tolerates both arm64 (Grace Blackwell) and amd64 (x86_64) taints
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: vllm
        image: vllm/vllm-openai:nightly
        imagePullPolicy: IfNotPresent
        command:
        - "vllm"
        - "serve"
        args:
        - "$(MODEL_NAME)"
        - "--tensor-parallel-size"
        - "2"
        - "--data-parallel-size"
        - "2"
        - "--max-model-len"
        - "262144"
        - "--gpu-memory-utilization"
        - "0.95"
        - "--enable-chunked-prefill"
        - "--disable-custom-all-reduce"
        - "--reasoning-parser"
        - "gemma4"
        - "--enable-auto-tool-choice"
        - "--tool-call-parser"
        - "gemma4"
        - "--trust-remote-code"
        - "--hf-overrides"
        - '{"vision_config": {"default_output_length": 1120}}'
        - "--mm-processor-kwargs"
        - '{"max_soft_tokens": 1120}'
        - "--limit-mm-per-prompt"
        - '{"image": 8}'
        - "--port"
        - "8000"
        env:
        - name: MODEL_NAME
          value: "" # Injected dynamically at runtime by eval_cluster.sh
        - name: HF_HOME
          value: "/root/.cache/huggingface"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
              optional: true
        - name: NCCL_NVLS_ENABLE
          value: "0" # Disables NVLS multicast on GKE virtualized GPU instances
        - name: NCCL_DEBUG
          value: "INFO"
        - name: NCCL_SOCKET_IFNAME
          value: "eth0,lo"
        resources:
          limits:
            nvidia.com/gpu: "4"
            memory: "350Gi"
          requests:
            nvidia.com/gpu: "4"
            memory: "250Gi"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          failureThreshold: 30
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
        - name: hf-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 32Gi
      - name: hf-cache
        hostPath:
          path: /var/cache/huggingface # Container-Optimized OS writable host directory
          type: DirectoryOrCreate
```

Apply the manifest:
```bash
kubectl apply -f vllm-gke.yaml
```

---

### Step 3: Multi-Pod Parallel Tunneling & Threaded Load-Balancing Proxy

When running `gbench` from your workstation VM, `kubectl port-forward svc/...` defaults to a single pod. To achieve true 4x parallel load balancing across all 4 Blackwell nodes:

#### A. Tunnel Each Pod to a Local Port (`8001-8004`)
```bash
PODS=($(kubectl get pods -l app=vllm -o jsonpath='{.items[*].metadata.name}'))
PORT=8001
for pod in "${PODS[@]}"; do
  kubectl port-forward pod/$pod $PORT:8000 > /dev/null 2>&1 &
  ((PORT++))
done
```

#### B. Start the Multi-Threaded Round-Robin Proxy on Port 8000
```python
python3 -c '
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request, itertools, threading

ports = itertools.cycle([8001, 8002, 8003, 8004])
lock = threading.Lock()

class ThreadedProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self): self.forward()
    def do_POST(self): self.forward()
    def forward(self):
        with lock:
            target_port = next(ports)
        target = f"http://127.0.0.1:{target_port}{self.path}"
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else None
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        req = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req) as res:
                self.send_response(res.status)
                for k, v in res.headers.items(): self.send_header(k, v)
                self.end_headers()
                self.wfile.write(res.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

print("Multi-threaded proxy running on :8000 (Parallel load balancing across 8001-8004)")
ThreadingHTTPServer(("0.0.0.0", 8000), ThreadedProxyHandler).serve_forever()
' &
```

---

### Step 4: Run `gbench`

Launch the distributed evaluation sweep:
```bash
source .venv/bin/activate

nohup gbench --evals-only \
  --evals gpqa_diamond mmlu_pro aime \
  --tokenizer google/gemma-4-26B-A4B-it \
  --batch-sizes 512 --sandboxes 96 \
  --thinking \
  --max-output-tokens 65536 \
  --remote-endpoint http://127.0.0.1:8000/v1 \
  --results-dir ./results/evalcloud-think-4x \
  > /tmp/evalcloud-think-4x.log 2>&1 &
```

---

### Step 5: Automated Single-Command Execution (`eval_cluster.sh`)

Alternatively, execute the full end-to-end cascading allocation, evaluation, and auto-teardown using [`eval_cluster.sh`](../scripts/eval_cluster.sh):

```bash
cd gbench/scripts
./eval_cluster.sh google/gemma-4-26B-A4B-it "gpqa_diamond mmlu_pro aime" ./results/eval_sweep
```

**How it works automatically**:
1. Checks **Tier 0 (Blackwell)** -> **Tier 1 (H100)** -> **Tier 2 (A100-40GB)** -> **Tier 3 (L4)**.
2. Allocates nodes on the highest available pool (4 nodes for Blackwell/Hopper/Ampere, 8 nodes for L4).
3. Sets `MODEL_NAME` dynamically in the deployment and scales pods to match nodes.
4. Waits for all pods to report `1/1 Ready`.
5. Tunnels all pods and runs `gbench` at batch size 512.
6. Automatically scales down the active GPU pool and deployment back to 0 nodes on exit ($0 GPU cost).
