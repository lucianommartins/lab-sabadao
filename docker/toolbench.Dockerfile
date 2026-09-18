# Canonical StableToolBench harness for gbench's `toolbench` suite, built LOCALLY (never pulled).
#
# Bundles StableToolBench (pinned) + the 236 MB response cache + the cached `/virtual` server, so
# `docker run` is self-contained. The entrypoint starts the server, runs the DFSDT loop against the
# model-under-test's served /v1 endpoint (backbone=chatgpt_function), converts the answer trees, and
# writes them to /out. The SoPR/SoWR JUDGE is NOT here - gbench scores /out with its Gemini cascade.
#
# LEAN DEPS: StableToolBench's requirements.txt pulls the full training stack (torch, transformers,
# vllm, deepspeed, bitsandbytes, peft, accelerate, langchain, gradio) for its LOCAL toolllama
# backbone - which we never use. deepspeed compiles CUDA at install and HANGS the build. We instead
# install only what the chatgpt_function inference + ToolEval convert + the cached server need
# (toolbench/utils.py imports torch+transformers at module load, so those two are required), and we
# STUB the 5 unused heavy backbones so their vllm/peft/etc. imports never load.
#
# Build:
#   docker build -t gbench-toolbench -f gbench/docker/toolbench.Dockerfile gbench/docker
FROM python:3.10-slim

ARG STB_REF=aa4ed9f4737ad98bd706663f01d63623c3427812
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONPATH=/stb

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/THUNLP-MT/StableToolBench /stb && \
    cd /stb && git checkout "${STB_REF}"
WORKDIR /stb

# torch (CPU) + transformers: required only because toolbench/utils.py does
# `import torch, transformers, transformers.models.llama.modeling_llama` at module load (the
# functions that USE them are toolllama-only and never run on the chatgpt_function path).
# NOTE: `--extra-index-url` (not `--index-url`) so PyPI stays available for build deps like
# flit_core; the `+cpu` local pin forces the small CPU wheel from the pytorch index.
RUN pip install --no-cache-dir torch==2.4.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
# Pin transformers to a stable release: unpinned pulls a bleeding-edge build whose
# integrations/accelerate.py references nn.Module without importing nn (NameError at import).
# 4.44.2 has models.llama.modeling_llama and imports cleanly on torch 2.4.1+cpu.
RUN pip install --no-cache-dir transformers==4.44.2 tiktoken sentencepiece
# Light runtime: chatgpt_function inference, ToolEval convert, and the cached FastAPI server.
RUN pip install --no-cache-dir \
        openai tenacity tqdm termcolor requests httpx "pydantic<2" PyYAML \
        fastapi "uvicorn[standard]" slowapi backoff shortuuid rich numpy huggingface_hub

# Stub the 5 unused backbones so `rapidapi_multithread.py`'s eager imports succeed WITHOUT vllm /
# peft / deepspeed / sentence_transformers. Only chatgpt_function (real, openai-based) runs.
RUN set -e; for pair in \
        "tool_llama_model:ToolLLaMA" "tool_llama_vllm_model:ToolLLaMA_vllm" \
        "tool_llama_lora_model:ToolLLaMALoRA" "davinci_model:Davinci" "retriever:ToolRetriever"; do \
      f="${pair%%:*}"; cls="${pair##*:}"; \
      printf 'class %s:\n    """gbench stub: unused backbone (chatgpt_function path only)."""\n    def __init__(self, *a, **k):\n        raise RuntimeError("%s backbone is stubbed in the gbench toolbench image; use backbone_model=chatgpt_function")\n' \
        "$cls" "$cls" > "/stb/toolbench/inference/LLM/${f}.py"; \
    done

# The 236 MB response cache -> server/tool_response_cache + server/tools (deterministic execution).
RUN curl -sSL -o /tmp/server_cache.zip \
        "https://huggingface.co/datasets/stabletoolbench/Cache/resolve/main/server_cache.zip" && \
    unzip -q /tmp/server_cache.zip -d /stb/server && rm /tmp/server_cache.zip && \
    ( [ -d /stb/server/tool_response_cache ] || \
      ( d=$(find /stb/server -maxdepth 2 -type d -name tool_response_cache | head -1); \
        [ -n "$d" ] && ln -s "$d" /stb/server/tool_response_cache ) ) && \
    ( [ -d /stb/server/tools ] || \
      ( d=$(find /stb/server -maxdepth 2 -type d -name tools | head -1); \
        [ -n "$d" ] && ln -s "$d" /stb/server/tools ) ) && true

# Patch StableToolBench's chat_completion_request: (1) route non-GPT models (our served
# google/gemma-4-...) through the OpenAI-compatible client instead of raising "Model not supported",
# and (2) drop the interactive `pdb.set_trace()` that hangs/kills a non-interactive container.
# Asserts each edit matched (build fails on upstream drift).
COPY toolbench_patch_llm.py /tmp/toolbench_patch_llm.py
RUN python /tmp/toolbench_patch_llm.py

# Patch ToolEval's convert_to_answer_format.py: guard unguarded message['content'] reads so an
# assistant turn with no content key (valid OpenAI schema; model-output dependent) degrades to an
# empty node instead of crashing the judge with KeyError: 'content' (an intermittent rc=1).
COPY toolbench_patch_convert.py /tmp/toolbench_patch_convert.py
RUN python /tmp/toolbench_patch_convert.py

COPY toolbench_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
