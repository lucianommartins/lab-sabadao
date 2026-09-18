# gbench resources directory

## 1. Core gbench suite
* **gbench GitHub repository**: Source code, runners, Web UI dashboard, and Terraform infrastructure (`https://github.com/google-gemma/gbench`)

## 2. Supported serving backends
* **vLLM**: Production LLM and VLM serving runtime (`https://github.com/vllm-project/vllm` | `https://docs.vllm.ai/`)
* **SGLang**: Structured generation language and fast backend runtime (`https://github.com/sgl-project/sglang` | `https://sglang.ai/`)
* **TGI (Text Generation Inference)**: HuggingFace LLM serving toolkit (`https://github.com/huggingface/text-generation-inference`)
* **TensorRT-LLM**: NVIDIA tensor-optimized inference runtime (`https://github.com/NVIDIA/TensorRT-LLM`)
* **Ollama**: Local foundation model runner (`https://github.com/ollama/ollama` | `https://ollama.com/`)
* **Google Cloud Run**: Managed serverless GPU container hosting (`https://cloud.google.com/run`)

## 3. Native Academic & Domain Evaluation Suites (85 Built-in Suites)

> **Dataset licenses, gated access, and redistribution:** see
> [THIRD_PARTY_DATASETS.md](THIRD_PARTY_DATASETS.md). gbench pulls each dataset at runtime from its
> upstream source and redistributes none of it; some are gated (need an HF token) or carry
> non-commercial / copyleft / undeclared licenses. Running a benchmark is subject to that dataset's
> terms.

Dataset links below reflect the exact source each suite's loader reads. Suites with a
Setup Guide need extra tooling (Docker, gold data, or an opt-in) and hard-error
until it is provisioned.

### Pillar 1: General Knowledge & Scientific Reasoning
* **ARC-AGI**: Abstraction and Reasoning Corpus - Visual Pattern Induction (`https://huggingface.co/datasets/dataartist/arc-agi` | `https://arcprize.org/`)
* **CausalBench**: Causal Discovery & Counterfactual Logic (`https://huggingface.co/datasets/causal-nlp/corr2cause`)
* **CIMemories**: Context-Integrated Continual Memory Persistence (`https://huggingface.co/datasets/facebook/CIMemories`)
* **CyberSecEval**: Meta Cybersecurity Risks, Vulnerabilities & Exploits (`https://huggingface.co/datasets/walledai/CyberSecEval`)
* **CyberGym**: Cybersecurity Vulnerability & Exploit Reasoning (`https://huggingface.co/datasets/sunblaze-ucb/cybergym`)
* **FRAMES**: Factuality, Retrieval, and Multi-hop Entity Synthesis (`https://huggingface.co/datasets/google/frames-benchmark`)
* **GPQA**: Google-Proof Graduate-Level Science - full split (`https://huggingface.co/datasets/Idavidrein/gpqa`)
* **GPQA Diamond**: Google-Proof Graduate-Level Science - diamond split (`https://huggingface.co/datasets/Idavidrein/gpqa`)
* **HealthBench**: Clinical Medical Diagnostics & Pharmacological Reasoning (`https://huggingface.co/datasets/openai/healthbench`)
* **Humanity's Last Exam (HLE)**: Frontier Multi-Disciplinary Academic Benchmark (`https://huggingface.co/datasets/cais/hle` | `https://lastexam.ai/`)
* **I18N Translation**: High-Precision Machine Translation (WMT19) (`https://huggingface.co/datasets/wmt/wmt19`)
* **IFEval**: Instruction-Following Evaluation with Verifiable Constraints (`https://huggingface.co/datasets/google/IFEval`)
* **LAB-Bench**: Biology & Chemistry Experimental Reasoning (`https://huggingface.co/datasets/futurehouse/lab-bench`)
* **LiveBench**: Contamination-Free Continuously Updated Academic Benchmark (`https://huggingface.co/datasets/livebench/math`)
* **LMSYS Hard Non-Coding**: Multi-Turn Complex Human Prompt Following (WildBench) (`https://huggingface.co/datasets/WildEval/WildBench`)
* **MedXpertQA**: Expert Clinical Diagnostic Case Studies (`https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA`)
* **MMLU**: Massive Multitask Language Understanding - canonical split (`https://huggingface.co/datasets/cais/mmlu`)
* **MMLU-Pro**: Multi-discipline Professional Reasoning (14 Subjects, 10 Options) (`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`)
* **MMLU-Redux**: Error-Corrected Canonical MMLU Benchmark Split (`https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux`)
* **Multilingual MMLU**: Global Language Understanding across 14+ Languages (`https://huggingface.co/datasets/openai/MMMLU`)
* **SimpleQA**: Short-form Factual Precision & Hallucination Abstention (`https://huggingface.co/datasets/basicv8vc/SimpleQA`)
* **WMDP**: Weapons of Mass Destruction Proxy - CBRN Biosecurity & Cyber Defense (`https://huggingface.co/datasets/cais/wmdp`)

### Pillar 2: Mathematics & Proofs
* **AIME**: American Invitational Mathematics Examination (`https://huggingface.co/datasets/AI-MO/aimo-validation-aime`)
* **GSM8K**: Grade School Math 8K Reasoning (`https://huggingface.co/datasets/openai/gsm8k`)
* **HMMT**: Harvard-MIT Mathematics Tournament Competition Math (`https://huggingface.co/datasets/MathArena/hmmt_feb_2025`)
* **IMO-AnswerBench**: Olympiad-Level Open-Form Short-Answer Problems (`https://huggingface.co/datasets/OpenEvals/IMO-AnswerBench`)
* **AMC / AIME Combined**: Modern Competition Mathematics Matrix (`https://huggingface.co/datasets/AI-MO/aimo-validation-amc`)
* **PutnamBench**: Collegiate Mathematics Competition Proofs (`https://huggingface.co/datasets/amitayusht/PutnamBench`)
* **Putnam-Formal**: Formalized Lean/Isabelle Mathematical Proof Verification (`https://huggingface.co/datasets/amitayusht/PutnamBench`)

### Pillar 3: Coding & Algorithmic Design
* **ACEBench**: Function-Calling / Tool-Use (Normal+Special subset; agent category excluded) (`https://huggingface.co/datasets/oliveirabruno01/acebench`)
* **Aider Polyglot**: Multi-language SEARCH/REPLACE Code Editing (`https://github.com/Aider-AI/polyglot-benchmark`)
* **BigCodeBench**: Complex Function & 130+ Library-Centric Python Synthesis ([Setup Guide](evals/bigcodebench.md) | `https://huggingface.co/datasets/bigcode/bigcodebench`)
* **Codeforces**: Multi-tier Competitive Programming Evaluation (`https://huggingface.co/datasets/open-r1/codeforces`)
* **CopilotBench / SWE-bench Verified**: Autonomous Repository Patching ([Setup Guide](evals/copilot_bench_swe.md) | `https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified`)
* **CRUXEval**: Code Reasoning, Execution Tracing & Output Prediction (`https://huggingface.co/datasets/cruxeval-org/cruxeval`)
* **LiveCodeBench (LCB)**: Holistic Coding Benchmark with Execution (`https://huggingface.co/datasets/livecodebench/code_generation_lite`)
* **Multi-SWE-bench**: Multi-Repository & Cross-Project Issue Resolution ([Setup Guide](evals/multi_swe_bench.md) | `https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench`)
* **MultiPL-E**: Multi-Language Execution-Based Synthesis ([Setup Guide](evals/multipl_e.md) | `https://huggingface.co/datasets/nuprl/MultiPL-E`)
* **OJBench**: Online-Judge Competitive Programming (DMOJ sandbox) ([Setup Guide](evals/ojbench.md) | `https://huggingface.co/datasets/He-Ren/OJBench_testdata`)
* **SciCode**: Scientific Computing, Physics, Chemistry & Math Research Algorithms (`https://huggingface.co/datasets/SciCode1/SciCode`)
* **SWE-bench Live**: Live GitHub Issue Resolution on Fresh Commits ([Setup Guide](evals/swe_bench_live.md) | `https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live`)
* **SWE-bench Multilingual**: Cross-Language Enterprise Bug Resolution ([Setup Guide](evals/swe_bench_multilingual.md) | `https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual`)
* **SWE-bench Pro**: Scale AI Enterprise Software Engineering Benchmark ([Setup Guide](evals/swe_bench_pro.md) | `https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro`)
* **SWE-Lancer**: Commercial Real-World Software Engineering Tasks - **deregistered (roadmap-only, not runnable)**: blocked by upstream Expensify/App image drift, see [roadmap](evals_roadmap.md) (`https://huggingface.co/datasets/DCAgent2/swe-lancer`)

### Pillar 4: Long Context & Retrieval
* **AA-LCR**: Artificial Analysis Long-Context Reasoning (`https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR`)
* **BEAM 128k**: Long-Context Benchmark at 128k tokens (`https://huggingface.co/datasets/Mohammadta/BEAM`)
* **CULER**: Ultra Long-Context Document Understanding (LongBench-v2) (`https://huggingface.co/datasets/zai-org/LongBench-v2`)
* **Loft x ArXiv**: Multi-Document Long-Context Synthesis (LOFT) (`https://github.com/google-deepmind/loft`)
* **MRCR**: Multi-Round Context Retrieval at 131k context window (`https://huggingface.co/datasets/openai/mrcr`)
* **RULER**: 4k-128k Long-Context Evaluation Matrix (`https://huggingface.co/datasets/rayonlabs/ruler-all`)

### Pillar 5: Tool Use & Agentic Workflows
* **AgentDojo**: Security & Prompt Injection Defense for Tool-Calling Agents (`https://huggingface.co/datasets/ffuuugor/agentdojo-dump`)
* **API-Bank**: Multi-Level API Calling, Tool Retrieval & Response Synthesis (`https://huggingface.co/datasets/liminghao1630/API-Bank`)
* **BFCL (Berkeley Function Calling Leaderboard)**: 11 Execution & Parsing Tiers (`https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `https://gorilla.cs.berkeley.edu/leaderboard.html`)
* **BFCL v4 Agentic**: Multi-Turn Function Calling, Context & Hallucination Abstention (`https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard`)
* **BFCL v3 Live**: Live (real-world) single-turn function-calling slice (`https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard`)
* **BrowseComp**: Complex Multi-Step Web Browsing & Information Synthesis (`https://huggingface.co/datasets/smolagents/browse_comp`)
* **ComplexFuncBench**: THUDM Multi-Axis Complex Parameter Function Calling (`https://huggingface.co/datasets/THUDM/ComplexFuncBench`)
* **DeepSearchQA**: Multi-Step Deep Web Search & XOR-Protected Verification (`https://huggingface.co/datasets/xbench/DeepSearch-2510`)
* **GAIA**: General AI Assistants - Multi-Modal Complex Tool-Assisted Tasks ([Setup Guide](evals/gaia.md) | `https://huggingface.co/datasets/gaia-benchmark/GAIA`)
* **GAIA-2**: Meta Multi-Turn Stateful ARE Environment Benchmark (canonical `are-benchmark` harness; hard-errors until provisioned) ([Setup Guide](evals/gaia2.md) | `https://huggingface.co/datasets/meta-agents-research-environments/gaia2`)
* **GDPval**: Economic knowledge-work deliverables - pairwise win-rate vs the expert reference (`https://huggingface.co/datasets/openai/gdpval`)
* **Gorilla APIBench**: Real-World HuggingFace, TorchHub, and TensorHub API Invocation (`https://huggingface.co/datasets/gorilla-llm/APIBench`)
* **WebArena (formerly `lmarena_web_agent`)**: Sandboxed Interactive Web Application Agent - canonical harness ROADMAPPED; the suite is **deregistered (roadmap-only, not runnable)**, see [roadmap](evals_roadmap.md) (`https://github.com/web-arena-x/webarena`)
* **MCP-Atlas**: Model Context Protocol Multi-Server Tool Orchestration (`https://huggingface.co/datasets/ScaleAI/MCP-Atlas`)
* **MCP-Bench**: Standardized Model Context Protocol Server Evaluation ([Setup Guide](evals/mcp_bench.md) | `https://github.com/Accenture/mcp-bench`)
* **NESTFUL**: Nested Function Calling & Dependent Parameter DAG Evaluation (`https://huggingface.co/datasets/ibm-research/nestful`)
* **Nexus Function Calling**: Zero-Shot Function & Parameter Schema Selection (NexusRaven) (`https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation`)
* **SEAL-Tools**: Complex Multi-Step API Parameter Mapping & Validation (`https://huggingface.co/datasets/casey-martin/Seal-Tools`)
* **SkillsBench**: Multi-Skill Agentic Capability Evaluation (`https://huggingface.co/datasets/benchflow/skillsbench`)
* **Spider 2.0**: Enterprise Text-to-SQL & Multi-Database Analytic Queries ([Setup Guide](evals/spider2.md) | `https://huggingface.co/datasets/xlangai/spider2-lite`)
* **T-Eval**: Step-by-Step Tool Usage, Instruction Following & Plan Verification (`https://huggingface.co/datasets/lovesnowbest/T-Eval`)
* **TAU-2 Bench**: Multi-Domain Conversational Airline / Retail / Telecom Agent Policy ([Setup Guide](evals/tau2.md) | `https://github.com/sierra-research/tau2-bench`)
* **TAU-3 Bench**: Telecom Track of tau-bench - Stateful Tool Routing Workflows ([Setup Guide](evals/tau3.md) | `https://github.com/sierra-research/tau2-bench`)
* **TerminalBench**: Sandboxed Linux Bash Task Execution ([Setup Guide](evals/terminal_bench.md) | `https://huggingface.co/datasets/terminal-bench/terminal-bench-2-1`)
* **ToolBench**: Massive 16,000+ Real-World REST API Tool Calling (`https://huggingface.co/datasets/Yhyu13/ToolBench_toolllama_G123_dfs`)
* **WildClawBench**: Wild Multi-Turn Agentic Tool-Use Scenarios (`https://huggingface.co/datasets/internlm/WildClawBench`)

### Pillar 6: Multimodal Vision & Grounding
* **Bundled Detection**: MS-COCO Normalized Bounding Box Detection (`https://huggingface.co/datasets/detection-datasets/coco`)
* **ChartQA**: Plot and Chart Visual Reasoning & Quantitative Extraction (`https://huggingface.co/datasets/ahmed-masry/ChartQA`)
* **CharXiv**: Princeton Complex Academic Chart Reasoning & Scientific Plot Reading (`https://huggingface.co/datasets/princeton-nlp/CharXiv`)
* **COCO Caption**: Image Captioning and Complex Scene Description (CIDEr) ([Setup Guide](evals/coco_caption.md) | `https://huggingface.co/datasets/lmms-lab/COCO-Caption`)
* **DocVQA**: Document Visual Question Answering on Scans & PDF Forms (`https://huggingface.co/datasets/lmms-lab-encoder/DocVQA`)
* **InfographicVQA**: Visual Question Answering on Infographic Posters (`https://huggingface.co/datasets/mm-eval/InfographicVQA`)
* **MMMU Pro**: Multimodal Multidisciplinary Reasoning Benchmark (10-Option) (`https://huggingface.co/datasets/MMMU/MMMU_Pro`)
* **OmniDocBench v1.5**: Multimodal Document Parsing, Layout & LaTeX Formula Recognition (`https://huggingface.co/datasets/opendatalab/OmniDocBench`)
* **ScreenSpot V2**: Pixel-Level GUI Coordinate Grounding (`https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2`)
* **Semantic Keypoint Grounding**: Continuous 2D GUI Spatial Coordinate Localization (`https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2`)
* **TextVQA**: Scene OCR Visual Question Answering Benchmark (`https://huggingface.co/datasets/lmms-lab/textvqa`)
* **OSWorld Desktop**: Computer-Use GUI Action and Automation Loop - **deregistered (roadmap-only, not runnable)**: the graded harness needs /dev/kvm (nested virtualization), see [roadmap](evals_roadmap.md) (`https://github.com/xlang-ai/OSWorld`)

## 4. Agentic quality and capabilities (Tiers 2 and 3)
* **GemmaClaw / OpenClaw QA suite**: Autonomous multi-turn agent simulation harness (48+ scenarios, session recall, MCP plugin routing)
* **Golden set invariants**: Deterministic functional capability gates (16/16 PASS rules over `gbench/golden_dataset/`)

## 5. Gemma models and datasets
* **Gemma HuggingFace collection**: Official Google Gemma model weights and GGUF checkpoints (`https://huggingface.co/google`)
* **ShareGPT V3 dataset**: Multi-turn conversational prompt dataset for the chat-like workload campaign (`https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered`)
* **MLPerf Inference standard**: Standardized inference benchmark geometry (`https://mlcommons.org/benchmarks/inference/`)
