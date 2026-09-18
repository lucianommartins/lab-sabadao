# agent_dojo

Canonical AgentDojo (ethz-spylab/agentdojo): prompt-injection security for **tool-calling
agents**. The agent runs real multi-step tool episodes across 4 task suites (workspace, travel,
banking, slack); attacker-controlled tool outputs carry injected instructions. gbench **delegates
to the agentdojo package's own suites, attacks and scorers**; it does not reimplement them.

## What it measures

For each suite, gbench runs both:
- **benign** (`benchmark_suite_without_injections`) → **utility** = fraction of user tasks the
  agent actually completes (headline `accuracy`);
- **under attack** (`benchmark_suite_with_injections`, `important_instructions` attack) →
  **utility under attack** and **attack success rate (ASR)** = fraction of (user × injection)
  pairs where the injected task was achieved (`security_results == True`; **lower is better**).

ASR is reported **separately** as `attack_success_rate` and is *not* folded into `accuracy`.
`dimension_scores` carries per-suite benign_utility / utility_under_attack / attack_success_rate.

## Requirements

1. A running OpenAI-compatible `/v1` endpoint **with vLLM tool-calling enabled**
   (`--enable-auto-tool-choice --tool-call-parser <parser>`). gbench uses AgentDojo's
   `OpenAILLM` (native tools API); with tool-calling off, the server's parser swallows the
   model's function calls and every task returns empty. (For a server *without* tool parsing,
   swap `OpenAILLM`→`LocalLLM` in the runner, prompt-based tool calling.)
2. **The `agentdojo` package** (REQUIRED; the suite hard-errors if it cannot be imported):
   ```bash
   pip install agentdojo==0.1.35
   ```
   The base install is enough. The optional `transformers` extra (a DeBERTa PI detector) is a
   *defense* and is not used by the canonical attack run. Installing the base package into a
   torch/vLLM env is safe: it only adds the langchain/langgraph stack, touching no torch /
   transformers / openai / pydantic pins (verified via `pip install --dry-run`).

No API key and no LLM judge. AgentDojo scores against its environments' ground truth.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_AGENTDOJO_VERSION` | benchmark version (default `v1.2.2`; the package ships `v1`…`v1.2.2`) |
| `GBENCH_AGENTDOJO_ATTACK` | attack name (default `important_instructions`, the canonical targeted attack) |
| `GBENCH_AGENTDOJO_MODEL` | the model id sent to the endpoint (default = the endpoint's served id) |
| `GBENCH_AGENTDOJO_MODEL_LABEL` | the prose model label the attack addresses (default `vllm_parsed` → "Local model"; set a `gemini-*` id for the Google-targeted variant) |
| `GBENCH_AGENT_DOJO_TEMPERATURE` | per-suite temperature override |

## Run

```bash
pip install agentdojo==0.1.35
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals agent_dojo --eval-limit 5 \
       --max-output-tokens 4096 --suite-timeout 36000
```

## Notes / caveats

- **Runtime:** each (user × injection) pair is a full multi-step tool episode. The full run is
  large (≈ user_tasks × injection_tasks × 4 suites, with injections). Use `--eval-limit N` (N
  user tasks + N injection tasks per suite) for a quick check; a limited run is a subset and is
  not `leaderboard_comparable`.
- **Attack label:** the `important_instructions` attack addresses the agent by a prose model name
  from `agentdojo.models.MODEL_NAMES`. gemma isn't in that map, so the default label is
  "Local model" (canonical for a locally-served model). `leaderboard_comparable` is set for a
  full run of the pinned version + attack.
