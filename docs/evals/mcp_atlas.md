# mcp_atlas

Canonical MCP-Atlas (`ScaleAI/MCP-Atlas`): Model Context Protocol multi-server tool
orchestration. **Scoring: per-claim verification by an LLM judge.** The row's `GTFA_CLAIMS` are the
facts a correct answer must assert; a sample is correct when *every* claim is
supported. `mean_claim_recall` reports partial credit. Without `GEMINI_API_KEY` the
suite falls back to verbatim claim containment, which under-credits paraphrase.

> **Closed book:** MCP-Atlas tasks need live MCP servers (GitHub API, whois, fetch) to
> reach their answers. gbench lists the enabled tool names in the prompt but executes
> nothing, so `closed_book: true` is set on the result and the score is a lower bound,
> not comparable with a figure from a real MCP harness.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mcp_atlas --eval-limit 20
```
