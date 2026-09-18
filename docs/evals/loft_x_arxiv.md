# loft_x_arxiv

Canonical LOFT (Long Context Frontiers, arXiv:2406.13121) in-context retrieval. LOFT
ships no "arXiv" task, so gbench uses the SciFact scientific-paper retrieval corpus
(`google-deepmind/loft`): documents are packed into the context and the model must
retrieve the relevant passage id for a claim. **Scoring: canonical LOFT Recall@1**.
The first predicted id must be in the gold pid set.

## Requirements
- A serving endpoint with a **large context window** (the corpus is packed in-context).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals loft_x_arxiv --eval-limit 20
```
