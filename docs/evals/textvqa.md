# textvqa

Canonical TextVQA (`lmms-lab/textvqa`): visual question answering that requires reading
text (scene OCR) in natural images. Images are sent losslessly (PNG).

**Scoring: the canonical TextVQA metric - the MEAN soft VQA-accuracy over samples, where a
sample scores `min(#annotators-whose-answer-matches / 3, 1)`** (VQAv2 accuracy). The headline
`accuracy` is that soft mean ×100. A per-sample binary form (`min(...) >= 0.5`) is also kept
as `binary_pass_rate`, and each trace records its `vqa_soft_score`. Reporting only the binary
pass-rate under-reports the model because it drops the defensible partial credit the canonical
metric awards (e.g. an answer 2 of 10 annotators also gave).

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals textvqa --eval-limit 20
```
