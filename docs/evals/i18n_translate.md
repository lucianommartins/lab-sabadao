# i18n_translate

Canonical multilingual machine-translation quality (`wmt/wmt19`): the model translates
source sentences across WMT19 language pairs. **Scoring: corpus chrF** (character n-gram
F-score, the metric WMT reports - `sacrebleu.corpus_chrf` when installed, otherwise an
equivalent local pooled implementation). chrF is a corpus statistic (n-gram counts are pooled
across all segments before the F-score, not averaged per sentence), it is case-sensitive, and a
failed/empty response stays in the pool as an empty hypothesis (penalising recall) rather than
being dropped. `pass_rate_chrf40` reports the share of segments at sentence chrF >= 40 as a
secondary.

> **Scope / comparability (`leaderboard_comparable=false`):** into-English only
> (de/zh/ru/cs -> en; WMT is multi-pair and bidirectional); scored on the `wmt/wmt19`
> **validation** split (newstest2018) because the dataset exposes no newstest2019 test split;
> and sub-sampled per pair. This does not measure generation *into* those languages.

## Requirements
A running OpenAI-compatible `/v1` endpoint. `sacrebleu` is used for the canonical corpus chrF
when present; without it the suite falls back to an equivalent local implementation (no failure).
A language pair that cannot be loaded from `wmt/wmt19` hard-errors (`infra_required`) rather than
being silently dropped.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals i18n_translate --eval-limit 20
```
