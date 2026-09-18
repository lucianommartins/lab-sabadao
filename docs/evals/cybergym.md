# cybergym setup

CyberGym (`sunblaze-ucb/cybergym`): cybersecurity vulnerability reasoning. Each item is
one ARVO/OSS-Fuzz vulnerability directory; the model is given the vulnerability
description plus its sanitizer crash trace and asked for the root cause and a
proof-of-concept or patch. Because responses are open-form technical analysis,
correctness is decided by an **LLM judge** against the reference, not string matching.

> **Scope: text proxy, not the CyberGym harness.** The upstream benchmark ships each
> vulnerability with a repository tarball and an execution sandbox, and scores a candidate
> PoC by *running* it to see whether it reproduces the crash. gbench does none of that: it
> reads `description.txt`, `error.txt` and `patch.diff` out of the dataset repo and grades
> the model's prose. So this measures vulnerability *reasoning*, not exploit generation,
> and the number is not comparable with a published CyberGym reproduce-rate.
>
> Items are enumerated as `data/<suite>/<id>` directories (ARVO ids), and `--eval-limit`
> takes the first N in sorted order, not a stratified sample across suites.

## Requirements
- **`GEMINI_API_KEY`**: the judge model. The suite **hard-errors** if it is unset.
- Network access to the HF Hub: the loader lists the dataset repo and downloads three
  small files per item, so a large `--eval-limit` means many Hub round-trips.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals cybergym --eval-limit 20
```
`--max-output-tokens` ≥ 4096. This is a defensive-security reasoning eval.
