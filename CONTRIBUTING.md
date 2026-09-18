# Contributing to gbench

Thank you for your interest in contributing to `gbench`. This suite provides performance benchmarking and capability evaluation for open Large Language Models (LLMs) and Vision-Language Models (VLMs).

We welcome contributions from developers, researchers, hardware partners, and framework maintainers. Please read these guidelines before opening an issue or submitting a pull request.

## 1. Development environment setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/google-gemma/gbench.git
   cd gbench
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev,local]"
   ```

## 2. Code formatting and type checking

Before submitting code changes, verify formatting and type safety:

* Run the linter and auto-formatter:
  ```bash
  ruff check gbench/
  ```
* Run static type checks:
  ```bash
  mypy gbench/
  ```

## 3. Running unit tests and capability verification

All pull requests must pass the existing test suite and preserve core evaluation invariants:

* Run unit and integration tests:
  ```bash
  pytest tests/
  ```
* **Golden set capability gate**: Modifications must maintain 100% pass rates across all 16 Golden Set capability invariants (`16/16 PASS`). Do not weaken two-sided presence matchers or bypass unit test execution assertions in `gbench/runners/golden.py`.
* **Statistical repeatability**: Any change to runner execution loops or timing logic must preserve statistical repeatability, ensuring Coefficient of Variation (`CV%`) remains below 5.0% across multi-iteration benchmark runs.

## 4. Documentation style

Documentation (the `README` and everything under `docs/`) is written to read as human authored:

* **Plain ASCII punctuation.** No em dashes and no en dashes. Use a comma, parentheses, a colon, or
  two sentences instead; for a numeric or token range use a hyphen (for example `4k-128k`).
* **No LaTeX in prose.** Write `O(n)` and `x >= 0` in plain text; keep math notation only where it is
  a benchmark's actual data.
* **Plain vocabulary.** Avoid machine-tell filler (delve, leverage, seamless, boasts, realm,
  tapestry, testament, and the like). Say what a thing does directly.
* **Code identifiers stay in backticks.** Do not spell `snake_case` or `camelCase` names as prose
  words; wrap real identifiers in backticks and use ordinary words otherwise.
* **Accurate policy language.** A suite that lacks a prerequisite hard-errors (`infra_required`)
  rather than emitting a fabricated score; it does not report a generic `skipped` status. The one
  exception is a code-execution suite with no available sandbox, which reports a genuine `skipped`
  because it cannot run untrusted code safely. Do not describe behavior the suites no longer have.

Code (docstrings and comments) is not held to the ASCII-punctuation rule, but the plain-vocabulary
and accurate-policy points apply there too.

## 5. Pull request guidelines

* **Keep diffs small and focused**: Avoid combining formatting cleanups with functional logic changes. Small, atomic pull requests are easier to review and merge quickly.
* **Provide clear reproducer commands**: When reporting a bug or submitting a fix, include the exact command line arguments used so reviewers can reproduce the behavior.
* **Add Apache 2.0 copyright headers**: All new source and configuration files (Python, Shell, Terraform, Docker, TypeScript) must include the standard Apache 2.0 Google LLC copyright notice at the top of the file.
* **Update documentation**: If you add new CLI arguments, dataset providers, or evaluation metrics, update `README.md` and the relevant documentation files in `docs/`.

## 6. License and contributor license agreement

By contributing to `gbench`, you agree that your contributions will be licensed under the Apache License, Version 2.0.
