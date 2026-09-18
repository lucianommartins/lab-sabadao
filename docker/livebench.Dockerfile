# gbench-livebench: isolated image for LiveBench's core-category scoring.
#
# LiveBench's base package (latex2sympy2, spacy, litellm, ...) and its coding-execution env
# (code_runner/requirements_eval.txt: tensorflow, numba, opencv, scikit-image, ...) would
# churn a torch/vLLM env, so gbench runs LiveBench's own pipeline INSIDE this image instead
# of installing it into the serving venv. gbench generates nothing here itself - it invokes
# LiveBench's run_livebench.py (inference via --api-base against the gbench-served model, then
# grading) and reads the resulting all_groups.csv.
#
# Build (context = your LiveBench checkout, so its LFS-pulled *.json data is included):
#   docker build -t gbench-livebench \
#       -f gbench/docker/livebench.Dockerfile /path/to/LiveBench
#
# The `agentic_coding` category is intentionally NOT part of this image or the default run
# (it needs the separate ~150GB Multi-SWE-Bench container harness).

FROM python:3.11-slim

# System libs commonly needed by the coding-eval stack (opencv, librosa/soundfile, lxml, LaTeX
# tokenizing, git for any VCS installs). Adjust here if a requirements_eval package needs more.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/LiveBench
# Build context must be the LiveBench checkout (with its LFS *.json already pulled).
COPY . /opt/LiveBench

# LiveBench package + the coding-execution requirements (heavy: tensorflow/numba/opencv/...).
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r livebench/code_runner/requirements_eval.txt

# NLTK data used by the instruction_following (IFBench) checker.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true

# run_livebench.py / show_livebench_result.py resolve package-relative paths from here and
# write their result CSVs (all_tasks.csv / all_groups.csv) to the CWD.
WORKDIR /opt/LiveBench/livebench
