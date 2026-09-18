# gbench-aider-benchmark: isolated image for Aider Polyglot's native-edit-format pass@2 scoring.
#
# This is aider's own benchmark image (Aider-AI/aider benchmark/Dockerfile: python3.11 +
# openjdk-21 + go + rust + node + gcc + aider) with ONE addition: a pinned setuptools-scm version.
# The gbench-prereqs/aider checkout has no git version metadata, so the upstream
# `uv pip install -e /aider[dev]` fails with "setuptools-scm was unable to detect version"; we set
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT so the editable build succeeds. gbench does not
# install aider into the serving venv (it would downgrade openai and move numpy/hf_hub/pydantic).
#
# Build (context = the aider checkout, so `COPY . /aider` picks it up):
#   docker build -t aider-benchmark \
#       -f gbench/docker/aider_polyglot.Dockerfile /path/to/aider
#
# The runner (aider_polyglot.py) then `docker run`s this image against the gbench-served endpoint.

FROM buildpack-deps:jammy

# Python 3.11 + Java (openjdk-21) + C/C++ build deps.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    cmake \
    libboost-all-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ca-certificates-java \
    openjdk-21-jdk \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Go (arch-detected).
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then GOARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then GOARCH="arm64"; \
    else false; fi && \
    curl -L "https://golang.org/dl/go1.21.5.linux-$GOARCH.tar.gz" -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Rust.
ADD https://sh.rustup.rs /tmp/rustup.sh
RUN chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Node.js + the JS exercise deps (jest/babel/eslint).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /npm-install && \
    cd /npm-install && \
    npm init -y && \
    npm install \
    jest \
    @babel/core@7.25.2 \
    @exercism/babel-preset-javascript@0.2.1 \
    @exercism/eslint-config-javascript@0.6.0 \
    @types/jest@29.5.12 \
    @types/node@20.12.12 \
    babel-jest@29.6.4 \
    core-js@3.37.1 \
    eslint@8.49.0

COPY . /aider
RUN pip3 install --no-cache-dir --upgrade pip uv
# The checkout lacks git version metadata -> pin a version so the setuptools-scm build succeeds.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT=0.0.0
RUN uv pip install --system --no-cache-dir -e /aider[dev]
# benchmark.py reads `repo.head.object.hexsha` (the aider repo's commit) for its run label. The
# copied checkout may have no HEAD, or a `.git` written with a newer git's `refstorage=reftable`
# extension that this image's older git cannot read - so drop it and make a fresh single commit.
RUN git config --global --add safe.directory '*' \
    && git config --global user.email "gbench@localhost" \
    && git config --global user.name "gbench" \
    && cd /aider \
    && { git rev-parse --verify HEAD >/dev/null 2>&1 \
         || { rm -rf /aider/.git && git init -q -b main && git add -A && git commit -qm "gbench snapshot"; }; }
WORKDIR /aider
