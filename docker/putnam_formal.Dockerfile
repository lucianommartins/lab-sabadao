# gbench-putnam-formal: Lean 4 + Mathlib image for the PutnamBench formal track.
#
# putnam_formal verifies each proof with a bare `lean <file>` run (putnam_formal.py: docker run
# --entrypoint "" --network none --memory 4g ... lean /workspace/Proof.lean), and every
# PutnamBench statement `import Mathlib`. The stock leanprovercommunity/lean4 image ships the
# toolchain only (no Mathlib), so all proofs fail structurally. This image bakes Mathlib's
# prebuilt olean cache and wraps `lean` so it always runs under the Mathlib lake environment.
#
# Build (context = this docker/ dir; nothing from it is used, but keep it consistent):
#   docker build -t gbench-putnam-formal -f docker/putnam_formal.Dockerfile docker
# Then point the suite at it:
#   export GBENCH_PUTNAM_FORMAL_LEAN_IMAGE=gbench-putnam-formal
#
# The Mathlib olean cache is several GB, so the build downloads a lot and the image is large.

FROM leanprovercommunity/lean4:latest

USER root

# The base image ships elan but no default toolchain, so `lake` refuses to run. Set a bootstrap
# stable toolchain; the `math` template / Mathlib then pins the exact toolchain it needs (elan
# auto-installs it for the subsequent lake commands run inside the project).
RUN elan default stable

# Official "new project that uses Mathlib" recipe: the `math` template adds Mathlib as a
# dependency and pins the matching lean-toolchain (elan auto-installs it); `lake exe cache get`
# downloads the prebuilt Mathlib oleans (no multi-hour compile); `lake build` compiles the tiny
# project against that cache so `import Mathlib` resolves from this project.
WORKDIR /opt
RUN lake new putnamenv math
WORKDIR /opt/putnamenv
# `lake new` scaffolds with the bootstrap stable toolchain, but the pulled Mathlib pins a
# different lean; the prebuilt olean cache only works when the project's toolchain matches
# Mathlib's. Fetch deps, copy Mathlib's toolchain into the project (elan auto-installs it on the
# next lake command), then download the cache and build against it.
RUN lake update
RUN cp .lake/packages/mathlib/lean-toolchain ./lean-toolchain
# `lake exe cache get` decompresses thousands of .ltar batches in parallel and exhausts the
# default open-file limit (EMFILE). Raise the soft limit to the hard max for this step. If the
# daemon's hard limit is itself low, also pass `--ulimit nofile=1048576:1048576` to `docker build`.
RUN ulimit -n "$(ulimit -Hn)" && lake exe cache get && lake build

# putnam_formal invokes a bare `lean <file>` (with --entrypoint "" so no image entrypoint runs,
# and --network none). Wrap `lean` so it always executes under the Mathlib lake environment
# (LEAN_PATH etc. from `lake env`), from any CWD. Placed first on PATH.
RUN printf '#!/bin/sh\ncd /opt/putnamenv && exec lake env lean "$@"\n' > /usr/local/bin/lean \
    && chmod 0755 /usr/local/bin/lean
ENV PATH="/usr/local/bin:${PATH}"

# Sanity: a bare `lean` on `import Mathlib` must succeed (this is exactly the suite's prereq probe).
RUN printf 'import Mathlib\n' > /tmp/probe.lean && lean /tmp/probe.lean && rm /tmp/probe.lean
