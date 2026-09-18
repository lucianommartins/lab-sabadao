# gbench-omnidocbench: LOCAL build of OmniDocBench's evaluator for the end2end composite (CDM
# formula + TEDS table + edit-distance text + reading order). gbench builds its images locally
# (it never pulls registry images), so this mirrors OmniDocBench's own repro image / README
# install: Python 3.10 + Ghostscript + TeX Live (with CJK) + ImageMagick 7 + the checkout's deps.
#
# It scores only (no model calls): gbench generates the markdown predictions in the host environment and
# runs this image's pdf_validation.py over them. OmniDocBench pins Python <3.12 + numpy 1.24.4,
# which would break the torch/vLLM serving env - hence the isolated image.
#
# Build (context = the OmniDocBench checkout):
#   docker build -t gbench-omnidocbench \
#       -f gbench/docker/omnidocbench.Dockerfile $GBENCH_PREREQS_DIR/OmniDocBench

FROM python:3.10-slim

# Ghostscript + TeX Live (with CJK for CDM) + latexml (LaTeX tables -> HTML) + runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ghostscript \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
        texlive-lang-chinese texlive-lang-cjk latex-cjk-all \
        latexml \
        wget ca-certificates git fontconfig libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# libfribidi is an ImageMagick 7 runtime dependency the portable AppImage does NOT bundle (its
# `magick` fails with "libfribidi.so.0: cannot open shared object file" without it). Kept in its
# own layer so rebuilding for it does not re-download the TeX Live layer above.
RUN apt-get update && apt-get install -y --no-install-recommends libfribidi0 \
    && rm -rf /var/lib/apt/lists/*

# ImageMagick 7 (CDM requires 7.x; Debian ships 6.x). Portable AppImage, extracted (no FUSE),
# with PDF read/write allowed so CDM can rasterize rendered formula PDFs.
RUN wget -q https://download.imagemagick.org/archive/binaries/magick -O /tmp/magick \
    && chmod +x /tmp/magick \
    && cd /opt && /tmp/magick --appimage-extract >/dev/null 2>&1 \
    && mv /opt/squashfs-root /opt/imagemagick7 && rm -f /tmp/magick \
    && ln -sf /opt/imagemagick7/AppRun /usr/local/bin/magick \
    && ln -sf /opt/imagemagick7/AppRun /usr/local/bin/convert \
    && for p in $(find /opt/imagemagick7 -name policy.xml 2>/dev/null); do \
         sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' "$p" || true; \
       done

# OmniDocBench + its pinned deps (numpy 1.24.4 etc. - isolated here).
COPY . /workspace
WORKDIR /workspace
RUN pip install --no-cache-dir -e .
# NLTK data for the text metrics (evaluate / nltk).
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true
