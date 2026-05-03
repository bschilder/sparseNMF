# Two-stage build:
# 1. ``builder`` installs the package + its build deps in a fat image
# 2. ``runtime`` copies only the resulting site-packages into a slim
#    image so the published artifact stays small
#
# Built and pushed by ``.github/workflows/docker.yml`` to
# ``ghcr.io/bschilder/sparsenmf`` on every main push and tag.
#
# CPU torch wheel by default — keeps the published image under 1 GB
# so it pulls / scans / scans-on-PR fast. GPU users who want CUDA
# layer on top:
#
#     FROM ghcr.io/bschilder/sparsenmf:latest
#     RUN pip install --index-url https://download.pytorch.org/whl/cu124 --upgrade torch
#
# (Or they can build their own CUDA image from this repo by changing
# ``--index-url`` below.)

# ─── Stage 1: build ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# CPU torch wheel first so the resolver doesn't pull the ~2 GB CUDA
# wheel from PyPI. ``[viz]`` adds matplotlib / seaborn for the
# plotting helpers.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install ".[viz]"

# ─── Stage 2: runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built site-packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /workspace

# Sanity check — fails the build if the package isn't importable.
RUN python -c "import sparse_nmf; print(f'sparse_nmf {sparse_nmf.__version__} ready')"

LABEL org.opencontainers.image.source="https://github.com/bschilder/sparseNMF"
LABEL org.opencontainers.image.description="GPU-accelerated sparse NMF — see https://github.com/bschilder/sparseNMF"
LABEL org.opencontainers.image.licenses="MIT"

CMD ["python", "-c", "import sparse_nmf; print(sparse_nmf.__doc__)"]
