# Two-stage build:
# 1. ``builder`` installs the package + its build deps in a fat image
# 2. ``runtime`` copies only the resulting site-packages into a slim
#    image so the published artifact stays small
#
# Built and pushed by ``.github/workflows/docker.yml`` to
# ``ghcr.io/bschilder/sparsenmf`` on every main push and tag. CUDA
# wheels are pulled from PyTorch's index — the published image
# supports CUDA 12.x out of the box; on a CPU-only host ``torch``
# silently falls back to CPU paths.

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

# CUDA 12.4 wheel of torch first so the resolver picks it up; the
# package's own ``torch`` dep then sees a satisfying install.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu124 torch \
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
