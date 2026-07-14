# =========================================================================
# ARCHITECTURE FIX: previously ran through mambaorg/micromamba with a
# conda-forge environment.yml. That stack (numpy/pandas/scipy/pyarrow/
# faiss-cpu + BLAS/LAPACK) was unused or duplicated real pip dependencies
# already declared in pyproject.toml, and accounted for most of this
# image's ~3GB size. Plain pip + python:3.11-slim installs exactly what
# pyproject.toml actually declares, nothing more.
#
# BUILD CONTEXT: monorepo root (see docker-compose.yml -> context: ../..),
# required because pyproject.toml depends on llmops-common via a relative
# "file:../../LLMOpsCommon/llmops-common" path.
# =========================================================================

# =========================
# Builder
# =========================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY LLMOpsCommon/llmops-common ./LLMOpsCommon/llmops-common
COPY RAGBenchmarkLab/rag-benchmark-lab ./RAGBenchmarkLab/rag-benchmark-lab

WORKDIR /workspace/RAGBenchmarkLab/rag-benchmark-lab

RUN pip install --no-cache-dir --user ../../LLMOpsCommon/llmops-common .

# =========================
# Runtime
# =========================
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /workspace/RAGBenchmarkLab/rag-benchmark-lab /app

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["uvicorn", "rag_benchmark_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
