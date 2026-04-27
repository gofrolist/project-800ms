# syntax=docker/dockerfile:1.7

# Retriever APPLICATION image — thin layer atop `Dockerfile.base`.
#
# The base image carries CUDA + system libs + Python deps + an import
# smoke layer, rebuilt only when pyproject.toml / uv.lock change. This
# file only copies source on top.
#
# Build order (local dev):
#   docker build -f services/retriever/Dockerfile.base \
#     -t project-800ms-retriever-base:local services/retriever/
#   docker compose -f infra/docker-compose.yml build retriever

ARG BASE_IMAGE=project-800ms-retriever-base:local
FROM ${BASE_IMAGE}

# Base image already set WORKDIR=/app, UID 1001 appuser, PATH, HF_HOME,
# and the venv at /opt/venv. Lay source in and start.
#
# BGE-M3 weights (~2 GB) are NOT baked into the image. sentence-transformers
# fetches them into HF_HOME on first embedder.load(); the hf_cache_retriever
# named volume persists that across container restarts.
COPY --chown=appuser:appuser . /app

# Listen on the internal port exposed via docker-compose (not published
# to the host).
EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
