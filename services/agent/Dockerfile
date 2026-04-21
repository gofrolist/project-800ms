# syntax=docker/dockerfile:1.7

# Agent APPLICATION image — thin layer atop `Dockerfile.base`.
#
# The base image carries CUDA + system libs + Python deps and is rebuilt
# only when pyproject.toml / uv.lock change. This file only copies source
# on top, so source-only rebuilds finish in seconds.
#
# Build order (local dev):
#   docker build -f services/agent/Dockerfile.base \
#     -t project-800ms-agent-base:local services/agent/
#   docker compose -f infra/docker-compose.yml build agent
#
# In CI the base is published to
# ghcr.io/<owner>/project-800ms-agent-base and pinned by digest or lockhash.

ARG BASE_IMAGE=project-800ms-agent-base:local
FROM ${BASE_IMAGE}

# The base image already set WORKDIR=/app, UID 1001 appuser, PATH, HF_HOME,
# and the venv at /opt/venv. We just lay the source in and start.
#
# Weights are NOT baked into the image. The gigaam / faster-whisper packages
# fetch models into HF_HOME on first call; the `hf_cache_agent` named volume
# in docker-compose.yml persists those across container restarts so the
# first-run cost is paid only once per host.
COPY --chown=appuser:appuser . /app

CMD ["python", "main.py"]
