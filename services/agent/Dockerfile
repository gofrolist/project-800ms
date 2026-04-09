# syntax=docker/dockerfile:1.7

# CUDA 12.x + cuDNN for Faster-Whisper in-process on GPU.
# ctranslate2 (used by faster-whisper) is linked against libcublas.so.12, so
# we need CUDA 12.x — CUDA 13 ships libcublas.so.13 only and won't load.

# ---- builder ---------------------------------------------------------------
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/bin/python3.12

# System python3.12 is Ubuntu 24.04's default — use it instead of downloading
# a second standalone interpreter. python3.12-venv gives uv a `venv` module to
# materialise /opt/venv from.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install deps first — cached unless pyproject.toml or uv.lock changes.
# BuildKit cache mount keeps wheels across rebuilds on the same host.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then the project source.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime ---------------------------------------------------------------
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface \
    PIPER_VOICES_DIR=/home/appuser/.cache/piper

# Runtime apt deps:
#   - python3.12: the venv's shebangs point at /usr/bin/python3.12
#   - ffmpeg, libsndfile1: audio I/O for Whisper and Piper
#   - espeak-ng: Piper's phonemizer backend for Russian + English
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        ffmpeg libsndfile1 espeak-ng \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. The NVIDIA container runtime exposes GPU device
# files with permissions accessible to non-root processes, so dropping root
# does not break CUDA. The HF_HOME / PIPER_VOICES_DIR cache dirs live under
# the user's home and are pre-created here so the volume mounts in
# docker-compose.yml have a writable target with the right ownership.
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 \
            --home-dir /home/appuser --create-home \
            --shell /usr/sbin/nologin appuser && \
    mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/piper && \
    chown -R appuser:appuser /home/appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

CMD ["python", "main.py"]
