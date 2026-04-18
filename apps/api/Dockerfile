# syntax=docker/dockerfile:1.7

# ---- builder ---------------------------------------------------------------
FROM python:3.14-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

# Build tools for C-extension deps without py3.14 wheels yet (asyncpg in
# particular). Builder-stage only — the runtime image is a separate FROM
# and doesn't inherit these packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv: single static binary, copied from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install deps first — cached unless pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then the project source. Only this layer is invalidated on code changes.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime ---------------------------------------------------------------
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as a non-root user. UID/GID 1001 to avoid colliding with debian's
# default `users` group at 100. The venv and source dirs are chowned in
# the COPY steps below; nothing the app needs to write at runtime.
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
