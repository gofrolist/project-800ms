#!/bin/bash
# project-800ms GCP bootstrap — runs once on first boot of the compute instance.
# On spot preemption the boot disk is preserved and the startup script does
# NOT re-run (GCP only runs metadata_startup_script on first boot), so
# everything below is idempotent-on-first-run only.
#
# Security: `set -x` is intentionally OMITTED so secret values fetched from
# Secret Manager are NOT echoed into the bootstrap log.
set -euo pipefail

LOG=/var/log/project-800ms-bootstrap.log
# Log is root:root 0600 so non-root SSH sessions can't read it.
install -m 600 /dev/null "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "[bootstrap] starting $(date -Is)"

APP_DIR=/opt/project-800ms
REPO="${git_repo}"
REF="${git_ref}"
IMAGE_TAG="${image_tag}"
PROJECT="${project_name}"
REGION="${region}"
LIVEKIT_PUBLIC_URL="${livekit_public_url}"
TLS_ENABLED="${tls_enabled}"
DOMAIN="${domain}"
TLS_EMAIL="${tls_email}"
SECRET_PREFIX="${secret_prefix}"

# Compose files depend on whether TLS is enabled.
if [ "$TLS_ENABLED" = "true" ]; then
  COMPOSE_FILES=(
    -f infra/docker-compose.yml
    -f infra/docker-compose.prod.yml
    -f infra/docker-compose.tls.yml
  )
else
  COMPOSE_FILES=(
    -f infra/docker-compose.yml
    -f infra/docker-compose.prod.yml
  )
fi

# -----------------------------------------------------------------------------
# 1. Docker + NVIDIA runtime + gcloud + git.
#
# Deep Learning VM images ship with all of these. Verify and install on miss
# so non-DLVM base images also work.
# -----------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "[bootstrap] installing docker"
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

# Check for the actual binary rather than dpkg metadata — DLVM images may ship
# the toolkit via a non-standard package name but the binary is canonical.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "[bootstrap] installing nvidia-container-toolkit"
  # --yes overrides the "file exists" prompt if the keyring was pre-staged.
  rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi

# Always ensure Docker knows about the nvidia runtime. On DLVM images the
# toolkit is installed but `nvidia-ctk runtime configure` may not have been
# run against Docker yet — safe to re-run (idempotent).
if ! docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia"; then
  echo "[bootstrap] configuring docker nvidia runtime"
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "[bootstrap] installing google-cloud-sdk"
  apt-get update
  apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --batch --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update
  apt-get install -y google-cloud-cli
fi

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi

# Fail fast if the GPU isn't reachable — nothing below will work without it.
nvidia-smi

# -----------------------------------------------------------------------------
# 2. Fetch secrets from Secret Manager.
#
# Uses the instance service account's token from the metadata server.
# Retries because SA propagation after instance create can take a few seconds.
# -----------------------------------------------------------------------------
secret_get() {
  local name="$1"
  local attempt=0
  local value
  while [ $attempt -lt 10 ]; do
    if value=$(gcloud secrets versions access latest \
      --secret="$${SECRET_PREFIX}$${name//_/-}" \
      2>/dev/null); then
      printf '%s' "$value"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 3
  done
  echo "[bootstrap] ERROR: failed to fetch secret $${SECRET_PREFIX}$${name//_/-} after 10 attempts" >&2
  return 1
}

POSTGRES_PASSWORD=$(secret_get postgres_password)
REDIS_PASSWORD=$(secret_get redis_password)
LIVEKIT_API_KEY=$(secret_get livekit_api_key)
LIVEKIT_API_SECRET=$(secret_get livekit_api_secret)
VLLM_API_KEY=$(secret_get vllm_api_key)
HUGGING_FACE_HUB_TOKEN=$(secret_get hugging_face_hub_token)
# Treat the sentinel value as unset.
if [ "$HUGGING_FACE_HUB_TOKEN" = "__UNSET__" ]; then
  HUGGING_FACE_HUB_TOKEN=""
fi

# -----------------------------------------------------------------------------
# 3. Clone the app repo.
# -----------------------------------------------------------------------------
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO" "$APP_DIR"
fi

cd "$APP_DIR"
git fetch --all --tags --prune
git checkout "$REF"
git pull --ff-only || true

# -----------------------------------------------------------------------------
# 4. Write .env.
#
# Written via `printf` instead of a heredoc so that even if someone re-runs
# this script under `bash -x`, the secret values stay on one line and don't
# appear as a multi-line expansion in logs.
# -----------------------------------------------------------------------------
install -d -m 750 infra

umask 077
{
  printf 'POSTGRES_USER=voice\n'
  printf 'POSTGRES_PASSWORD=%s\n' "$POSTGRES_PASSWORD"
  printf 'POSTGRES_DB=voice\n'
  printf 'REDIS_PASSWORD=%s\n' "$REDIS_PASSWORD"
  printf 'LIVEKIT_API_KEY=%s\n' "$LIVEKIT_API_KEY"
  printf 'LIVEKIT_API_SECRET=%s\n' "$LIVEKIT_API_SECRET"
  printf 'LIVEKIT_PUBLIC_URL=%s\n' "$LIVEKIT_PUBLIC_URL"
  # vLLM reads VLLM_API_KEY, agent reads LLM_API_KEY — same value, different
  # env var names (agent env was renamed to signal that LLM is pluggable).
  printf 'VLLM_API_KEY=%s\n' "$VLLM_API_KEY"
  printf 'LLM_API_KEY=%s\n' "$VLLM_API_KEY"
  printf 'HUGGING_FACE_HUB_TOKEN=%s\n' "$HUGGING_FACE_HUB_TOKEN"
  printf 'LOG_LEVEL=INFO\n'
  if [ "$TLS_ENABLED" = "true" ] && [ -n "$DOMAIN" ]; then
    printf 'CORS_ALLOWED_ORIGINS=["https://%s"]\n' "$DOMAIN"
  else
    printf 'CORS_ALLOWED_ORIGINS=["*"]\n'
  fi
  printf 'ENV=production\n'
  printf 'IMAGE_TAG=%s\n' "$IMAGE_TAG"
  if [ "$TLS_ENABLED" = "true" ]; then
    printf 'DOMAIN=%s\n' "$DOMAIN"
    printf 'TLS_EMAIL=%s\n' "$TLS_EMAIL"
  fi
} > infra/.env
chmod 600 infra/.env
umask 022

# -----------------------------------------------------------------------------
# 5. Bring up the stack.
#
# All images — including web — are pulled from GHCR. The web container
# rewrites __API_URL__ placeholders at start using the API_URL env var,
# so one published image works for any deployment domain.
#
# --ignore-pull-failures is kept as defense-in-depth: if the web image
# happens to be temporarily unavailable on GHCR (e.g. during CI rollout
# before first push of that image), compose still brings up the rest of
# the stack and we can retry web later.
# -----------------------------------------------------------------------------
docker compose --env-file infra/.env "$${COMPOSE_FILES[@]}" pull --ignore-pull-failures
docker compose --env-file infra/.env "$${COMPOSE_FILES[@]}" up -d

echo "[bootstrap] done $(date -Is)"
