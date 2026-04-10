#!/bin/bash
# project-800ms bootstrap — runs once on first boot of the spot instance.
# On spot stop/start the root volume is preserved and cloud-init does NOT
# re-run, so everything below is idempotent-on-first-run only.
#
# Security: `set -x` is intentionally OMITTED so secret values fetched from
# SSM are NOT echoed into the bootstrap log.
set -euo pipefail

LOG=/var/log/project-800ms-bootstrap.log
# Log is root:root 0600 so non-root SSM sessions can't read it.
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
# 1. Docker + NVIDIA runtime + AWS CLI.
#
# The AWS Deep Learning AMI ships with all three preinstalled, but we verify
# and install on miss so non-DLAMI base images also work.
# -----------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "[bootstrap] installing docker"
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "[bootstrap] installing nvidia-container-toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "[bootstrap] installing aws cli"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  apt-get install -y unzip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi

# Fail fast if the GPU isn't reachable — nothing below will work without it.
nvidia-smi

# -----------------------------------------------------------------------------
# 2. Fetch secrets from SSM Parameter Store.
#
# Uses the instance IAM role. Retries because IAM role propagation can take a
# few seconds after launch even though the policy attachment is ordered
# before the instance create.
# -----------------------------------------------------------------------------
ssm_get() {
  local name="$1"
  local attempt=0
  local value
  while [ $attempt -lt 10 ]; do
    if value=$(aws --region "$REGION" ssm get-parameter \
      --name "/$PROJECT/$name" \
      --with-decryption \
      --query 'Parameter.Value' \
      --output text 2>/dev/null); then
      printf '%s' "$value"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 3
  done
  echo "[bootstrap] ERROR: failed to fetch /$PROJECT/$name from SSM after 10 attempts" >&2
  return 1
}

POSTGRES_PASSWORD=$(ssm_get postgres_password)
REDIS_PASSWORD=$(ssm_get redis_password)
LIVEKIT_API_KEY=$(ssm_get livekit_api_key)
LIVEKIT_API_SECRET=$(ssm_get livekit_api_secret)
VLLM_API_KEY=$(ssm_get vllm_api_key)
HUGGING_FACE_HUB_TOKEN=$(ssm_get hugging_face_hub_token)
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
  printf 'VLLM_API_KEY=%s\n' "$VLLM_API_KEY"
  printf 'HUGGING_FACE_HUB_TOKEN=%s\n' "$HUGGING_FACE_HUB_TOKEN"
  printf 'LOG_LEVEL=INFO\n'
  printf 'DEMO_ROOM=demo\n'
  if [ "$TLS_ENABLED" = "true" ] && [ -n "$DOMAIN" ]; then
    printf 'CORS_ALLOWED_ORIGINS=["https://api.%s"]\n' "$DOMAIN"
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
# The prod overlay replaces `build:` with `image: ghcr.io/...` for api and
# agent. The optional tls overlay adds Caddy and swaps in livekit.prod.yaml.
#
# First run downloads ~5GB of model weights into the hf_cache volume —
# expect several minutes before vllm reports healthy.
# -----------------------------------------------------------------------------
docker compose --env-file infra/.env "$${COMPOSE_FILES[@]}" pull
docker compose --env-file infra/.env "$${COMPOSE_FILES[@]}" up -d

echo "[bootstrap] done $(date -Is)"
