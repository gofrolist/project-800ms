# -----------------------------------------------------------------------------
# Project / placement
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID (globally unique). Create via `gcloud projects create`."
  type        = string
}

variable "project_name" {
  description = "Name prefix for resources"
  type        = string
  default     = "project-800ms"
}

variable "region" {
  description = "GCP region. us-central1 has the best L4 stock and lowest latency to US East; us-west1/us-west4 are alternatives."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone. L4 GPUs live in specific zones — us-central1-a/b/c all have them."
  type        = string
  default     = "us-central1-a"
}

# -----------------------------------------------------------------------------
# Instance
# -----------------------------------------------------------------------------

variable "machine_type" {
  description = <<-EOT
    GCP GPU machine type. g2-standard-8 (1x L4, 8 vCPU, 32 GB) is the direct
    equivalent of AWS g6.2xlarge. g2-standard-4 (1x L4, 4 vCPU, 16 GB) is
    cheaper but tighter — comparable to g6.xlarge.
  EOT
  type        = string
  default     = "g2-standard-8"
}

variable "root_volume_size" {
  description = "Boot disk size in GB. Deep Learning VM + Docker images + HF model cache need ~100-200 GB."
  type        = number
  default     = 200
}

variable "use_spot" {
  description = <<-EOT
    true  = request a Spot VM (cheap, ~60-70%% off, AWS may preempt with 30s
            notice).
    false = launch a regular on-demand VM (~3x cost but no preemption risk).
    On GCP, Spot VMs do not have the hard capacity failures AWS spot has
    — they are scheduled on standard capacity pools with preemption as the
    primary tradeoff, not scarcity.
  EOT
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Deep Learning VM image
# -----------------------------------------------------------------------------

variable "image_family" {
  description = <<-EOT
    Deep Learning VM image family. common-cu129-ubuntu-2204-nvidia-580 ships
    Ubuntu 22.04 with NVIDIA driver R580, Docker, and nvidia-container-toolkit
    preinstalled. The host CUDA toolkit (12.9) is irrelevant — the agent
    container brings its own CUDA 13 runtime via the nvidia/cuda base image.
    What matters is the kernel-mode driver version: R580 is the first driver
    family to support CUDA 13 forward-compat, which is our runtime floor.
    Verify current families with:
      gcloud compute images list --project deeplearning-platform-release \
        --filter="family ~ cu1" --format="value(family)" | sort -u
  EOT
  type        = string
  default     = "common-cu129-ubuntu-2204-nvidia-580"
}

variable "image_project" {
  description = "Deep Learning VM images are published by the deeplearning-platform-release project."
  type        = string
  default     = "deeplearning-platform-release"
}

# -----------------------------------------------------------------------------
# Access — narrow before real use
# -----------------------------------------------------------------------------

variable "allowed_app_cidr" {
  description = <<-EOT
    CIDR allowed to reach public app ports.
    With TLS off: 7880/tcp, 7881/tcp, 8000/tcp, 50000-50099/udp.
    With TLS on : 80/tcp, 443/tcp, 50000-50099/udp (signaling+API behind Caddy).
    Default 0.0.0.0/0 is required for a public voice app — the real security
    boundary is TLS + app-level auth.
  EOT
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed for SSH (port 22). Leave empty to rely solely on gcloud OS Login / IAP."
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# TLS (optional) — mirrors AWS module behaviour
# -----------------------------------------------------------------------------

variable "domain" {
  description = "Base domain for TLS. Empty disables TLS. When set, Caddy serves api.$domain + livekit.$domain with Let's Encrypt HTTP-01 certs."
  type        = string
  default     = ""

  validation {
    condition     = var.domain == "" || can(regex("^[a-z0-9][a-z0-9.-]*[a-z0-9]$", var.domain))
    error_message = "domain must be a valid lowercase hostname (e.g. voice.example.com), or empty to disable TLS."
  }
}

variable "tls_email" {
  description = "Contact email for Let's Encrypt registration. Required when domain is set."
  type        = string
  default     = ""

  validation {
    condition     = var.tls_email == "" || can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.tls_email))
    error_message = "tls_email must be a valid email address (or empty if TLS is disabled)."
  }
}

# -----------------------------------------------------------------------------
# Cloudflare DNS (optional) — mirrors AWS module behaviour
#
# proxied=false is enforced — CF proxy cannot pass WebRTC UDP or Let's Encrypt
# HTTP-01 challenges.
# -----------------------------------------------------------------------------

variable "cloudflare_api_token" {
  description = "Cloudflare API token with Zone:DNS:Edit scoped to the target zone. Empty = don't manage DNS via Terraform."
  type        = string
  sensitive   = true
  default     = ""
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for the domain. Empty = don't manage DNS via Terraform."
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# App deployment
# -----------------------------------------------------------------------------

variable "git_repo" {
  description = "HTTPS git URL of the project-800ms repo to clone on the instance."
  type        = string
}

variable "git_ref" {
  description = "Git ref (branch / tag / sha) to deploy."
  type        = string
  default     = "main"
}

variable "image_tag" {
  description = "GHCR image tag for api/agent."
  type        = string
  default     = "latest"
}

variable "tts_preload_engines" {
  description = <<-EOT
    Comma-separated list of TTS engines the agent should preload and accept
    at dispatch. Valid values: piper, silero, qwen3, xtts. Including `qwen3`
    automatically activates the `tts-qwen3` compose profile and writes
    `QWEN3_TTS_BASE_URL=http://qwen3-tts:8000/v1` to infra/.env. Including
    `xtts` requires populated voice_library/profiles/ on disk — the agent
    hard-fails at boot otherwise. Default `piper` preserves the
    single-engine operator workflow.
  EOT
  type        = string
  default     = "piper"
}

variable "livekit_public_url" {
  description = "Public LiveKit URL the browser will connect to. Empty = derive from EIP."
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Secrets
#
# Terraform writes these to Secret Manager; the instance fetches them at boot
# via its service account. They still live in terraform.tfstate — use a GCS
# remote backend for any real deployment.
# -----------------------------------------------------------------------------

variable "postgres_password" {
  type      = string
  sensitive = true

  validation {
    condition     = length(var.postgres_password) >= 16
    error_message = "postgres_password must be at least 16 characters. Generate with: openssl rand -hex 32"
  }
}

variable "livekit_api_key" {
  description = "LiveKit API key. No default — must be set explicitly."
  type        = string

  validation {
    condition     = length(var.livekit_api_key) >= 6 && var.livekit_api_key != "devkey"
    error_message = "livekit_api_key must be at least 6 characters and not the literal string 'devkey'."
  }
}

variable "livekit_api_secret" {
  type      = string
  sensitive = true

  validation {
    condition     = length(var.livekit_api_secret) >= 32
    error_message = "livekit_api_secret must be at least 32 characters. Generate with: openssl rand -hex 32"
  }
}

variable "vllm_api_key" {
  type      = string
  sensitive = true

  validation {
    condition     = length(var.vllm_api_key) >= 16
    error_message = "vllm_api_key must be at least 16 characters. Generate with: openssl rand -hex 32"
  }
}

variable "hugging_face_hub_token" {
  type      = string
  sensitive = true
  default   = ""
}

# -----------------------------------------------------------------------------
# LLM provider (optional override)
#
# Default (empty) routes the agent to the local vLLM container running Qwen.
# Set these three to swap to an external OpenAI-compatible LLM (Groq, OpenAI,
# Together, Fireworks, etc.) without changing infra.
#
# Typical Groq values:
#   llm_base_url = "https://api.groq.com/openai/v1"
#   llm_model    = "llama-3.3-70b-versatile"
#   llm_api_key  = "gsk_..."  # from https://console.groq.com/keys
#
# The local vllm container stays running regardless (wastes some GPU memory
# but keeps it easy to flip back). Drop vllm from compose later if you
# commit to the external provider.
# -----------------------------------------------------------------------------

variable "llm_base_url" {
  description = "External OpenAI-compatible LLM endpoint. Empty = use local vLLM."
  type        = string
  default     = ""
}

variable "llm_model" {
  description = "Model name for the external LLM (e.g. llama-3.3-70b-versatile). Empty = use local default qwen-7b."
  type        = string
  default     = ""
}

variable "llm_api_key" {
  description = "API key for the external LLM. Empty = use the local vllm_api_key (agent self-auths to its own vLLM)."
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# API seed key
#
# When set, the Alembic seed migration (0002_seed_demo_tenant) materializes
# an API key for the pre-created "demo" tenant. The raw value is hashed
# and discarded by the migration — this is the only moment it is
# recoverable.
#
# The web SPA defaults WEB_API_KEY to this value in the startup script,
# and compose refuses an empty WEB_API_KEY, so in practice this is
# required for any deployment that exposes the SPA.
# -----------------------------------------------------------------------------

variable "seed_demo_api_key" {
  description = "Raw API key for the seeded 'demo' tenant. Empty to skip."
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# Agent <-> API internal token (optional)
#
# Shared secret for server-to-server calls from the agent worker to the
# API's /internal/* endpoints (today: transcript writes). Empty disables
# transcript persistence; the agent still sends transcripts over LiveKit's
# data channel to connected clients.
# -----------------------------------------------------------------------------

variable "agent_internal_token" {
  description = "Shared secret for agent -> api /internal/* calls. Empty disables transcript persistence."
  type        = string
  sensitive   = true
  default     = ""
}

variable "retriever_internal_token" {
  description = <<-EOT
    Shared secret the agent presents to the retriever's `/retrieve`
    endpoint as `X-Internal-Token`, and the retriever validates against
    its own copy. Both agent and retriever are wired to fail closed
    when the token is missing -- empty here triggers the same auto-
    generate-and-persist fallback used for `agent_internal_token`,
    keeping the value stable across spot preemptions via
    `/etc/project-800ms/retriever-token`. Set to a deliberate value
    only when rotating per the playbook in
    `docs/solutions/security-issues/retriever-internal-token-rotation-2026-04-27.md`.
  EOT
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# Admin API key (optional)
#
# Master credential for /v1/admin/* endpoints. NOT a tenant key. Leaving
# this empty disables the admin surface entirely — appropriate for
# deployments that provision tenants + keys out-of-band via SQL.
# -----------------------------------------------------------------------------

variable "admin_api_key" {
  description = "Master key for /v1/admin/* endpoints. Empty disables the admin API."
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# Chatwoot help-base ingestion (optional)
#
# Bearer token for `GET https://chatwoot.arizona-rp.com/api-help-base/get`,
# consumed by `tools/fetch_chatwoot_kb.py` (one-shot or scheduled fetch
# that dumps articles under `data/kb/<project>/`). Stored in Secret Manager
# alongside the other app secrets and projected into infra/.env as
# `CHATWOOT_HELP_BASE_TOKEN`. Empty disables the integration — fetch script
# refuses to run without the token.
# -----------------------------------------------------------------------------

variable "chatwoot_help_base_token" {
  description = "Bearer token for the Chatwoot help-base API. Empty = disable Chatwoot KB ingestion."
  type        = string
  sensitive   = true
  default     = ""

  validation {
    condition     = length(var.chatwoot_help_base_token) == 0 || length(var.chatwoot_help_base_token) >= 32
    error_message = "chatwoot_help_base_token must be either empty (disabled) or at least 32 characters. Mirrors the strength bar applied to postgres_password and livekit_api_secret."
  }
}
