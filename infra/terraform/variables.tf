variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix for resources"
  type        = string
  default     = "project-800ms"
}

variable "instance_type" {
  description = "GPU EC2 instance type. g6.xlarge (L4, 24GB) is the MVP sweet spot."
  type        = string
  default     = "g6.xlarge"
}

variable "max_spot_price" {
  description = "Max $/hr for spot. Empty = on-demand price cap (recommended)."
  type        = string
  default     = ""
}

variable "root_volume_size" {
  description = "Root EBS size in GB. Needs room for DLAMI + images + HF model cache."
  type        = number
  default     = 200
}

variable "key_name" {
  description = "EC2 key pair name for SSH. Leave empty to use SSM Session Manager only."
  type        = string
  default     = ""
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed for SSH (22). Ignored if key_name is empty."
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_app_cidr" {
  description = <<-EOT
    CIDR allowed to reach public app ports.
    With TLS off: 7880/tcp, 7881/tcp, 8000/tcp, 50000-50099/udp.
    With TLS on : 80/tcp, 443/tcp, 50000-50099/udp (signaling+API move behind Caddy).
    Default 0.0.0.0/0 is required for a public voice app — the real security
    boundary is TLS + app-level auth, not the CIDR. Narrow to a single IP
    only for private demos.
  EOT
  type        = string
  default     = "0.0.0.0/0"
}

# -----------------------------------------------------------------------------
# TLS (optional)
#
# When `domain` is set, Caddy runs on the instance and terminates TLS for
# api.${domain} and livekit.${domain}, automatically obtaining certs from
# Let's Encrypt via HTTP-01 on port 80. The plaintext 7880/7881/8000 ports
# are then closed at the security group level.
#
# Prerequisites when enabling:
#   1. You own the domain and can create DNS records.
#   2. Create two A records after `terraform apply`, pointing to the EIP:
#        api.${domain}     -> <eip>
#        livekit.${domain} -> <eip>
#   3. Wait for DNS propagation before the first HTTPS request.
# -----------------------------------------------------------------------------

variable "domain" {
  description = "Base domain for TLS. Empty disables TLS. When set, exposes api.$domain and livekit.$domain with Let's Encrypt certs."
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
# App deployment
# -----------------------------------------------------------------------------

variable "git_repo" {
  description = "HTTPS git URL of the project-800ms repo to clone on the instance."
  type        = string
}

variable "git_ref" {
  description = "Git ref (branch / tag / sha) to deploy. Controls which docker-compose files the instance uses."
  type        = string
  default     = "main"
}

variable "image_tag" {
  description = "GHCR image tag for api/agent (e.g. latest, main, sha-abc1234). Pin to a sha for reproducible deploys."
  type        = string
  default     = "latest"
}

variable "livekit_public_url" {
  description = "Public LiveKit URL the browser will connect to. Leave empty to derive ws://<eip>:7880."
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Secrets
#
# Terraform stores these as SSM Parameter Store SecureStrings and the instance
# fetches them at boot via its IAM role. Secrets do NOT end up in the cloud-init
# user-data blob on the instance.
#
# They DO still end up in terraform.tfstate — use an encrypted remote backend
# (S3 + DynamoDB locking) for any real deployment.
# -----------------------------------------------------------------------------

variable "postgres_password" {
  type      = string
  sensitive = true

  validation {
    condition     = length(var.postgres_password) >= 16
    error_message = "postgres_password must be at least 16 characters. Generate with: openssl rand -hex 32"
  }
}

variable "redis_password" {
  type      = string
  sensitive = true

  validation {
    condition     = length(var.redis_password) >= 16
    error_message = "redis_password must be at least 16 characters. Generate with: openssl rand -hex 32"
  }
}

variable "livekit_api_key" {
  description = "LiveKit API key. No default — must be set explicitly to avoid shipping with a well-known key."
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
