# =============================================================================
# Network — dedicated VPC so we don't depend on GCP's default VPC.
# =============================================================================

resource "google_compute_network" "main" {
  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false
  description             = "Dedicated VPC for project-800ms voice MVP"
}

resource "google_compute_subnetwork" "public" {
  name          = "${var.project_name}-public"
  network       = google_compute_network.main.id
  region        = var.region
  ip_cidr_range = "10.0.1.0/24"

  # Enable private Google access so the instance can reach Secret Manager +
  # Artifact Registry without traversing the public internet (even when the
  # NIC also has an external IP).
  private_ip_google_access = true
}

# =============================================================================
# Firewall rules
#
# GCP firewalls are network-scoped and filter by target tags. Every rule
# targets the "project-800ms" network tag applied to the VM.
#
# Port layout mirrors the AWS SG:
#   With TLS off: 7880/tcp, 7881/tcp, 8000/tcp, 50000-50099/udp.
#   With TLS on : 80/tcp, 443/tcp, 50000-50099/udp (signaling+API via Caddy).
# =============================================================================

locals {
  tls_enabled     = var.domain != ""
  network_tag     = var.project_name
  instance_labels = { project = var.project_name }

  # LiveKit WebRTC media is raw UDP (SRTP-encrypted at the application layer).
  # It cannot be reverse-proxied through Caddy — the browser sends media
  # directly to the instance's external IP regardless of whether TLS is on.
}

resource "google_compute_firewall" "webrtc_udp" {
  name        = "${var.project_name}-webrtc-udp"
  network     = google_compute_network.main.name
  description = "LiveKit WebRTC media (UDP)"
  direction   = "INGRESS"
  priority    = 1000

  source_ranges = [var.allowed_app_cidr]
  target_tags   = [local.network_tag]

  allow {
    protocol = "udp"
    ports    = ["50000-50099"]
  }
}

# Plaintext ports — exposed ONLY when TLS is disabled.
resource "google_compute_firewall" "plaintext_app" {
  count = local.tls_enabled ? 0 : 1

  name        = "${var.project_name}-plaintext-app"
  network     = google_compute_network.main.name
  description = "LiveKit signaling, TCP RTC fallback, FastAPI (TLS off)"
  direction   = "INGRESS"
  priority    = 1000

  source_ranges = [var.allowed_app_cidr]
  target_tags   = [local.network_tag]

  allow {
    protocol = "tcp"
    ports    = ["7880", "7881", "8000"]
  }
}

# HTTPS + HTTP (for Let's Encrypt HTTP-01) — exposed ONLY when TLS is on.
resource "google_compute_firewall" "caddy" {
  count = local.tls_enabled ? 1 : 0

  name        = "${var.project_name}-caddy"
  network     = google_compute_network.main.name
  description = "Caddy HTTPS (443) + HTTP (80, for Let's Encrypt HTTP-01 + 301 redirect)"
  direction   = "INGRESS"
  priority    = 1000

  source_ranges = [var.allowed_app_cidr]
  target_tags   = [local.network_tag]

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }
}

# Optional SSH — if allowed_ssh_cidr is empty, rely on gcloud IAP tunneling
# (no public port needed).
resource "google_compute_firewall" "ssh" {
  count = var.allowed_ssh_cidr == "" ? 0 : 1

  name        = "${var.project_name}-ssh"
  network     = google_compute_network.main.name
  description = "SSH from narrow CIDR"
  direction   = "INGRESS"
  priority    = 1000

  source_ranges = [var.allowed_ssh_cidr]
  target_tags   = [local.network_tag]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

# IAP SSH — always enabled so we can `gcloud compute ssh --tunnel-through-iap`
# even without public SSH. GCP's IAP IP range is fixed.
resource "google_compute_firewall" "iap_ssh" {
  name        = "${var.project_name}-iap-ssh"
  network     = google_compute_network.main.name
  description = "SSH via Identity-Aware Proxy tunnel"
  direction   = "INGRESS"
  priority    = 1000

  source_ranges = ["35.235.240.0/20"] # IAP's well-known source range
  target_tags   = [local.network_tag]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

# =============================================================================
# Static external IP — stable public endpoint survives instance stop/start.
# =============================================================================

resource "google_compute_address" "main" {
  name         = "${var.project_name}-ip"
  region       = var.region
  address_type = "EXTERNAL"
  description  = "Public IP for project-800ms voice MVP — stable across spot preemption"
}

locals {
  # When TLS is enabled Caddy serves api.$domain and livekit.$domain on 443.
  # Otherwise the browser hits the external IP directly on plain http/ws.
  api_host     = local.tls_enabled ? "api.${var.domain}" : google_compute_address.main.address
  livekit_host = local.tls_enabled ? "livekit.${var.domain}" : google_compute_address.main.address

  api_url     = local.tls_enabled ? "https://${local.api_host}" : "http://${local.api_host}:8000"
  livekit_url = local.tls_enabled ? "wss://${local.livekit_host}" : "ws://${local.livekit_host}:7880"

  livekit_public_url = var.livekit_public_url != "" ? var.livekit_public_url : local.livekit_url

  cloudflare_enabled = local.tls_enabled && nonsensitive(var.cloudflare_api_token != "") && var.cloudflare_zone_id != ""
}

# =============================================================================
# Service account — dedicated SA with narrowly-scoped roles.
#
# Secrets live in Secret Manager. The SA gets secretAccessor on each secret
# individually (not project-wide), following least privilege.
# =============================================================================

resource "google_service_account" "instance" {
  account_id   = "${var.project_name}-vm"
  display_name = "project-800ms voice VM"
  description  = "Service account for the compute instance. Has Secret Manager accessor on app secrets only."
}

# Logging + monitoring are needed by the Ops Agent (bundled in Deep Learning
# VM images) to ship stdout/stderr and metrics. Narrow roles, not Editor.
resource "google_project_iam_member" "logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.instance.email}"
}

resource "google_project_iam_member" "metrics_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.instance.email}"
}

# =============================================================================
# Secret Manager — one secret per credential, with a version containing the
# value. Access is granted to the instance SA via per-secret IAM binding.
# =============================================================================

locals {
  app_secrets = {
    postgres_password      = var.postgres_password
    redis_password         = var.redis_password
    livekit_api_key        = var.livekit_api_key
    livekit_api_secret     = var.livekit_api_secret
    vllm_api_key           = var.vllm_api_key
    hugging_face_hub_token = var.hugging_face_hub_token == "" ? "__UNSET__" : var.hugging_face_hub_token
    llm_api_key            = var.llm_api_key == "" ? "__UNSET__" : var.llm_api_key
    # Optional API seed key. Sentinel + startup script treat __UNSET__ as
    # empty so downstream compose falls back to defaults.
    seed_demo_api_key = var.seed_demo_api_key == "" ? "__UNSET__" : var.seed_demo_api_key
  }
}

resource "google_secret_manager_secret" "app" {
  for_each  = local.app_secrets
  secret_id = "${var.project_name}-${replace(each.key, "_", "-")}"

  replication {
    auto {}
  }

  labels = local.instance_labels
}

resource "google_secret_manager_secret_version" "app" {
  for_each    = local.app_secrets
  secret      = google_secret_manager_secret.app[each.key].id
  secret_data = each.value
}

resource "google_secret_manager_secret_iam_member" "instance_accessor" {
  for_each  = local.app_secrets
  secret_id = google_secret_manager_secret.app[each.key].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.instance.email}"
}

# =============================================================================
# Boot image — Deep Learning VM with CUDA 12.4, PyTorch, Docker, and
# nvidia-container-toolkit preinstalled.
# =============================================================================

data "google_compute_image" "dlvm" {
  family  = var.image_family
  project = var.image_project
}

# =============================================================================
# Startup script — bootstrap the docker-compose stack.
#
# Secrets are fetched from Secret Manager at boot using the instance SA's
# token from the metadata server. Plaintext secrets never appear in the
# instance metadata.
# =============================================================================

locals {
  startup_script = templatefile("${path.module}/startup_script.sh", {
    git_repo           = var.git_repo
    git_ref            = var.git_ref
    image_tag          = var.image_tag
    project_name       = var.project_name
    region             = var.region
    livekit_public_url = local.livekit_public_url
    tls_enabled        = local.tls_enabled
    domain             = var.domain
    tls_email          = var.tls_email
    secret_prefix      = "${var.project_name}-"
    llm_base_url       = var.llm_base_url
    llm_model          = var.llm_model
  })
}

# =============================================================================
# Compute instance — Spot VM or on-demand, toggled by `use_spot`.
#
# GCP Spot VMs differ from AWS spot in two important ways:
#   1. They don't have a separate quota hard-limit — same region quota as
#      on-demand, but lower priority.
#   2. When preempted, the instance is STOPPED (not terminated) — boot disk
#      is preserved, so Postgres data and HF model cache survive. On capacity
#      return, the VM can be restarted (via automatic_restart=false + external
#      orchestration, or manually).
# =============================================================================

resource "google_compute_instance" "main" {
  name         = "${var.project_name}-instance"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = [local.network_tag]
  labels       = local.instance_labels

  # Don't let a spot preemption delete the VM; we want the boot disk retained.
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = data.google_compute_image.dlvm.self_link
      size  = var.root_volume_size
      type  = "pd-balanced"
    }
  }

  # L4 GPU attachment.
  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  network_interface {
    subnetwork = google_compute_subnetwork.public.id
    access_config {
      nat_ip = google_compute_address.main.address
    }
  }

  scheduling {
    # GPU VMs don't support live migration — must terminate on host maintenance.
    on_host_maintenance = "TERMINATE"
    automatic_restart   = var.use_spot ? false : true
    preemptible         = var.use_spot
    provisioning_model  = var.use_spot ? "SPOT" : "STANDARD"
  }

  service_account {
    email  = google_service_account.instance.email
    scopes = ["cloud-platform"]
  }

  # Shielded VM — rough equivalent of AWS IMDSv2 hardening. Enforces signed
  # bootloader + kernel, protects against boot-time tampering.
  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  metadata = {
    enable-oslogin = "TRUE"
    # Block legacy V1 metadata access — only V2 (header-based) is allowed.
    block-project-ssh-keys = "TRUE"
  }

  metadata_startup_script = local.startup_script

  depends_on = [
    google_project_iam_member.logs_writer,
    google_project_iam_member.metrics_writer,
    google_secret_manager_secret_iam_member.instance_accessor,
    google_secret_manager_secret_version.app,
  ]
}

# =============================================================================
# Cloudflare DNS records (optional) — mirrors AWS module.
#
# proxied=false is mandatory: CF proxy cannot pass WebRTC UDP or Let's Encrypt
# HTTP-01 challenges.
# =============================================================================

resource "cloudflare_record" "api" {
  count   = local.cloudflare_enabled ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "api.${var.domain}"
  type    = "A"
  content = google_compute_address.main.address
  ttl     = 60
  proxied = false
  comment = "Managed by Terraform (project-800ms GCP) — do not proxy, WebRTC + HTTP-01 break behind CF proxy"
}

resource "cloudflare_record" "livekit" {
  count   = local.cloudflare_enabled ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "livekit.${var.domain}"
  type    = "A"
  content = google_compute_address.main.address
  ttl     = 60
  proxied = false
  comment = "Managed by Terraform (project-800ms GCP) — do not proxy, WebRTC + HTTP-01 break behind CF proxy"
}

# Apex — Caddy serves the React SPA here. Without this record, Let's Encrypt's
# HTTP-01 challenge for ${domain} hits whatever stale IP is in DNS and fails,
# which also poisons cert issuance for the subdomains (ACME rate-limits /
# retries per account).
resource "cloudflare_record" "apex" {
  count   = local.cloudflare_enabled ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = var.domain
  type    = "A"
  content = google_compute_address.main.address
  ttl     = 60
  proxied = false
  comment = "Managed by Terraform (project-800ms GCP) — apex SPA host, do not proxy (HTTP-01)"
}

# www — Caddyfile.prod 301-redirects www.${domain} → ${domain}. Needs to
# resolve to the instance for the redirect block to be reachable; without
# it, Caddy still tries to issue a cert and HTTP-01 fails against stale DNS.
resource "cloudflare_record" "www" {
  count   = local.cloudflare_enabled ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "www.${var.domain}"
  type    = "A"
  content = google_compute_address.main.address
  ttl     = 60
  proxied = false
  comment = "Managed by Terraform (project-800ms GCP) — www → apex 301 via Caddy"
}
