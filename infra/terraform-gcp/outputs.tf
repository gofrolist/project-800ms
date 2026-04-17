output "public_ip" {
  description = "Static external IP of the voice box"
  value       = google_compute_address.main.address
}

output "instance_name" {
  description = "Compute instance name (use with gcloud commands)"
  value       = google_compute_instance.main.name
}

output "instance_zone" {
  description = "Zone the instance lives in"
  value       = var.zone
}

output "use_spot" {
  description = "Whether the instance is a Spot VM (preemptible) or on-demand"
  value       = var.use_spot
}

output "tls_enabled" {
  description = "Whether Caddy is terminating TLS on the instance"
  value       = local.tls_enabled
}

output "api_url" {
  description = "FastAPI endpoint the browser should hit"
  value       = local.api_url
}

output "livekit_ws_url" {
  description = "LiveKit signaling URL for the browser client"
  value       = local.livekit_url
}

output "dns_records_needed" {
  description = "A records you must create in your DNS provider before the first HTTPS request succeeds. Empty when TLS is disabled. When cloudflare_api_token + cloudflare_zone_id are set, apex + www + api + livekit are ALL auto-managed by Terraform — nothing manual remains."
  value = local.tls_enabled ? (
    local.cloudflare_enabled ? {} : {
      "${var.domain}"         = google_compute_address.main.address
      "www.${var.domain}"     = google_compute_address.main.address
      "api.${var.domain}"     = google_compute_address.main.address
      "livekit.${var.domain}" = google_compute_address.main.address
    }
  ) : {}
}

output "cloudflare_managed" {
  description = "Whether Terraform is managing the apex, www, api, and livekit DNS records via Cloudflare."
  value       = local.cloudflare_enabled
}

output "ssh_connect" {
  description = "Open an interactive shell via gcloud + Identity-Aware Proxy (no public SSH needed)"
  value       = "gcloud compute ssh ${google_compute_instance.main.name} --zone ${var.zone} --project ${var.project_id} --tunnel-through-iap"
}

output "bootstrap_log_hint" {
  description = "Where to watch bootstrap progress after connecting"
  value       = "sudo tail -f /var/log/project-800ms-bootstrap.log"
}

output "secret_prefix" {
  description = "Secret Manager secret name prefix holding the app credentials. Rotate via `gcloud secrets versions add`."
  value       = "${var.project_name}-"
}
