output "public_ip" {
  description = "Elastic IP of the voice box"
  value       = aws_eip.main.public_ip
}

output "instance_id" {
  description = "Underlying EC2 instance ID (spot-fulfilled or on-demand, depending on use_spot)"
  value       = local.instance_id
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
      "${var.domain}"         = aws_eip.main.public_ip
      "www.${var.domain}"     = aws_eip.main.public_ip
      "api.${var.domain}"     = aws_eip.main.public_ip
      "livekit.${var.domain}" = aws_eip.main.public_ip
    }
  ) : {}
}

output "cloudflare_managed" {
  description = "Whether Terraform is managing the apex, www, api, and livekit DNS records via Cloudflare."
  value       = local.cloudflare_enabled
}

output "ssm_connect" {
  description = "Open an interactive shell via SSM Session Manager (no SSH needed)"
  value       = "aws ssm start-session --target ${local.instance_id} --region ${var.region}"
}

output "bootstrap_log_hint" {
  description = "Where to watch bootstrap progress after connecting"
  value       = "sudo tail -f /var/log/project-800ms-bootstrap.log"
}

output "ssm_parameter_prefix" {
  description = "SSM Parameter Store prefix holding the app secrets. Rotate via `aws ssm put-parameter --overwrite`, then re-run docker compose up on the instance."
  value       = "/${var.project_name}/"
}
