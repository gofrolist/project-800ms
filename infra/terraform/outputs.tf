output "public_ip" {
  description = "Elastic IP of the voice box"
  value       = aws_eip.main.public_ip
}

output "instance_id" {
  description = "Underlying EC2 instance ID fulfilled by the spot request"
  value       = aws_spot_instance_request.main.spot_instance_id
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
  description = "A records you must create in your DNS provider before the first HTTPS request succeeds. Empty when TLS is disabled. When cloudflare_api_token + cloudflare_zone_id are set, the api/livekit records are auto-managed by Terraform — only the apex record (if you want one) is left for you."
  value = local.tls_enabled ? (
    local.cloudflare_enabled ? {
      "${var.domain}" = aws_eip.main.public_ip # apex — manual, not auto-managed
      } : {
      "${var.domain}"         = aws_eip.main.public_ip
      "api.${var.domain}"     = aws_eip.main.public_ip
      "livekit.${var.domain}" = aws_eip.main.public_ip
    }
  ) : {}
}

output "cloudflare_managed" {
  description = "Whether Terraform is managing the api/livekit DNS records via Cloudflare."
  value       = local.cloudflare_enabled
}

output "ssm_connect" {
  description = "Open an interactive shell via SSM Session Manager (no SSH needed)"
  value       = "aws ssm start-session --target ${aws_spot_instance_request.main.spot_instance_id} --region ${var.region}"
}

output "bootstrap_log_hint" {
  description = "Where to watch bootstrap progress after connecting"
  value       = "sudo tail -f /var/log/project-800ms-bootstrap.log"
}

output "ssm_parameter_prefix" {
  description = "SSM Parameter Store prefix holding the app secrets. Rotate via `aws ssm put-parameter --overwrite`, then re-run docker compose up on the instance."
  value       = "/${var.project_name}/"
}
