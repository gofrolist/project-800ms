terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }

  # IMPORTANT: Configure a remote backend before running `terraform apply`.
  # Local state stores all secrets (passwords, API keys) in plaintext.
  #
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "project-800ms/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-locks"
  # }
}

provider "aws" {
  region = var.region
}

# Cloudflare DNS automation (optional). When cloudflare_api_token and
# cloudflare_zone_id are both set and TLS is enabled, Terraform manages the
# api / livekit A records automatically. Token needs Zone:DNS:Edit scoped to
# the target zone only.
provider "cloudflare" {
  api_token = var.cloudflare_api_token
}
