terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
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
