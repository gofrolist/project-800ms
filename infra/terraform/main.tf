# -----------------------------------------------------------------------------
# Network — dedicated VPC so we don't depend on a pre-existing default VPC.
# -----------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name    = "${var.project_name}-vpc"
    Project = var.project_name
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name    = "${var.project_name}-igw"
    Project = var.project_name
  }
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
  # Explicit AZ override wins; otherwise fall back to the region's first AZ.
  # GPU spot capacity is AZ-specific — use `aws ec2 describe-spot-price-history`
  # to pick an AZ with live pricing for your instance type.
  availability_zone       = var.availability_zone != "" ? var.availability_zone : data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-public"
    Project = var.project_name
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name    = "${var.project_name}-rt-public"
    Project = var.project_name
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

data "aws_caller_identity" "current" {}

# -----------------------------------------------------------------------------
# AMI — AWS Deep Learning AMI (Ubuntu 22.04, PyTorch).
# Ships with NVIDIA driver, Docker, and nvidia-container-toolkit preinstalled.
# Account ID 898082745236 is the official AWS DLAMI publisher.
# -----------------------------------------------------------------------------

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["898082745236"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  tls_enabled = var.domain != ""

  # When TLS is enabled Caddy serves api.$domain and livekit.$domain on 443.
  # Otherwise the browser hits the EIP directly on plain http/ws.
  api_host     = local.tls_enabled ? "api.${var.domain}" : aws_eip.main.public_ip
  livekit_host = local.tls_enabled ? "livekit.${var.domain}" : aws_eip.main.public_ip

  api_url     = local.tls_enabled ? "https://${local.api_host}" : "http://${local.api_host}:8000"
  livekit_url = local.tls_enabled ? "wss://${local.livekit_host}" : "ws://${local.livekit_host}:7880"

  livekit_public_url = var.livekit_public_url != "" ? var.livekit_public_url : local.livekit_url
}

# -----------------------------------------------------------------------------
# Security group
# -----------------------------------------------------------------------------

resource "aws_security_group" "main" {
  name        = "${var.project_name}-sg"
  description = "project-800ms voice MVP"
  vpc_id      = aws_vpc.main.id

  # LiveKit WebRTC media is raw UDP (SRTP-encrypted at the application layer).
  # It cannot be reverse-proxied through Caddy; the browser sends media
  # directly to the EIP regardless of whether TLS is enabled elsewhere.
  ingress {
    description = "LiveKit WebRTC media (UDP)"
    from_port   = 50000
    to_port     = 50099
    protocol    = "udp"
    cidr_blocks = [var.allowed_app_cidr]
  }

  # Plain HTTP signaling + FastAPI are exposed ONLY when TLS is disabled.
  # With TLS on, Caddy fronts everything on 443 and the internal services
  # stay reachable only via the docker network.
  dynamic "ingress" {
    for_each = local.tls_enabled ? [] : [1]
    content {
      description = "LiveKit signaling (plain HTTP/WS, MVP only)"
      from_port   = 7880
      to_port     = 7880
      protocol    = "tcp"
      cidr_blocks = [var.allowed_app_cidr]
    }
  }

  dynamic "ingress" {
    for_each = local.tls_enabled ? [] : [1]
    content {
      description = "LiveKit TCP RTC fallback (plain, MVP only)"
      from_port   = 7881
      to_port     = 7881
      protocol    = "tcp"
      cidr_blocks = [var.allowed_app_cidr]
    }
  }

  dynamic "ingress" {
    for_each = local.tls_enabled ? [] : [1]
    content {
      description = "FastAPI (plain HTTP, MVP only)"
      from_port   = 8000
      to_port     = 8000
      protocol    = "tcp"
      cidr_blocks = [var.allowed_app_cidr]
    }
  }

  # Caddy / HTTPS — only opened when TLS is on. Port 80 is required by
  # Let's Encrypt's HTTP-01 challenge for automatic cert issuance.
  dynamic "ingress" {
    for_each = local.tls_enabled ? [1] : []
    content {
      description = "HTTPS (Caddy)"
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = [var.allowed_app_cidr]
    }
  }

  dynamic "ingress" {
    for_each = local.tls_enabled ? [1] : []
    content {
      description = "HTTP (Caddy - Lets Encrypt HTTP-01 + 301 to HTTPS)"
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = [var.allowed_app_cidr]
    }
  }

  dynamic "ingress" {
    for_each = var.key_name != "" && var.allowed_ssh_cidr != "" ? [1] : []

    content {
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [var.allowed_ssh_cidr]
    }
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# -----------------------------------------------------------------------------
# SSM Parameter Store — encrypted secrets.
#
# Secrets are stored as SecureString parameters under /${project_name}/ and
# fetched by user_data at boot. This keeps the cloud-init user-data blob
# non-sensitive and lets us rotate secrets without re-running Terraform
# (re-run `docker compose up` on the instance after rotation).
#
# Values still live in terraform.tfstate — use an encrypted remote backend.
# -----------------------------------------------------------------------------

resource "aws_ssm_parameter" "postgres_password" {
  name  = "/${var.project_name}/postgres_password"
  type  = "SecureString"
  value = var.postgres_password
  tags  = { Project = var.project_name }
}

resource "aws_ssm_parameter" "redis_password" {
  name  = "/${var.project_name}/redis_password"
  type  = "SecureString"
  value = var.redis_password
  tags  = { Project = var.project_name }
}

resource "aws_ssm_parameter" "livekit_api_key" {
  name  = "/${var.project_name}/livekit_api_key"
  type  = "SecureString"
  value = var.livekit_api_key
  tags  = { Project = var.project_name }
}

resource "aws_ssm_parameter" "livekit_api_secret" {
  name  = "/${var.project_name}/livekit_api_secret"
  type  = "SecureString"
  value = var.livekit_api_secret
  tags  = { Project = var.project_name }
}

resource "aws_ssm_parameter" "vllm_api_key" {
  name  = "/${var.project_name}/vllm_api_key"
  type  = "SecureString"
  value = var.vllm_api_key
  tags  = { Project = var.project_name }
}

resource "aws_ssm_parameter" "hugging_face_hub_token" {
  name = "/${var.project_name}/hugging_face_hub_token"
  type = "SecureString"
  # SSM rejects empty string — store a sentinel we filter out in user_data.
  value = var.hugging_face_hub_token == "" ? "__UNSET__" : var.hugging_face_hub_token
  tags  = { Project = var.project_name }
}

# -----------------------------------------------------------------------------
# IAM — SSM Session Manager access + scoped read on the project's SSM params.
# -----------------------------------------------------------------------------

data "aws_iam_policy_document" "assume_ec2" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.project_name}-ec2"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

# SSM Session Manager — lets us open an interactive shell without SSH.
resource "aws_iam_role_policy_attachment" "ssm_managed" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Scoped read on /${project_name}/* parameters + decrypt via the default
# SSM KMS key. No write, no list outside the project prefix, no access to
# anything else in the account.
data "aws_iam_policy_document" "ssm_read" {
  statement {
    sid    = "ReadProjectParameters"
    effect = "Allow"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
    ]
    resources = [
      "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/*"
    ]
  }

  statement {
    sid       = "DecryptSsmParameters"
    effect    = "Allow"
    actions   = ["kms:Decrypt"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "ssm_read" {
  name   = "${var.project_name}-ssm-read"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.ssm_read.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.project_name}-ec2"
  role = aws_iam_role.instance.name
}

# -----------------------------------------------------------------------------
# Elastic IP — survives spot stop/start, so the public endpoint is stable.
# -----------------------------------------------------------------------------

resource "aws_eip" "main" {
  domain = "vpc"

  tags = {
    Name    = "${var.project_name}-eip"
    Project = var.project_name
  }
}

# -----------------------------------------------------------------------------
# User data — bootstrap the compose stack.
#
# Secrets are NOT passed here — they're fetched from SSM at boot using the
# instance IAM role. This keeps plaintext secrets out of the cloud-init
# user-data blob (readable at /var/lib/cloud/instance/user-data.txt).
# -----------------------------------------------------------------------------

locals {
  user_data = templatefile("${path.module}/user_data.sh", {
    git_repo           = var.git_repo
    git_ref            = var.git_ref
    image_tag          = var.image_tag
    project_name       = var.project_name
    region             = var.region
    livekit_public_url = local.livekit_public_url
    tls_enabled        = local.tls_enabled
    domain             = var.domain
    tls_email          = var.tls_email
  })
}

# -----------------------------------------------------------------------------
# Spot instance — persistent request with stop-on-interruption.
#
# On spot interruption, AWS stops (not terminates) the instance, preserves the
# EBS root volume, and restarts it automatically when capacity returns. The
# docker-compose services come back up via `restart: unless-stopped`, and the
# Elastic IP is re-associated so the public endpoint is stable.
# -----------------------------------------------------------------------------

resource "aws_spot_instance_request" "main" {
  count = var.use_spot ? 1 : 0

  ami                  = data.aws_ami.dlami.id
  instance_type        = var.instance_type
  subnet_id            = aws_subnet.public.id
  iam_instance_profile = aws_iam_instance_profile.instance.name
  key_name             = var.key_name == "" ? null : var.key_name

  vpc_security_group_ids = [aws_security_group.main.id]

  spot_type                      = "persistent"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true
  spot_price                     = var.max_spot_price == "" ? null : var.max_spot_price

  user_data = local.user_data

  # Enforce IMDSv2. `http_tokens = required` blocks IMDSv1. `hop_limit = 1`
  # prevents container networks (which add a routing hop) from reaching
  # IMDS — so even if a container is compromised, it cannot steal the
  # instance role credentials.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Name    = "${var.project_name}-spot-request"
    Project = var.project_name
  }

  # Make sure the IAM policy and SSM parameters exist before the instance
  # boots user_data, which depends on both.
  depends_on = [
    aws_iam_role_policy.ssm_read,
    aws_iam_role_policy_attachment.ssm_managed,
    aws_ssm_parameter.postgres_password,
    aws_ssm_parameter.redis_password,
    aws_ssm_parameter.livekit_api_key,
    aws_ssm_parameter.livekit_api_secret,
    aws_ssm_parameter.vllm_api_key,
    aws_ssm_parameter.hugging_face_hub_token,
  ]
}

# -----------------------------------------------------------------------------
# On-demand fallback — used when use_spot = false. Same AMI, same networking,
# same user_data. Costs ~3x but bypasses spot capacity constraints entirely.
# -----------------------------------------------------------------------------

resource "aws_instance" "main" {
  count = var.use_spot ? 0 : 1

  ami                  = data.aws_ami.dlami.id
  instance_type        = var.instance_type
  subnet_id            = aws_subnet.public.id
  iam_instance_profile = aws_iam_instance_profile.instance.name
  key_name             = var.key_name == "" ? null : var.key_name

  vpc_security_group_ids = [aws_security_group.main.id]

  user_data = local.user_data

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Name    = "${var.project_name}-instance"
    Project = var.project_name
  }

  depends_on = [
    aws_iam_role_policy.ssm_read,
    aws_iam_role_policy_attachment.ssm_managed,
    aws_ssm_parameter.postgres_password,
    aws_ssm_parameter.redis_password,
    aws_ssm_parameter.livekit_api_key,
    aws_ssm_parameter.livekit_api_secret,
    aws_ssm_parameter.vllm_api_key,
    aws_ssm_parameter.hugging_face_hub_token,
  ]
}

# Single source of truth for the instance ID, regardless of spot vs on-demand.
locals {
  instance_id = var.use_spot ? aws_spot_instance_request.main[0].spot_instance_id : aws_instance.main[0].id
}

# Tags on aws_spot_instance_request don't propagate to the underlying instance,
# so apply them explicitly. On-demand already has tags on the resource itself.
resource "aws_ec2_tag" "instance_name" {
  count       = var.use_spot ? 1 : 0
  resource_id = local.instance_id
  key         = "Name"
  value       = "${var.project_name}-instance"
}

resource "aws_ec2_tag" "instance_project" {
  count       = var.use_spot ? 1 : 0
  resource_id = local.instance_id
  key         = "Project"
  value       = var.project_name
}

resource "aws_eip_association" "main" {
  instance_id   = local.instance_id
  allocation_id = aws_eip.main.id
}

# -----------------------------------------------------------------------------
# Cloudflare DNS records (optional)
#
# Created only when TLS is enabled AND a Cloudflare zone/token are configured.
# `proxied = false` is mandatory — see variables.tf for why.
# -----------------------------------------------------------------------------

locals {
  # nonsensitive() — the token itself is secret, but "is the token set?" is a
  # boolean we want to surface in outputs and use in resource counts without
  # Terraform tainting every downstream value as sensitive.
  cloudflare_enabled = local.tls_enabled && nonsensitive(var.cloudflare_api_token != "") && var.cloudflare_zone_id != ""
}

resource "cloudflare_record" "api" {
  count   = local.cloudflare_enabled ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "api.${var.domain}"
  type    = "A"
  content = aws_eip.main.public_ip
  ttl     = 60
  proxied = false
  comment = "Managed by Terraform (project-800ms) — do not proxy, WebRTC + HTTP-01 break behind CF proxy"
}

resource "cloudflare_record" "livekit" {
  count   = local.cloudflare_enabled ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "livekit.${var.domain}"
  type    = "A"
  content = aws_eip.main.public_ip
  ttl     = 60
  proxied = false
  comment = "Managed by Terraform (project-800ms) — do not proxy, WebRTC + HTTP-01 break behind CF proxy"
}
