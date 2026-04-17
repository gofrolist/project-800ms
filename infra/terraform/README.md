# infra/terraform

Deploys the project-800ms voice stack to a single GPU EC2 spot instance.

## What it creates

- **Persistent spot request** for a GPU instance (default: `g6.xlarge`)
- **Elastic IP** so the public endpoint survives spot stop/start
- **Security group** — ports depend on whether TLS is enabled (see below)
- **IAM role** with SSM Session Manager + scoped read on the project's SSM parameters
- **SSM Parameter Store SecureStrings** for every secret, KMS-encrypted
- **cloud-init** that fetches secrets from SSM at boot, clones the repo, writes `.env`, and runs `docker compose up`
- **Optional:** Caddy for automatic Let's Encrypt TLS (when `domain` is set)

## Why spot + stop-on-interrupt (not ASG)

The stack is stateful (Postgres, Redis, LiveKit) and monolithic (single docker-compose
file, GPU shared by `vllm` and `agent`). A standard ASG would destroy state on every
interruption.

Instead we use a **persistent** spot request with `instance_interruption_behavior = "stop"`:

- On spot interruption, AWS **stops** the instance (doesn't terminate).
- The **EBS root volume is preserved**, including Postgres data and the model cache.
- When spot capacity returns, AWS **restarts** the same instance automatically.
- The EIP re-associates, and `docker compose` services come back via `restart: unless-stopped`.

Expected downtime per interruption: ~2 minutes of warning + however long until capacity returns.

## Prerequisites

1. **Terraform** >= 1.5
2. **AWS credentials** configured (`aws configure sso` recommended, see main project docs)
3. **Spot quota** for the G family — check Service Quotas → EC2 → *All G and VT Spot Instance Requests*. Default is 0; request 8 vCPUs minimum.
4. **A git URL** the instance can clone (your public repo)
5. **For TLS (strongly recommended):** a domain you control where you can create DNS A records.

## Deploy — plaintext (quick test)

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Generate secrets and paste into terraform.tfvars:
openssl rand -hex 32  # for each of postgres_password, redis_password, livekit_api_secret, vllm_api_key

terraform init
terraform plan
terraform apply
```

Outputs include `api_url` and `livekit_ws_url` — both plain http/ws on the EIP.

## Deploy — with TLS (recommended for anything real)

```bash
# 1. In terraform.tfvars, set:
#      domain    = "voice.example.com"
#      tls_email = "you@example.com"
#
# 2. Apply — this creates the EIP and instance but Caddy will fail to get a
#    cert until DNS is pointed at the EIP:
terraform apply

# 3. Look at outputs:
terraform output dns_records_needed
# {
#   "api.voice.example.com"     = "52.x.y.z"
#   "livekit.voice.example.com" = "52.x.y.z"
# }

# 4. Create those A records in your DNS provider (Route53, Cloudflare, etc.)
#    and wait for propagation:
dig +short api.voice.example.com
# 52.x.y.z

# 5. Caddy on the instance polls for DNS + requests certs automatically.
#    Watch it:
$(terraform output -raw ssm_connect)
sudo docker compose -f /opt/project-800ms/infra/docker-compose.yml \
                    -f /opt/project-800ms/infra/docker-compose.prod.yml \
                    -f /opt/project-800ms/infra/docker-compose.tls.yml \
                    logs -f caddy
# Look for: "certificate obtained successfully"

# 6. From your laptop:
curl https://api.voice.example.com/health
```

## Deploy — with TLS + Cloudflare DNS automation

If the zone is on Cloudflare, skip the manual DNS step. Terraform manages the
api/livekit A records for you.

```bash
# 1. In the Cloudflare dashboard, get:
#    - Zone ID: your domain → Overview → API section (right sidebar)
#    - API Token: My Profile → API Tokens → Create Token → "Edit zone DNS"
#      template, restricted to the target zone only. Not the Global API Key.

# 2. In terraform.tfvars, set:
#      domain               = "voice.example.com"
#      tls_email            = "you@example.com"
#      cloudflare_api_token = "..."
#      cloudflare_zone_id   = "..."

# 3. Apply — Terraform now creates the EIP, instance, AND the DNS records:
terraform apply

# 4. Verify DNS + cert:
dig +short api.voice.example.com   # → the EIP
terraform output cloudflare_managed  # → true
$(terraform output -raw ssm_connect)
sudo docker compose -f /opt/project-800ms/infra/docker-compose.yml \
                    -f /opt/project-800ms/infra/docker-compose.prod.yml \
                    -f /opt/project-800ms/infra/docker-compose.tls.yml \
                    logs -f caddy
# Look for: "certificate obtained successfully"

# 5. From your laptop:
curl https://api.voice.example.com/health
```

> **Do not turn on the Cloudflare proxy (orange cloud).** Records are created
> with `proxied = false` and must stay that way. The CF proxy cannot pass
> WebRTC UDP traffic (LiveKit media on 50000-50099) and it breaks Let's
> Encrypt HTTP-01 challenges. Orange cloud = audio drops + no certs.

## What verification looks like

```bash
# Outputs after apply
terraform output

# Shell into the instance (no SSH key needed)
$(terraform output -raw ssm_connect)

# Inside the instance:
sudo tail -f /var/log/project-800ms-bootstrap.log
sudo docker ps

# Plaintext mode:
curl "$(terraform output -raw api_url)/health"

# TLS mode:
curl "$(terraform output -raw api_url)/health"  # https://api.$domain/health
```

## Updating the deployed code

Two options:

1. **Hot path** — push to `main`, SSM into the box, pull new images:
   ```bash
   cd /opt/project-800ms
   sudo git pull
   COMPOSE="-f infra/docker-compose.yml -f infra/docker-compose.prod.yml"
   # Add -f infra/docker-compose.tls.yml if TLS is on
   sudo docker compose --env-file infra/.env $COMPOSE pull
   sudo docker compose --env-file infra/.env $COMPOSE up -d
   ```

2. **Via Terraform** — bump `image_tag` to a specific `sha-xxxxxxx` and re-apply. Note: `user_data` only runs on first boot, so changing it alone won't re-deploy — taint the instance (`terraform taint aws_spot_instance_request.main`) to force a fresh boot, which loses state.

## Rotating secrets

SSM params are the source of truth — no re-apply needed:

```bash
# Rotate postgres_password
NEW_PW=$(openssl rand -hex 32)
aws ssm put-parameter \
  --name /project-800ms/postgres_password \
  --value "$NEW_PW" \
  --type SecureString \
  --overwrite \
  --region us-west-2

# Re-run the bootstrap's .env-write step manually on the instance, then
# recreate the affected containers:
$(terraform output -raw ssm_connect)
# ... fetch new value and update infra/.env
# sudo docker compose ... up -d --force-recreate postgres
```

Keep `terraform.tfvars` in sync with the new value, or you'll overwrite it back on the next `terraform apply`.

## Cost

us-west-2 ballpark (2026):

| | On-demand | Spot |
|---|---|---|
| `g6.xlarge` | ~$0.805/hr | ~$0.25-0.35/hr |
| `g5.xlarge` | ~$1.006/hr | ~$0.30-0.40/hr |

Plus:
- ~$0.08/GB/month for the EBS root volume (default 200 GB ≈ $16/mo)
- Elastic IP: free while attached
- SSM Parameter Store SecureString: free for the first 10k standard params
- Caddy (when TLS on): zero extra cost (runs on the same instance)

## Security notes

> This module is built for a single-box MVP demo. It's reasonable to expose
> to real users **if and only if** you set `domain` / `tls_email`. Without
> TLS, treat it as a private throwaway.

### What is locked down

- **IMDSv2 required** (`http_tokens = required`, `hop_limit = 1`) — containers on the default docker bridge network cannot reach the instance metadata service, blocking SSRF-to-credentials attacks.
- **Secrets in SSM Parameter Store SecureStrings** — KMS-encrypted at rest, fetched at boot via the instance role. Plaintext secrets do NOT appear in the cloud-init user-data blob on the host.
- **IAM role scoped tight** — SSM Session Manager + read-only on `/${project_name}/*` parameters + decrypt via the SSM KMS key. Nothing else.
- **Root EBS encrypted** with the default AWS KMS key.
- **Bootstrap script** runs without `set -x` and its log file is `root:root 0600`.
- **Secret length validated at plan time** — short/placeholder passwords fail `terraform plan`.
- **Compose-level auth** on Postgres, Redis, and vLLM — even loopback traffic requires credentials.
- **Internal services stay internal** — Postgres, Redis, vLLM are bound to `127.0.0.1` inside the compose file and not exposed by the security group.
- **With TLS on:** plaintext ports 7880 / 7881 / 8000 are closed at the SG level; everything flows through Caddy on 443 with HSTS and modern-only TLS.

### Known gaps

- **Secrets in Terraform state.** SSM moved them out of user-data but they still live in `terraform.tfstate`. Use an encrypted remote backend (S3 + DynamoDB locking) for any real deployment, and don't commit `terraform.tfvars`.
- **Open ingress by default.** `allowed_app_cidr = "0.0.0.0/0"` — correct for a public voice app, but narrow it for private demos.
- **No TURN/TLS on 5349.** If your users are on networks that block the 50000-50099 UDP range, media will fail. Add a TURN server (coturn or LiveKit's built-in) if you see this in practice.
- **Apt packages not upgraded** at boot — short-lived spot box, DLAMI is patched recently enough. Bake a custom AMI if you need hardened base.

## Destroy

```bash
terraform destroy
```

> **Gotcha:** `aws_spot_instance_request` cancels the spot request on destroy but occasionally the underlying instance lingers. Check the EC2 console after destroy and terminate by hand if needed.
