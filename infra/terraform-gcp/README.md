# infra/terraform-gcp

Deploys the project-800ms voice stack to a single GPU Compute Engine instance on GCP.

Parallel to `infra/terraform/` (AWS). Same app stack, same Caddy/Cloudflare integration, same docker-compose layout — only the provider changes.

## Why GCP

- **Better GPU availability** than AWS today — new accounts can land L4 on-demand in minutes.
- **Cleaner spot model**: Spot VMs use the same capacity pools as on-demand (just lower priority), so you don't hit `capacity-not-available` errors like AWS spot. Preemption is the tradeoff, not scarcity.
- **Simpler IAM**: service accounts with per-secret bindings vs AWS's role + policy + SSM prefix dance.

## What it creates

- **VPC + subnetwork** (dedicated, not the default)
- **Firewall rules** gated on network tag `project-800ms` — ports depend on TLS (see below)
- **Static external IP** so the public endpoint survives preemption
- **Dedicated service account** with per-secret `secretAccessor` (not project-wide) + `logging.logWriter` + `monitoring.metricWriter`
- **Secret Manager secrets** (one per credential) with the instance SA bound to each
- **Compute instance** with 1x L4 GPU on Deep Learning VM image, Spot or on-demand toggled by `use_spot`
- **Optional Caddy TLS** via the same docker-compose overlay you use on AWS
- **Optional Cloudflare DNS records** for `api.$domain` and `livekit.$domain` (proxied=false enforced)

## Prerequisites

1. **Terraform** >= 1.5
2. **gcloud CLI** authenticated: `gcloud auth application-default login`
3. **A GCP project with billing linked:**
   ```bash
   gcloud projects create <PROJECT_ID>
   gcloud billing projects link <PROJECT_ID> --billing-account=<BILLING_ACCOUNT_ID>
   gcloud config set project <PROJECT_ID>
   gcloud services enable compute.googleapis.com secretmanager.googleapis.com \
     iam.googleapis.com cloudresourcemanager.googleapis.com
   ```
4. **GPU quota.** New projects start at 0 for the global `GPUS_ALL_REGIONS` quota. Request increase to 1 via:
   > Cloud Console → IAM & Admin → Quotas → filter "GPUs (all regions)" → Edit Quotas → request 1
   Typically auto-approves in 5-15 min. Region-level L4 quota (`NVIDIA_L4_GPUS` in your chosen region) is usually already 1 by default.

## Deploy — plaintext (quick test)

```bash
cd infra/terraform-gcp
cp terraform.tfvars.example terraform.tfvars
# Set project_id, git_repo, and generate secrets:
openssl rand -hex 32  # for each of postgres_password, redis_password, livekit_api_secret, vllm_api_key

terraform init
terraform plan
terraform apply
```

Outputs include `api_url` and `livekit_ws_url` — both plain http/ws on the static IP.

## Deploy — with TLS + Cloudflare (recommended for anything real)

```bash
# In terraform.tfvars, set:
#   domain               = "voice.example.com"
#   tls_email            = "you@example.com"
#   cloudflare_api_token = "<Edit zone DNS token>"
#   cloudflare_zone_id   = "<from CF dashboard>"

terraform apply

# Verify:
dig +short api.voice.example.com   # should return the static IP
terraform output cloudflare_managed  # true

# Watch Caddy get certs:
$(terraform output -raw ssh_connect)
sudo docker compose -f /opt/project-800ms/infra/docker-compose.yml \
                    -f /opt/project-800ms/infra/docker-compose.prod.yml \
                    -f /opt/project-800ms/infra/docker-compose.tls.yml \
                    logs -f caddy
# Look for: "certificate obtained successfully"

curl https://api.voice.example.com/health
```

> **Do not enable the Cloudflare proxy (orange cloud).** Records are created with `proxied=false` and must stay that way. CF proxy drops WebRTC UDP and breaks Let's Encrypt HTTP-01.

## Verification

```bash
# Shell into the instance (no SSH keys needed — IAP tunnel)
$(terraform output -raw ssh_connect)

# Inside the instance:
sudo tail -f /var/log/project-800ms-bootstrap.log
sudo docker ps

# From your laptop (plaintext):
curl "$(terraform output -raw api_url)/health"

# From your laptop (TLS):
curl "$(terraform output -raw api_url)/health"  # https://api.$domain/health
```

## Updating the deployed code

1. **Hot path** — push to `main`, SSH into the box, pull new images:
   ```bash
   $(terraform output -raw ssh_connect)
   cd /opt/project-800ms
   sudo git pull
   COMPOSE="-f infra/docker-compose.yml -f infra/docker-compose.prod.yml"
   # Add -f infra/docker-compose.tls.yml if TLS is on
   sudo docker compose --env-file infra/.env $COMPOSE pull
   sudo docker compose --env-file infra/.env $COMPOSE up -d
   ```

2. **Via Terraform** — bump `image_tag` to a specific `sha-xxxxxxx` and re-apply. `metadata_startup_script` only runs on first boot, so changing it alone won't re-deploy. Recreate the instance via `terraform taint google_compute_instance.main` to force a fresh boot (loses state).

## Switching LLM provider (Groq / OpenAI / etc.)

The agent talks to any OpenAI-compatible endpoint. Default is the bundled local vLLM running Qwen-7B on the L4. To swap to an external provider, set three Terraform variables:

```hcl
# terraform.tfvars
llm_base_url = "https://api.groq.com/openai/v1"
llm_model    = "llama-3.3-70b-versatile"
llm_api_key  = "gsk_..."  # https://console.groq.com/keys
```

Run `terraform apply` — the instance is recreated (startup script changed), the agent boots pointed at Groq, and the local vLLM container keeps running in the background (unused, wasting ~11 GB GPU mem but easy to revert). Once you're committed to the external provider, you can drop vLLM from compose to free the GPU — or stop paying for a GPU box entirely and move to the hybrid architecture noted in the root README.

Empty values for all three = use local vLLM (default).

## Rotating secrets

Secret Manager holds the truth — no re-apply needed:

```bash
# Rotate postgres_password
NEW_PW=$(openssl rand -hex 32)
echo -n "$NEW_PW" | gcloud secrets versions add project-800ms-postgres-password --data-file=-

# Re-run the bootstrap's .env-write step manually, then recreate containers:
$(terraform output -raw ssh_connect)
# ... re-fetch, update infra/.env, docker compose up -d --force-recreate postgres
```

Keep `terraform.tfvars` in sync or the next `terraform apply` overwrites back.

## Cost

us-central1 (2026):

| | On-demand | Spot |
|---|---|---|
| `g2-standard-8` (L4, 8 vCPU, 32 GB) | ~$0.76/hr | ~$0.23-0.30/hr |
| `g2-standard-4` (L4, 4 vCPU, 16 GB) | ~$0.57/hr | ~$0.17-0.22/hr |

Plus:
- ~$0.10/GB/month for the pd-balanced boot disk (default 200 GB ≈ $20/mo)
- Static external IP: free while attached
- Secret Manager: free for the first 6 active secret versions / 10k access calls
- Egress to the internet: $0.085/GB for the first 10 TB/month (us-central1)

## Security notes

### What is locked down

- **Shielded VM** — secure boot + vTPM + integrity monitoring.
- **Service account least privilege** — per-secret `secretAccessor`, narrow logging/monitoring roles. No Editor.
- **Secrets in Secret Manager** — Google-managed KMS at rest, fetched at boot via SA token from metadata server. Plaintext secrets never appear in instance metadata.
- **Boot disk encrypted** with Google-managed keys by default.
- **Bootstrap log** is `root:root 0600`.
- **IAP SSH always on, public SSH optional** — `gcloud compute ssh --tunnel-through-iap` works without a public SSH port; `allowed_ssh_cidr` defaults to "" (closed).
- **OS Login enforced** — project SSH keys disabled; access via IAM, auditable.
- **With TLS on:** plaintext 7880/7881/8000 firewall rule is not created; everything flows through Caddy on 443 with HSTS.

### Known gaps

- **Secrets in Terraform state.** They live in Secret Manager on the cloud side, but also in `terraform.tfstate`. Use a GCS remote backend for any real deployment and don't commit `terraform.tfvars`.
- **Open ingress by default** — `allowed_app_cidr = "0.0.0.0/0"` is correct for a public voice app. Narrow for private demos.
- **No TURN/TLS on 5349.** If your users are on networks that block the 50000-50099 UDP range, media will fail. Add a TURN server (coturn or LiveKit's built-in) if needed.

## Destroy

```bash
terraform destroy
```

Secret Manager secrets are soft-deleted for 30 days by default (configurable) — if you `terraform apply` with the same project name during that window, Terraform may fail because the secret ID is taken. Purge with `gcloud secrets delete <name>` to recreate immediately.

## AWS vs GCP parity cheat sheet

| AWS | GCP |
|---|---|
| VPC + subnet + IGW + route table | VPC + subnetwork (default routes implicit) |
| Security group | Firewall rule, tag-targeted |
| Elastic IP | Static external IP (`google_compute_address`) |
| SSM Parameter Store | Secret Manager |
| IAM role + instance profile | Service account |
| Deep Learning AMI | Deep Learning VM image |
| Spot instance request | Spot VM (via scheduling block) |
| `user_data` | `metadata_startup_script` |
| SSM Session Manager | `gcloud compute ssh --tunnel-through-iap` |
