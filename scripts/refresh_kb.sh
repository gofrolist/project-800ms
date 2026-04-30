#!/usr/bin/env bash
# refresh_kb.sh — fetch the Chatwoot help-base feed and run KB ingestion.
#
# Designed to be cron-callable. Logs a single-line JSON record per phase
# to stdout (parseable by ops grep / log aggregator) and full diagnostic
# loguru output to stderr.
#
# Usage:
#   ./scripts/refresh_kb.sh              # default: tenant=demo, project=arizona
#   ./scripts/refresh_kb.sh demo arizona # explicit
#
# Exit codes:
#   0   Both fetch and ingest succeeded.
#   2   Fetch step failed (network / auth / API). DB unchanged.
#   3   Ingest step failed or partial-success. Some KB drift may have
#       landed; the next run resumes from where this one stopped.
#
# Requirements on the host:
#   - infra/.env with CHATWOOT_HELP_BASE_TOKEN set (written by
#     terraform-gcp's startup_script.sh).
#   - python3 in $PATH (any 3.10+ — the fetch script is stdlib-only).
#   - docker compose with a healthy `retriever` container (the
#     ingest step runs inside it).
#
# To wire as a daily cron on the VM:
#   sudo crontab -e
#   0 4 * * * /opt/project-800ms/scripts/refresh_kb.sh demo arizona >> \
#       /var/log/project-800ms-kb-refresh.log 2>&1

set -euo pipefail

TENANT="${1:-demo}"
PROJECT="${2:-arizona}"

# Validate the positional args before they flow into filesystem paths
# and python3 invocations. The strict alphabet (letters/digits/dot/dash/
# underscore) blocks shell metacharacters, single-quotes (which would
# break out of the python3 -c string literal below), path separators,
# and whitespace. Cron entries on the VM use fixed args; this guard
# protects ad-hoc operator invocations + future automation.
if [[ ! "${TENANT}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "refresh_kb: invalid TENANT ${TENANT@Q} — must match [A-Za-z0-9._-]+" >&2
    exit 2
fi
if [[ ! "${PROJECT}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "refresh_kb: invalid PROJECT ${PROJECT@Q} — must match [A-Za-z0-9._-]+" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/infra/.env"
DATA_DIR="${REPO_ROOT}/data/kb/${PROJECT}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "refresh_kb: missing ${ENV_FILE} — bootstrap the VM first." >&2
    exit 2
fi

# Source the .env to pull CHATWOOT_HELP_BASE_TOKEN into this shell. The
# `set -a` / `set +a` pair auto-exports each assignment so it crosses
# the python3 boundary; otherwise the var would only exist in the
# current shell and the fetch script would see it empty.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -z "${CHATWOOT_HELP_BASE_TOKEN:-}" ]]; then
    echo "refresh_kb: CHATWOOT_HELP_BASE_TOKEN is empty in ${ENV_FILE}." >&2
    echo "             Set chatwoot_help_base_token in terraform.tfvars and apply." >&2
    exit 2
fi

echo "refresh_kb: phase=fetch project=${PROJECT}" >&2
if ! python3 "${REPO_ROOT}/tools/fetch_chatwoot_kb.py" \
        --project "${PROJECT}" \
        --out-dir "${DATA_DIR}"; then
    echo "refresh_kb: fetch step failed" >&2
    exit 2
fi

# Sanity check: the manifest should exist and report >0 articles. An
# upstream auth flip can return success with zero articles, which the
# ingest step's mass-deletion safeguard would catch — but we surface
# the empty case here so the operator sees a clear log line.
MANIFEST="${DATA_DIR}/_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "refresh_kb: manifest absent at ${MANIFEST} — fetch produced no output" >&2
    exit 2
fi
# Pass MANIFEST as argv[1] rather than embedding it into the python -c
# source string. Inline interpolation would expose the script to shell-
# to-Python injection if PROJECT ever contained a single-quote (now
# impossible thanks to the regex guard above, but the argv form removes
# the risk class entirely and survives future regex edits). Capturing
# the exit status separately keeps ``set -e`` honest — bash does NOT
# propagate non-zero exits from command-substitution-into-assignment
# even with -e + -o pipefail.
if ! COUNT="$(python3 -c '
import json
import sys
print(json.load(open(sys.argv[1]))["count"])
' "${MANIFEST}")"; then
    echo "refresh_kb: failed to read count from ${MANIFEST}" >&2
    exit 2
fi
if [[ -z "${COUNT}" || "${COUNT}" == "0" ]]; then
    echo "refresh_kb: upstream returned 0 articles — refusing to run ingest" >&2
    exit 2
fi
echo "refresh_kb: phase=fetch ok count=${COUNT}" >&2

# The ingest step runs INSIDE the retriever container so it reuses the
# already-loaded BGE-M3 embedder (saves ~6s of cold load per run) and
# inherits DB credentials from the container env. The data dir is
# bind-mounted at /app/data (see infra/docker-compose.yml).
echo "refresh_kb: phase=ingest tenant=${TENANT} namespace=chatwoot" >&2
INGEST_CMD=(
    docker compose
    --env-file "${ENV_FILE}"
    -f "${REPO_ROOT}/infra/docker-compose.yml"
    exec -T retriever
    uv run python -m ingest
    --tenant "${TENANT}"
    --source "/app/data/kb/${PROJECT}/"
    --mode incremental
)

# Apply prod overlay when present — same pattern the startup script uses.
if [[ -f "${REPO_ROOT}/infra/docker-compose.prod.yml" ]]; then
    INGEST_CMD=(
        docker compose
        --env-file "${ENV_FILE}"
        -f "${REPO_ROOT}/infra/docker-compose.yml"
        -f "${REPO_ROOT}/infra/docker-compose.prod.yml"
        exec -T retriever
        uv run python -m ingest
        --tenant "${TENANT}"
        --source "/app/data/kb/${PROJECT}/"
        --mode incremental
    )
fi

if ! "${INGEST_CMD[@]}"; then
    EXIT=$?
    if [[ "${EXIT}" == "75" ]]; then
        echo "refresh_kb: ingest reported partial success — see retriever logs" >&2
        exit 3
    fi
    echo "refresh_kb: ingest failed (exit ${EXIT})" >&2
    exit 3
fi

echo "refresh_kb: phase=ingest ok" >&2
