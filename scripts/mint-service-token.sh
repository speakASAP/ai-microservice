#!/usr/bin/env bash
#
# Mints an RS256 service token for a caller of ai-microservice, writes it to
# that caller's Vault path, and restarts the caller so it picks the token up.
#
# This exists because the 2026-08-01 rotation changed the shared JWT_SECRET
# without re-minting the tokens signed by it. Nine services kept serving stale
# tokens whose `exp` was still a year out, so nothing looked expired — they just
# failed signature verification with a 401. Re-minting must never again be a
# manual step someone can forget.
#
# Usage:
#   ./scripts/mint-service-token.sh <service-slug> [--ttl-days N] [--dry-run]
#   ./scripts/mint-service-token.sh --all [--ttl-days N] [--dry-run]
#
# Requires: vault (authenticated), kubectl, openssl, jq.

set -euo pipefail

VAULT_KEY_PATH="secret/prod/ai-microservice"
PRIVATE_KEY_FIELD="JWT_PRIVATE_KEY"
NAMESPACE="statex-apps"
TTL_DAYS=365
DRY_RUN=0
TARGETS=()

# Services that authenticate to ai-microservice with AI_SERVICE_TOKEN.
ALL_SERVICES=(
  agentic-email-processing-system
  business-orchestrator
  crypto-ai-agent
  domain-research
  flipflop-service
  notifications-microservice
  runlayer
  shop-assistant
  statex
)

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)      TARGETS=("${ALL_SERVICES[@]}"); shift ;;
    --ttl-days) TTL_DAYS="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    -*)         die "unknown flag: $1" ;;
    *)          TARGETS+=("$1"); shift ;;
  esac
done

[[ ${#TARGETS[@]} -eq 0 ]] && die "no service specified (use a slug or --all)"

command -v vault  >/dev/null || die "vault CLI not found"
command -v jq     >/dev/null || die "jq not found"
command -v openssl>/dev/null || die "openssl not found"

base64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

PRIVATE_KEY="$(vault kv get -format=json "$VAULT_KEY_PATH" 2>/dev/null \
  | jq -r --arg f "$PRIVATE_KEY_FIELD" '.data.data[$f] // empty')"
[[ -n "$PRIVATE_KEY" ]] || die "no $PRIVATE_KEY_FIELD at $VAULT_KEY_PATH — run bootstrap-jwt-keypair.sh first"

KEYFILE="$(mktemp)"
chmod 600 "$KEYFILE"
trap 'rm -f "$KEYFILE"' EXIT
printf '%s' "$PRIVATE_KEY" > "$KEYFILE"

mint_token() {
  local service_id="$1"
  local now exp header payload signing_input signature
  now="$(date +%s)"
  exp=$(( now + TTL_DAYS * 24 * 3600 ))

  header="$(printf '{"alg":"RS256","typ":"JWT"}' | base64url)"
  payload="$(printf '{"serviceId":"%s","iss":"ai-microservice","iat":%s,"exp":%s}' \
    "$service_id" "$now" "$exp" | base64url)"
  signing_input="${header}.${payload}"
  signature="$(printf '%s' "$signing_input" \
    | openssl dgst -sha256 -sign "$KEYFILE" -binary | base64url)"

  printf '%s.%s' "$signing_input" "$signature"
}

for service in "${TARGETS[@]}"; do
  echo "==> $service"
  token="$(mint_token "$service")"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] would write AI_SERVICE_TOKEN to secret/prod/$service"
    echo "    [dry-run] would restart deployment/$service in $NAMESPACE"
    continue
  fi

  # Read-modify-write: vault kv put replaces the whole map, so every other key
  # at this path must be carried over or it is silently destroyed.
  existing="$(vault kv get -format=json "secret/prod/$service" 2>/dev/null | jq '.data.data')"
  [[ -n "$existing" && "$existing" != "null" ]] || die "cannot read secret/prod/$service"

  updated="$(jq --arg t "$token" '.AI_SERVICE_TOKEN = $t' <<<"$existing")"
  changed="$(jq -n --argjson a "$existing" --argjson b "$updated" \
    '[$a | to_entries[] | select(.value != ($b[.key])) | .key] | join(",")' -r)"
  [[ "$changed" == "AI_SERVICE_TOKEN" || "$changed" == "" ]] \
    || die "refusing to write: unexpected key changes ($changed)"

  vault kv put "secret/prod/$service" - <<<"$updated" >/dev/null
  echo "    vault: AI_SERVICE_TOKEN updated"

  # ESO syncs the K8s Secret, but env vars are only read at container start —
  # without a restart the pod keeps serving the old token.
  if kubectl -n "$NAMESPACE" get externalsecret "${service}-secret" >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" annotate externalsecret "${service}-secret" \
      force-sync="$(date +%s)" --overwrite >/dev/null
    echo "    eso: resync triggered"
  fi

  if kubectl -n "$NAMESPACE" get deployment "$service" >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" rollout restart "deployment/$service" >/dev/null
    echo "    k8s: rollout restarted"
  else
    echo "    k8s: no deployment/$service in $NAMESPACE — restart it wherever it runs"
  fi
done

echo
echo "Done. Verify with: ./scripts/verify-service-tokens.sh"
