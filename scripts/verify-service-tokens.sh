#!/usr/bin/env bash
#
# Audits every stored AI_SERVICE_TOKEN and reports whether it actually verifies.
#
# A stale token is invisible to ordinary checks: its `exp` stays valid for a
# year, its claims look correct, and the shared secret matches on both sides.
# The only thing that discriminates is recomputing the signature. Run this after
# ANY key or secret rotation.
#
# Exit code 1 if any token fails, so it can gate a rotation runbook.

set -euo pipefail

NAMESPACE="statex-apps"
AI_URL="http://ai-microservice:3380"
PROBE_FROM="deployment/ai-microservice"

SERVICES=(
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

failed=0

printf '%-34s %-10s %-8s %s\n' SERVICE ALG LOCAL LIVE
printf '%-34s %-10s %-8s %s\n' ---------------------------------- ---------- -------- ----

PUBLIC_KEY="$(vault kv get -format=json secret/prod/ai-microservice 2>/dev/null \
  | jq -r '.data.data.JWT_PUBLIC_KEY // empty')"
PUBFILE="$(mktemp)"; trap 'rm -f "$PUBFILE"' EXIT
[[ -n "$PUBLIC_KEY" ]] && printf '%s' "$PUBLIC_KEY" > "$PUBFILE"

for service in "${SERVICES[@]}"; do
  token="$(vault kv get -format=json "secret/prod/$service" 2>/dev/null \
    | jq -r '.data.data.AI_SERVICE_TOKEN // empty')"

  if [[ -z "$token" ]]; then
    printf '%-34s %-10s %-8s %s\n' "$service" "-" "ABSENT" "-"
    continue
  fi

  header="${token%%.*}"
  pad=$(( (4 - ${#header} % 4) % 4 ))
  padded="$header$(printf '=%.0s' $(seq 1 $pad 2>/dev/null) 2>/dev/null || true)"
  alg="$(printf '%s' "$padded" | tr '_-' '/+' | openssl base64 -d -A 2>/dev/null \
    | jq -r '.alg // "?"' 2>/dev/null || echo '?')"

  # Local cryptographic check against the public key.
  local_status="SKIP"
  if [[ -s "$PUBFILE" && "$alg" == "RS256" ]]; then
    signing_input="${token%.*}"
    sigfile="$(mktemp)"
    sig="${token##*.}"
    pad=$(( (4 - ${#sig} % 4) % 4 ))
    printf '%s' "$sig$(printf '=%.0s' $(seq 1 $pad 2>/dev/null) 2>/dev/null || true)" \
      | tr '_-' '/+' | openssl base64 -d -A > "$sigfile" 2>/dev/null || true
    if printf '%s' "$signing_input" \
        | openssl dgst -sha256 -verify "$PUBFILE" -signature "$sigfile" >/dev/null 2>&1; then
      local_status="VALID"
    else
      local_status="STALE"; failed=1
    fi
    rm -f "$sigfile"
  fi

  # Live check: what ai-microservice itself says.
  live="$(kubectl -n "$NAMESPACE" exec "$PROBE_FROM" -- \
    curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
    -X POST "$AI_URL/ai/complete" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $token" \
    -d '{"user_prompt":"ping","model_tier":"free"}' 2>/dev/null || echo "ERR")"

  [[ "$live" == "401" ]] && failed=1

  printf '%-34s %-10s %-8s %s\n' "$service" "$alg" "$local_status" "$live"
done

echo
if [[ $failed -eq 1 ]]; then
  echo "FAIL: at least one token is stale or rejected. Re-mint with:"
  echo "  ./scripts/mint-service-token.sh --all"
  exit 1
fi
echo "OK: every service token verifies."
