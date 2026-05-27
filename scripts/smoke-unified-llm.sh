#!/usr/bin/env bash
# Smoke test for ai-microservice NestJS endpoints.
# Tests /health and POST /ai/complete via the ingress (https://ai.alfares.cz) or localhost.
set -euo pipefail

HOST="${AI_SERVICE_HOST:-localhost}"
PORT="${AI_SERVICE_PORT:-3380}"
BASE="http://${HOST}:${PORT}"

pass() { echo "OK  $*"; }
fail() { echo "FAIL $*" >&2; FAILED=1; }
FAILED=0

have_curl() { command -v curl >/dev/null 2>&1; }

if ! have_curl; then
  echo "SKIP: curl not available"
  exit 0
fi

# 1. Health check
code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 8 "${BASE}/health" || echo 000)"
[[ "$code" == "200" ]] && pass "/health returned 200" || fail "/health returned HTTP ${code}"

# 2. POST /ai/complete — only if ANTHROPIC_API_KEY available in env
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  body='{"model_tier":"free","user_prompt":"Reply with the single word: OK"}'
  response="$(curl -sS --connect-timeout 5 --max-time 30 \
    -X POST "${BASE}/ai/complete" \
    -H 'Content-Type: application/json' \
    -d "$body" || echo '{}')"
  if echo "$response" | grep -q '"model_used"'; then
    model_used="$(echo "$response" | grep -o '"model_used":"[^"]*"' | head -1)"
    pass "/ai/complete responded — ${model_used}"
  else
    fail "/ai/complete unexpected response: ${response:0:200}"
  fi
else
  echo "SKIP /ai/complete: ANTHROPIC_API_KEY not set in environment"
fi

echo "Smoke test finished${FAILED:+ — FAILURES DETECTED}"
exit $FAILED
