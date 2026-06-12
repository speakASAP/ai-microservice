#!/usr/bin/env bash
# Smoke test for ai-microservice production-safe endpoints and contracts.
set -euo pipefail

HOST="${AI_SERVICE_HOST:-localhost}"
PORT="${AI_SERVICE_PORT:-3380}"
BASE="${AI_SERVICE_BASE_URL:-http://${HOST}:${PORT}}"
BASE="${BASE%/}"

pass() { echo "OK  $*"; }
fail() { echo "FAIL $*" >&2; FAILED=1; }
FAILED=0

have_curl() { command -v curl >/dev/null 2>&1; }

curl_json() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local out_file="$4"
  local code
  local auth_args=()

  if [[ -n "${AI_SERVICE_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${AI_SERVICE_TOKEN}")
  fi

  if [[ "$method" == "GET" ]]; then
    code="$(curl -sS -o "$out_file" -w '%{http_code}' --connect-timeout 5 --max-time 12 "${auth_args[@]}" "${BASE}${path}" || echo 000)"
  else
    code="$(curl -sS -o "$out_file" -w '%{http_code}' --connect-timeout 5 --max-time 30 \
      -X "$method" "${BASE}${path}" \
      "${auth_args[@]}" \
      -H 'Content-Type: application/json' \
      -d "$body" || echo 000)"
  fi
  printf '%s' "$code"
}

assert_body_contains() {
  local file="$1"
  local needle="$2"
  local label="$3"
  if grep -q "$needle" "$file"; then
    pass "$label"
  else
    fail "$label missing ${needle}; body=$(head -c 220 "$file")"
  fi
}

if ! have_curl; then
  echo "SKIP: curl not available"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Smoke target: ${BASE}"

health_body="$TMP_DIR/health.json"
code="$(curl_json GET /health '' "$health_body")"
[[ "$code" == "200" ]] && pass "/health returned 200" || fail "/health returned HTTP ${code}"
assert_body_contains "$health_body" '"status":"ok"' "/health status"
assert_body_contains "$health_body" '"service":"ai-microservice"' "/health service"

if [[ -z "${AI_SERVICE_TOKEN:-}" ]]; then
  echo "SKIP protected /ai/* contract checks: set AI_SERVICE_TOKEN"
  if [[ "$FAILED" -eq 0 ]]; then
    echo "Smoke test finished"
  else
    echo "Smoke test finished - FAILURES DETECTED"
  fi
  exit "$FAILED"
fi

premium_body="$TMP_DIR/premium.json"
code="$(curl_json POST /ai/complete '{"model_tier":"premium","user_prompt":"deployment smoke premium approval guard"}' "$premium_body")"
[[ "$code" == "200" ]] && pass "/ai/complete premium guard returned 200 contract response" || fail "/ai/complete premium guard HTTP ${code}"
assert_body_contains "$premium_body" '"error_code":"AI_AUTH_ERROR"' "/ai/complete premium approval block"
assert_body_contains "$premium_body" 'Premium tier requires explicit human approval' "/ai/complete premium approval message"

if [[ "${AI_SMOKE_CHECK_AGENT_ROUTING:-false}" == "true" ]]; then
  agent_body="$TMP_DIR/agent.json"
  code="$(curl_json POST /ai/complete '{"model_tier":"free","user_prompt":"deployment smoke agent miss","agent_slug":"deployment-smoke-agent-missing"}' "$agent_body")"
  [[ "$code" == "200" ]] && pass "/ai/complete agent routing error returned 200 contract response" || fail "/ai/complete agent routing HTTP ${code}"
  assert_body_contains "$agent_body" '"error_code":"AGENT_NOT_AVAILABLE"' "/ai/complete inactive agent block"
  assert_body_contains "$agent_body" '"model_used":"agent-registry"' "/ai/complete agent registry model marker"
else
  echo "SKIP /ai/complete agent routing smoke: set AI_SMOKE_CHECK_AGENT_ROUTING=true after GOAL-05 is deployed"
fi

invalid_job_body="$TMP_DIR/invalid-job.json"
code="$(curl_json POST /ai/claude-code-execute '{"taskId":"not-a-uuid","repoPath":"","branch":"","instructions":""}' "$invalid_job_body")"
[[ "$code" == "400" ]] && pass "/ai/claude-code-execute rejects invalid enqueue payload" || fail "/ai/claude-code-execute invalid payload HTTP ${code}"

if [[ "${AI_SMOKE_RUN_LIVE_AI:-false}" == "true" ]]; then
  live_body="$TMP_DIR/live-ai.json"
  code="$(curl_json POST /ai/complete '{"model_tier":"free","user_prompt":"Reply with the single word: OK"}' "$live_body")"
  [[ "$code" == "200" ]] && pass "/ai/complete live free tier returned 200" || fail "/ai/complete live free tier HTTP ${code}"
  assert_body_contains "$live_body" '"model_used"' "/ai/complete live model marker"
else
  echo "SKIP /ai/complete live inference: set AI_SMOKE_RUN_LIVE_AI=true"
fi

if [[ "$FAILED" -eq 0 ]]; then
  echo "Smoke test finished"
else
  echo "Smoke test finished - FAILURES DETECTED"
fi
exit "$FAILED"
