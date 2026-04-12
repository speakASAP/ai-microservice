#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
load_env() {
  local f="$ROOT/.env"
  [[ -f "$f" ]] || return 0
  set -a
  # shellcheck source=/dev/null
  source "$f" 2>/dev/null || true
  set +a
}
load_env
HOST="${AI_SERVICE_HOST:-localhost}"
ORCH_PORT="${AI_ORCHESTRATOR_PORT:-3380}"
LITELLM_PORT="${LITELLM_LOCAL_PORT:-4000}"
pass() { echo "OK $*"; }
skip() { echo "SKIP $*"; }
have_curl() { command -v curl >/dev/null 2>&1; }
have_docker() { command -v docker >/dev/null 2>&1; }
litellm_container() {
  if [[ -n "${LITELLM_DOCKER_CONTAINER:-}" ]]; then
    echo "${LITELLM_DOCKER_CONTAINER}"
    return
  fi
  docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'litellm' | head -1 || true
}
if have_curl; then
  code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 8 "http://${HOST}:${ORCH_PORT}/health" 2>/dev/null || echo 000)"
  [[ "$code" == "200" ]] && pass "orchestrator /health" || echo "WARN orchestrator /health HTTP ${code}" >&2
  code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 "http://${HOST}:${LITELLM_PORT}/health/liveliness" 2>/dev/null || echo 000)"
  if [[ "$code" == "200" ]]; then
    pass "litellm liveliness (host ${HOST}:${LITELLM_PORT})"
  else
    c="$(litellm_container)"
    if [[ -n "$c" ]] && have_docker; then
      if docker exec "$c" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health/liveliness', timeout=6).read()" 2>/dev/null; then
        pass "litellm liveliness (docker exec $c localhost:4000)"
      else
        skip "litellm host HTTP ${code} and docker exec $c failed (publish 4000 or set LITELLM_DOCKER_CONTAINER)"
      fi
    else
      skip "litellm not on ${HOST}:${LITELLM_PORT} (HTTP ${code}); no running litellm container for docker fallback"
    fi
  fi
fi
echo "OK: smoke-unified-llm finished"
exit 0
