# LLM unified gateway — install scripts and run validation (markdown copy)

**Continue from here (when not in Cursor Plan mode):** paste §1 and §2 into the files below, then run the one-liner in **§6**.

```bash
# One-liner after pasting §1 -> scripts/validate-llm-gateway-tasks.sh and §2 -> scripts/smoke-unified-llm.sh
cd ai-microservice && chmod +x scripts/validate-llm-gateway-tasks.sh scripts/smoke-unified-llm.sh && ./scripts/validate-llm-gateway-tasks.sh && ./scripts/smoke-unified-llm.sh
```

Plan mode cannot create `.sh` files in this workspace. **Copy** the blocks below into executable files when implementing outside plan mode, or run the **inline grep / curl** checks manually.

---

## 1) Save as `scripts/validate-llm-gateway-tasks.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INDEX="docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md"
STAGES="docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md"
MASTER="docs/agents/master-prompt-llm-gateway.md"
SMOKE="scripts/smoke-unified-llm.sh"
fail() { echo "FAIL: $*" >&2; exit 1; }
[[ -f "$INDEX" ]] || fail "missing $INDEX"
[[ -f "$STAGES" ]] || fail "missing $STAGES"
[[ -f "$MASTER" ]] || fail "missing $MASTER"
[[ -f "$SMOKE" ]] || fail "missing $SMOKE (copy from section 2 of LLM_GATEWAY_SETUP.md first)"
[[ -x "$SMOKE" ]] || fail "$SMOKE must be executable (chmod +x)"
for id in T-UG-00 T-UG-01 T-UG-02 T-UG-03 T-UG-04 T-UG-05 T-UG-06 T-UG-07; do
  grep -q "$id" "$INDEX" || fail "task id $id not found in $INDEX"
done
for vid in V-UG-00 V-UG-01 V-UG-03 V-UG-04 V-UG-06 V-UG-07; do
  grep -q "$vid" "$INDEX" || fail "validator id $vid not found in $INDEX"
done
grep -q "Stage 0" "$STAGES" || fail "Stage 0 missing in $STAGES"
grep -q "Stage 5" "$STAGES" || fail "Stage 5 missing in $STAGES"
echo "OK: LLM gateway task artifacts validated"
exit 0
```

Then: `chmod +x scripts/validate-llm-gateway-tasks.sh`

---

## 2) Save as `scripts/smoke-unified-llm.sh`

```bash
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
```

Then: `chmod +x scripts/smoke-unified-llm.sh`

---

## 3) Markdown-only validation (no scripts on disk yet)

From `ai-microservice` repo root:

```bash
test -f docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md
test -f docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md
test -f docs/agents/master-prompt-llm-gateway.md
test -f docs/superpowers/LLM_GATEWAY_SETUP.md
grep -q 'T-UG-07' docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md
grep -q 'V-UG-07' docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md
grep -q 'Stage 5' docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md
echo OK markdown-only validation
```

---

## 4) Smoke (curl only)

With stack running and ports mapped:

```bash
export AI_SERVICE_HOST=localhost
curl -sf --max-time 8 "http://${AI_SERVICE_HOST:-localhost}:${AI_ORCHESTRATOR_PORT:-3380}/health" | head -c 200
```

---

## 5) Full test suite (after scripts exist)

```bash
cd ai-microservice && ./scripts/validate-llm-gateway-tasks.sh && ./scripts/smoke-unified-llm.sh && python3 scripts/test-ai-services.py
```

---

## 6) Continue — after files exist on disk

```bash
cd ai-microservice && chmod +x scripts/validate-llm-gateway-tasks.sh scripts/smoke-unified-llm.sh && ./scripts/validate-llm-gateway-tasks.sh && ./scripts/smoke-unified-llm.sh
```

---

## 7) Manual checks (validators V-UG-01 … V-UG-03)

Use **active color** container names (`*-green` / `*-blue`) as deployed.

**Ollama models in volume**

```bash
docker exec ai-microservice-ollama-green ollama list
```

**LiteLLM models (needs `LITELLM_MASTER_KEY` from `.env`)**

```bash
# From host only if 4000 is published; else run inside litellm:
docker exec ai-microservice-litellm-green python3 -c \
  "import os,urllib.request; k=os.environ.get('LITELLM_MASTER_KEY',''); r=urllib.request.Request('http://127.0.0.1:4000/v1/models',headers={'Authorization':f'Bearer {k}'}); print(urllib.request.urlopen(r,timeout=10).read()[:800])"
```

**Orchestrator tier smoke (JWT required)**

```bash
# See docs/model-tier-endpoints.md for full curl; use model_tier free|cheap|smart
```
