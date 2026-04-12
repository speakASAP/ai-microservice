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
