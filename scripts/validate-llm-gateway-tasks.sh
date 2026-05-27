#!/usr/bin/env bash
# Validates that key ai-microservice source artifacts exist and are consistent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK:   $*"; }

# TypeScript build compiles cleanly
cd "$ROOT"
npm run build 2>&1 || fail "TypeScript build failed"
ok "TypeScript build"

# Required source files exist
for f in \
  src/main.ts \
  src/app.module.ts \
  src/health.controller.ts \
  src/ai/ai.service.ts \
  src/ai/ai.controller.ts \
  src/claude-code/claude-code.service.ts \
  Dockerfile \
  k8s/deployment.yaml \
  k8s/external-secret.yaml; do
  [[ -f "$ROOT/$f" ]] || fail "missing $ROOT/$f"
  ok "$f"
done

# ANTHROPIC_API_KEY is referenced in ai.service.ts (not hardcoded)
grep -q "ANTHROPIC_API_KEY" "$ROOT/src/ai/ai.service.ts" || fail "ANTHROPIC_API_KEY not referenced in ai.service.ts"
ok "ANTHROPIC_API_KEY env reference in ai.service.ts"

# No Python files in src/
PY_COUNT=$(find "$ROOT/src" -name "*.py" | wc -l)
[[ "$PY_COUNT" -eq 0 ]] || fail "Python files found in src/ — expected zero"
ok "No Python in src/"

# Smoke test script is executable
[[ -x "$ROOT/scripts/smoke-unified-llm.sh" ]] || fail "smoke-unified-llm.sh is not executable"
ok "smoke-unified-llm.sh is executable"

echo ""
echo "All checks passed."
