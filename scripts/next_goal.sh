#!/usr/bin/env bash
set -euo pipefail

STATE_FILE="docs/IMPLEMENTATION_STATE.md"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "Missing $STATE_FILE"
  exit 1
fi

echo "Read AGENTS.md, docs/IMPLEMENTATION_ORCHESTRATOR.md, and $STATE_FILE."
echo "Continue from the 'Next Action' section:"
awk '/^## Next Action/{flag=1; next} /^## /{flag=0} flag {print}' "$STATE_FILE"
