#!/bin/bash
# Start ai-microservice in Kubernetes (primary) or local dev mode.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-k8s}"

case "$MODE" in
  k8s)
    echo "Starting ai-microservice in Kubernetes..."
    kubectl rollout restart deployment/ai-microservice -n statex-apps
    kubectl rollout status deployment/ai-microservice -n statex-apps --timeout=120s
    echo "Done. Check: kubectl logs -n statex-apps -l app=ai-microservice -f"
    ;;
  dev)
    cd "$PROJECT_DIR"
    if [ ! -f .env ]; then
      echo "Warning: .env not found. Copy .env.example to .env and fill in ANTHROPIC_API_KEY."
    fi
    npm run build && node dist/main.js
    ;;
  *)
    echo "Usage: $0 [k8s|dev]"
    echo "  k8s  - restart K8s deployment (default)"
    echo "  dev  - build and run locally"
    exit 1
    ;;
esac
