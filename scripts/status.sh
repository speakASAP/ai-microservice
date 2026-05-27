#!/bin/bash
# Show live status of ai-microservice in Kubernetes.
set -e

NAMESPACE="statex-apps"
SERVICE="ai-microservice"

echo "=========================================="
echo "AI Microservice Status"
echo "=========================================="
echo ""

echo "Pods:"
kubectl get pods -n "$NAMESPACE" -l app="$SERVICE" -o wide

echo ""
echo "Recent logs (20 lines):"
kubectl logs -n "$NAMESPACE" -l app="$SERVICE" --tail=20 2>/dev/null || echo "(no logs)"

echo ""
echo "Health check:"
POD=$(kubectl get pod -n "$NAMESPACE" -l app="$SERVICE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$POD" ]; then
  kubectl exec -n "$NAMESPACE" "$POD" -- wget -qO- http://localhost:3380/health 2>/dev/null || echo "Health check failed"
else
  echo "No running pod found"
fi

echo ""
echo "=========================================="
