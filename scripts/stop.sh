#!/bin/bash
# Scale ai-microservice to zero in Kubernetes.
set -e

echo "Scaling ai-microservice to 0 replicas..."
kubectl scale deployment/ai-microservice -n statex-apps --replicas=0
echo "Done. Restore with: kubectl scale deployment/ai-microservice -n statex-apps --replicas=1"
