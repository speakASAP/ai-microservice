#!/bin/bash
# deploy.sh — Kubernetes deployment for ai-microservice
# Usage: ./scripts/deploy.sh [image-tag]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SERVICE_NAME="ai-microservice"
NAMESPACE="statex-apps"
REGISTRY="localhost:5000"
IMAGE_TAG="${1:-latest}"
IMAGE="${REGISTRY}/${SERVICE_NAME}:${IMAGE_TAG}"
PORT="3380"

# ═══════════════════════════════════════════════════════════
#  ai-microservice - Kubernetes Deployment
# ═══════════════════════════════════════════════════════════

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ${SERVICE_NAME}"
echo "║  Kubernetes Deployment"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Phase 1: Build Docker image ──────────────────────────────
echo -e "${YELLOW}[1/5] Building image: ${IMAGE}...${NC}"
docker build -t "$IMAGE" "$PROJECT_ROOT"
echo -e "${GREEN}✅ Image built${NC}"

# ── Phase 2: Push to local registry ──────────────────────────
echo -e "${YELLOW}[2/5] Pushing to registry...${NC}"
docker push "$IMAGE"
echo -e "${GREEN}✅ Image pushed: ${IMAGE}${NC}"

# ── Phase 3: Apply K8s manifests ─────────────────────────────
echo -e "${YELLOW}[3/5] Applying K8s manifests...${NC}"
kubectl apply -f "$PROJECT_ROOT/k8s/configmap.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/secret.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/deployment.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/service.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/ingress.yaml"
echo -e "${GREEN}✅ Manifests applied${NC}"

# ── Phase 4: Rollout & Status ────────────────────────────────
echo -e "${YELLOW}[4/5] Restarting deployment and waiting for rollout...${NC}"
kubectl rollout restart deployment/${SERVICE_NAME} -n "${NAMESPACE}"
kubectl rollout status deployment/${SERVICE_NAME} \
  -n "${NAMESPACE}" \
  --timeout=120s
echo -e "${GREEN}✅ Rollout complete${NC}"

# ── Phase 5: Health check ────────────────────────────────────
echo -e "${YELLOW}[5/5] Verifying health...${NC}"
POD=$(kubectl get pod -n "${NAMESPACE}" \
  -l app=${SERVICE_NAME} \
  -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD" ]; then
  echo -e "${RED}❌ No pod found for ${SERVICE_NAME}${NC}"
  exit 1
fi

kubectl exec -n "${NAMESPACE}" "$POD" -- \
  curl -s http://localhost:${PORT}/health || {
  echo -e "${RED}⚠️  Health check failed (service may still be starting)${NC}"
}
echo -e ""

# ── Done ─────────────────────────────────────────────────────
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║            ✅ Deployment successful!                   ║"
echo "║  Service:  ${SERVICE_NAME}"
echo "║  Image:    ${IMAGE}"
echo "║  Namespace: ${NAMESPACE}"
echo "║  Pods:     $(kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME} --no-headers | wc -l) running"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"
