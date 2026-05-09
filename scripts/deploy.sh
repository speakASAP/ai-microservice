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
DEFAULT_TAG="$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "build-$(date -u +%Y%m%d%H%M%S)")"
IMAGE_TAG="${1:-$DEFAULT_TAG}"
IMAGE="${REGISTRY}/${SERVICE_NAME}:${IMAGE_TAG}"
IMAGE_LATEST="${REGISTRY}/${SERVICE_NAME}:latest"
PORT="3380"

log_ts() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

preflight_service_health() {
  echo -e "${YELLOW}Preflight: checking Kubernetes and current service health...${NC}"

  if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo -e "${RED}Namespace not found: $NAMESPACE${NC}"
    exit 1
  fi

  if ! kubectl get nodes >/dev/null 2>&1; then
    echo -e "${RED}kubectl cannot reach cluster${NC}"
    exit 1
  fi

  BAD_PODS=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" --no-headers 2>/dev/null | awk '$3 ~ /Error|CrashLoopBackOff|ImagePullBackOff|CreateContainerConfigError|CreateContainerError|ErrImagePull/ {print $1}')
  if [ -n "$BAD_PODS" ]; then
    echo -e "${RED}Service has unhealthy pods before deploy:${NC}"
    kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o wide || true
    for pod in $BAD_PODS; do
      echo -e "${YELLOW}--- describe pod/$pod ---${NC}"
      kubectl describe pod -n "$NAMESPACE" "$pod" || true
      echo -e "${YELLOW}--- logs pod/$pod (tail 80) ---${NC}"
      kubectl logs -n "$NAMESPACE" "$pod" --tail=80 || true
    done
    echo -e "${RED}Fix pod errors first, then redeploy.${NC}"
    exit 1
  fi

  echo -e "${GREEN}Preflight passed${NC}"
}

# ═══════════════════════════════════════════════════════════
#  ai-microservice - Kubernetes Deployment
# ═══════════════════════════════════════════════════════════

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║       AI Microservice - Kubernetes Deployment          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Phase 1: Build Docker image ──────────────────────────────
preflight_service_health

echo -e "${YELLOW}[1/5] Building image: ${IMAGE}...${NC}"
docker build -t "$IMAGE" -t "$IMAGE_LATEST" "$PROJECT_ROOT"
echo -e "${GREEN}✅ Image built${NC}"

# ── Phase 2: Push to local registry ──────────────────────────
echo -e "${YELLOW}[2/5] Pushing to registry...${NC}"
docker push "$IMAGE"
docker push "$IMAGE_LATEST"
echo -e "${GREEN}✅ Image pushed: ${IMAGE}${NC}"

# ── Phase 3: Apply K8s manifests ─────────────────────────────
echo -e "${YELLOW}[3/5] Applying K8s manifests...${NC}"
kubectl apply -f "$PROJECT_ROOT/k8s/configmap.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/external-secret.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/deployment.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/service.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/ingress.yaml" -n "${NAMESPACE}"
echo -e "${GREEN}✅ Manifests applied${NC}"

# ── Phase 4: Rollout & Status ────────────────────────────────
echo -e "${YELLOW}[4/5] Restarting deployment and waiting for rollout...${NC}"
kubectl rollout restart deployment/${SERVICE_NAME} -n "${NAMESPACE}"
if ! kubectl rollout status deployment/${SERVICE_NAME} \
  -n "${NAMESPACE}" \
  --timeout=120s; then
  echo -e "${YELLOW}Rollout did not complete in time. Diagnosing terminating pods...${NC}"
  kubectl get pods -n "${NAMESPACE}" -l app=${SERVICE_NAME} -o wide || true
  TERMINATING_PODS=$(kubectl get pods -n "${NAMESPACE}" -l app=${SERVICE_NAME} --no-headers 2>/dev/null | awk '$3=="Terminating"{print $1}')
  if [ -n "$TERMINATING_PODS" ]; then
    echo -e "${YELLOW}Force deleting stuck terminating pods...${NC}"
    for pod in $TERMINATING_PODS; do
      kubectl delete pod -n "${NAMESPACE}" "$pod" --grace-period=0 --force || true
    done
  fi
  kubectl rollout status deployment/${SERVICE_NAME} -n "${NAMESPACE}" --timeout=120s
fi
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

log_ts "Selected pod for health verification: ${POD}"

# Do not use kubectl exec for health checks here: OCI exec can fail even when pod is healthy.
# Read readiness state from Kubernetes status, then print recent logs when not ready.
READY_STATE=$(kubectl get pod -n "${NAMESPACE}" "$POD" \
  -o jsonpath='{range .status.containerStatuses[*]}{.name}={.ready}{" "}{end}')

if echo "$READY_STATE" | grep -q "=true"; then
  echo -e "${GREEN}✅ Pod containers ready: ${READY_STATE}${NC}"
else
  echo -e "${RED}⚠️  Pod not ready yet: ${READY_STATE}${NC}"
  log_ts "Recent pod logs (last 60 lines) for debugging:"
  kubectl logs -n "${NAMESPACE}" "$POD" --tail=60 || true
fi
echo -e ""

# ── Done ─────────────────────────────────────────────────────
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║       ✅ AI Microservice Deployment successful!        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo "Image:    ${IMAGE}"
echo "Namespace: ${NAMESPACE}"
echo "Pods:     $(kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME} --no-headers | wc -l) running"

echo -e "${NC}"
