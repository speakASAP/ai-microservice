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

# shellcheck disable=SC1091
source "$(dirname "$PROJECT_ROOT")/shared/scripts/load-deploy-phase-timing.sh" "$PROJECT_ROOT" 2>/dev/null \
  || source "$HOME/Documents/Github/shared/scripts/load-deploy-phase-timing.sh" "$PROJECT_ROOT" \
  || { echo "Error: deploy timing library not found" >&2; exit 1; }

SERVICE_NAME="ai-microservice"
NAMESPACE="statex-apps"
REGISTRY="localhost:5000"
DEFAULT_TAG="$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "build-$(date -u +%Y%m%d%H%M%S)")"
IMAGE_TAG="${1:-$DEFAULT_TAG}"
IMAGE="${REGISTRY}/${SERVICE_NAME}:${IMAGE_TAG}"
IMAGE_LATEST="${REGISTRY}/${SERVICE_NAME}:latest"
PORT="3380"
PUBLIC_BASE_URL="${AI_SERVICE_PUBLIC_URL:-https://ai.alfares.cz}"
ROLLBACK_PREVIOUS_IMAGE=""
ROLLBACK_PREVIOUS_REVISION=""

log_ts() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

deployment_readiness_gate() {
  echo -e "${YELLOW}Preflight: running deployment readiness gate...${NC}"
  python3 "$PROJECT_ROOT/scripts/deployment_readiness_gate.py" --root "$PROJECT_ROOT"
  echo -e "${GREEN}Deployment readiness gate passed${NC}"
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

capture_rollback_context() {
  echo -e "${YELLOW}Capturing rollback context...${NC}"
  ROLLBACK_PREVIOUS_IMAGE="$(kubectl get deployment/"${SERVICE_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)"
  ROLLBACK_PREVIOUS_REVISION="$(kubectl rollout history deployment/"${SERVICE_NAME}" -n "${NAMESPACE}" 2>/dev/null \
    | awk 'NF && $1 ~ /^[0-9]+$/ {revision=$1} END {print revision}')"

  if [ -n "$ROLLBACK_PREVIOUS_IMAGE" ]; then
    log_ts "Previous image: ${ROLLBACK_PREVIOUS_IMAGE}"
  else
    log_ts "Previous image: unavailable"
  fi

  if [ -n "$ROLLBACK_PREVIOUS_REVISION" ]; then
    log_ts "Previous rollout revision: ${ROLLBACK_PREVIOUS_REVISION}"
  else
    log_ts "Previous rollout revision: unavailable"
  fi
}

run_smoke_checks() {
  echo -e "${YELLOW}Running production smoke checks against ${PUBLIC_BASE_URL}...${NC}"
  AI_SERVICE_BASE_URL="$PUBLIC_BASE_URL" "$PROJECT_ROOT/scripts/smoke-unified-llm.sh"
  echo -e "${GREEN}✅ Smoke checks passed${NC}"
}

print_rollback_evidence() {
  echo -e "${BLUE}Rollback evidence:${NC}"
  echo "Current image:  ${IMAGE}"
  echo "Previous image: ${ROLLBACK_PREVIOUS_IMAGE:-unknown}"
  echo "Rollout history:"
  kubectl rollout history deployment/"${SERVICE_NAME}" -n "${NAMESPACE}" || true

  if [ -n "$ROLLBACK_PREVIOUS_REVISION" ]; then
    echo "Rollback command: kubectl rollout undo deployment/${SERVICE_NAME} -n ${NAMESPACE} --to-revision=${ROLLBACK_PREVIOUS_REVISION}"
  else
    echo "Rollback command: kubectl rollout undo deployment/${SERVICE_NAME} -n ${NAMESPACE}"
  fi
}

# ═══════════════════════════════════════════════════════════
#  ai-microservice - Kubernetes Deployment
# ═══════════════════════════════════════════════════════════

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║       AI Microservice - Kubernetes Deployment          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

deploy_timing_init "$SERVICE_NAME"
deploy_timing_run_phase "Readiness gate" deployment_readiness_gate
deploy_timing_run_phase "Preflight" preflight_service_health
deploy_timing_run_phase "Rollback context" capture_rollback_context

deploy_timing_phase_start "Build image"
echo -e "${YELLOW}Building image: ${IMAGE}...${NC}"
docker build --no-cache -t "$IMAGE" -t "$IMAGE_LATEST" "$PROJECT_ROOT"
echo -e "${GREEN}✅ Image built${NC}"
deploy_timing_phase_end "Build image"

deploy_timing_phase_start "Push image"
echo -e "${YELLOW}Pushing to registry...${NC}"
docker push "$IMAGE"
docker push "$IMAGE_LATEST"
echo -e "${GREEN}✅ Image pushed: ${IMAGE}${NC}"
deploy_timing_phase_end "Push image"

deploy_timing_phase_start "Apply K8s manifests"
echo -e "${YELLOW}Applying K8s manifests...${NC}"
kubectl apply -f "$PROJECT_ROOT/k8s/configmap.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/external-secret.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/deployment.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/service.yaml" -n "${NAMESPACE}"
kubectl apply -f "$PROJECT_ROOT/k8s/ingress.yaml" -n "${NAMESPACE}"
kubectl patch deployment/"${SERVICE_NAME}" -n "${NAMESPACE}" --type=json \
  -p='[{"op":"remove","path":"/spec/template/spec/containers/0/env"}]' 2>/dev/null || true
echo -e "${GREEN}✅ Manifests applied${NC}"
deploy_timing_phase_end "Apply K8s manifests"

deploy_timing_phase_start "Wait for rollout"
echo -e "${YELLOW}Restarting deployment and waiting for rollout...${NC}"
kubectl rollout restart deployment/${SERVICE_NAME} -n "${NAMESPACE}"
deploy_timing_k8s_rollout_wait kubectl "$SERVICE_NAME" "$NAMESPACE"
echo -e "${GREEN}✅ Rollout complete${NC}"
deploy_timing_phase_end "Wait for rollout"

deploy_timing_phase_start "Health check"
echo -e "${YELLOW}Verifying health...${NC}"
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
  LITELLM_URL=$(kubectl exec -n "${NAMESPACE}" "$POD" -- printenv LITELLM_BASE_URL 2>/dev/null || true)
  if [ -z "$LITELLM_URL" ]; then
    echo -e "${RED}❌ LITELLM_BASE_URL empty in pod (ConfigMap shadowed?) — /ai/complete will use rate-limited OpenRouter fallback${NC}"
    exit 1
  fi
  log_ts "LITELLM_BASE_URL=${LITELLM_URL}"
else
  echo -e "${RED}⚠️  Pod not ready yet: ${READY_STATE}${NC}"
  log_ts "Recent pod logs (last 60 lines) for debugging:"
  kubectl logs -n "${NAMESPACE}" "$POD" --tail=60 || true
  exit 1
fi
echo -e ""
deploy_timing_phase_end "Health check"

deploy_timing_run_phase "Smoke checks" run_smoke_checks
print_rollback_evidence

deploy_timing_finish_success "AI Microservice"
echo "Image:    ${IMAGE}"
echo "Namespace: ${NAMESPACE}"
echo "Pods:     $(kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME} --no-headers | wc -l) running"
DEPLOY_TIMING_FINISHED=1
exit 0
