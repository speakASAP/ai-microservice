# deploy.config.sh — declaration consumed by shared/scripts/deploy.sh.
# See shared/docs/DEPLOY_STANDARD.md for the config format.
#
# Migrated from scripts/deploy.sh on 2026-08-14 so this service can take part in
# the automatic deploy queue (shared/scripts/deploy-queue/).

SERVICE_NAME="ai-microservice"
PORT="3380"

IMAGES=(
  "ai-microservice|.||"
)

DEPLOYMENTS=(
  "ai-microservice|app|ai-microservice"
)

# MANIFESTS left at the runner default (configmap, external-secret, deployment,
# service, ingress) — matches the legacy script's manifest loop exactly.

# The legacy script stripped the inline `env:` block from the deployment after
# applying it, so the pod takes its configuration from envFrom (ConfigMap +
# ExternalSecret) alone. An inline env entry would shadow those and pin a stale
# value. Kept here verbatim; `|| true` because the block is absent after the
# first deploy and `remove` on a missing path is an error.
deploy_post_manifests() {
  kubectl patch deployment/"${SERVICE_NAME}" -n "${NAMESPACE}" --type=json \
    -p='[{"op":"remove","path":"/spec/template/spec/containers/0/env"}]' 2>/dev/null || true
}
