# Kubernetes Vault Setup — Phase 2 Environment Variables

This document describes how to configure Phase 2 environment variables via Kubernetes Vault.

## Overview

Phase 2 introduces the `LOGGING_SERVICE_URL` environment variable for structured event logging integration. This variable is managed via Kubernetes ExternalSecrets and HashiCorp Vault, not through `.env` files.

## Configuration

### LOGGING_SERVICE_URL

**Purpose:** URL for the logging-microservice HTTP API endpoint where Claude Code job events are posted.

**Vault Path:** `secret/prod/ai-microservice`  
**Vault Key:** `LOGGING_SERVICE_URL`

**Values:**
- **Development/K8s:** `http://logging-microservice:3367`
- **Production:** `https://logging.alfares.cz`

**ExternalSecret Configuration:**
```yaml
# File: k8s/external-secret.yaml (already updated)
- secretKey: LOGGING_SERVICE_URL
  remoteRef:
    key: secret/prod/ai-microservice
    property: LOGGING_SERVICE_URL
```

## Vault Setup Steps

To set this value in Kubernetes Vault, use the HashiCorp Vault CLI:

```bash
# Authenticate to Vault
vault login -method=kubernetes role=ai-microservice

# Set the secret (development)
vault kv put secret/prod/ai-microservice LOGGING_SERVICE_URL="http://logging-microservice:3367"

# Or set the secret (production)
vault kv put secret/prod/ai-microservice LOGGING_SERVICE_URL="https://logging.alfares.cz"

# Verify the secret
vault kv get secret/prod/ai-microservice
```

## Kubernetes Pod Environment

Once the secret is stored in Vault, the ExternalSecrets operator will:

1. Read the secret from Vault (`secret/prod/ai-microservice`)
2. Create a Kubernetes Secret named `ai-microservice-secret`
3. Mount it as environment variables in the pod

The pod deployment should reference this secret:

```yaml
# deployment.yaml
env:
  - name: LOGGING_SERVICE_URL
    valueFrom:
      secretKeyRef:
        name: ai-microservice-secret
        key: LOGGING_SERVICE_URL
```

## Deployment Steps

1. **Update Vault** (as shown above) with the LOGGING_SERVICE_URL value
2. **Apply ExternalSecret** to sync the value:
   ```bash
   kubectl apply -f k8s/external-secret.yaml
   ```
3. **Verify Secret** was created:
   ```bash
   kubectl get secret ai-microservice-secret -n statex-apps -o yaml
   ```
4. **Restart pods** to pick up the new secret:
   ```bash
   kubectl rollout restart deployment/ai-microservice -n statex-apps
   ```

## Verification

Check that the pod received the environment variable:

```bash
kubectl exec -it <pod-name> -n statex-apps -- env | grep LOGGING_SERVICE_URL
```

## Related Files

- **Implementation:** `src/claude-code/logging.client.ts`
  - Uses `LOGGING_SERVICE_URL` env var with fallback to `http://logging-microservice:3367`
  - Posts job events to `${LOGGING_SERVICE_URL}/api/logs`
  
- **Configuration:** `k8s/external-secret.yaml`
  - Defines Vault path: `secret/prod/ai-microservice`
  - Syncs to K8s Secret: `ai-microservice-secret`

## Notes

- The `.env` file has `LOGGING_SERVICE_URL` as reference only (not used in K8s)
- Vault is the single source of truth for secrets in Kubernetes
- ExternalSecrets refreshes every 5 minutes (see external-secret.yaml `refreshInterval`)
- No secrets are hardcoded in deployment files or git
