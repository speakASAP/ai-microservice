# Phase 2 Environment Variable Updates

**File:** `.env`  
**Section:** External Services (Lines 42-45)

## Required Update

### Current (Line 42-45)

```bash
# External Services (Shared Microservices)
# Logging Service - All services must send logs to centralized logging microservice
LOGGING_SERVICE_URL=https://logging.alfares.cz
LOGGING_SERVICE_API_PATH=/api/logs
```

### Updated (Replace With)

```bash
# External Services (Shared Microservices)
# Logging Service - All services must send logs to centralized logging microservice
# Phase 2: Claude Code job events (executing, completed, retry, failed) are posted here
# Production: https://logging.alfares.cz
# Development/K8s: http://logging-microservice:3367
# NOTE: In Kubernetes, this is managed via Vault (secret/prod/ai-microservice)
LOGGING_SERVICE_URL=https://logging.alfares.cz
LOGGING_SERVICE_API_PATH=/api/logs
```

## Changes Made

1. **Added Phase 2 documentation** explaining that LOGGING_SERVICE_URL is now used for Claude Code job event logging
2. **Added URL options** showing both production and development/K8s values
3. **Added Vault note** explaining that in Kubernetes, this value comes from Vault instead of .env

## Values

| Environment | LOGGING_SERVICE_URL |
| ----------- | ----------------- |
| Development/Local | `http://logging-microservice:3367` |
| Production | `https://logging.alfares.cz` |
| Kubernetes | Managed via Vault: `secret/prod/ai-microservice` |

## How to Update (Manual)

1. Open `/home/ssf/Documents/Github/ai-microservice/.env`
2. Find line 42-45 (External Services section)
3. Replace the LOGGING_SERVICE_URL comment block with the updated version above
4. Save the file

## Verification

After updating, the `.env` file should have:

- ✅ Original LOGGING_SERVICE_URL value (unchanged)
- ✅ New comments explaining Phase 2 usage
- ✅ Documentation of development vs production URLs
- ✅ Note about Kubernetes Vault management

## Related Files

- `k8s/external-secret.yaml` — Kubernetes vault binding (already updated)
- `shared/docs/VAULT.md` — Vault configuration guide
- `src/claude-code/logging.client.ts` — Implementation using this variable
- `src/claude-code/README.md` — Documentation of logging integration

---

**Note:** The `.env` value `https://logging.alfares.cz` is correct for production. In development/K8s, the LoggingClient falls back to `http://logging-microservice:3367` if the environment variable is not set.
