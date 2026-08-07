# System: ai-microservice

## Architecture

NestJS. Modules: Orchestrator, NLP, ASR, Document AI, Prototype Generator, Free AI, Gemini, Data Viz.

- Tier routing: free (Ollama) → cheap (OpenRouter via LiteLLM when enabled) → smart (Gemini 2.0 Flash via LiteLLM when enabled) → premium (Claude, human approval)
- Implementation providers: `claude-code` (default) and `codex` via `/ai/claude-code-execute`; provider choice is separate from `/ai/complete` model-tier routing.
- Intent preservation: every AI-microservice goal should include `intent`; implementation jobs persist `intent`, `intentChecksum`, and `implementationProvider`.
- LiteLLM handles automatic failover when `LITELLM_BASE_URL` is set (e.g. OpenRouter issues → Ollama in compose via `OLLAMA_API_BASE`)
- Endpoint: `POST /ai/complete` — body: `{ model_tier, system_prompt, user_prompt, output_schema?, max_tokens?, correlation_id? }` (see `docs/model-tier-endpoints.md`)

## Service authentication (RS256)

Callers present `Authorization: Bearer <AI_SERVICE_TOKEN>`. Tokens are **RS256**: this service holds the private key (`JWT_PRIVATE_KEY`) and signs; verification uses `JWT_PUBLIC_KEY`. A leaked public key cannot mint tokens, so compromising one caller does not affect any other.

This replaced a **shared** HS256 `JWT_SECRET` that 11 services held in common — symmetric, so any of them could impersonate the others. Rotating it on 2026-08-01 without re-minting the dependent tokens broke 9 services with `401 Invalid signature` while their `exp` still read 2027.

| Variable | Meaning |
|---|---|
| `JWT_PRIVATE_KEY` | RSA-2048 PEM. **Only this service holds it.** Signs service tokens. |
| `JWT_PUBLIC_KEY` | RSA public PEM. Verifies incoming tokens. Not secret. |
| `ALLOW_HS256_FALLBACK` | `false` closes the legacy shared-secret path. Defaults to `true` so an unconfigured deploy cannot lock every caller out. |

Never rotate a signing key without re-minting the tokens signed by it:

```bash
./scripts/mint-service-token.sh --all      # re-mint, write Vault, resync ESO, restart
./scripts/verify-service-tokens.sh         # audit; exits 1 if any token is stale
```

Both verify paths pin `alg` from the token header before verifying — without that, a token can be relabelled `HS256` and signed with the public key (algorithm-confusion attack).

## Integrations

| Dependency | URL |
|-----------|-----|
| database-server | db-server-postgres:5432 + Redis |
| logging-microservice | logging-microservice:3367 |
| Ollama (Docker) | Port 11435 on host. Sidecar pod in K8s. Controlled by `OLLAMA_DOCKER_PORT` env var (default 11435). `OLLAMA_API_BASE` overrides the internal URL. |
| Ollama (systemd) | Host port 11434 — separate instance managed by systemd `ollama.service`. |
| LiteLLM proxy | Sidecar pod `:4000` — routes `free` / `cheap` / `smart` tiers (see `litellm_config.yaml`) |
| Codex CLI | `CODEX_CLI_PATH` (default `/home/ssf/.local/bin/codex`) for `implementationProvider=codex` jobs |

## Current State
<!-- AI-maintained -->
Stage: production

## Known Issues
<!-- AI-maintained -->
- None
