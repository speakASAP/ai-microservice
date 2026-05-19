# System: ai-microservice

## Architecture

NestJS. Modules: Orchestrator, NLP, ASR, Document AI, Prototype Generator, Free AI, Gemini, Data Viz.

- Tier routing: free (Ollama) → cheap (OpenRouter via LiteLLM when enabled) → smart (Gemini 2.0 Flash via LiteLLM when enabled) → premium (Claude, human approval)
- LiteLLM handles automatic failover when `LITELLM_BASE_URL` is set (e.g. OpenRouter issues → Ollama in compose via `OLLAMA_API_BASE`)
- Endpoint: `POST /ai/complete` — body: `{ model_tier, system_prompt, user_prompt, output_schema?, max_tokens?, correlation_id? }` (see `docs/model-tier-endpoints.md`)

## Integrations

| Dependency | URL |
|-----------|-----|
| database-server | db-server-postgres:5432 + Redis |
| logging-microservice | logging-microservice:3367 |
| Ollama (Docker) | Port 11435 on host. Sidecar pod in K8s. Controlled by `OLLAMA_DOCKER_PORT` env var (default 11435). `OLLAMA_API_BASE` overrides the internal URL. |
| Ollama (systemd) | Host port 11434 — separate instance managed by systemd `ollama.service`. |
| LiteLLM proxy | Sidecar pod `:4000` — routes `free` / `cheap` / `smart` tiers (see `litellm_config.yaml`) |

## Current State
<!-- AI-maintained -->
Stage: production

## Known Issues
<!-- AI-maintained -->
- None
