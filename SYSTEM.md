# System: ai-microservice

## Architecture

FastAPI (Python). Modules: Orchestrator, NLP, ASR, Document AI, Prototype Generator, Free AI, Gemini, Data Viz.

- Tier routing: free (Ollama) → cheap (OpenRouter via LiteLLM when enabled) → smart (Gemini Flash via LiteLLM when enabled) → premium (Claude, human approval)
- LiteLLM handles automatic failover when `LITELLM_BASE_URL` is set (e.g. OpenRouter rate limits → Ollama on host)
- Endpoint: `POST /ai/complete` — body: `{ model_tier, system_prompt, user_prompt, output_schema?, max_tokens?, correlation_id? }` (see `docs/model-tier-endpoints.md`)

## Integrations

| Dependency | URL |
|-----------|-----|
| database-server | db-server-postgres:5432 + Redis |
| logging-microservice | logging-microservice:3367 |
| Ollama | localhost:11434 (local models) |
| LiteLLM proxy | ai-microservice-litellm:4000 (optional fallback gateway — Ollama → OpenRouter → Gemini; internal Docker only) |

## Current State
<!-- AI-maintained -->
Stage: production

## Known Issues
<!-- AI-maintained -->
- None
