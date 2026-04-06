# System: ai-microservice

## Architecture

FastAPI (Python). Modules: Orchestrator, NLP, ASR, Document AI, Prototype Generator, Free AI, Gemini, Data Viz.

- Tier routing: free (Ollama) → cheap (OpenRouter) → smart (Gemini Flash) → premium (Claude)
- Endpoint: `POST /ai/process` `{model_tier, system_prompt, user_prompt, output_schema}`

## Integrations

| Dependency | URL |
|-----------|-----|
| database-server | db-server-postgres:5432 + Redis |
| logging-microservice | logging-microservice:3367 |
| Ollama | localhost:11434 (local models) |

## Current State
<!-- AI-maintained -->
Stage: production

## Known Issues
<!-- AI-maintained -->
- None
