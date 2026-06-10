# Business: ai-microservice

> ⚠️ **IMMUTABLE BY AI** — do not change Goal, Consumers, or SLA without product-owner approval. Factual **routing/stack** lines may be updated to match production (same spirit as `AGENTS.md` / `SYSTEM.md`).

## Goal

Centralized AI inference gateway for all Statex services. Routes LLM calls by model tier, provides NLP, ASR, Document AI, and prototype generation.

## Constraints

- Other Statex services must **not** call external LLM providers directly — they call **this service** (orchestrator, free-ai, or other published agents on `nginx-network`).
- **Tier routing** is implemented via the **LiteLLM** sidecar when enabled: route names `free`, `cheap`, `smart` map to upstreams in `litellm_config.yaml` (Docker **Ollama**, OpenRouter, Gemini). See **`AGENTS.md`** for the current model list; secrets in Vault (`secret/prod/ai-microservice`) via ESO; local dev uses `.env` from `vault-env-gen.sh`.
- **Premium** tier still requires explicit **human approval** per invocation (not exposed as an unattended LiteLLM route).
- **API costs** remain a product requirement: track per service / **business_id** in inference logs (see `TASKS.md` backlog).

## Consumers

runlayer, statex, shop-assistant, crypto-ai-agent, agentic-email.

## SLA

- Port: 3380 (<http://ai-microservice:3380>)
- Production: <https://ai.alfares.cz>
