# Agents: ai-microservice

## Knowledge Retrieval (query before reading files)
Query the RAG service first — saves 2000-5000 tokens per query:
- URL: `http://docs-rag-microservice.statex-apps.svc.cluster.local:3397`
- Endpoint: `POST /retrieval/agent-context` with `{"query": "...", "maxTokens": 3000}`
- Auth: `Authorization: Bearer <JWT_TOKEN>`

Infrastructure service — provides LLM inference to other agents, does not self-coordinate.

## Model Tier → Model Mapping

Canonical router definitions live in **`litellm_config.yaml`** (edit there first; keep this table in sync).

```yaml
free:    ollama/qwen2.5-coder:0.5b          # Ollama via OLLAMA_API_BASE (compose service `ollama`, not host-only)
cheap:   openrouter/google/gemma-3-27b-it:free   # OpenRouter; LiteLLM fallback → cheap-fallback (same Ollama model)
smart:   gemini/gemini-2.0-flash             # Gemini API key; LiteLLM fallback → smart-fallback (same Ollama model)
premium: anthropic/claude-sonnet-4-6          # BLOCKED — human approval required per call (not routed in LiteLLM)
```

## Fallback chain (LiteLLM proxy)

When **`LITELLM_BASE_URL`** is set, orchestrator **`POST /ai/complete`** and **free-ai-service `/analyze`** (when `LITELLM_*` set) use LiteLLM’s OpenAI-compatible API; route names are the tier ids `free`, `cheap`, `smart`.

Router fallbacks (see `router_settings.fallbacks` in `litellm_config.yaml`):

```
Caller → LiteLLM (e.g. ai-microservice-litellm-green:4000)
  free   → ollama/qwen2.5-coder:0.5b  ; on failure → route "cheap"
  cheap  → openrouter/.../gemma-3-27b-it:free ; on failure → cheap-fallback → same Ollama model
  smart  → gemini/gemini-2.0-flash    ; on failure → smart-fallback → same Ollama model
```

**Ollama** is the compose-built service (`services/ollama/Dockerfile`); **`OLLAMA_API_BASE`** points at `http://ai-microservice-ollama(-blue|-green):11434` by default. Pull weights into the volume after deploy (see `litellm_config.yaml` header comment).

If **`LITELLM_BASE_URL`** is unset on the orchestrator, **`/ai/complete`** keeps the legacy OpenRouter multi-model chain in `main.py`. **free-ai** without both `LITELLM_BASE_URL` and `LITELLM_MASTER_KEY` keeps direct OpenRouter/Ollama paths.

See `docs/superpowers/cursor-tasks/task-02-litellm-fallback-gateway.md` for task history and verify commands.

## Active Agents
<!-- Coordinator-maintained -->
None — consumer services spawn agents, not this service.
