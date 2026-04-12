# Agents: ai-microservice

Infrastructure service — provides LLM inference to other agents, does not self-coordinate.

## Model Tier → Model Mapping

```yaml
free:    ollama/gemma2:2b                              # local, zero cost, always available
cheap:   openrouter/meta-llama/llama-3.1-8b-instruct:free  # free tier; falls back to Ollama via LiteLLM
smart:   google/gemini-flash-1.5                       # free quota; falls back to Ollama via LiteLLM
premium: anthropic/claude-sonnet-4-6                   # BLOCKED — human approval required per call
```

## Fallback chain (LiteLLM proxy)

When `LITELLM_BASE_URL` is set, `ai-microservice` routes all calls through LiteLLM:

```
Caller → ai-microservice /ai/complete
           → LiteLLM proxy (container ai-microservice-litellm:4000)
               ├─ cheap → OpenRouter (free tier) → on error → Ollama (host:11434)
               ├─ smart → Gemini Flash → on error → Ollama (host:11434)
               └─ free → Ollama directly (zero cost)
```

See `docs/superpowers/cursor-tasks/task-02-litellm-fallback-gateway.md` for implementation spec.

## Active Agents
<!-- Coordinator-maintained -->
None — consumer services spawn agents, not this service.
