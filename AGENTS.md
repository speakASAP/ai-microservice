# Agents: ai-microservice

Infrastructure service — provides LLM inference to other agents, does not self-coordinate.

## Model Tier → Model Mapping

```yaml
free:    ollama/qwen2.5-coder:0.5b                     # local via LiteLLM (pull on host if missing)
cheap:   openrouter/google/gemma-3-27b-it:free         # OpenRouter free; falls back to Ollama via LiteLLM
smart:   gemini/gemini-2.0-flash                       # Google AI Studio; falls back to Ollama via LiteLLM
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
