# Agents: ai-microservice

Infrastructure service — provides LLM inference to other agents, does not self-coordinate.

## Model Tier → Model Mapping

```yaml
free:    ollama/gemma2:2b
cheap:   openrouter/meta-llama/llama-3.1-8b-instruct:free
smart:   google/gemini-flash-1.5
premium: anthropic/claude-sonnet-4-6  # BLOCKED — human approval required
```

## Active Agents
<!-- Coordinator-maintained -->
None — consumer services spawn agents, not this service.
