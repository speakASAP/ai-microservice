# Agents: ai-microservice

## Remote-First Working Rule

All implementation and orchestration work for this project must happen on the remote `alfares` server in:

```bash
/home/ssf/Documents/Github/ai-microservice
```

Use local files only as a temporary staging mirror when needed, then copy changes to the remote repo and validate on `alfares`.

## Knowledge Retrieval (query before reading files)
Query the RAG service first — saves 2000-5000 tokens per query:
- URL: `http://docs-rag-microservice.statex-apps.svc.cluster.local:3397`
- Endpoint: `POST /retrieval/agent-context` with `{"query": "...", "maxTokens": 3000}`
- Auth: `Authorization: Bearer <JWT_TOKEN>`

Infrastructure service — provides LLM inference to other agents, does not self-coordinate.

## Intent Preservation

Every goal against AI microservice must preserve user intent from intake to execution:
- Capture `intent` before planning or enqueueing implementation work.
- Include `intent` on `/ai/claude-code-execute` jobs; include `intentChecksum` when the caller already has one.
- Treat returned `intentChecksum` as the continuity marker for follow-up jobs, validation, and review.
- Change intent only when a newer user instruction explicitly changes the objective.
- Validate outputs against the intent, not only build/test success.

See `docs/INTENT_PRESERVATION.md`.

## Model Tier → Model Mapping

Canonical router definitions live in **`litellm_config.yaml`** (edit there first; keep this table in sync).

```yaml
free:    ollama/qwen2.5-coder:0.5b          # Ollama via OLLAMA_API_BASE (compose service `ollama`, not host-only)
cheap:   openrouter/google/gemma-3-27b-it:free   # OpenRouter; LiteLLM fallback → cheap-fallback (same Ollama model)
smart:   gemini/gemini-2.0-flash             # Gemini API key; LiteLLM fallback → smart-fallback (same Ollama model)
premium: anthropic/claude-sonnet-4-6          # BLOCKED — human approval required per call (not routed in LiteLLM)
```

## Implementation Providers

Implementation jobs use `/ai/claude-code-execute` and accept `implementationProvider`:

```yaml
claude-code: default; uses CC_CLI_PATH and CLAUDE_CONFIG_DIR
codex:       optional; uses CODEX_CLI_PATH, CODEX_HOME, CODEX_SANDBOX, CODEX_APPROVAL_POLICY
```

The endpoint name remains `claude-code-execute` for backward compatibility, but callers should set `implementationProvider=codex` when they want Codex to implement a job.

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

See `TASKS.md`, `docs/IMPLEMENTATION_STATE.md`, and `implementation-goals/` for task history and verify commands.

## Active Agents
<!-- Coordinator-maintained -->
None — consumer services spawn agents, not this service.
