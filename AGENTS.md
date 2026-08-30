# Repository Agent Instructions

Shared rules live here:

- Codex profile: `/home/ssf/.codex/AGENTS.md`
- Cross-agent standard: `/home/ssf/.ai-agent-standards/CROSS_AGENT_AUTOMATION_STANDARD.md`
- Repository operations: `AGENT_OPERATIONS.md`

Read those first, then follow the repository-specific notes below and the current planning/status files.


## Repository-Specific Notes

# Agents: ai-microservice

## Remote-First Working Rule

All implementation and orchestration work for this project must happen on the remote `alfares` server in:

```bash
/home/ssf/Documents/Github/ai-microservice
```

Work directly in the authoritative repository on `alfares`; do not copy staged source into production.

## Knowledge Retrieval

Use `docs-rag-microservice` for bounded discovery when it is healthy, then
verify deployment, security, database, integration and public-contract facts
against the cited Git source. Git remains authoritative.

Authority and fallback rules:
`/home/ssf/Documents/Github/shared/docs/DOCUMENTATION_AUTHORITY.md`.

Do not generate tokens in documentation or assume an unconfident/failed RAG
response means that source documentation does not exist.

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
free:           ollama/qwen2.5-coder:0.5b                          # Ollama via OLLAMA_API_BASE (compose service `ollama`, not host-only)
cheap:          openrouter/google/gemma-4-26b-a4b-it:free          # OpenRouter; LiteLLM fallback → cheap-fallback
cheap-fallback: openrouter/nvidia/nemotron-3-nano-30b-a3b:free     # deliberately a different vendor from `cheap`
smart:          openrouter/google/gemma-4-31b-it:free              # OpenRouter; LiteLLM fallback → smart-fallback
smart-fallback: openrouter/nvidia/nemotron-3-super-120b-a12b:free  # deliberately a different vendor from `smart`
premium:        openrouter/anthropic/claude-sonnet-4.6             # human_approval=true required on every call
```

> ⚠️ `free` is a 0.5B **code** model. It is unsuitable for natural-language prose
> generation — treat it as a last-resort fallback, not a quality tier.

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
  free   → ollama/qwen2.5-coder:0.5b            ; on failure → route "cheap"
  cheap  → openrouter/google/gemma-4-26b-a4b-it:free ; on failure → cheap-fallback → openrouter/nvidia/nemotron-3-nano-30b-a3b:free
  smart  → openrouter/google/gemma-4-31b-it:free     ; on failure → smart-fallback → openrouter/nvidia/nemotron-3-super-120b-a12b:free
```

**Silent quality degradation is a real failure mode.** A fallback returns a *different
model* than the tier requested, and the response still looks well-formed. Callers doing
quality-critical generation must read `model_used` from the response (surfaced as `model`
by `src/teacher-assistant/llm.client.ts:193`), compare it against the requested tier, and
reject rather than accept a silent downgrade.

**Timeout budgets must nest outward.** See the dated incident comment in
`litellm_config.yaml` `router_settings`: a caller timeout shorter than this proxy's
`request_timeout` means the fallback chain never runs at all, and the aborted attempts
leave no trace in the proxy access log. Set caller timeouts *above* the proxy's.

**Ollama** runs as the host container `ai-microservice-ollama-green`, defined in `docker-compose.ollama.yml` (image `ai-microservice-ollama:local`; there is no `services/ollama/Dockerfile`). **`OLLAMA_API_BASE`** is `http://ai-microservice-ollama-green:11435` — port 11435, not Ollama’s 11434 default, because the container sets `OLLAMA_HOST=0.0.0.0:11435`. Nothing serves 11434 on this host. Pull weights into the volume after deploy (see `litellm_config.yaml` header comment).

If **`LITELLM_BASE_URL`** is unset on the orchestrator, **`/ai/complete`** keeps the legacy OpenRouter multi-model chain in `main.py`. **free-ai** without both `LITELLM_BASE_URL` and `LITELLM_MASTER_KEY` keeps direct OpenRouter/Ollama paths.

See `TASKS.md`, `docs/IMPLEMENTATION_STATE.md`, and `implementation-goals/` for task history and verify commands.

## Active Agents
<!-- Coordinator-maintained -->
None — consumer services spawn agents, not this service.
