# Tasks: ai-microservice

## Program: Unified LLM gateway (reference)

| What | Where |
| ---- | ----- |
| Staged plan (exit criteria) | [`docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md`](docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md) |
| Task index + validators **V-UG-*** | [`docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md`](docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md) |
| Lead agent prompt (status table) | [`docs/agents/master-prompt-llm-gateway.md`](docs/agents/master-prompt-llm-gateway.md) |
| Scripts + markdown checks | [`docs/superpowers/LLM_GATEWAY_SETUP.md`](docs/superpowers/LLM_GATEWAY_SETUP.md) |

## Backlog

- [x] 2026-04-11 Add LiteLLM fallback gateway sidecar — automatic Ollama fallback when OpenRouter hits limits — `docs/superpowers/cursor-tasks/task-02-litellm-fallback-gateway.md`; config fallback placement fixed post-review
- [ ] Add cost tracking per business_id to inference logs (priority: 2) — next feature work after unified gateway

## Completed
<!-- AI appends here. Never modifies previous entries. -->
- [x] 2026-06-12 Added intent preservation system for AI-microservice goals and implementation jobs; added Codex as selectable implementation provider beside Claude Code.
- [x] 2026-04-11 Documented model tier HTTP API with examples — `docs/model-tier-endpoints.md`; corrected `SYSTEM.md` path (`/ai/complete`).
- [x] 2026-04-11 `POST /ai/complete` on ai-orchestrator (task-bo-01) — already present; task doc marked finished
- [x] 2026-04-05 Documentation standard applied
- [x] 2026-04-12 Unified LLM gateway (staged) — LiteLLM + Docker Ollama + free-ai → LiteLLM; T-UG-00…T-UG-07 done per `docs/superpowers/LLM_UNIFIED_GATEWAY_TASK_INDEX.md`; `scripts/validate-llm-gateway-tasks.sh`, `scripts/smoke-unified-llm.sh`; stages `docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md`
