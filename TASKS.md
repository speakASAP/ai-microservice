# Tasks: ai-microservice

## Program: Unified LLM gateway (reference)

| What | Where |
| ---- | ----- |
| Current implementation checkpoint | [`docs/IMPLEMENTATION_STATE.md`](docs/IMPLEMENTATION_STATE.md) |
| Goal queue and validation reports | [`implementation-goals/`](implementation-goals/) |
| Model tier routing source of truth | [`litellm_config.yaml`](litellm_config.yaml) |
| Deployment and smoke checks | [`scripts/deploy.sh`](scripts/deploy.sh), [`scripts/smoke-unified-llm.sh`](scripts/smoke-unified-llm.sh) |

## Backlog

- [x] 2026-04-11 Add LiteLLM fallback gateway sidecar — automatic Ollama fallback when OpenRouter hits limits — config fallback placement fixed post-review
- [x] Add cost tracking per business_id to inference logs (priority: 2) — completed in GOAL-03

## Completed
<!-- AI appends here. Never modifies previous entries. -->
- [x] 2026-06-12 Added intent preservation system for AI-microservice goals and implementation jobs; added Codex as selectable implementation provider beside Claude Code.
- [x] 2026-04-11 Documented model tier HTTP API with examples — `docs/model-tier-endpoints.md`; corrected `SYSTEM.md` path (`/ai/complete`).
- [x] 2026-04-11 `POST /ai/complete` on ai-orchestrator (task-bo-01) — already present; task doc marked finished
- [x] 2026-04-05 Documentation standard applied
- [x] 2026-04-12 Unified LLM gateway (staged) — LiteLLM + Docker Ollama + free-ai → LiteLLM; validation now tracked in `docs/IMPLEMENTATION_STATE.md` and `implementation-goals/`; smoke coverage in `scripts/smoke-unified-llm.sh`

## Project Completion Marker

- 2026-06-21: Project marked completed/frozen after remote inventory. There are no active goals, active plans, open tasks, blockers, or pending human/AI actions. Do not ask for a new goal during routine status checks unless the owner explicitly creates one.
