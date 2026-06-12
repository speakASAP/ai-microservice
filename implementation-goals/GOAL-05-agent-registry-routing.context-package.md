# Context Package: GOAL-05

```yaml
id: CP-GOAL-05
status: complete
source_goal: implementation-goals/GOAL-05-agent-registry-routing.md
source_execution_plan: implementation-goals/GOAL-05-agent-registry-routing.execution-plan.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Intent

Connect persisted admin agent definitions to controlled routing where appropriate, without replacing core `/ai/complete` tier routing.

## Intent Checksum

`95744219fb3427c4df75d107955235ce43d2f483719e08633336272df0af5c08`

## Current State

`docs/IMPLEMENTATION_STATE.md` marks GOAL-05 as queued after GOAL-01 and GOAL-02. GOAL-01 added the persisted `ai_agents` registry and GOAL-03/GOAL-04 are already completed. The remote working tree contains dirty files from prior goal work; preserve them and modify only GOAL-05-owned files.

RAG lookup was required but unavailable in this session because no `JWT_TOKEN` was available on `alfares`; repository docs and existing source files are the fallback context.

## Relevant Contracts

- `POST /ai/complete` remains the runtime inference endpoint.
- `model_tier` continues to route through LiteLLM route names and legacy fallback behavior.
- `ai_agents.status` values are `draft`, `active`, and `disabled`; only `active` can be used at runtime.
- `ai_agents.modelTier` values are `free`, `cheap`, `smart`, and `premium`; premium remains blocked without approval.

## Files To Read First

- `implementation-goals/GOAL-05-agent-registry-routing.md`
- `docs/IMPLEMENTATION_STATE.md`
- `docs/governance/PROJECT_INVARIANTS.md`
- `src/database/entities/ai-agent.entity.ts`
- `src/admin/admin-agents.service.ts`
- `src/ai/ai.service.ts`
- `src/ai/ai.module.ts`
- `src/contracts/ai-complete.contract.ts`
- `src/ai/ai.service.spec.ts`

## Constraints

- Preserve existing `/ai/complete` callers.
- Do not change `litellm_config.yaml`; model tier route names remain canonical there.
- Do not allow draft, disabled, missing, route-incompatible, or premium agents to execute.
- Do not log full prompts or registry metadata.
- Do not deploy unless explicitly requested.
- Do not revert unrelated dirty files.

## Sensitive-Data Rules

Treat agent system prompts, user prompt templates, output schemas, and request prompts as sensitive. Tests may use toy prompts only. Validation reports should include command outcomes and non-sensitive audit field names, not full prompt content from real agents.

## Validation Evidence Required

- Pre-coding gate pass for GOAL-05.
- TypeScript build pass.
- Focused AI service tests pass.
- Full test suite pass or documented reason if blocked.
- Validation report and implementation state updated before closure.
