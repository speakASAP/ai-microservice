# Coding Prompt: GOAL-05

You are a bounded implementation worker for AI Microservice.

## Execution Plan

`implementation-goals/GOAL-05-agent-registry-routing.execution-plan.md`

## Goal

`GOAL-05-agent-registry-routing`

## Intent

Connect persisted admin agent definitions to controlled routing where appropriate, without replacing core `/ai/complete` tier routing.

## Intent Checksum

`95744219fb3427c4df75d107955235ce43d2f483719e08633336272df0af5c08`

## Required Context

Read `implementation-goals/GOAL-05-agent-registry-routing.context-package.md` and the files listed in its "Files To Read First" section before editing runtime code.

## Scope

Add backward-compatible, opt-in registry routing for `/ai/complete`.

## Allowed Changes

- GOAL-05 artifacts and validation report.
- `src/ai/ai.service.ts`
- `src/ai/ai.module.ts`
- `src/contracts/ai-complete.contract.ts`
- `src/ai/ai.service.spec.ts`
- `docs/IMPLEMENTATION_STATE.md`

## Forbidden Changes

- Do not change `litellm_config.yaml`.
- Do not change the canonical model-tier fallback chain.
- Do not allow premium agent routing without human approval.
- Do not edit `src/claude-code/claude-code.consumer.ts`.
- Do not change admin auth.
- Do not deploy.

## Required Reading

- `docs/governance/PROJECT_INVARIANTS.md`
- `docs/process/OPERATIONAL_GATES.md`
- `src/database/entities/ai-agent.entity.ts`
- `src/admin/admin-agents.service.ts`
- `src/ai/ai.service.ts`
- `src/contracts/ai-complete.contract.ts`

## Implementation Steps

1. Run the pre-coding gate for GOAL-05 and stop if it fails.
2. Add optional `/ai/complete` request fields for explicit agent routing.
3. Resolve only active persisted agents by slug and optional service scope.
4. Reject missing, inactive, incompatible, or premium agents with safe error responses.
5. Apply the selected agent's tier, system prompt, user template, output schema, and max tokens to existing routing paths.
6. Return audit metadata for the selected agent.
7. Add focused tests.
8. Run validation commands and write the validation report.

## Validation

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-05-agent-registry-routing.md
npm run build
npm test -- --runTestsByPath src/ai/ai.service.spec.ts
npm test
```

## Acceptance Criteria

- Registry-backed routing is explicit and auditable.
- Disabled or draft agents cannot be accidentally used.
- Existing callers remain compatible.

## Completion Report

Report:

- implemented changes;
- files changed;
- tests run;
- validation evidence;
- blockers;
- risks;
- intent compliance.
