# EP-AI-05: Agent Registry Routing

```yaml
id: EP-AI-05
status: approved-for-coding
source_goal: implementation-goals/GOAL-05-agent-registry-routing.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
branch: main
```

## Metadata

This execution plan implements GOAL-05 for AI Microservice on the remote `alfares` repository at `/home/ssf/Documents/Github/ai-microservice`.

## Upstream Traceability

- User request: continue with `GOAL-05-agent-registry-routing`.
- Implementation state: `docs/IMPLEMENTATION_STATE.md`.
- Selected goal: `implementation-goals/GOAL-05-agent-registry-routing.md`.
- Contracts: `POST /ai/complete` request and response schema.

## Goal Impact

Consumers can opt into persisted, operator-managed agent definitions for `/ai/complete` without losing the existing model-tier gateway behavior. The registry read path gives operators a controlled prompt/schema/tier configuration surface while keeping normal callers compatible.

## Project Invariants

- INV-01 Preserve Runtime Contracts.
- INV-03 Auth Stays Explicit.
- INV-04 Model Routing Source Of Truth.
- INV-05 Premium Requires Approval.
- INV-06 Secrets Stay Out Of Artifacts.
- INV-07 Deployment Is A Goal.
- INV-08 Dirty Worktree Safety.

## Sensitive-Data Handling

Agent prompts and output schemas can contain sensitive operational instructions. Do not log full prompts, request bodies, authorization headers, or registry metadata. Validation evidence should include only command outcomes and non-sensitive routing metadata.

## Contract/Schema Impact

Add optional `/ai/complete` request fields for explicit registry routing and optional response audit fields showing the resolved agent. Existing request fields and response fields remain valid. No database schema change is required because GOAL-01 already created the `ai_agents` table.

## Replay/Determinism Impact

Outputs remain model-dependent and non-deterministic. Agent resolution is deterministic for a given active `agent_slug`, optional service scope, and registry state.

## Scope

- Add an explicit optional registry read path to `/ai/complete`.
- Resolve only `active` agents and reject draft, disabled, missing, or route-incompatible definitions.
- Apply persisted agent tier, prompts, output schema, and max tokens when an agent is selected.
- Return non-sensitive audit fields for the resolved agent.
- Keep ordinary model-tier calls unchanged.

## Non-Goals

- Do not alter `litellm_config.yaml` route definitions.
- Do not route premium without human approval.
- Do not change admin CRUD behavior except where type compatibility requires it.
- Do not deploy unless explicitly requested.

## Files To Inspect

- `src/database/entities/ai-agent.entity.ts`
- `src/admin/admin-agents.service.ts`
- `src/ai/ai.service.ts`
- `src/ai/ai.module.ts`
- `src/contracts/ai-complete.contract.ts`
- `src/ai/ai.service.spec.ts`

## Files To Create

- `implementation-goals/GOAL-05-agent-registry-routing.validation-report.md`

## Files To Modify

- `implementation-goals/GOAL-05-agent-registry-routing.md`
- `implementation-goals/GOAL-05-agent-registry-routing.execution-plan.md`
- `implementation-goals/GOAL-05-agent-registry-routing.context-package.md`
- `implementation-goals/GOAL-05-agent-registry-routing.coding-prompt.md`
- `src/ai/ai.service.ts`
- `src/ai/ai.module.ts`
- `src/contracts/ai-complete.contract.ts`
- `src/ai/ai.service.spec.ts`
- `docs/IMPLEMENTATION_STATE.md`

## Files That Must Not Be Modified

- `litellm_config.yaml`
- `src/claude-code/claude-code.consumer.ts`
- deployment scripts
- Kubernetes manifests
- unrelated admin frontend files

## Implementation Steps

1. Add GOAL-05 planning artifacts and pass the pre-coding gate.
2. Inject the `AiAgent` repository into `AiService`.
3. Add optional contract fields for `agent_slug` and `agent_service_scope`.
4. Resolve an active agent only when `agent_slug` is supplied.
5. Apply registry tier, prompt template, output schema, and max-token settings to the existing routing flow.
6. Return explicit audit metadata and safe error responses for missing, inactive, incompatible, or premium agents.
7. Add focused tests for compatibility, active-agent routing, draft/disabled exclusion, and premium blocking.
8. Run build and tests, then update the validation report and implementation state.

## Test Plan

- `npm run build`
- `npm test -- --runTestsByPath src/ai/ai.service.spec.ts`
- `npm test`

## Validation Plan

Record command results in the validation report. Confirm ordinary `/ai/complete` requests still route by `model_tier` and registry-backed requests are explicit, auditable, and limited to active agents.

## Gate Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-05-agent-registry-routing.md
python3 scripts/deployment_readiness_gate.py --root .
```

## Gate Evidence Location

Evidence is recorded in `implementation-goals/GOAL-05-agent-registry-routing.validation-report.md` and summarized in `docs/IMPLEMENTATION_STATE.md`.

## Documentation Updates

Update the selected goal artifacts, validation report, and implementation state. Do not update deployment docs unless deployment is explicitly requested.

## Rollback Plan

Revert the GOAL-05 code and contract additions. No migration rollback is required because this goal does not change schema.

## Agent Handoff Prompt

Implement GOAL-05 exactly as scoped here. Preserve ordinary `/ai/complete` model-tier routing, add only explicit active-agent registry routing, block inactive and premium agents, validate with build and focused tests, and update the validation report.

## Completion Checklist

- [ ] Implementation complete
- [ ] Tests complete
- [ ] Validation evidence collected
- [ ] Documentation updated
- [ ] Deviations documented
