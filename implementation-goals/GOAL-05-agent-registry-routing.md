# GOAL-05: Agent Registry Routing

```yaml
id: GOAL-05
status: in-progress
owner: orchestrator
dependencies:
  - GOAL-01
  - GOAL-02
```

## Intent

Connect persisted admin agent definitions to controlled routing where appropriate, without replacing core `/ai/complete` tier routing.

## Scope

- Inspect admin agent registry contracts.
- Define safe read path for service-scoped agent definitions.
- Keep model-tier routing source of truth intact.

## Non-Goals

- Do not replace normal `/ai/complete` `model_tier` routing.
- Do not edit `litellm_config.yaml` or change LiteLLM route definitions.
- Do not allow premium model routing without explicit human approval.
- Do not change admin authentication or make draft/disabled agents executable.
- Do not deploy unless explicitly requested after validation.

## Acceptance Criteria

- Registry-backed routing is explicit and auditable.
- Disabled or draft agents cannot be accidentally used.
- Existing callers remain compatible.

## Required Artifacts Before Coding

- `implementation-goals/GOAL-05-agent-registry-routing.execution-plan.md`
- `implementation-goals/GOAL-05-agent-registry-routing.context-package.md`
- `implementation-goals/GOAL-05-agent-registry-routing.coding-prompt.md`

## Validation Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-05-agent-registry-routing.md
npm run build
npm test -- --runTestsByPath src/ai/ai.service.spec.ts
npm test
```
