# Context Package: GOAL-06

```yaml
id: CP-GOAL-06
status: complete
source_goal: implementation-goals/GOAL-06-deployment-hardening.md
source_execution_plan: implementation-goals/GOAL-06-deployment-hardening.execution-plan.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Intent

Strengthen deployment readiness, smoke checks, rollback evidence, and operator handoff for production AI Microservice changes.

## Intent Checksum

Runtime checksum is not available in this manual implementation session. Intent continuity is preserved by copying the selected goal intent verbatim into this package and the validation report.

## Current State

GOAL-03 through GOAL-06 are complete in `docs/IMPLEMENTATION_STATE.md`. GOAL-06 hardened deployment readiness, smoke checks, rollback evidence, and operator handoff, and the production deployment has been verified.

## Relevant Contracts

- `GET /health` returns the service health contract.
- `POST /ai/complete` must keep model-tier behavior, business metadata handling, agent registry routing, and the premium approval block. Agent-routing smoke remains opt-in unless an operator explicitly wants to exercise the registry route against production.
- `POST /ai/claude-code-execute` must preserve intent fields and provider compatibility.
- Kubernetes deployment should continue using rolling updates with readiness/liveness/startup probes.

## Files To Read First

1. `implementation-goals/GOAL-06-deployment-hardening.md`
2. `docs/IMPLEMENTATION_STATE.md`
3. `docs/process/OPERATIONAL_GATES.md`
4. `scripts/deploy.sh`
5. `scripts/deployment_readiness_gate.py`
6. `scripts/smoke-unified-llm.sh`
7. `k8s/deployment.yaml`
8. `k8s/configmap.yaml`
9. `src/contracts/ai-complete.contract.ts`
10. `src/contracts/claude-code-job.contract.ts`

## Constraints

- Remote-first work must happen on `alfares` under `/home/ssf/Documents/Github/ai-microservice`.
- Do not deploy without explicit user request or deployment goal scope. GOAL-06 is a deployment-hardening goal, but implementation validation does not require a production rollout.
- Do not weaken gates to pass validation.
- Do not expose secrets or real customer prompts in smoke output.
- Do not route premium models without human approval.

## Sensitive-Data Rules

Use synthetic smoke payloads only. Do not print `LITELLM_MASTER_KEY`, provider keys, database password, JWTs, or raw implementation-job output.

## Validation Evidence Required

- Pre-coding gate passes for GOAL-06.
- Python syntax and shell syntax checks pass.
- Deployment readiness gate passes and validates project-specific files.
- Build and relevant tests pass.
- Smoke script is either run against a service endpoint or explicitly recorded as not run because no local/production deployment was performed in this session.
