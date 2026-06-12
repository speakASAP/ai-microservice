# Coding Prompt: GOAL-06

You are a bounded implementation worker for AI Microservice.

## Execution Plan

`implementation-goals/GOAL-06-deployment-hardening.execution-plan.md`

## Goal

`implementation-goals/GOAL-06-deployment-hardening.md`

## Intent

Strengthen deployment readiness, smoke checks, rollback evidence, and operator handoff for production AI Microservice changes.

## Intent Checksum

Runtime checksum is unavailable in this manual implementation session. Preserve the intent text exactly.

## Required Context

Read `implementation-goals/GOAL-06-deployment-hardening.context-package.md` and the files listed there before editing.

## Scope

Harden deployment scripts and documentation only. Smoke checks may exercise existing contracts but must not change endpoint behavior.

## Allowed Changes

- `scripts/deploy.sh`
- `scripts/deployment_readiness_gate.py`
- `scripts/smoke-unified-llm.sh`
- GOAL-06 implementation docs
- `docs/IMPLEMENTATION_STATE.md`

## Forbidden Changes

- No premium model routing.
- No DTO or database schema changes.
- No real implementation-job enqueue in default smoke checks.
- No secret output in logs or reports.

## Required Reading

- `scripts/deploy.sh`
- `scripts/deployment_readiness_gate.py`
- `scripts/smoke-unified-llm.sh`
- `k8s/deployment.yaml`
- `k8s/configmap.yaml`
- `src/contracts/ai-complete.contract.ts`
- `src/contracts/claude-code-job.contract.ts`

## Implementation Steps

1. Make deployment readiness gate project-specific.
2. Add production-safe smoke coverage for health, premium approval guard, missing agent routing, and invalid implementation-job payload validation.
3. Make deploy script run the readiness gate and smoke checks.
4. Capture previous image and rollout revision and print rollback commands.
5. Update validation and implementation-state docs.

## Validation

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-06-deployment-hardening.md
python3 -m py_compile scripts/deployment_readiness_gate.py scripts/pre_coding_gate.py
bash -n scripts/deploy.sh scripts/smoke-unified-llm.sh
python3 scripts/deployment_readiness_gate.py --root .
npm run build
npm test
```

## Acceptance Criteria

- Deployment readiness gate has project-specific checks.
- Rollback path is documented.
- Smoke checks cover health and changed behavior.

## Completion Report

Report implemented changes, files changed, tests run, validation evidence, blockers, risks, and intent compliance.
