# GOAL-06: Deployment Hardening

```yaml
id: GOAL-06
status: done
owner: orchestrator
dependencies:
  - GOAL-03
  - GOAL-04
```

## Intent

Strengthen deployment readiness, smoke checks, rollback evidence, and operator handoff for production AI Microservice changes.

## Scope

- Inspect existing scripts and Kubernetes manifests.
- Define deployment readiness evidence.
- Add or improve smoke checks where low risk.

## Non-Goals

- Do not change runtime model routing behavior.
- Do not deploy premium models or route premium requests without explicit human approval.
- Do not enqueue real implementation jobs from default smoke checks.
- Do not modify database schema or public DTO contracts unless validation exposes a required defect.

## Acceptance Criteria

- Deployment readiness gate has project-specific checks.
- Rollback path is documented.
- Smoke checks cover health and changed behavior.

## Required Artifacts Before Coding

- `implementation-goals/GOAL-06-deployment-hardening.execution-plan.md`
- `implementation-goals/GOAL-06-deployment-hardening.context-package.md`
- `implementation-goals/GOAL-06-deployment-hardening.coding-prompt.md`

## Validation Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-06-deployment-hardening.md
python3 -m py_compile scripts/deployment_readiness_gate.py scripts/pre_coding_gate.py
bash -n scripts/deploy.sh scripts/smoke-unified-llm.sh
python3 scripts/deployment_readiness_gate.py --root .
npm run build
npm test
```
