# EP-AI-06: Deployment Hardening

```yaml
id: EP-AI-06
status: approved
source_goal: implementation-goals/GOAL-06-deployment-hardening.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Metadata

Branch: `codex/ai-goal-06-deployment-hardening`.
Lifecycle state: implementation.

## Upstream Traceability

- User request: `implement GOAL-06-deployment-hardening`.
- Selected goal: `implementation-goals/GOAL-06-deployment-hardening.md`.
- Implementation state: `docs/IMPLEMENTATION_STATE.md`.
- Operational gate standard: `docs/process/OPERATIONAL_GATES.md`.

## Goal Impact

This goal reduces release risk for AI Microservice by making deployment readiness project-specific, requiring smoke checks for recent behavior, and printing rollback evidence during deployment.

## Project Invariants

- Preserve existing AI runtime endpoints and deployment flow.
- Preserve intent fields on `/ai/claude-code-execute` jobs.
- Keep `implementationProvider=claude-code` and `implementationProvider=codex` compatible.
- Keep model-tier routing canonical in `litellm_config.yaml`.
- Do not route premium models without human approval.

## Sensitive-Data Handling

Deployment smoke checks must not print secrets, prompts from real users, raw job output, API keys, or database credentials. Checks use synthetic payloads only.

## Contract/Schema Impact

No DTO, database, or public endpoint schema changes are planned. Smoke checks exercise existing endpoint contracts for `/health`, `/ai/complete`, and `/ai/claude-code-execute`.

## Replay/Determinism Impact

No replay semantics change. Live model inference remains opt-in in smoke checks to avoid nondeterministic or paid behavior in normal deployment validation.

## Scope

- Strengthen `scripts/deployment_readiness_gate.py`.
- Improve `scripts/smoke-unified-llm.sh`.
- Integrate readiness, smoke, and rollback evidence into `scripts/deploy.sh`.
- Document rollback and validation evidence for GOAL-06.

## Non-Goals

- Do not change model routing behavior.
- Do not run premium models.
- Do not enqueue real implementation jobs in smoke tests.
- Do not change Kubernetes service exposure or database schema.

## Files To Inspect

- `scripts/deploy.sh`
- `scripts/deployment_readiness_gate.py`
- `scripts/smoke-unified-llm.sh`
- `k8s/deployment.yaml`
- `k8s/configmap.yaml`
- `k8s/service.yaml`
- `k8s/ingress.yaml`
- `src/contracts/ai-complete.contract.ts`
- `src/contracts/claude-code-job.contract.ts`

## Files To Create

- `implementation-goals/GOAL-06-deployment-hardening.execution-plan.md`
- `implementation-goals/GOAL-06-deployment-hardening.context-package.md`
- `implementation-goals/GOAL-06-deployment-hardening.coding-prompt.md`
- `implementation-goals/GOAL-06-deployment-hardening.validation-report.md`

## Files To Modify

- `scripts/deploy.sh`
- `scripts/deployment_readiness_gate.py`
- `scripts/smoke-unified-llm.sh`
- `implementation-goals/GOAL-06-deployment-hardening.md`
- `docs/IMPLEMENTATION_STATE.md`

## Files That Must Not Be Modified

- `litellm_config.yaml`
- Database migrations
- Runtime DTO contracts unless validation exposes a required defect
- Secret manifests containing sensitive values

## Implementation Steps

1. Add GOAL-06 planning, context, coding prompt, and validation report artifacts.
2. Expand deployment readiness gate to check project-specific manifests, smoke checks, rollback evidence, and documentation artifacts.
3. Expand smoke checks to cover health, premium approval block, missing registry agent handling, and invalid implementation-job payload validation.
4. Update deploy script to run the gate, capture rollback context, run smoke checks after rollout, and print rollback commands.
5. Validate with Python compilation, shell syntax, focused smoke checks, readiness gates, build, and tests.

## Test Plan

- `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-06-deployment-hardening.md`
- `python3 -m py_compile scripts/deployment_readiness_gate.py scripts/pre_coding_gate.py`
- `bash -n scripts/deploy.sh scripts/smoke-unified-llm.sh`
- `python3 scripts/deployment_readiness_gate.py --root .`
- `npm run build`
- `npm test`
- Run `scripts/smoke-unified-llm.sh` against localhost if a local service is running, otherwise record why it was not executed locally.

## Validation Plan

Evidence is recorded in `implementation-goals/GOAL-06-deployment-hardening.validation-report.md` and `docs/IMPLEMENTATION_STATE.md`.

## Gate Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-06-deployment-hardening.md
python3 scripts/deployment_readiness_gate.py --root .
```

## Gate Evidence Location

- `implementation-goals/GOAL-06-deployment-hardening.validation-report.md`
- `docs/IMPLEMENTATION_STATE.md`

## Documentation Updates

Update GOAL-06 status and implementation state before closure.

## Rollback Plan

If deployment fails after image rollout, run the rollback command printed by `scripts/deploy.sh`, preferring the captured previous revision:

```bash
kubectl rollout undo deployment/ai-microservice -n statex-apps --to-revision=<previous-revision>
kubectl rollout status deployment/ai-microservice -n statex-apps
AI_SERVICE_BASE_URL=https://ai.alfares.cz ./scripts/smoke-unified-llm.sh
```

If the previous revision is unavailable, run `kubectl rollout history deployment/ai-microservice -n statex-apps` and select the last known good revision.

## Agent Handoff Prompt

Implement GOAL-06 by hardening deployment readiness, smoke checks, rollback evidence, and operator handoff while preserving AI runtime behavior and avoiding premium or real implementation-job execution in default smoke checks.

## Completion Checklist

- [x] Implementation complete
- [ ] Tests complete
- [ ] Validation evidence collected
- [ ] Documentation updated
- [ ] Deviations documented
