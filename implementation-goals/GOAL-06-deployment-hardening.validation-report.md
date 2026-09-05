# Validation Report: GOAL-06

```yaml
id: VAL-GOAL-06
status: pass
artifact_validated: implementation-goals/GOAL-06-deployment-hardening.md
owner: validator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Artifact Validated

Selected goal: `implementation-goals/GOAL-06-deployment-hardening.md`.
Branch: `codex/ai-goal-06-deployment-hardening`.

## Validation Scope

Validate deployment-readiness gate coverage, smoke check safety, rollback evidence, build, and tests for deployment hardening changes.

## Preserved Intent

Strengthen deployment readiness, smoke checks, rollback evidence, and operator handoff for production AI Microservice changes.

## Intent Checksum Evidence

Runtime checksum is unavailable in this manual implementation session. Intent continuity is preserved through matching goal, context package, execution plan, coding prompt, and report text.

## Intent Compliance Decision

Pass. The implementation strengthens deployment readiness, smoke checks, rollback evidence, and operator handoff without changing runtime model routing, DTOs, database schema, or premium approval policy.

## Command Evidence

## Gate Evidence

Pre-coding and deployment readiness gates passed on `alfares`.

## Invariant Evidence

Runtime endpoint behavior is unchanged. Premium requests remain blocked with explicit human approval messaging. Implementation job schema validation and intent fields remain owned by existing contracts.

## Sensitive-Data Evidence

Pass. Smoke uses synthetic payloads. The deploy script generates a short-lived deployment smoke token in memory from `JWT_SECRET` and does not print the token or secret. Scripts do not print provider keys, database credentials, JWTs, or raw implementation-job output.

## Contract/Schema Evidence

No DTO, database, or endpoint schema changes. Smoke validates existing `/health`, `/ai/complete`, and `/ai/claude-code-execute` behavior.

## Replay/Determinism Evidence

Live model inference remains opt-in with `AI_SMOKE_RUN_LIVE_AI=true`. Agent-routing smoke remains opt-in with `AI_SMOKE_CHECK_AGENT_ROUTING=true` for explicit production registry-route checks.

## Passed Criteria

- Deployment readiness gate has project-specific checks.
- Rollback path is documented in the execution plan and printed by `scripts/deploy.sh`.
- Smoke checks cover health and changed behavior without default premium routing or real job enqueue.

## Failed Criteria

None.

## Manual Checks

Authenticated smoke against `https://ai.alfares.cz` passed for `/health`, premium approval block, and invalid implementation-job payload validation.

Production deployment completed with one ready pod, zero restarts, and running image digest `localhost:5000/ai-microservice@sha256:da24fd454caf4336f3f28d3931ea5c554b210046e62eb9b6373095a7f3e526a5`.

## Skipped Checks

Default smoke skipped live `/ai/complete` inference because `AI_SMOKE_RUN_LIVE_AI=true` was not set. Agent-routing smoke was skipped during the original GOAL-06 deployment because it was intentionally opt-in for production registry-route checks.

## Deviations

Agent-routing smoke remains opt-in to avoid accidental live inference during routine deployment validation.

## Risks

The deploy script now depends on `kubectl get secret ai-microservice-secret` and local `node` to generate authenticated deployment smoke tokens. If unavailable, protected smoke checks are skipped with an explicit message. Because Kubernetes deployment history stores mutable `latest`, the deploy script was patched after deployment to capture pod image digests for future rollback evidence.

## Decision

Pass.
