# GOAL-04: Implementation Job Observability

```yaml
id: GOAL-04
status: done
owner: orchestrator
dependencies:
  - GOAL-02
```

## Intent

Improve status, logs, and audit summaries for implementation jobs using `claude-code` and `codex` providers.

## Scope

- Inspect current implementation job entity, consumer, service, and DTOs.
- Preserve existing endpoint names and response compatibility.
- Add only observable metadata that improves operator understanding and intent validation.

## Problem Statement

Implementation jobs currently expose raw execution fields but do not provide enough concise metadata for operators to understand lifecycle, executor choice, intent continuity, validation evidence, or failure detail without reading potentially sensitive raw output.

## Non-Goals

- Do not rename `/ai/claude-code-execute`.
- Do not remove raw fields that existing callers use.
- Do not change JWT auth, queue names, deployment flow, provider routing, or model-tier routing.
- Do not deploy as part of this goal unless explicitly requested.

## Files To Inspect

- `src/claude-code/claude-code.service.ts`
- `src/claude-code/claude-code.consumer.ts`
- `src/claude-code/dto/job-enqueue-response.dto.ts`
- `src/claude-code/dto/job-status-response.dto.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/claude-code-job.entity.ts`
- `test/claude-code/claude-code.controller.spec.ts`
- `test/claude-code/claude-code.e2e.spec.ts`

## Acceptance Criteria

- Status responses clearly show provider choice, intent checksum, lifecycle, and meaningful failure/success detail.
- Sensitive logs are redacted or summarized.
- Validation evidence is recorded.

## Required Artifacts Before Coding

- `implementation-goals/GOAL-04-implementation-job-observability.execution-plan.md`
- `implementation-goals/GOAL-04-implementation-job-observability.context-package.md`
- `implementation-goals/GOAL-04-implementation-job-observability.coding-prompt.md`

## Validation Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-04-implementation-job-observability.md
npm run build
npm test -- --runTestsByPath test/claude-code/claude-code.controller.spec.ts test/claude-code/claude-code.e2e.spec.ts
npm test
```

## Risks And Follow-Ups

- Existing raw stdout, stderr, diffs, and validation output may remain sensitive for backward compatibility; newly added summary fields must redact obvious secrets.
- Database migration should remain additive and nullable.
- Deployment is intentionally deferred unless the user explicitly requests it.

## Completion Report Requirements

Produce `implementation-goals/GOAL-04-implementation-job-observability.validation-report.md` and update `docs/IMPLEMENTATION_STATE.md` with intent compliance, command evidence, changed files, skipped checks, risks, and next action.
