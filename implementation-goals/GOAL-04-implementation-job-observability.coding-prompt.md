# Coding Prompt: GOAL-04

You are a bounded implementation worker for AI Microservice.

## Execution Plan

`implementation-goals/GOAL-04-implementation-job-observability.execution-plan.md`

## Goal

`GOAL-04-implementation-job-observability`

## Intent

Improve status, logs, and audit summaries for implementation jobs using `claude-code` and `codex` providers.

## Intent Checksum

`66fbbb3b24aca532d3a763bf5032de027f224bb4fdc43978a77aeeeb407b5490`

## Required Context

Read `implementation-goals/GOAL-04-implementation-job-observability.context-package.md` and the files listed in its "Files To Read First" section before editing runtime code.

## Scope

Add backward-compatible implementation-job observability for `/ai/claude-code-execute` enqueue and status responses.

## Allowed Changes

- GOAL-04 artifacts and validation report.
- Claude Code job entity, service, consumer, DTOs, contracts, migration, and focused tests.
- `docs/IMPLEMENTATION_STATE.md` state and evidence entries for GOAL-04.

## Forbidden Changes

- Do not rename endpoints, queues, or providers.
- Do not remove existing response fields.
- Do not change auth.
- Do not edit model routing files or deployment manifests.
- Do not deploy.

## Required Reading

- `docs/governance/PROJECT_INVARIANTS.md`
- `docs/process/OPERATIONAL_GATES.md`
- `src/claude-code/claude-code.service.ts`
- `src/claude-code/claude-code.consumer.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/claude-code-job.entity.ts`
- `test/claude-code/claude-code.controller.spec.ts`
- `test/claude-code/claude-code.e2e.spec.ts`

## Implementation Steps

1. Run the pre-coding gate for GOAL-04 and stop if it fails.
2. Add nullable observability columns and a migration.
3. Extend contracts and DTOs with optional status metadata.
4. Populate metadata on enqueue, executing, retrying, validation, success, failure, and timeout paths.
5. Summarize and redact newly introduced output/failure fields.
6. Update focused tests.
7. Run validation commands and write the validation report.

## Validation

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-04-implementation-job-observability.md
npm run build
npm test -- --runTestsByPath test/claude-code/claude-code.controller.spec.ts test/claude-code/claude-code.e2e.spec.ts
npm test
```

## Acceptance Criteria

- Status responses clearly show provider choice, intent checksum, lifecycle, and meaningful failure/success detail.
- Sensitive logs are redacted or summarized.
- Validation evidence is recorded.
- Existing endpoint names and response compatibility are preserved.

## Completion Report

Report:

- implemented changes;
- files changed;
- tests run;
- validation evidence;
- blockers;
- risks;
- intent compliance.
