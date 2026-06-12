# Context Package: GOAL-04

```yaml
id: CP-GOAL-04
status: complete
source_goal: implementation-goals/GOAL-04-implementation-job-observability.md
source_execution_plan: implementation-goals/GOAL-04-implementation-job-observability.execution-plan.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Intent

Improve status, logs, and audit summaries for implementation jobs using `claude-code` and `codex` providers.

## Intent Checksum

`66fbbb3b24aca532d3a763bf5032de027f224bb4fdc43978a77aeeeb407b5490`

## Current State

`docs/IMPLEMENTATION_STATE.md` marks GOAL-04 as queued after GOAL-03 completion. The working tree contains unrelated in-progress files from previous goals; preserve them and modify only the GOAL-04 files listed in the execution plan.

RAG lookup was required but unavailable in this session because no `JWT_TOKEN` was present in the remote environment.

## Relevant Contracts

- `POST /ai/claude-code-execute` enqueues implementation jobs.
- `GET /ai/claude-code-execute/:jobId` returns job status and execution details.
- Entity: `src/database/entities/claude-code-job.entity.ts`.
- Contract: `src/contracts/claude-code-job.contract.ts`.

## Files To Read First

- `implementation-goals/GOAL-04-implementation-job-observability.md`
- `docs/IMPLEMENTATION_STATE.md`
- `docs/governance/PROJECT_INVARIANTS.md`
- `src/claude-code/claude-code.service.ts`
- `src/claude-code/claude-code.consumer.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/claude-code-job.entity.ts`
- `test/claude-code/claude-code.controller.spec.ts`
- `test/claude-code/claude-code.e2e.spec.ts`

## Constraints

- Preserve endpoint names, auth expectations, queue names, and existing response fields.
- Preserve `intent` and `intentChecksum`.
- Do not alter provider routing or premium model controls.
- Do not deploy unless explicitly requested.
- Do not revert unrelated dirty files.

## Sensitive-Data Rules

Raw job fields may already contain sensitive output and remain for compatibility. Newly added summaries must redact obvious tokens, authorization headers, private keys, passwords, and high-entropy secret-like values. Logs should identify job, status, provider, and summaries without printing full instructions.

## Validation Evidence Required

- Pre-coding gate pass for GOAL-04.
- TypeScript build pass.
- Focused Claude Code controller/e2e tests pass.
- Full test suite pass or documented reason if blocked.
- Validation report and implementation state updated before closure.
