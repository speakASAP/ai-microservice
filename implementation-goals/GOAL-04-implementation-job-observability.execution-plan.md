# EP-AI-04: Implementation Job Observability

```yaml
id: EP-AI-04
status: approved-for-coding
source_goal: implementation-goals/GOAL-04-implementation-job-observability.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
branch: codex/ai-goal-04-implementation-job-observability
```

## Metadata

This execution plan implements GOAL-04 for AI Microservice on the remote `alfares` repository at `/home/ssf/Documents/Github/ai-microservice`.

## Upstream Traceability

- User request: continue with `GOAL-04-implementation-job-observability`.
- Implementation state: `docs/IMPLEMENTATION_STATE.md`.
- Selected goal: `implementation-goals/GOAL-04-implementation-job-observability.md`.
- Contracts: `/ai/claude-code-execute` enqueue and status responses.

## Goal Impact

Operators need status responses that explain which executor ran, whether intent continuity was preserved, where the job is in its lifecycle, and what validation or failure evidence exists without exposing raw sensitive output by default.

## Project Invariants

- INV-01 Preserve Runtime Contracts.
- INV-02 Preserve Intent.
- INV-03 Auth Stays Explicit.
- INV-06 Secrets Stay Out Of Artifacts.
- INV-07 Deployment Is A Goal.
- INV-08 Dirty Worktree Safety.

## Sensitive-Data Handling

Job instructions, stdout, stderr, diffs, and validation output can contain secrets. Keep existing raw fields for compatibility, but add summarized/redacted metadata for operator views. Do not log full instructions, raw tokens, authorization headers, private keys, or long raw command output.

## Contract/Schema Impact

The `/ai/claude-code-execute` endpoint name and existing response fields stay compatible. Add optional response metadata fields and database columns for lifecycle, provider, audit summary, redacted output summaries, validation evidence, and failure details.

## Replay/Determinism Impact

Job execution remains non-deterministic because external CLIs and repository state can vary. Observability fields must be derived deterministically from stored job state where possible, and retry behavior must not change.

## Scope

- Inspect and update the Claude Code job entity, service, consumer, DTOs, contracts, tests, and migration.
- Add optional observability fields to enqueue/status responses.
- Record lifecycle timestamps and validation evidence.
- Redact or summarize sensitive output in newly added summary fields.

## Non-Goals

- Do not rename `/ai/claude-code-execute`.
- Do not change auth, queue names, deployment scripts, model routing, or provider selection rules.
- Do not remove raw response fields that existing callers may use.
- Do not deploy unless explicitly requested after validation.

## Files To Inspect

- `src/claude-code/claude-code.service.ts`
- `src/claude-code/claude-code.consumer.ts`
- `src/claude-code/dto/job-enqueue-response.dto.ts`
- `src/claude-code/dto/job-status-response.dto.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/claude-code-job.entity.ts`
- `test/claude-code/claude-code.controller.spec.ts`
- `test/claude-code/claude-code.e2e.spec.ts`

## Files To Create

- `src/database/migrations/006-claude-code-job-observability.sql`
- `implementation-goals/GOAL-04-implementation-job-observability.validation-report.md`

## Files To Modify

- `implementation-goals/GOAL-04-implementation-job-observability.md`
- `implementation-goals/GOAL-04-implementation-job-observability.execution-plan.md`
- `implementation-goals/GOAL-04-implementation-job-observability.context-package.md`
- `implementation-goals/GOAL-04-implementation-job-observability.coding-prompt.md`
- `src/claude-code/claude-code.service.ts`
- `src/claude-code/claude-code.consumer.ts`
- `src/claude-code/dto/job-enqueue-response.dto.ts`
- `src/claude-code/dto/job-status-response.dto.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/claude-code-job.entity.ts`
- `test/claude-code/claude-code.controller.spec.ts`
- `test/claude-code/claude-code.e2e.spec.ts`
- `docs/IMPLEMENTATION_STATE.md`

## Files That Must Not Be Modified

- `litellm_config.yaml`
- `k8s/*`
- deployment scripts
- unrelated admin frontend and cost-tracking files

## Implementation Steps

1. Add required GOAL-04 artifacts and pass the pre-coding gate.
2. Add nullable observability columns to `claude_code_jobs`.
3. Extend contracts and DTOs with optional metadata fields.
4. Update enqueue/status mapping to include provider, intent checksum, lifecycle, audit, validation, and failure summaries.
5. Update the consumer to populate lifecycle, execution attempts, redacted summaries, validation evidence, and terminal failure detail.
6. Add or update focused controller/e2e tests for the new observable fields.
7. Run build and focused tests, then produce the validation report and state update.

## Test Plan

- `npm run build`
- `npm test -- --runTestsByPath test/claude-code/claude-code.controller.spec.ts test/claude-code/claude-code.e2e.spec.ts`
- `npm test`

## Validation Plan

Record command results in the validation report. Confirm backward compatibility by ensuring existing response fields remain present and new fields are optional.

## Gate Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-04-implementation-job-observability.md
python3 scripts/deployment_readiness_gate.py --root .
```

## Gate Evidence Location

Evidence is recorded in `implementation-goals/GOAL-04-implementation-job-observability.validation-report.md` and summarized in `docs/IMPLEMENTATION_STATE.md`.

## Documentation Updates

Update the selected goal artifacts, validation report, and implementation state. Do not update deployment docs unless deployment is explicitly requested.

## Rollback Plan

Revert the GOAL-04 runtime changes and avoid applying migration `006` if validation fails before deployment. If migration is applied later, rollback by dropping only the nullable observability columns and indexes added by `006`.

## Agent Handoff Prompt

Implement GOAL-04 exactly as scoped here. Preserve `/ai/claude-code-execute` compatibility, preserve intent fields, add optional observability metadata, redact newly introduced summaries, validate with build and focused tests, and update the validation report.

## Completion Checklist

- [ ] Implementation complete
- [ ] Tests complete
- [ ] Validation evidence collected
- [ ] Documentation updated
- [ ] Deviations documented
