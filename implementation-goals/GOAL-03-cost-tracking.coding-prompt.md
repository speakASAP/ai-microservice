# Coding Prompt: GOAL-03

You are a bounded implementation worker for AI Microservice.

## Execution Plan

Use `implementation-goals/GOAL-03-cost-tracking.execution-plan.md`.

## Goal

Implement `GOAL-03-cost-tracking`: add cost tracking per `business_id` to inference logs.

## Intent

Add cost tracking per `business_id` to inference logs so AI usage can be audited by business/customer without breaking existing inference contracts.

## Intent Checksum

If invoking `/ai/claude-code-execute`, pass the preserved intent above. Pass an existing checksum only if supplied by the caller; otherwise allow AI Microservice to compute it.

## Required Context

Read:

- `implementation-goals/GOAL-03-cost-tracking.context-package.md`
- `docs/INTENT_PRESERVATION.md`
- `docs/governance/PROJECT_INVARIANTS.md`
- `docs/process/OPERATIONAL_GATES.md`
- `src/app.module.ts`
- `src/database/entities/*.ts`
- `src/database/migrations/*.sql`

## Scope

Add optional business-level accounting to inference logs in the narrowest backward-compatible place. Existing `/ai/complete` callers that do not send business metadata must continue to work.

## Allowed Changes

- Inference gateway request DTO/controller/service files once located or restored.
- Inference log entity/interceptor/persistence files once located or restored.
- Database migration for nullable business accounting fields.
- Focused tests or validation fixtures using synthetic business identifiers.
- Documentation for the implemented behavior and validation evidence.
- `docs/IMPLEMENTATION_STATE.md` before finishing.

## Forbidden Changes

- Do not change model-tier routing or `litellm_config.yaml`.
- Do not require `business_id` globally unless existing contracts already require it.
- Do not change premium model approval behavior.
- Do not bypass or weaken JWT/admin auth.
- Do not deploy.
- Do not revert unrelated dirty or untracked files.
- Do not include secrets, production prompts, raw customer data, or unredacted production logs in artifacts.

## Required Reading

Start with `src/app.module.ts`. It references inference logging and AI gateway modules that are absent from this local checkout. Before implementing behavior, identify whether those files need to be restored from project history or created from existing documented contracts.

## Implementation Steps

1. Run `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md`.
2. Reconcile the missing local source files referenced by `src/app.module.ts`.
3. Locate or create the `/ai/complete` request contract in a backward-compatible way.
4. Add optional `business_id` or `businessId` handling.
5. Persist the business identifier in inference logs through a nullable schema field.
6. Add validation for requests with and without business metadata.
7. Update documentation and `docs/IMPLEMENTATION_STATE.md`.
8. Create `implementation-goals/GOAL-03-cost-tracking.validation-report.md`.

## Validation

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md
npm run build
npm test
```

## Acceptance Criteria

- Cost tracking behavior is documented.
- Existing `/ai/complete` behavior remains compatible.
- Business-specific usage can be queried or audited from the chosen logging/persistence path.
- Validation evidence is recorded in a validation report.

## Completion Report

Report:

- implemented changes;
- files changed;
- tests run;
- validation evidence;
- blockers;
- risks;
- intent compliance;
- any deviations from the execution plan.
