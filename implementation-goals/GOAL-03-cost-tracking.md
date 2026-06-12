# GOAL-03: Cost Tracking Per Business

```yaml
id: GOAL-03
status: done
owner: orchestrator
dependencies:
  - GOAL-02
```

## Intent

Add cost tracking per `business_id` to inference logs so AI usage can be audited by business/customer without breaking existing inference contracts.

## Problem

`TASKS.md` tracked cost tracking per `business_id` as follow-up feature work after the unified gateway. GOAL-03 implemented the current inference logging path and added accounting without disrupting `/ai/complete` callers.

## Scope

- Inspect existing inference DTOs, controllers, services, persistence, and logging.
- Identify where `business_id` is accepted, inferred, or should remain optional.
- Add cost/accounting fields in the narrowest compatible place.
- Preserve backwards compatibility for callers that do not send `business_id`.
- Add tests or validation covering existing callers and business-specific logging.

## Non-Goals

- Do not redesign model-tier routing.
- Do not require `business_id` for all callers unless an existing contract already does.
- Do not change premium model approval rules.
- Do not deploy unless explicitly requested.

## Files To Inspect

- `AGENTS.md`
- `README.md`
- `SYSTEM.md`
- `TASKS.md`
- `docs/INTENT_PRESERVATION.md`
- `src/app.module.ts`
- `src/**`
- database migrations and entities

## Acceptance Criteria

- Cost tracking behavior is documented.
- Existing `/ai/complete` behavior remains compatible.
- Business-specific usage can be queried or audited from the chosen logging/persistence path.
- Validation evidence is recorded in a validation report.

## Required Artifacts Before Coding

- `implementation-goals/GOAL-03-cost-tracking.execution-plan.md`
- `implementation-goals/GOAL-03-cost-tracking.context-package.md`
- `implementation-goals/GOAL-03-cost-tracking.coding-prompt.md`

## Validation Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md
npm run build
npm test
```
