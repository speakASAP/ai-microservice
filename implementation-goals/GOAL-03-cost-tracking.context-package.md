# Context Package: GOAL-03

```yaml
id: CP-GOAL-03
status: complete
source_goal: implementation-goals/GOAL-03-cost-tracking.md
source_execution_plan: implementation-goals/GOAL-03-cost-tracking.execution-plan.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Intent

Add cost tracking per `business_id` to inference logs so AI usage can be audited by business/customer without breaking existing inference contracts.

## Intent Checksum

Runtime implementation jobs should provide this intent text and may omit `intentChecksum`; AI Microservice will compute the checksum when enqueueing the job. If a caller already has a checksum, it must pass it through unchanged.

## Current State

`docs/IMPLEMENTATION_STATE.md` marks `GOAL-03-cost-tracking` complete. The implementation added optional business-level accounting metadata to `/ai/complete` inference logs, applied the migration, and verified production deployment.

The remote checkout contains the AI gateway, inference logging interceptor, `InferenceLog` entity, and migration files used by this completed goal.

## Relevant Contracts

- `POST /ai/complete` is the central LLM gateway and must remain backward compatible.
- `docs/governance/PROJECT_INVARIANTS.md` requires runtime contract preservation, explicit auth, canonical model routing in `litellm_config.yaml`, premium model approval, secret safety, deployment as a goal, and dirty-worktree safety.
- `docs/INTENT_PRESERVATION.md` requires intent continuity from goal through validation.
- Existing implementation-job contracts are in `src/contracts/claude-code-job.contract.ts` and are not the target of this goal.

## Files To Read First

1. `AGENTS.md`
2. `README.md`
3. `SYSTEM.md`
4. `TASKS.md`
5. `docs/INTENT_PRESERVATION.md`
6. `docs/IMPLEMENTATION_STATE.md`
7. `docs/governance/PROJECT_INVARIANTS.md`
8. `docs/process/OPERATIONAL_GATES.md`
9. `implementation-goals/GOAL-03-cost-tracking.md`
10. `implementation-goals/GOAL-03-cost-tracking.execution-plan.md`
11. `src/app.module.ts`
12. `src/database/entities/*.ts`
13. `src/database/migrations/*.sql`

## Constraints

- Preserve `/ai/complete` compatibility.
- Keep `business_id` optional unless existing source proves it is already required.
- Do not redesign model-tier routing.
- Do not change premium model approval behavior.
- Do not bypass auth.
- Do not deploy unless explicitly requested.
- Do not revert unrelated dirty or untracked files.
- If required local source files are missing, reconcile that gap before implementing behavior.

## Sensitive-Data Rules

`business_id` is sensitive tenant/customer metadata. Use synthetic identifiers such as `biz_test_001` in tests and validation reports. Do not include bearer tokens, production prompts, raw customer records, production request payloads, or unredacted logs in artifacts.

## Validation Evidence Required

Required commands:

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md
npm run build
npm test
```

Required report evidence:

- existing caller without `business_id` remains compatible;
- caller with synthetic `business_id` records auditable business metadata;
- schema or migration impact is documented;
- skipped checks include reasons;
- intent compliance is pass, fail, or blocked.
