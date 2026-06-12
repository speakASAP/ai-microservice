# EP-AI-03: Cost Tracking Per Business

```yaml
id: EP-AI-03
status: draft
source_goal: implementation-goals/GOAL-03-cost-tracking.md
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
```

## Metadata

Branch: current working tree on `main`.

Lifecycle state: ready for pre-coding gate after context package and coding prompt are present.

Selected goal: `implementation-goals/GOAL-03-cost-tracking.md`.

## Upstream Traceability

User intent: add cost tracking per `business_id` to inference logs so AI usage can be audited by business/customer without breaking existing inference contracts.

Project state: `docs/IMPLEMENTATION_STATE.md` identifies `GOAL-03-cost-tracking` as the next ready goal after `GOAL-02`.

Backlog source: `TASKS.md` lists "Add cost tracking per business_id to inference logs" as the next feature work after the unified gateway.

Governance source: `docs/INTENT_PRESERVATION.md`, `docs/governance/PROJECT_INVARIANTS.md`, and `docs/process/OPERATIONAL_GATES.md`.

## Goal Impact

This goal adds auditable business/customer usage accounting to AI inference logs while preserving AI Microservice as the central inference gateway. The intended impact is operational accountability, not model-routing redesign.

## Project Invariants

- `INV-01 Preserve Runtime Contracts`: `/ai/complete` and existing endpoints must remain backward compatible.
- `INV-02 Preserve Intent`: intent and validation evidence must remain attached to this goal.
- `INV-03 Auth Stays Explicit`: no auth bypass or weakening.
- `INV-04 Model Routing Source Of Truth`: no model-tier routing changes outside `litellm_config.yaml`.
- `INV-05 Premium Requires Approval`: do not route premium models without approval.
- `INV-06 Secrets Stay Out Of Artifacts`: no secrets or raw production data in reports or tests.
- `INV-07 Deployment Is A Goal`: do not deploy unless explicitly requested.
- `INV-08 Dirty Worktree Safety`: do not revert unrelated files.

## Sensitive-Data Handling

Data classification: operational metadata. `business_id` is a tenant/customer identifier and must be treated as sensitive metadata.

Rules:

- Use synthetic `business_id` values in tests and validation reports.
- Do not copy production request bodies, bearer tokens, customer data, or raw inference prompts into documentation or test fixtures.
- Log only the fields needed for accounting and debugging.

## Contract/Schema Impact

Expected impact:

- `/ai/complete` request handling may accept optional `business_id` or `businessId`.
- Inference logging persistence may gain a nullable business identifier column.
- Existing callers that omit the field must continue to work.

Known local checkout gap:

- `src/app.module.ts` references `InferenceLog`, `InferenceLogInterceptor`, and `AiModule`, but the corresponding local files are absent in this checkout. Implementation must first reconcile the local source tree with the referenced modules or restore/create the missing files in the narrowest compatible way.

## Replay/Determinism Impact

No deterministic model output behavior should change. Job retry/replay semantics are not in scope except that repeated requests should produce consistent accounting metadata for the same supplied `business_id`.

## Scope

- Inspect current AI gateway, inference logging, entities, migrations, and module wiring.
- Add nullable business-level accounting to inference logs.
- Preserve compatibility for requests without business metadata.
- Add or update migration/entity code for the chosen persistence path.
- Add validation that existing callers still work and business-specific usage can be audited.
- Document the resulting behavior.

## Non-Goals

- Do not redesign model-tier routing.
- Do not require `business_id` for all callers unless an existing contract already requires it.
- Do not change premium approval rules.
- Do not deploy production changes unless explicitly requested.
- Do not implement unrelated admin UI or agent registry behavior.

## Files To Inspect

- `AGENTS.md`
- `README.md`
- `SYSTEM.md`
- `TASKS.md`
- `docs/INTENT_PRESERVATION.md`
- `docs/IMPLEMENTATION_STATE.md`
- `docs/governance/PROJECT_INVARIANTS.md`
- `src/app.module.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/*.ts`
- `src/database/migrations/*.sql`
- any restored or newly discovered `src/ai`, `src/service-identity`, or inference logging files

## Files To Create

- A database migration for nullable business-level inference accounting if persistence requires one.
- Tests or validation fixtures for business-specific inference logging.
- `implementation-goals/GOAL-03-cost-tracking.validation-report.md` after implementation.

## Files To Modify

- `src/app.module.ts` only if module/entity wiring requires correction.
- Inference request DTO/controller/service files once their local paths are present or restored.
- Inference log entity/interceptor/persistence files once their local paths are present or restored.
- `README.md`, `SYSTEM.md`, or `docs/INTENT_PRESERVATION.md` only if public behavior or governance changes.
- `docs/IMPLEMENTATION_STATE.md` before ending the session.

## Files That Must Not Be Modified

- Runtime model-tier routing in `litellm_config.yaml` unless this goal discovers documentation drift and the user approves.
- Admin agent registry behavior outside incidental type compatibility.
- Claude Code/Codex implementation-job provider behavior, except documentation references if needed.
- Deployment scripts and Kubernetes manifests unless a later deployment-hardening goal owns them.

## Implementation Steps

1. Run the pre-coding gate for `GOAL-03` and confirm this plan, context package, and coding prompt are present.
2. Inspect `src/app.module.ts` imports and reconcile missing local files for inference logging and the AI gateway.
3. Identify the actual `/ai/complete` request DTO/controller/service path.
4. Add optional `business_id` support in the narrowest backward-compatible contract location.
5. Persist business metadata in inference logs using a nullable field and migration.
6. Add an audit/query path only if an existing logging or admin endpoint already owns that responsibility.
7. Add tests or validation commands that prove existing callers work and business-specific accounting is recorded.
8. Update docs and `docs/IMPLEMENTATION_STATE.md`.
9. Produce `implementation-goals/GOAL-03-cost-tracking.validation-report.md`.

## Test Plan

- Compile TypeScript with `npm run build`.
- Run available tests with `npm test`.
- Add focused tests for requests with and without `business_id` when the project test harness is available.
- If no automated tests exist for the gateway, run a documented manual validation against local request handling using synthetic data.

## Validation Plan

Expected evidence:

- `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md`
- `npm run build`
- `npm test`
- validation report showing preserved `/ai/complete` compatibility and business-specific inference logging.

## Gate Commands

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md
npm run build
npm test
python3 scripts/deployment_readiness_gate.py --root .
```

## Gate Evidence Location

Record gate evidence in `implementation-goals/GOAL-03-cost-tracking.validation-report.md` and summarize it in `docs/IMPLEMENTATION_STATE.md`.

## Documentation Updates

- Update `README.md` or `SYSTEM.md` if the request contract is documented there.
- Update `docs/IMPLEMENTATION_STATE.md` with validation evidence, changed files, risks, and next action.
- Update this execution plan if implementation discovery changes file ownership or validation commands.

## Rollback Plan

Rollback by reverting the code, migration, and documentation changes owned by `GOAL-03`. If a migration has been applied outside local development, add a forward rollback migration that removes or ignores the added nullable accounting fields only after confirming no production consumer depends on them.

## Agent Handoff Prompt

Implement `GOAL-03-cost-tracking` for AI Microservice. Preserve the intent: add cost tracking per `business_id` to inference logs so AI usage can be audited by business/customer without breaking existing inference contracts. Read this execution plan, the context package, the coding prompt, `docs/INTENT_PRESERVATION.md`, `docs/governance/PROJECT_INVARIANTS.md`, and `src/app.module.ts` before editing. First reconcile the missing local inference logging and AI gateway files referenced by `src/app.module.ts`. Keep `business_id` optional, use synthetic data in tests, do not change model routing or premium approval, and do not deploy. Run the pre-coding gate before code edits and record validation evidence in a validation report.

## Completion Checklist

- [ ] Implementation complete
- [ ] Tests complete
- [ ] Validation evidence collected
- [ ] Documentation updated
- [ ] Deviations documented
