# EP-TASK-001-bootstrap-service: Bootstrap ai-microservice

```yaml
id: EP-TASK-001-bootstrap-service
status: implemented
owner: project owner
created: 2026-08-30
last_updated: 2026-08-30
completeness_level: validated
source_task: ../11_tasks/TASK-001-bootstrap-service.md
goal_impact:
  - ../22_goal_impact/GOAL-IMPACT-TASK-001.md
validation:
  - ../12_validation/VAL-TASK-001-bootstrap-service.md
```

## Upstream traceability

`BUSINESS.md` and `VISION.md` establish purpose; `SYSTEM.md` and `PROJECT_INVARIANTS.md` establish technical and safety boundaries; `TASK-001-bootstrap-service.md` bounds delivery.

## Scope

Scaffold and complete required canonical adoption documents, profile, and state while preserving real production facts.

## Non-goals

No source, infrastructure, secret, Docker, provider-routing, migration, or deployment operation.

## Project invariants

Preserve direct-provider centralization, RS256 algorithm pinning, re-minting after signing-key rotation, compatibility, and premium approval.

## Sensitive-data handling

Use configuration names and architecture descriptions only; never print or write secret values, raw tokens, or production data.

## Contract validation plan

Compare required capability decisions with source modules and repository configuration; retain Docker-only dependency boundaries.

## Replay and determinism plan

Run the dependency-free validator against a static documentation profile; no runtime replay semantics change.

## Files to inspect

Root contracts, `src/`, `admin-panel/`, `docker-compose.ollama.yml`, `litellm_config.yaml`, registry artifacts, and central validator source.

## Files to create

Canonical constitution, vision, integration contract, invariants, bootstrap task, goal impact, execution plan, validation report, and adoption profile.

## Files to modify

Root documentation contracts, state, task register, and validation debt ledger.

## Files that must not be modified

Runtime source, deployment files, Docker configuration, LiteLLM configuration, secret material, and registry master plans outside this repository.

## Implementation steps

1. Read standards and source evidence.
2. Run the non-destructive scaffolder.
3. Restructure approved facts into required canonical documents.
4. Review capability decisions and validate.

## Parallel execution

| Workstream | Status | Owner role | Allowed files | Dependencies | Validation | Merge order |
| --- | --- | --- | --- | --- | --- | --- |
| Documentation and contracts | completed | worker agent | canonical and root docs | approved sources | adoption validator | first |
| Integration profile | completed | integration validator | `ips-adoption.json` | source review | capability review | second |
| Final validation | final integration | integration validator | validation report | completed documents | planning validator | last |

## Blockers

No blocker is recorded.

## Test plan

Run the adoption validator; source code is not changed, so no new runtime test is required for this documentation task.

## Validation plan

Map artifact completeness and all capability decisions to the planning validator; review security invariants and Docker boundaries in canonical documents.

## Gate commands

`python3 ../intent-preservation-system/scripts/validate_adoption_profile.py --root . --phase planning`

## Documentation updates

Update all required canonical adoption artifacts and root authority documents.

## Rollback plan

Revert the documentation-only commit if canonical content is found inconsistent with approved source facts; no runtime rollback exists.

## Handoff

Handoff includes validator output, commit hash, file list, scope confirmation, and any blocker.

## Completion checklist

Protected intent approved, adoption profile valid, integration decisions complete, invariants documented, and validation report completed.
