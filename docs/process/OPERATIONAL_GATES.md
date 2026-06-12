# Operational Gates

```yaml
id: DOC-AI-OPERATIONAL-GATES
status: reviewed
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
upstream:
  - docs/INTENT_PRESERVATION.md
  - docs/process/DOCUMENTATION_COMPLETENESS_STANDARD.md
  - /Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system/23_documentation_contracts/OPERATIONAL_GATE_STANDARD.md
downstream:
  - scripts/pre_coding_gate.py
  - scripts/deployment_readiness_gate.py
  - implementation-goals/templates/VALIDATION_REPORT.md
```

Operational gates make intent preservation enforceable. They block coding, integration, deployment, or closure when the required intent, traceability, scope, validation, or safety evidence is missing.

## Gate Types

| Gate | Timing | Blocks on |
| --- | --- | --- |
| Pre-coding gate | Before editing code for a selected goal. | Missing goal, execution plan, context package, coding prompt, intent, traceability, validation plan, project invariants, or sensitive-data handling. |
| Build/test gate | After code changes and before review or closure. | Failed build, failed tests, skipped relevant checks without documented reason, or behavior that violates preserved intent. |
| Integration-readiness gate | Before combining independently developed changes. | Failed contracts, invariant violations, incomplete test evidence, or unresolved ownership overlap. |
| Deployment-readiness gate | Before release, merge, deployment, or closure. | Failed pre-coding gate, missing validation report, unresolved markers, missing rollback note, protected document uncertainty, or deployment outside selected goal scope. |

## Required Evidence

Each gate must produce or reference:

- command executed;
- repository root;
- target artifact;
- status;
- missing files;
- failed checks;
- invariant evidence;
- sensitive-data scan or review result;
- next action.

Reports should be written under `reports/validation/` as JSON or Markdown when the gate is part of a coding session. Reports are evidence, not source-of-truth governance documents.

## Pre-Coding Gate

Run before editing code for a selected goal:

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-XX-name.md
```

The gate checks:

- required orchestration documents exist;
- selected goal exists and includes intent, scope, non-goals, and acceptance criteria;
- selected goal has no unresolved `[MISSING: ...]` markers;
- execution plan, context package, and coding prompt exist for the selected goal;
- required artifacts have intent, scope or constraints, and validation sections;
- execution-critical markers are resolved before coding.

## Build/Test Gate

Run the narrowest relevant validation for the changed surface:

```bash
npm run build
npm test
```

For documentation-only changes, run relevant syntax and gate checks instead. If a command cannot run locally, record the reason in the validation report and `docs/IMPLEMENTATION_STATE.md`.

## Integration-Readiness Gate

Run before merging subagent or branch work into the orchestrator-controlled state. The orchestrator must verify:

- each worker stayed inside its declared file ownership;
- no worker reverted unrelated dirty files;
- contract and DTO changes are documented;
- validation evidence maps to the selected goal;
- the preserved intent still matches the implementation.

## Deployment Readiness Gate

Run before production deployment:

```bash
python3 scripts/deployment_readiness_gate.py --root .
```

Deployment requires:

- selected goal allows deployment or the user explicitly requested it;
- build/test evidence exists;
- rollback note exists;
- no known secret exposure;
- changed files and validation evidence are recorded;
- `docs/IMPLEMENTATION_STATE.md` contains the concrete deployment status and next action.

## Failure Policy

Fail closed on missing intent, missing validation criteria, auth uncertainty, secret exposure risk, unclear production impact, unresolved execution-critical markers, or missing required artifacts. Do not weaken a gate to make a task pass. Fix the artifact, document an exception, or split the goal so the gate can evaluate it precisely.
