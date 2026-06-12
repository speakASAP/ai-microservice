# Documentation Completeness Standard

```yaml
id: DOC-AI-DOCUMENTATION-COMPLETENESS
status: reviewed
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
upstream:
  - docs/INTENT_PRESERVATION.md
  - /Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system/23_documentation_contracts/DOCUMENTATION_COMPLETENESS_STANDARD.md
downstream:
  - implementation-goals/
  - implementation-goals/templates/
  - scripts/pre_coding_gate.py
```

Implementation documents must be specific enough for a later session to resume without transcript history. Incompleteness must be explicit. Vague prose is not a substitute for traceability, scope, validation, or evidence.

## Completeness Levels

Use these values in metadata blocks when a document type has metadata:

```yaml
completeness_level: missing | skeletal | partial | complete | validated
```

Definitions:

- `missing`: document does not exist.
- `skeletal`: document exists but mostly contains headings or placeholders.
- `partial`: document contains useful content but lacks required sections.
- `complete`: required sections exist and contain meaningful content.
- `validated`: complete and reviewed against upstream intent and validation evidence.

## Marker Policy

Use `[MISSING: ...]` only while drafting and only to make a gap explicit.

Use `[UNKNOWN: ...]` when information genuinely cannot be derived from available documentation and the discovery path is known.

A goal cannot move to coding while execution-critical markers remain in the selected goal, execution plan, context package, or coding prompt. Execution-critical markers include missing intent, traceability, scope, non-goals, file ownership, implementation steps, validation commands, rollback, sensitive-data handling, or contract impact.

## Meaningful Content Rule

A required section is incomplete if it contains only:

- `TBD`;
- `N/A` without explanation;
- placeholder text;
- empty bullet lists;
- generic statements that do not identify this project, this goal, or this validation path.

## Required Goal Sections

Every `implementation-goals/GOAL-XX-*.md` must include:

- goal id and status;
- owner and dependencies;
- user intent;
- problem statement;
- scope;
- non-goals;
- files to inspect;
- acceptance criteria;
- required artifacts before coding;
- validation commands;
- risks or follow-ups when known;
- completion report requirements.

## Required Execution Plan Sections

Every coding execution plan must include:

- metadata;
- upstream traceability;
- goal impact;
- applicable project invariants;
- sensitive-data handling;
- contract/schema impact;
- replay/determinism impact, or why it is not applicable;
- scope and non-goals;
- files to inspect, create, modify, and avoid;
- implementation steps;
- test plan;
- validation plan;
- gate commands;
- documentation updates;
- rollback plan;
- agent handoff prompt;
- completion checklist.

## Required Context Package Sections

Every context package must include:

- source goal and execution plan;
- preserved intent;
- current implementation state;
- relevant contracts;
- files to read first;
- constraints and non-goals;
- sensitive-data rules;
- validation evidence required.

Context packages are bounded input artifacts. They must not copy large unrelated documents when a path reference is sufficient.

## Required Coding Prompt Sections

Every coding prompt must include:

- task summary;
- execution plan link;
- preserved intent;
- required context;
- allowed changes;
- forbidden changes;
- implementation instructions;
- acceptance criteria;
- validation commands;
- expected completion report.

Coding prompts are generated from the execution plan and context package. They must not add scope that is absent from the selected goal.

## Required Validation Report Sections

Every validation report must include:

- artifact validated;
- preserved intent and checksum evidence when available;
- validation scope;
- command evidence;
- gate evidence;
- invariant evidence;
- sensitive-data scan or review evidence;
- contract/schema evidence;
- replay/determinism evidence when applicable;
- passed criteria;
- failed criteria;
- skipped checks and reasons;
- deviations from the execution plan;
- recommendation: pass, fail, or blocked.

## State Updates

`docs/IMPLEMENTATION_STATE.md` must be updated before ending a coding or orchestration session with:

- active, completed, or blocked goal;
- validation evidence;
- changed files;
- blockers or risks;
- deployment or rollback status when applicable;
- concrete next action.

## Audit Output Requirements

Audit or gate tools should report:

- missing documents;
- missing required sections;
- unresolved markers;
- unknown upstream traceability;
- missing goal impact;
- missing validation evidence;
- suggested remediation.
