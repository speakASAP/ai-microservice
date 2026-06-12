# AI Microservice Implementation Goals

This directory contains executable goal prompts for separate implementation sessions.

Use the master command:

```text
AI-MICROSERVICE ORCHESTRATOR: continue implementation
```

To print the current checkpoint:

```bash
./scripts/next_goal.sh
```

## Goals

1. `GOAL-01-admin-agent-registry.md` - completed admin registry and frontend.
2. `GOAL-02-orchestrator-operating-model.md` - completed Goalkeeper-style operating model.
3. `GOAL-03-cost-tracking.md` - add cost tracking per `business_id` to inference logs.
4. `GOAL-04-implementation-job-observability.md` - improve implementation job status, logs, and audit summaries.
5. `GOAL-05-agent-registry-routing.md` - connect persisted agent definitions to controlled routing.
6. `GOAL-06-deployment-hardening.md` - strengthen deployment, smoke checks, rollback, and evidence.

## Required Workflow

Every coding goal session must:

1. Query RAG when available.
2. Read required orchestrator docs.
3. Run `git status --short --branch`.
4. Create or update execution plan, context package, and coding prompt.
5. Run the pre-coding gate.
6. Keep implementation within selected goal scope.
7. Run relevant validation.
8. Produce a validation report and intent compliance report.
9. Update `docs/IMPLEMENTATION_STATE.md`.

## Required Artifact Chain

Every coding goal must preserve:

```text
User Intent -> Project Invariants -> Goal -> Execution Plan
-> Context Package -> Coding Prompt -> Code -> Validation Report
-> Implementation State
```

Required files before coding:

- `implementation-goals/GOAL-XX-name.md`
- `implementation-goals/GOAL-XX-name.execution-plan.md`
- `implementation-goals/GOAL-XX-name.context-package.md`
- `implementation-goals/GOAL-XX-name.coding-prompt.md`

Required file before closure:

- `implementation-goals/GOAL-XX-name.validation-report.md`

Run:

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-XX-name.md
```

The gate must pass before editing runtime code.

## Final Report Shape

```markdown
## Intent Compliance Report

### Goal
...

### Implemented
...

### Not Implemented
...

### Boundary Check
...

### Subagents Used
...

### Validation Evidence
...

### Risks
...

### Files Changed
...

### Next Action
...
```
