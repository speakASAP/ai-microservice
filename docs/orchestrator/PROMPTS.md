# Orchestrator Prompts

## Continue

```text
AI-MICROSERVICE ORCHESTRATOR: continue implementation
```

## Implement Specific Goal

```text
AI-MICROSERVICE ORCHESTRATOR: implement goal number N
```

## Validation Only

```text
AI-MICROSERVICE ORCHESTRATOR: validate the active goal and update docs/IMPLEMENTATION_STATE.md
```

## Worker Handoff Shape

```text
You are a bounded implementation worker for AI Microservice.
Read AGENTS.md, docs/IMPLEMENTATION_STATE.md, docs/IMPLEMENTATION_ORCHESTRATOR.md, and the selected goal.
Only modify the assigned files.
Preserve user intent and do not revert unrelated changes.
Report changed files, validation commands, blockers, and intent evidence.
```
