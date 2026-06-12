# Master Prompt: AI Microservice Orchestrator

Use this prompt to resume coordinated work in this repository:

```text
AI-MICROSERVICE ORCHESTRATOR: continue implementation using docs/IMPLEMENTATION_ORCHESTRATOR.md. Query RAG first when available, read docs/IMPLEMENTATION_STATE.md, select the next ready goal from implementation-goals, create or update the execution plan/context package/coding prompt before coding, run gates, validate, update state, and report the next action.
```

The orchestrator owns goal selection, task splitting, executor coordination, validation, and state updates. Subagents may work only on bounded tasks with explicit file ownership.
