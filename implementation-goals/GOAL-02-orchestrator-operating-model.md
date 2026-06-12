# GOAL-02: Orchestrator Operating Model

```yaml
id: GOAL-02
status: done
owner: orchestrator
completed: 2026-06-12
```

## Intent

Organize AI Microservice work like Goalkeeper: one master orchestrator agent checks jobs, sets goals, splits goals into plans, coordinates tasks, and preserves state in repository documentation.

## Scope

- Add master orchestrator documentation.
- Add implementation state checkpoint.
- Add process, governance, branch workflow, and goal queue.
- Add templates for execution plans, context packages, coding prompts, and validation reports.
- Add local helper scripts for next-goal selection and gates.

## Acceptance Criteria

- New sessions can resume from `docs/IMPLEMENTATION_STATE.md`.
- Goals are represented under `implementation-goals/`.
- Coding work requires execution plan, context package, coding prompt, and pre-coding gate.
- Deployment is gated and documented.
