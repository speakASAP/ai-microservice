# Plan: AI Microservice Orchestrator

1. Keep `docs/IMPLEMENTATION_STATE.md` as the single source of truth for continuation.
2. Select the next goal from `implementation-goals/` unless the user explicitly overrides it.
3. Before coding, create or update:
   - execution plan;
   - context package;
   - coding prompt;
   - validation report target.
4. Run `python3 scripts/pre_coding_gate.py --root . --goal <goal-file>` before edits.
5. Coordinate bounded subagents only when file ownership is disjoint.
6. Validate with the narrowest relevant commands and update `docs/IMPLEMENTATION_STATE.md`.

Implementation constraints:
- Do not touch the existing dirty `src/claude-code/claude-code.consumer.ts` file.
- Do not change the runtime inference contracts.
- Keep all UI controls code-native and editable.
- Keep deployment in the main backend image because the current `admin-panel` Next app is only a skeleton and has no deployment manifests.
- Do not deploy unless the selected goal or user explicitly asks for deployment.

Next planned implementation:
- `GOAL-03-cost-tracking`: add cost tracking per `business_id` to inference logs.
