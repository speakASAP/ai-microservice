# Plan: AI Microservice Admin

1. Add an `AiAgent` entity for persisted agent definitions.
2. Add an admin module with a CRUD service and controller under `/admin/api/agents`.
3. Serve a static browser admin app under `/admin`.
4. Use the existing JWT service auth for write/read API calls. The frontend stores the token in local storage on the operator's machine.
5. Build and deploy through the existing Kubernetes script.

Implementation constraints:
- Do not touch the existing dirty `src/claude-code/claude-code.consumer.ts` file.
- Do not change the runtime inference contracts.
- Keep all UI controls code-native and editable.
- Keep deployment in the main backend image because the current `admin-panel` Next app is only a skeleton and has no deployment manifests.
