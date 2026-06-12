# Goals: AI Microservice Orchestrator

This file is the compact operator view. The canonical implementation checkpoint is `docs/IMPLEMENTATION_STATE.md`.

## G1 - Inspect ecosystem
- Status: complete
- Confirmed a large multi-repo ecosystem under `/home/ssf/Documents/Github`.
- Confirmed `ai-microservice` is the central AI inference gateway and is deployed at `ai.alfares.cz`.

## G2 - Add backend agent registry
- Status: implemented
- Add `ai_agents` persistence through TypeORM.
- Add protected `/admin/api/agents` CRUD endpoints.
- Keep endpoint authentication on the existing service JWT guard.

## G3 - Add admin frontend
- Status: implemented
- Serve `/admin` from the Nest app.
- Provide list, search/filter, create, edit, duplicate, delete, JSON prompt/config editing, and service token management.

## G4 - Verify and deploy
- Status: complete
- Build backend.
- Copy changed files to the remote repo.
- Deploy with `scripts/deploy.sh`.
- Verify `/health` and `/admin`.

## G5 - Add Goalkeeper-style orchestrator operating model
- Status: complete
- Add one master implementation orchestrator.
- Store continuation state in `docs/IMPLEMENTATION_STATE.md`.
- Store executable goals in `implementation-goals/`.
- Require execution plans, context packages, coding prompts, validation reports, and gates before coding.

## G6 - Add cost tracking per business
- Status: complete
- Source: `TASKS.md` backlog.
- Goal file: `implementation-goals/GOAL-03-cost-tracking.md`.
- Outcome: optional business-level accounting metadata added to `/ai/complete` inference logs; migration applied and production verified.

## G7 - Implementation job observability
- Status: complete
- Goal file: `implementation-goals/GOAL-04-implementation-job-observability.md`.
- Outcome: added lifecycle, audit, redacted summary, and validation metadata for implementation jobs.

## G8 - Agent registry routing
- Status: complete
- Goal file: `implementation-goals/GOAL-05-agent-registry-routing.md`.
- Outcome: added explicit active-agent routing for `/ai/complete`; validated but not separately deployed.

## G9 - Deployment hardening
- Status: complete
- Goal file: `implementation-goals/GOAL-06-deployment-hardening.md`.
- Outcome: hardened readiness gates, smoke checks, rollback evidence, and deploy handoff.

## Next
- No active implementation goal is queued in `docs/IMPLEMENTATION_STATE.md`.
