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
- Status: ready
- Source: `TASKS.md` backlog.
- Goal file: `implementation-goals/GOAL-03-cost-tracking.md`.
- Next required artifact: `implementation-goals/GOAL-03-cost-tracking.execution-plan.md`.
