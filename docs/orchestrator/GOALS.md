# Goals: AI Microservice Admin

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
