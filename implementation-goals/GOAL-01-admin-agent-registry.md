# GOAL-01: Admin Agent Registry

```yaml
id: GOAL-01
status: done
owner: orchestrator
completed: 2026-06-12
```

## Intent

Provide a usable admin frontend for managing existing and future AI agent definitions from a browser.

## Outcome

- Public browser entry at `/admin`.
- CRUD for AI agent definitions.
- Editable prompts, model tier, provider model override, token limits, temperature, output schema, metadata, tags, route path, service scope, and status.
- Existing AI runtime endpoints preserved.

## Validation Evidence

- Backend build passed remotely.
- Production health passed at `https://ai.alfares.cz/health`.
- Production admin UI available at `https://ai.alfares.cz/admin`.
- Authenticated agent CRUD smoke passed.
