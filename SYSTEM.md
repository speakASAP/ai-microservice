# System: ai-microservice

```yaml
id: SYSTEM-ai-microservice
status: validated
owner: ai-microservice maintainers
created: 2026-06-13
last_updated: 2026-08-30
completeness_level: validated
upstream:
  - BUSINESS.md
  - docs/01_vision/VISION.md
downstream:
  - docs/06_architecture/INTEGRATION_CONTRACT.md
  - docs/17_governance/PROJECT_INVARIANTS.md
```

## Purpose

Provide the production NestJS AI inference gateway for Statex on port 3380 and `https://ai.alfares.cz`.

## Responsibilities

Provide Orchestrator, NLP, ASR, Document AI, Prototype Generator, Free AI, Gemini, and Data Viz modules; serve `POST /ai/complete`; persist AI agents, implementation jobs, and inference logs through TypeORM; and preserve implementation-job `intent`, `intentChecksum`, and `implementationProvider`.

## Non-responsibilities

The K8s ai-microservice pod runs one `app` container. Docker-only Ollama and LiteLLM are not sidecars. This service does not own commerce domains, and premium inference is never an unattended LiteLLM route.

## Inputs

RS256 bearer tokens, AI HTTP requests, RabbitMQ implementation-job messages, and environment-supplied database and provider configuration.

## Outputs

Model-tier completion responses, PostgreSQL agent/job/log records, structured logs, and published and consumed Claude Code job messages.

## Dependencies

TypeORM connects to database-server PostgreSQL using `DATABASE_URL` or `DB_*`/`POSTGRES_*`. The established database-server integration is `db-server-postgres:5432 + Redis`. `LoggingClient` posts to logging-microservice port 3367. RabbitMQ carries Claude Code jobs. Voice transcription fetches audio from configured MinIO-compatible object storage. Docker container `ai-microservice-ollama-green` runs on `:11435` and `nginx-network`, is defined by `docker-compose.ollama.yml`, and is addressed by `OLLAMA_API_BASE`. Docker container `ai-microservice-litellm-green` runs on `:4000` and `nginx-network`, with tier routes in `litellm_config.yaml`. `CODEX_CLI_PATH` selects Codex implementation jobs.

## Service authentication (RS256)

For machine service identity, follow the sole canonical [`SERVICE_IDENTITY_CONSUMER_STANDARD.md`](../auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md). It is not reproduced here.

## Upstream traceability

`BUSINESS.md`, `docs/01_vision/VISION.md`, and `docs/06_architecture/INTEGRATION_CONTRACT.md` define the approved purpose and contracts.

## Downstream artifacts

`docs/06_architecture/INTEGRATION_CONTRACT.md`, `docs/17_governance/PROJECT_INVARIANTS.md`, and the bootstrap task record implementation boundaries.

## Validation criteria

`GET /health` returns a healthy service response, the adoption validator accepts the profile, and `npm run build` plus maintained tests cover application behavior.

## Open questions

No open system-design question is recorded for this completed adoption.

## Architecture context

NestJS modules cover Orchestrator, NLP, ASR, Document AI, Prototype Generator, Free AI, Gemini, and Data Viz. Tier routing is free (Ollama), cheap (OpenRouter through LiteLLM when enabled), and smart (Gemini 2.0 Flash through LiteLLM when enabled); the currently configured LiteLLM upstreams and fallbacks are authoritative in `litellm_config.yaml`. Premium remains deferred until a funded production rollout and requires per-call human approval. LiteLLM provides automatic failover when `LITELLM_BASE_URL` is set. Implementation providers are `claude-code` by default and `codex` through `/ai/claude-code-execute`; provider choice is separate from `/ai/complete` model-tier routing.
