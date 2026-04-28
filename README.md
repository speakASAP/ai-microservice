# AI Microservice

Centralized AI processing service for the Statex ecosystem. Routes LLM calls by model tier, provides NLP, ASR, Document AI, and prototype generation.

**Stack**: NestJS · LiteLLM · Ollama · OpenRouter · Gemini — see `SYSTEM.md`

## Features

- AI Orchestrator — central coordination, `POST /ai/complete` (LLM gateway)
- NLP, ASR, Document AI, Prototype Generator, Template Repository
- Free AI Service, AI Workers, Gemini AI Service, Data Viz Service
- Email-triage agents (Ingest, Classifier, Extractor, Decider) for agentic-email-processing-system
- Centralized Logging, Shared Database, Blue/Green Deployment

## Ports

| Service | Port | Env var |
|---------|------|---------|
| AI Orchestrator | 3380 | `AI_ORCHESTRATOR_PORT` |
| NLP Service | 3381 | `NLP_SERVICE_PORT` |
| ASR Service | 3382 | `ASR_SERVICE_PORT` |
| Document AI | 3383 | `DOCUMENT_AI_PORT` |
| Prototype Generator | 3384 | `PROTOTYPE_GENERATOR_PORT` |
| Template Repository | 3385 | `TEMPLATE_REPOSITORY_PORT` |
| Free AI Service | 3386 | `FREE_AI_SERVICE_PORT` |
| AI Workers | 3387 | `AI_WORKERS_PORT` |
| Gemini AI Service | 3388 | `GEMINI_AI_SERVICE_PORT` |
| Data Viz Service | 3389 | `DATA_VIZ_SERVICE_PORT` |
| LiteLLM proxy | 4000 (internal) | `LITELLM_BASE_URL` |
| Ollama | 11434 (internal) | `OLLAMA_API_BASE` |

**LiteLLM** routes `free`/`cheap`/`smart` tiers; config in `litellm_config.yaml`. **Ollama** built from `services/ollama/Dockerfile`; after first deploy pull weights: `docker exec -it ai-microservice-ollama-green ollama pull qwen2.5-coder:0.5b`. Cold-start order: `ollama` → `litellm` → `free-ai-service` → `backend`.

## Environment / Secrets

Secrets managed by Vault via Kubernetes ESO (`secret/prod/ai-microservice`). Local dev: `./scripts/vault-env-gen.sh` generates `.env`.

## Access

```bash
# Production
curl https://ai.alfares.cz/health

# Docker network (from container on nginx-network)
curl http://ai-microservice:3380/health

# SSH to server
ssh statex && cd /home/statex/ai-microservice
```

## Quick Start

```bash
./scripts/start.sh
./scripts/status.sh
./scripts/stop.sh
docker compose -f docker-compose.blue.yml logs -f
./scripts/deploy.sh
```

## API Endpoints — AI Orchestrator

- `POST /ai/complete` — LLM gateway (`model_tier`, `system_prompt`, `user_prompt`, `output_schema?`, `max_tokens?`, `correlation_id?`) · JWT required · see `docs/model-tier-endpoints.md`
- `POST /api/process-submission` · `GET /api/status/{id}` · `GET /api/results/{id}` · `GET /health`
- `POST /api/shop-assistant/transcribe` — ASR
- `POST /api/shop-assistant/refine-query` — COMMUNICATION agent
- `POST /api/shop-assistant/search` — SEARCH agent
- `POST /api/shop-assistant/format-presentation` — PRESENTATION agent
- `POST /api/shop-assistant/compare-prices` — COMPARISON agent
- `POST /api/shop-assistant/extract-location` — LOCATION agent
- `POST /api/email-triage/ingest` — validate/normalize email
- `POST /api/email-triage/classify` — intent + confidence (`EMAIL_TRIAGE_LLM_CLASSIFIER=true` for LLM)
- `POST /api/email-triage/extract` — entity extraction
- `POST /api/email-triage/decide` — action decision (`EMAIL_TRIAGE_LLM_DECIDER=true` for LLM)
- `POST /api/v1/translate` — stable translation API

## Shared Services

- **Database**: `db-server-postgres:5432`, db `statex_ai`; Redis at `db-server-redis`
- **Logging**: `logging-microservice:3367` (prod: `https://logging.alfares.cz`); fallback to local files

## Internal Connectivity

Containers on `nginx-network` call the orchestrator at `http://ai-microservice:3380`. The hostname is a Docker alias on the active blue/green backend — only one stack should run at a time. If consumers report timeouts, confirm: `curl http://ai-microservice:3380/health` from within a container on nginx-network.
