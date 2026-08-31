# ai-microservice

## Status

Production AI inference gateway; canonical IPS adoption is validated.

## Documentation authority

`BUSINESS.md` protects approved business intent, `SYSTEM.md` defines the technical contract, and the central Intent Preservation System standard governs adoption artifacts.

## Capabilities

Model-tier completion, implementation jobs, NLP, ASR, document extraction, prototype generation, AI-agent administration, and inference-cost logging.

## Interfaces

`POST /ai/complete`, `POST /ai/claude-code-execute`, `GET /health`, and documented feature endpoints are protected by service authentication except explicitly public health.

## Development

Run `npm run build` and the relevant Jest test command from this repository. Model tier definitions are maintained in `litellm_config.yaml`.

## Configuration

PostgreSQL uses `DATABASE_URL` or `DB_*`/`POSTGRES_*`; service authentication uses the configured RS256 keys; LiteLLM is selected by `LITELLM_BASE_URL`.

## Deployment

The app runs as a single-container K8s deployment. Ollama and LiteLLM are separate Docker-only dependencies defined by this repository and are not sidecars.

## Health and observability

`GET /health` returns service status. Structured logs go to logging-microservice and inference records retain caller and optional business attribution.
