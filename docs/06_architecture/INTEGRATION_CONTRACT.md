# Integration Contract

## Purpose

Define deliberate ecosystem integration decisions for the production AI gateway and its separate Docker-only inference dependencies.

## Capability decisions

| Capability | Decision | Evidence |
| --- | --- | --- |
| Auth | required | Service calls use Auth-issued per-pair RS256 JWTs with target-scoped roles; see `auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md`. |
| PostgreSQL | required | TypeORM persists agents, jobs, and inference logs. |
| Redis | required | The established database-server contract is PostgreSQL plus Redis. |
| Logging | required | `LoggingClient` posts structured logs to logging-microservice. |
| Notifications | required | Contract-violation filtering uses configured notifications delivery. |
| AI | not-applicable | This project is the AI domain gateway, not a client of another ai-microservice. |
| Payments, catalog, orders, warehouse, invoices | not-applicable | The gateway is called by domain services and contains no client integration for these domains. |
| Object storage | required | Voice transcription fetches audio from configured MinIO-compatible storage. |
| Event bus | required | Claude Code jobs publish and consume RabbitMQ messages. |
| docs-RAG | required | Repository documentation is indexed directly from Git; Git remains authoritative. |
| Monitoring | required | Public `GET /health` supports runtime health checks. |
| Backups | not-applicable | No backups-microservice client is implemented in this gateway. |

## Data ownership

The service owns its AI agents, Claude Code jobs, and inference-log records. Consumer services own their prompts, business workflows, and commerce-domain data.

## Authentication and authorization

Machine-accessible routes accept only Auth-issued, pair-specific RS256 bearer JWTs. They validate through Auth or an approved local RS256 verifier, create a separate service actor, declare target-scoped roles per route, and deny and error-log undecorated routes. Auth alone signs and re-mints credentials; delivery is Vault -> ExternalSecret -> Kubernetes Secret -> secretKeyRef. See auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md.

## Synchronous dependencies

PostgreSQL is required for persistence. Logging requests have a three-second timeout and do not crash the service. Voice transcription retrieves from configured object storage. `LITELLM_BASE_URL` enables the LiteLLM Docker proxy, which routes to configured OpenRouter and Ollama backends; Ollama is `ai-microservice-ollama-green:11435`, LiteLLM is `ai-microservice-litellm-green:4000`, both on `nginx-network` and neither is a K8s sidecar.

## Asynchronous dependencies

Claude Code execution uses RabbitMQ publish/subscribe. Jobs are persisted before publication, and the consumer re-publishes retryable work after backoff when RabbitMQ recovers. This service has no documented commerce-domain event contract.

## Degraded operation

A logging outage is best-effort and does not fail requests. RabbitMQ publication failure leaves the persisted job available for recovery. Object-storage and inference-provider failures are returned by their feature paths. If docs-RAG is unavailable or unconfident, Git source remains authoritative.

## Validation

Validate health with `GET /health`; validate TypeORM, service-identity, Claude Code, voice, and task behavior with maintained tests; validate adoption decisions with the IPS profile validator.
