Service-to-service authentication follows the [canonical service identity standard](../../../auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md).

# Project Invariants

```yaml
id: PROJECT-INVARIANTS-ai-microservice
status: validated
owner: project owner
created: 2026-08-30
last_updated: 2026-08-30
completeness_level: validated
upstream:
  - ../00_constitution/CONSTITUTION.md
  - ../01_vision/VISION.md
```

## Purpose

Make safety-critical gateway boundaries enforceable during planning, implementation, and validation.

## Applicability

These invariants apply to every ai-microservice change, including documentation, runtime code, routing configuration, and operational plans.

## Invariants

| ID | Level | Source | Rule | Forbidden outcome | Validation method | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| INV-01 | constitutional | `BUSINESS.md` | Other Statex services must not call external LLM providers directly; they use ai-microservice interfaces. | Bypassing centralized routing and governance. | Review contracts and consumer integration changes. | pre-coding |
| INV-02 | security | `SYSTEM.md` | Service-to-service authentication follows the [canonical service identity standard](../../../auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md). | Conflicting service authentication procedures create unsafe authorization boundaries. | Review the canonical standard. | pre-coding |
| INV-03 | security | `SYSTEM.md` | Service-to-service authentication follows the [canonical service identity standard](../../../auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md). | Conflicting service authentication procedures create unsafe authorization boundaries. | Review the canonical standard. | pre-coding |
| INV-04 | operational | `SYSTEM.md` | Preserve published health and AI API compatibility. | Consumer outage from a breaking endpoint change. | Relevant tests and health verification. | validation |
| INV-05 | governance | `BUSINESS.md` | Premium inference requires explicit human approval per invocation. | Unattended premium spend. | Request-contract review. | pre-coding |

## Exceptions

No approved exception is recorded.

## Review cadence

Review with every architecture, authentication, routing, or consumer-contract change.

## Service-to-service authentication

Follow the [canonical service identity standard](../../../auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md).
