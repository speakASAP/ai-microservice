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
| INV-02 | security | `SYSTEM.md` | Never rotate a signing key without re-minting tokens signed by it. | Dependent services fail authentication after a key rotation. | Run `scripts/verify-service-tokens.sh` after token minting. | deployment |
| INV-03 | security | `SYSTEM.md` | RS256 signing and verification must pin token-header `alg`. | Algorithm-confusion verification or caller impersonation. | Service-identity tests and security review. | pre-coding |
| INV-04 | operational | `SYSTEM.md` | Preserve published health and AI API compatibility. | Consumer outage from a breaking endpoint change. | Relevant tests and health verification. | validation |
| INV-05 | governance | `BUSINESS.md` | Premium inference requires explicit human approval per invocation. | Unattended premium spend. | Request-contract review. | pre-coding |

## Exceptions

No approved exception is recorded.

## Review cadence

Review with every architecture, authentication, routing, or consumer-contract change.
