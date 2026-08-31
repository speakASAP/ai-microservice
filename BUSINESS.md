# Business: ai-microservice

> **IMMUTABLE BY AI** - do not change the approved business baseline without product-owner approval.

```yaml
id: BUSINESS-ai-microservice
status: approved
owner: project owner
created: 2026-06-13
last_updated: 2026-08-30
completeness_level: complete
upstream:
  - docs/00_constitution/CONSTITUTION.md
  - docs/01_vision/VISION.md
downstream:
  - SYSTEM.md
  - docs/22_goal_impact/GOAL-IMPACT-TASK-001.md
  - docs/17_governance/PROJECT_INVARIANTS.md
```

## Problem

Statex services need one governed path for AI inference rather than separate, unmanaged external-provider integrations.

## Target users and stakeholders

Consumer services are runlayer, statex, shop-assistant, crypto-ai-agent, and agentic-email. Their operators and the project owner are stakeholders in reliable routing, controlled costs, and human approval of premium use.

## Value proposition

The service centralizes model-tier routing and exposes NLP, ASR, Document AI, prototype generation, and implementation-job execution behind a single gateway.

## Goals

- Route LLM requests by the `free`, `cheap`, and `smart` tiers through the configured provider chain.
- Track inference API costs by calling service and `business_id` in inference logs.
- Preserve caller intent through implementation jobs using `intent`, `intentChecksum`, and `implementationProvider`.

## Non-goals

- It is not a general catalog, orders, payments, warehouse, or invoice-domain service.
- Premium tier inference is not an unattended route; each invocation requires explicit human approval.
- Consumer services must not directly integrate with external LLM providers.

## Success metrics

- The published gateway remains reachable on port 3380 at `https://ai.alfares.cz`.
- Inference logs retain the calling service and optional `business_id` for cost tracking.
- `POST /ai/complete` continues to route published model tiers through its configured provider path.

## Business constraints

- Other Statex services call this service rather than external LLM providers directly.
- Tier routes are configured in `litellm_config.yaml`; LiteLLM is used when enabled.
- Premium use requires explicit human approval per invocation.
- API costs are tracked per service and `business_id` under GOAL-03-cost-tracking.

## Approval

Status: approved
Approved by: project owner
Approval evidence: owner-confirmation: ai-microservice-onboarding-approved
