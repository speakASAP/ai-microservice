# Vision: ai-microservice

```yaml
id: VISION-ai-microservice
status: approved
owner: project owner
created: 2026-08-30
last_updated: 2026-08-30
completeness_level: complete
upstream:
  - ../00_constitution/CONSTITUTION.md
downstream:
  - ../../BUSINESS.md
  - ../17_governance/PROJECT_INVARIANTS.md
  - ../22_goal_impact/GOAL-IMPACT-TASK-001.md
```

## One-sentence vision

Every Statex service can use governed AI capabilities through one reliable inference gateway.

## Problem statement

Independent provider calls fragment routing, cost attribution, security controls, and premium-approval enforcement.

## Target users

Statex consumer services, their operators, and the project owner use or govern the gateway.

## Core user need

Consumers need a stable API for appropriately routed AI capabilities without direct external-provider management.

## Key outcomes

Published tier routing, inference attribution by service and `business_id`, and preserved intent for implementation jobs.

## Non-goals

Unattended premium inference, commerce-domain ownership, and treating Docker-only Ollama or LiteLLM as K8s sidecars.

## Success criteria

`GET /health` is healthy, `POST /ai/complete` remains the documented tier interface, and consumers use this gateway rather than direct providers.

## Approval

Status: approved
Approved by: project owner
Approval evidence: owner-confirmation: ai-microservice-onboarding-approved
