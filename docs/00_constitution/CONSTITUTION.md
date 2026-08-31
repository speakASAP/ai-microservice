# Project Constitution: ai-microservice

```yaml
id: CONSTITUTION-ai-microservice
status: approved
owner: project owner
created: 2026-08-30
last_updated: 2026-08-30
completeness_level: complete
upstream: []
downstream:
  - ../01_vision/VISION.md
  - ../17_governance/PROJECT_INVARIANTS.md
```

## Purpose

Protect ai-microservice as the centralized, governed AI inference gateway for Statex services.

## Constitutional principles

### Intent preservation

Implementation work preserves caller intent through planning, execution, validation, and review.

### Human-controlled change

Premium inference requires explicit human approval for every invocation, and business-baseline changes require project-owner approval.

### Scope boundaries

Statex services use this gateway rather than direct external LLM providers. This service does not own commerce-domain workflows.

### Data and security

The RS256 service-authentication boundary and secret-handling rules in `SYSTEM.md` are non-negotiable.

### Validation

No task is complete without acceptance, invariant, and upstream-goal evidence.

## Amendment process

1. Create an amendment proposal under `docs/17_governance/amendments/`.
2. Explain its rationale, impact, and affected artifacts.
3. Obtain project-owner approval.
4. Update dependent records and rerun validation.

## Approval

Status: approved
Approved by: project owner
Approval evidence: owner-confirmation: ai-microservice-onboarding-approved
