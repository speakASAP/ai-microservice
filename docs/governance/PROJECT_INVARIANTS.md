# Project Invariants

These invariants are non-negotiable unless a future user instruction explicitly changes them.

## INV-01 Preserve Runtime Contracts

Existing production endpoints must remain backward compatible:

- `GET /health`
- `POST /ai/complete`
- `POST /ai/claude-code-execute`
- documented shop-assistant, email-triage, translation, and status endpoints.

## INV-02 Preserve Intent

Implementation work must carry user intent through planning, enqueueing, execution, status, validation, and review. Use `intent` and `intentChecksum` as continuity markers.

## INV-03 Auth Stays Explicit

Admin and implementation APIs must remain protected by the existing JWT bearer auth unless a selected goal implements a replacement auth model.

## INV-04 Model Routing Source Of Truth

Model-tier routing changes must be made in `litellm_config.yaml` first and then reflected in docs.

## INV-05 Premium Requires Approval

Premium model use, including Claude Sonnet class routing, requires human approval per call unless a future governance goal changes this rule.

## INV-06 Secrets Stay Out Of Artifacts

Do not commit secrets, raw tokens, production credentials, or unredacted sensitive logs in prompts, reports, screenshots, or validation output.

## INV-07 Deployment Is A Goal

Production deployment must be part of the selected goal scope or explicitly requested by the user. Otherwise, stop after local validation and documented next action.

## INV-08 Dirty Worktree Safety

Treat unrelated dirty or untracked files as user/project state. Do not revert or delete them unless explicitly requested.
