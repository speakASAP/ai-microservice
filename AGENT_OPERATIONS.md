# Agent Operations

## Roles

Readiness scanner classifies work; worker agent implements one bounded goal; worker monitor tracks ownership and conflicts; integration validator separates current-task failures from validation debt.

## Before work

Read the required contracts, verify task traceability, identify sensitive-data and contract impacts, name validation commands, and establish scope before editing.

## Parallel work

Do not edit the same public contract, schema, deployment file, generated index, or status artifact in parallel without a documented integration owner and merge order.

## Validation debt

Use `docs/orchestrator/VALIDATION_DEBT.md` only for pre-existing, out-of-scope failures. A failure affecting current files or acceptance criteria remains blocking.

## Handoff

Record the objective, changed files, commands and results, remaining blockers, validation debt, and next responsible owner.

## Project-specific operations

Preserve existing runtime interfaces and the RS256 service-authentication design. Model routing changes begin in `litellm_config.yaml`. Do not operate the Docker-only Ollama or LiteLLM dependencies as part of documentation work.
