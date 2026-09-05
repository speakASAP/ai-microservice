# Agents: ai-microservice

## Required reading

Read `BUSINESS.md`, `SYSTEM.md`, `AGENT_OPERATIONS.md`, `TASKS.md`, `STATE.json`, canonical IPS artifacts, and applicable implementation-goal records before work.

## Authority

Git source and approved project contracts are authoritative. docs-RAG may support discovery, but Git verification is required. Agents cannot alter protected intent or create their own deployment authority.

## Intent preservation system

Preserve the chain Vision -> Goal Impact -> System -> Feature -> Task -> Execution Plan -> Coding Prompt -> Code -> Validation. Capture `intent` before implementation jobs and retain `intentChecksum` across follow-up work.

## Safety and operations

Work in the authoritative remote repository. Do not print secrets, tokens, raw production data, or private evidence. Do not treat unconfident retrieval as absent documentation. Preserve the RS256 boundary and never rotate a signing key without re-minting dependent tokens.

## Project-specific rules

`litellm_config.yaml` is the model-routing source of truth. `free`, `cheap`, and `smart` are tier IDs; premium requires per-call human approval. `ai-microservice-ollama-green` on port 11435 and `ai-microservice-litellm-green` on port 4000 are Docker dependencies, not K8s sidecars. Implementation providers are `claude-code` by default and `codex` when selected through `/ai/claude-code-execute`.

## Required final report

Report changed files, validation evidence, validation debt, blockers, scope deviations, and the concrete next action.

## Service-to-service authentication
For machine service identity, follow the sole canonical [`SERVICE_IDENTITY_CONSUMER_STANDARD.md`](../auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md). It is not reproduced here.
