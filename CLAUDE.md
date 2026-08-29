# Claude Instructions

Shared rules live here:

- Claude profile: `/home/ssf/.claude/CLAUDE.md`
- Shared ecosystem instructions: `/home/ssf/Documents/Github/CLAUDE.md`
- Codex profile: `/home/ssf/.codex/AGENTS.md`
- Cross-agent standard: `/home/ssf/.ai-agent-standards/CROSS_AGENT_AUTOMATION_STANDARD.md`
- Repository operations: `AGENT_OPERATIONS.md`

Read those first, then follow the repository-specific notes below and the current planning/status files.


## Repository-Specific Notes

# CLAUDE.md (ai-microservice)

→ Ecosystem: [../shared/CLAUDE.md](../shared/CLAUDE.md) | Reading order: `BUSINESS.md` → `SYSTEM.md` → `AGENTS.md` → `TASKS.md` → `STATE.json`

---

## Knowledge Retrieval

Use `docs-rag-microservice` for bounded discovery when it is healthy, then
verify deployment, security, database, integration and public-contract facts
against the cited Git source. Git remains authoritative.

Authority and fallback rules:
`/home/ssf/Documents/Github/shared/docs/DOCUMENTATION_AUTHORITY.md`.

Do not generate tokens in documentation or assume an unconfident/failed RAG
response means that source documentation does not exist.

## ai-microservice

**Purpose**: Centralized AI inference gateway — all LLM calls route through here; no service calls providers directly.  
**Port**: 3380 | **Domain**: <https://ai.alfares.cz>  
**Stack**: NestJS · LiteLLM sidecar · Ollama · OpenRouter · Gemini  
**Model tiers**: `free` / `cheap` / `smart` / `premium` — full config in `AGENTS.md`  
**Consumers**: runlayer, statex, shop-assistant, crypto-ai-agent, agentic-email — see `BUSINESS.md`  
**Secrets**: Vault via ESO (`secret/prod/ai-microservice`); local dev: `./scripts/vault-env-gen.sh`

**Ops**: `curl http://ai-microservice:3380/health` · `kubectl logs -n statex-apps -l app=ai-microservice -f` · `./scripts/deploy.sh`
