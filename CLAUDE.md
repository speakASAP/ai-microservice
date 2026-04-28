# CLAUDE.md (ai-microservice)

Ecosystem defaults: sibling [`../CLAUDE.md`](../CLAUDE.md) and [`../shared/docs/PROJECT_AGENT_DOCS_STANDARD.md`](../shared/docs/PROJECT_AGENT_DOCS_STANDARD.md).

Read this repo's `BUSINESS.md` → `SYSTEM.md` → `AGENTS.md` → `TASKS.md` → `STATE.json` first.

---

## ai-microservice

**Purpose**: Centralized AI inference gateway — all LLM calls route through here; no service calls providers directly.  
**Port**: 3380 | **Domain**: <https://ai.alfares.cz>  
**Stack**: NestJS · LiteLLM sidecar · Ollama · OpenRouter · Gemini  
**Model tiers**: `free` / `cheap` / `smart` / `premium` — full config in `AGENTS.md`  
**Consumers**: business-orchestrator, statex, shop-assistant, crypto-ai-agent, agentic-email — see `BUSINESS.md`  
**Secrets**: Vault via ESO (`secret/prod/ai-microservice`); local dev: `./scripts/vault-env-gen.sh`

### Quick ops

```bash
curl http://ai-microservice:3380/health
./scripts/orch-test-ai.sh free
docker compose logs -f
./scripts/deploy.sh
```
