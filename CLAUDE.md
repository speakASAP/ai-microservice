# CLAUDE.md (ai-microservice)

→ Ecosystem: [../shared/CLAUDE.md](../shared/CLAUDE.md) | Reading order: `BUSINESS.md` → `SYSTEM.md` → `AGENTS.md` → `TASKS.md` → `STATE.json`

---

## ai-microservice

**Purpose**: Centralized AI inference gateway — all LLM calls route through here; no service calls providers directly.  
**Port**: 3380 | **Domain**: <https://ai.alfares.cz>  
**Stack**: NestJS · LiteLLM sidecar · Ollama · OpenRouter · Gemini  
**Model tiers**: `free` / `cheap` / `smart` / `premium` — full config in `AGENTS.md`  
**Consumers**: business-orchestrator, statex, shop-assistant, crypto-ai-agent, agentic-email — see `BUSINESS.md`  
**Secrets**: Vault via ESO (`secret/prod/ai-microservice`); local dev: `./scripts/vault-env-gen.sh`

**Ops**: `curl http://ai-microservice:3380/health` · `kubectl logs -n statex-apps -l app=ai-microservice -f` · `./scripts/deploy.sh`
