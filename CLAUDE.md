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

## Knowledge Retrieval — docs-rag-microservice (MANDATORY, query before reading files)

**Query the RAG before reading source files** — saves 2000-5000 tokens per answer.

```bash
kubectl -n statex-apps exec deployment/ai-microservice -- curl -s -X POST http://docs-rag-microservice:3397/retrieval/agent-context \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat ~/.claude/rag-token)" \
  -d '{"query": "YOUR QUESTION HERE", "maxTokens": 3000}'
```

---

## ai-microservice

**Purpose**: Centralized AI inference gateway — all LLM calls route through here; no service calls providers directly.  
**Port**: 3380 | **Domain**: <https://ai.alfares.cz>  
**Stack**: NestJS · LiteLLM sidecar · Ollama · OpenRouter · Gemini  
**Model tiers**: `free` / `cheap` / `smart` / `premium` — full config in `AGENTS.md`  
**Consumers**: runlayer, statex, shop-assistant, crypto-ai-agent, agentic-email — see `BUSINESS.md`  
**Secrets**: Vault via ESO (`secret/prod/ai-microservice`); local dev: `./scripts/vault-env-gen.sh`

**Ops**: `curl http://ai-microservice:3380/health` · `kubectl logs -n statex-apps -l app=ai-microservice -f` · `./scripts/deploy.sh`
