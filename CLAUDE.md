# CLAUDE.md (ai-microservice)

Ecosystem defaults: sibling [`../CLAUDE.md`](../CLAUDE.md) and [`../shared/docs/PROJECT_AGENT_DOCS_STANDARD.md`](../shared/docs/PROJECT_AGENT_DOCS_STANDARD.md).

Read this repo's `BUSINESS.md` → `SYSTEM.md` → `AGENTS.md` → `TASKS.md` → `STATE.json` first.

---

## ai-microservice

**Purpose**: Centralized AI inference gateway — all LLM calls in the ecosystem route through here; no service calls external LLM providers directly.  
**Port**: 3380  
**Domain**: https://ai.statex.cz  
**Stack**: NestJS · LiteLLM sidecar · Ollama (local) · OpenRouter · Gemini

### Model tiers (LiteLLM routes)
| Tier | Use | Notes |
|------|-----|-------|
| `free` | Bulk, low-stakes | Ollama / Docker |
| `cheap` | Standard tasks | OpenRouter cheap models |
| `smart` | Complex reasoning | OpenRouter smart models |
| `premium` | Critical/sensitive | Explicit human approval required per invocation |

### Key constraints
- Other Statex services must call this service — never direct LLM provider calls
- `premium` tier requires human approval — not for unattended/automated use
- Track API costs per service/business_id in inference logs
- Model config lives in `litellm_config.yaml` — secrets in `.env` only

### Consumers
business-orchestrator, statex, shop-assistant, crypto-ai-agent, agentic-email.

### Quick ops
```bash
curl http://ai-microservice:3380/health
./scripts/orch-test-ai.sh free   # smoke test from business-orchestrator
docker compose logs -f
./scripts/deploy.sh
```
