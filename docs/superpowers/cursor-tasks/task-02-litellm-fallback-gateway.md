# Task 02 — LiteLLM fallback gateway for ai-microservice

## Status (implemented)

Delivered as the **unified LLM gateway**: LiteLLM sidecar + **compose `ollama`** (image `services/ollama/Dockerfile`) + orchestrator **`/ai/complete`** + **free-ai `/analyze`** when `LITELLM_*` is set. Staged plan: [`../plans/2026-04-12-unified-llm-gateway-stages.md`](../plans/2026-04-12-unified-llm-gateway-stages.md). Task index: [`../LLM_UNIFIED_GATEWAY_TASK_INDEX.md`](../LLM_UNIFIED_GATEWAY_TASK_INDEX.md).

**Do not duplicate YAML here** — the source of truth is repo-root **`litellm_config.yaml`**. **`AGENTS.md`** summarizes tier → model and fallback intent.

---

## Goal (original)

Self-hosted **LiteLLM** proxy between ai-microservice and providers, with automatic fallbacks (Ollama, OpenRouter, Gemini) and no caller code changes for tiers `free` \| `cheap` \| `smart`.

---

## Inputs (read before coding)

- [`SYSTEM.md`](../../../SYSTEM.md) — integrations, Ollama/LiteLLM compose
- [`BUSINESS.md`](../../../BUSINESS.md) — constraints (LiteLLM tier routing, no direct upstream calls from other services)
- [`AGENTS.md`](../../../AGENTS.md) — tier → model (sync with `litellm_config.yaml`)
- [`services/ai-orchestrator/app/main.py`](../../../services/ai-orchestrator/app/main.py) — `LITELLM_BASE_URL` + legacy OpenRouter
- [`services/free-ai-service/app/main.py`](../../../services/free-ai-service/app/main.py) — LiteLLM path when `LITELLM_*` set
- [`.env.example`](../../../.env.example)

---

## Scope (where it lives)

| Item | Location |
|------|-----------|
| LiteLLM + Ollama services | `docker-compose.yml`, `docker-compose.blue.yml`, `docker-compose.green.yml` — **no** published `4000` on prod (internal only); `litellm` `depends_on` `ollama` |
| Router models + fallbacks | `litellm_config.yaml` — `os.environ/OLLAMA_API_BASE`, keys via env |
| Ollama image | `services/ollama/Dockerfile` — `OLLAMA_HOST=0.0.0.0:11434` |
| Orchestrator LLM | `services/ai-orchestrator/app/main.py` — `_litellm_chat_completions_url()`, bearer `LITELLM_MASTER_KEY` |
| Free AI LLM | `services/free-ai-service/app/main.py` — `_litellm_configured()`, `analyze_with_litellm` |
| Smoke / validate | `scripts/smoke-unified-llm.sh`, `scripts/validate-llm-gateway-tasks.sh`, [`../LLM_GATEWAY_SETUP.md`](../LLM_GATEWAY_SETUP.md) |
| HTTP contract | [`../../model-tier-endpoints.md`](../../model-tier-endpoints.md) |

---

## Do (policy)

- LiteLLM is **additive** on the orchestrator: unset `LITELLM_BASE_URL` → legacy OpenRouter chain remains.
- Secrets only via **env** in `litellm_config.yaml` (`os.environ/VAR`); never commit keys.
- **`LITELLM_MASTER_KEY`**: shared secret for proxy + callers (orchestrator, free-ai).
- **Premium** stays out of LiteLLM config — human approval per ecosystem rules.

## Do Not

- Do not expose LiteLLM **:4000** on the public internet without a deliberate decision.
- Do not add paid providers to the fallback chain without approval.

---

## Verify (from `ai-microservice/` repo root)

```bash
test -f litellm_config.yaml
grep -q "litellm:" docker-compose.yml
grep -q "LITELLM" .env.example
grep -q "LITELLM_BASE_URL" services/ai-orchestrator/app/main.py
grep -q "_litellm_configured" services/free-ai-service/app/main.py
./scripts/validate-llm-gateway-tasks.sh
./scripts/smoke-unified-llm.sh
python3 scripts/test-ai-services.py
```

After deploy, pull Ollama weights referenced in `litellm_config.yaml` into the **named volume** (see comments in that file).
