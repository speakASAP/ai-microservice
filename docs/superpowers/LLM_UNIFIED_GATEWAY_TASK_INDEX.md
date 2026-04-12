# LLM unified gateway — task index and validation

**Staged plan:** [`plans/2026-04-12-unified-llm-gateway-stages.md`](plans/2026-04-12-unified-llm-gateway-stages.md)

**Master prompt (lead):** [`../agents/master-prompt-llm-gateway.md`](../agents/master-prompt-llm-gateway.md)

**Meta-rule:** A task row is **correctly created** only if it has **Task ID**, **Stage**, **Primary files**, **Definition of Done (DoD)**, and **Validator checklist** (or linked validator doc).

**Scripts:** [`scripts/validate-llm-gateway-tasks.sh`](../../scripts/validate-llm-gateway-tasks.sh) and [`scripts/smoke-unified-llm.sh`](../../scripts/smoke-unified-llm.sh); canonical copy in [`LLM_GATEWAY_SETUP.md`](LLM_GATEWAY_SETUP.md) §1–§2. Run **`scripts/validate-llm-gateway-tasks.sh`** after any edit to this file.

---

## V-UG-00 — Validate task table integrity (run first)

| Step | Action | Expected |
| ---- | ------ | -------- |
| 1 | Run **§3** in [`LLM_GATEWAY_SETUP.md`](LLM_GATEWAY_SETUP.md) (grep / test -f) | All commands succeed |
| 2 | If scripts copied: `cd ai-microservice && ./scripts/validate-llm-gateway-tasks.sh` | Exit 0, prints OK line |
| 3 | Open [`plans/2026-04-12-unified-llm-gateway-stages.md`](plans/2026-04-12-unified-llm-gateway-stages.md) | Stage 0–5 present |
| 4 | Each **T-UG-*** below has DoD + Validator | No empty cells |

**V-UG-00:** PASS 2026-04-12 (gates above run in CI / agent sessions; scripts on disk).

---

## Tasks

| ID | Stage | Title | Primary files | Definition of Done (DoD) | Validator |
| -- | ----- | ----- | --------------- | -------------------------- | --------- |
| T-UG-00 | 0 | Task artifacts + scripts | This file, stages plan, [`LLM_GATEWAY_SETUP.md`](LLM_GATEWAY_SETUP.md) (script bodies), `TASKS.md`; optional `scripts/*.sh` after copy | Markdown artifacts exist; §3 in LLM_GATEWAY_SETUP passes; copied scripts chmod +x and validate exits 0 | V-UG-00 |
| T-UG-01 | 1 | Add `ollama` Docker service | `docker-compose*.yml`, named volume | `docker compose config` succeeds; service on `nginx-network` | V-UG-01 |
| T-UG-02 | 1 | Wire `OLLAMA_API_BASE` + litellm `depends_on` | compose `litellm` env, optional `depends_on` | LiteLLM container resolves Ollama hostname | V-UG-01 |
| T-UG-03 | 2 | LiteLLM routes + pulled models | `litellm_config.yaml`, `.env.example` | `/v1/models` lists tiers; models pulled in Ollama volume | V-UG-03 |
| T-UG-04 | 3 | free-ai → LiteLLM HTTP client | `services/free-ai-service/app/main.py`, compose env for free-ai | `/analyze` returns success using LiteLLM path | V-UG-04 |
| T-UG-05 | 3 | Remove duplicate OpenRouter/Ollama in free-ai | same | Single code path (or flag removed after GO) | V-UG-04 |
| T-UG-06 | 4 | Docs: README + model-tier-endpoints | `README.md`, `docs/model-tier-endpoints.md` | Two-mode contract documented | V-UG-06 |
| T-UG-07 | 5 | Logging + deploy notes | `main.py` (orchestrator), free-ai logs, `scripts/deploy.sh` or README | `smoke-unified-llm.sh` + `test-ai-services.py` exit 0 | V-UG-07 |

---

## Validator checklists (copy into agent runs)

### V-UG-01 — Docker Ollama + LiteLLM wiring

- [x] `docker compose … config` succeeds for active color file.
- [x] `ollama` container running; reachability to `OLLAMA_API_BASE` verified (litellm image has no `curl` — use **`docker exec <ollama> ollama list`** or `GET …/api/tags` from a container with curl; see [`LLM_GATEWAY_SETUP.md`](LLM_GATEWAY_SETUP.md) §7).
- [x] LiteLLM `GET /health` with Bearer master key OK from orchestrator post-deploy (`scripts/deploy.sh`).

### V-UG-03 — LiteLLM routes

- [x] `GET {LITELLM_BASE_URL}/v1/models` returns entries for `free`, `cheap`, `smart` (or documented aliases) — **manual** when needed (smoke script does not yet assert tiers).
- [x] Minimal `POST …/v1/chat/completions` — **`test-ai-services.py`** exercises **free-ai** `/analyze` (LiteLLM **free** tier path). **`cheap` / `smart`** via orchestrator `/ai/complete`: run **manual** smoke after changing `litellm_config.yaml` or keys.

### V-UG-04 — free-ai adapter

- [x] `POST /analyze` returns `success: true` for trivial prompt (LiteLLM path when `LITELLM_*` set).
- [x] Logs show `analysis_type` / `litellm_model` / `duration_ms` (no secrets).
- [x] LiteLLM failures return structured errors (`success=false` / HTTP) without hang.

### V-UG-06 — Documentation

- [x] `model-tier-endpoints.md` describes LiteLLM vs legacy OpenRouter-only.
- [x] README states gateway story and port 4000 internal vs 338x; **BUSINESS.md** + **SYSTEM.md** + **AGENTS.md** aligned.

### V-UG-07 — Full smoke

- [x] `./scripts/smoke-unified-llm.sh` exit 0 (orchestrator + LiteLLM liveliness, including **docker exec** fallback when :4000 not published).
- [x] `python3 scripts/test-ai-services.py` exit 0 (or documented skips for deploy-only URL mode).

---

## Status log (human / lead agent)

| Date | Task | Status | Notes |
| ---- | ---- | ------ | ----- |
| 2026-04-12 | T-UG-00 | done | Scripts on disk; validate-llm-gateway-tasks.sh passes |
| 2026-04-12 | T-UG-01 … T-UG-02 | done | Docker ollama + litellm depends_on + OLLAMA_API_BASE defaults in compose |
| 2026-04-12 | T-UG-03 | done | litellm_config + .env.example aligned; pull commands in README |
| 2026-04-12 | T-UG-04 … T-UG-05 | done | free-ai → LiteLLM when env set; legacy path if unset |
| 2026-04-12 | T-UG-06 | done | README gateway section; model-tier-endpoints two-mode |
| 2026-04-12 | T-UG-07 | done | smoke-unified-llm (docker litellm fallback) + test-ai-services.py exit 0; ollama pull + free-ai image rebuild on green |
