# Unified LLM gateway (LiteLLM + Docker Ollama + free-ai) — staged plan

**Canonical narrative:** Cursor workspace plan `ollama_multi-source_ai_0b086a32.plan.md` (under `.cursor/plans/` on the machine where the plan was authored). This document **splits execution into ordered stages** so nothing is dropped between sessions.

**Task index and validators:** [`../LLM_UNIFIED_GATEWAY_TASK_INDEX.md`](../LLM_UNIFIED_GATEWAY_TASK_INDEX.md)

**Lead agent prompt:** [`../../agents/master-prompt-llm-gateway.md`](../../agents/master-prompt-llm-gateway.md)

---

## Stage 0 — Task artifacts and scripts (no runtime change)

**Goal:** Frozen checklist, index rows, and smoke/validate scripts exist before infra edits.

**Deliverables:**

- [`LLM_UNIFIED_GATEWAY_TASK_INDEX.md`](../LLM_UNIFIED_GATEWAY_TASK_INDEX.md) populated with task IDs **T-UG-00 … T-UG-07** and validation procedure **V-UG-00**.
- [`LLM_GATEWAY_SETUP.md`](../LLM_GATEWAY_SETUP.md) contains **copy-paste** `validate` + `smoke` shell scripts and **markdown-only** grep validation (§3).
- [`TASKS.md`](../../../TASKS.md) links to this plan and the index.
- **Optional (outside plan mode):** copy §1–§2 from `LLM_GATEWAY_SETUP.md` into `scripts/*.sh`, `chmod +x`, then run `./scripts/validate-llm-gateway-tasks.sh`.

**Stage validation (must pass before Stage 1):**

1. Run **§3** in [`LLM_GATEWAY_SETUP.md`](../LLM_GATEWAY_SETUP.md).
2. If scripts are on disk: `cd ai-microservice && ./scripts/validate-llm-gateway-tasks.sh`.

**Exit criteria:** §3 succeeds; if scripts installed, validate script prints `OK: LLM gateway task artifacts validated` and exits 0.

---

## Stage 1 — Docker Ollama service

**Goal:** `ollama/ollama` runs on `nginx-network`; LiteLLM reaches it via `OLLAMA_API_BASE` (internal URL).

**Deliverables:**

- `ollama` service in `docker-compose.yml`, `docker-compose.blue.yml`, `docker-compose.green.yml` (volume, no published port to internet unless you explicitly need host debug).
- `litellm` `depends_on` / ordering; `OLLAMA_API_BASE` default for litellm points at Docker Ollama DNS name.
- [`litellm_config.yaml`](../../../litellm_config.yaml) Ollama `api_base` remains `os.environ/OLLAMA_API_BASE`.

**Stage validation:**

- After `docker compose … up -d ollama litellm`: `docker exec … curl -sS "${OLLAMA_API_BASE}/api/tags"` from litellm container.
- LiteLLM `GET /health` with master key: Ollama routes not connection-refused.

**Exit criteria:** T-UG-01 + T-UG-02 rows in task index marked done with validator **V-UG-01** PASS.

---

## Stage 2 — LiteLLM config and fallbacks (optional breadth)

**Goal:** Router fallbacks and model IDs match pulled models; extra free providers only if keys exist.

**Deliverables:**

- Updates to `litellm_config.yaml` and `.env.example` as needed.
- Document model pull commands in README or stage doc.

**Stage validation:** `smoke-unified-llm.sh` **tier** section or manual `/v1/models` + one chat per tier.

**Exit criteria:** T-UG-03 PASS.

---

## Stage 3 — free-ai-service → LiteLLM adapter

**Goal:** `POST /analyze` unchanged for callers; implementation calls `POST {LITELLM_BASE_URL}/v1/chat/completions` with mapped `model` tier.

**Deliverables:**

- `free-ai-service` compose env: `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY` (same pattern as backend).
- Code change in `services/free-ai-service/app/main.py`; remove duplicate OpenRouter/Ollama paths after verification (or single-flag rollback window).

**Stage validation:**

- Orchestrator health path that hits free-ai; `test-ai-services.py` analyze path; email-triage classify/decide smoke if LLM enabled.

**Exit criteria:** T-UG-04 + T-UG-05 PASS (**V-UG-04**).

---

## Stage 4 — Docs and HTTP contract

**Goal:** README + `model-tier-endpoints.md` + `AGENTS.md`/`SYSTEM.md` match production (LiteLLM vs legacy).

**Deliverables:**

- [`docs/model-tier-endpoints.md`](../../model-tier-endpoints.md) two-mode contract.
- README gateway + ports (338x vs 4000).

**Stage validation:** Human read + link check; `validate-llm-gateway-tasks.sh` still passes if index references updated paths.

**Exit criteria:** T-UG-06 PASS.

---

## Stage 5 — Logging, deploy order, full smoke

**Goal:** Structured logs (tier, transport, duration, mapped model); deploy order documented; full smoke green.

**Deliverables:**

- Orchestrator + free-ai logging fields per logging-microservice norms.
- `deploy.sh` / README: recreate order `ollama` → `litellm` → `free-ai-service` → `backend`.

**Stage validation:**

```bash
cd ai-microservice && ./scripts/smoke-unified-llm.sh
python3 scripts/test-ai-services.py
```

**Exit criteria:** T-UG-07 PASS; smoke + test script exit 0 against running stack.

---

## Smoke order (after Stage 0 complete)

1. [`LLM_GATEWAY_SETUP.md`](../LLM_GATEWAY_SETUP.md) §3 (markdown-only) and optionally installed `validate` / `smoke` scripts.
2. Bring stack up (your compose project).
3. §4 curl smoke or `./scripts/smoke-unified-llm.sh` if installed.
4. `python3 scripts/test-ai-services.py` (see [README](../../README.md)).

If Stages 1–3 are **not** implemented yet, expect smoke steps that require Ollama/LiteLLM/free-ai to **skip or fail** with clear messages — that is expected until those stages land.

---

## Continue (handoff)

**Closed 2026-04-12.** Stages 0–5 are implemented; **T-UG-00 … T-UG-07** and **V-UG-** checklists are marked PASS in [`LLM_UNIFIED_GATEWAY_TASK_INDEX.md`](../LLM_UNIFIED_GATEWAY_TASK_INDEX.md). Lead status: [`../../agents/master-prompt-llm-gateway.md`](../../agents/master-prompt-llm-gateway.md). Product follow-up: **`TASKS.md`** (e.g. cost tracking per `business_id`).

If you **re-open** the program (e.g. new LiteLLM routes or blue stack parity):

1. Follow [`LLM_GATEWAY_SETUP.md`](../LLM_GATEWAY_SETUP.md) §3 + `./scripts/validate-llm-gateway-tasks.sh`.
2. Run `./scripts/smoke-unified-llm.sh` and `python3 scripts/test-ai-services.py` after compose or image changes.
3. Optional canonical narrative: workspace plan `ollama_multi-source_ai_0b086a32.plan.md` under `.cursor/plans/` (path may differ per clone).
