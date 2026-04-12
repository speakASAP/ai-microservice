# Model tier HTTP API (ai-microservice)

Central reference for callers that pass **`model_tier`** (`free` \| `cheap` \| `smart`; ecosystem also defines `premium` for policy, not for blind LLM calls).

## Two modes

| Mode | How it is selected | `/ai/complete` behavior | free-ai `/analyze` behavior |
| ---- | -------------------- | ------------------------ | ---------------------------- |
| **LiteLLM (preferred)** | `LITELLM_BASE_URL` and `LITELLM_MASTER_KEY` set on the **orchestrator** and matching proxy reachable | `POST {LITELLM_BASE_URL}/v1/chat/completions` with `model` = tier name (`free`, `cheap`, `smart`). Routing and fallbacks are defined in `litellm_config.yaml` (Ollama + OpenRouter + Gemini). | **Only** LiteLLM: `POST {LITELLM_BASE_URL}/v1/chat/completions`. Tier from `analysis_type` (`email_*` → `cheap`, else `free`); optional request `model` overrides if it is a valid tier name. |
| **Legacy** | `LITELLM_BASE_URL` empty or master key missing (orchestrator); **free-ai** without both LiteLLM env vars | OpenRouter free-model fallback chain in `services/ai-orchestrator/app/main.py` (`FREE_MODEL_FALLBACKS` / `OPENROUTER_MODEL`). | Direct OpenRouter and optional Ollama/Hugging Face paths in `services/free-ai-service/app/main.py`. |

Ports: orchestrator and sibling services use **338x** on the host; LiteLLM is **4000 inside the stack** (compose service `litellm`, not published by default).

## Where `model_tier` is accepted

| Service | Path | Notes |
|---------|------|-------|
| **AI Orchestrator** | `POST /ai/complete` | JSON body includes `model_tier`; used when LiteLLM mode is active (see table above). |

Other in-repo agents (free-ai-service `/analyze`, email-triage, shop-assistant, translation, and so on) **do not** take `model_tier`; they use their own fields (`analysis_type`, optional `model`, and so on).

## Intended tier → capability (ecosystem)

See `AGENTS.md` in this repo for the **target** mapping used across Statex services.

## `POST /ai/complete` (AI Orchestrator)

- **Default URL (Docker):** `http://ai-microservice:3380/ai/complete` (port from `AI_ORCHESTRATOR_PORT`).
- **Production (typical):** `https://<DOMAIN>/ai/complete` when exposed behind nginx for `DOMAIN` (for example `ai.statex.cz`).

### Authentication

**Required:** `Authorization: Bearer <JWT>`.

Token must be valid for the same `JWT_SECRET` as auth-microservice, and the payload must include one of:

- `global:superadmin`
- `internal:ai-microservice:admin`

Paths not listed as public in `shared/auth.py` require this header; `/ai/complete` is protected.

### Request body (JSON)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_tier` | string | no | `free` | `free`, `cheap`, or `smart`. In **LiteLLM mode**, selects the proxy model name. In **legacy mode**, accepted for compatibility but OpenRouter fallback chain does not map tiers to different providers. |
| `system_prompt` | string | yes | — | Sent as a `system` message when non-empty. |
| `user_prompt` | string | yes | — | Sent as the `user` message. |
| `output_schema` | object | no | `null` | Caller metadata / cache keys; **not** sent as `response_format`. Enforce JSON in prompts. |
| `max_tokens` | integer | no | `1000` | Passed to LiteLLM or OpenRouter. |
| `correlation_id` | string | no | `null` | Tracing id; logged on orchestrator. |

### Success responses

1. **JSON object** — If the model’s reply (after stripping optional markdown fences) parses as JSON, that **parsed object is returned as the top-level JSON body** (not wrapped in `{ "data": ... }`).
2. **Non-JSON text** — `{ "text": "<raw model output>", "model_used": "<id from provider>" }`.

### Error responses (selection)

| HTTP | Meaning |
|------|---------|
| `401` | Missing/invalid JWT or insufficient roles. |
| `502` | LiteLLM or OpenRouter returned an error status not handled by fallback. |
| `503` | Missing API keys / master key, empty model output, timeouts, or all legacy models failed. |

### Example: minimal JSON-oriented call (`model_tier: free`)

```bash
JWT="<paste JWT with internal:ai-microservice:admin or global:superadmin>"
BASE="http://localhost:3380"

curl -sS -X POST "${BASE}/ai/complete" \
  -H "Authorization: Bearer ${JWT}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tier": "free",
    "system_prompt": "Reply only with valid JSON, no markdown.",
    "user_prompt": "Return {\"greeting\": \"hello\", \"language\": \"en\"}",
    "max_tokens": 128
  }'
```

### Example: `cheap` tier (same contract; tier selects LiteLLM route when configured)

```bash
curl -sS -X POST "${BASE}/ai/complete" \
  -H "Authorization: Bearer ${JWT}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tier": "cheap",
    "system_prompt": "You summarize briefly.",
    "user_prompt": "One sentence on why idempotency keys matter in payment APIs.",
    "max_tokens": 200,
    "correlation_id": "bo-task-88421"
  }'
```

### Example: `smart` tier with optional `output_schema` (caller metadata only)

```bash
curl -sS -X POST "${BASE}/ai/complete" \
  -H "Authorization: Bearer ${JWT}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tier": "smart",
    "system_prompt": "You are a precise assistant.",
    "user_prompt": "List two risks of storing cards in plain Redis; JSON array of strings.",
    "output_schema": {"type": "array", "items": {"type": "string"}},
    "max_tokens": 300
  }'
```

### Example: `401` without token

```bash
curl -sS -o /dev/stderr -w "%{http_code}" -X POST "${BASE}/ai/complete" \
  -H "Content-Type: application/json" \
  -d '{"model_tier":"free","system_prompt":"x","user_prompt":"y"}'
# expect 401 and JSON body {"detail":"..."}
```

## Related reading

- `AGENTS.md` — tier → model names (ecosystem contract).
- `docs/superpowers/plans/2026-04-12-unified-llm-gateway-stages.md` — staged rollout and smoke order.
- `litellm_config.yaml` — proxy routes and fallbacks.
- `business-orchestrator/src/worker/ai-http.client.ts` — production client: `POST .../ai/complete` with `model_tier` in body and cache key.
