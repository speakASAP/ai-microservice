# Model tier HTTP API (ai-microservice)

Central reference for callers that pass **`model_tier`** (`free` \| `cheap` \| `smart`; ecosystem also defines `premium` for policy, not for blind LLM calls).

## Where `model_tier` is accepted

| Service | Path | Notes |
|---------|------|-------|
| **AI Orchestrator** | `POST /ai/complete` | Only HTTP route in this repo whose JSON body includes `model_tier`. |

Other in-repo agents (free-ai-service `/analyze`, email-triage, shop-assistant, translation, and so on) **do not** take `model_tier`; they use their own fields (`analysis_type`, optional `model`, and so on).

## Intended tier → capability (ecosystem)

See `AGENTS.md` in this repo for the **target** mapping used across Statex services. The orchestrator’s `/ai/complete` handler today routes all successful calls through **OpenRouter** using the shared free-model fallback chain (`OPENROUTER_MODEL` plus `FREE_MODEL_FALLBACKS` in `services/ai-orchestrator/app/main.py`). The **`model_tier` field is accepted on the wire** (for example `business-orchestrator` includes it in the body and in Redis cache keys) but **does not yet change** which provider or model list is used.

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
| `model_tier` | string | no | `free` | Callers use `free`, `cheap`, or `smart` (retry escalation in business-orchestrator). Accepted for compatibility; selection logic is unchanged today. |
| `system_prompt` | string | yes | — | Sent as a `system` message when non-empty. |
| `user_prompt` | string | yes | — | Sent as the `user` message. |
| `output_schema` | object | no | `null` | Carried for callers/cache keys; **not** sent to OpenRouter as `response_format`. Enforce JSON in prompts. |
| `max_tokens` | integer | no | `1000` | Passed to OpenRouter. |
| `correlation_id` | string | no | `null` | Optional tracing id (ignored by OpenRouter path). |

### Success responses

1. **JSON object** — If the model’s reply (after stripping optional markdown fences) parses as JSON, that **parsed object is returned as the top-level JSON body** (not wrapped in `{ "data": ... }`).
2. **Non-JSON text** — `{ "text": "<raw model output>", "model_used": "<id from provider>" }`.

### Error responses (selection)

| HTTP | Meaning |
|------|---------|
| `401` | Missing/invalid JWT or insufficient roles. |
| `502` | OpenRouter returned an error status that was not handled by model fallback. |
| `503` | `OPENROUTER_API_KEY` unset, or all candidate models failed (rate limit, empty content, and so on). |

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

### Example: `cheap` tier (same contract; tier for client policy / future routing)

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
- `docs/superpowers/cursor-tasks/task-bo-01-ai-complete-endpoint.md` — history of `/ai/complete`.
- `business-orchestrator/src/worker/ai-http.client.ts` — production client: `POST .../ai/complete` with `model_tier` in body and cache key.
