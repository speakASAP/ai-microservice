# Task 02 — LiteLLM fallback gateway for ai-microservice

## Goal

Add **LiteLLM** as a self-hosted fallback/load-balancing proxy between `ai-microservice` and LLM providers. When OpenRouter hits rate limits or is unavailable, the service must transparently fall back to Ollama (local) or other configured free providers without any code change in callers.

LiteLLM is chosen because it is:

- 100 % open-source (MIT), self-hosted, zero recurring cost
- OpenAI-compatible API — drop-in replacement for the current OpenRouter calls
- Supports Ollama, OpenRouter, Groq, Together.ai, Mistral, Vercel AI Gateway, Eden AI, and 100+ others
- Has built-in automatic fallbacks, retries, and load balancing per route
- Can be deployed as a Docker sidecar alongside ai-microservice

---

## Inputs (read before coding)

- `ai-microservice/SYSTEM.md` — current tier routing (`free → cheap → smart`)
- `ai-microservice/BUSINESS.md` — constraints (no direct external LLM calls, tier routing)
- `ai-microservice/AGENTS.md` — current model tier → model mapping
- `ai-microservice/services/ai-orchestrator/app/main.py` — current OpenRouter call logic and `FREE_MODEL_FALLBACKS`
- `ai-microservice/.env.example` — existing env var patterns

---

## Scope

### 1. LiteLLM Docker sidecar

Add a `litellm` service to `ai-microservice/docker-compose.yml` (and `docker-compose.prod.yml` if present):

```yaml
litellm:
  image: ghcr.io/berriai/litellm:main-latest
  ports:
    - "4000:4000"
  volumes:
    - ./litellm_config.yaml:/app/config.yaml
  environment:
    - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}
  command: ["--config", "/app/config.yaml", "--port", "4000"]
  restart: unless-stopped
```

### 2. LiteLLM config file

Create `ai-microservice/litellm_config.yaml`:

```yaml
model_list:
  # free tier — Ollama local (zero cost, always available)
  - model_name: free
    litellm_params:
      model: ollama/gemma2:2b
      api_base: http://host.docker.internal:11434

  # cheap tier — OpenRouter free models with Ollama fallback
  - model_name: cheap
    litellm_params:
      model: openrouter/meta-llama/llama-3.1-8b-instruct:free
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: https://openrouter.ai/api/v1

  - model_name: cheap-fallback
    litellm_params:
      model: ollama/gemma2:2b
      api_base: http://host.docker.internal:11434

  # smart tier — Gemini Flash (free quota) with Ollama fallback
  - model_name: smart
    litellm_params:
      model: gemini/gemini-1.5-flash
      api_key: os.environ/GEMINI_API_KEY

  - model_name: smart-fallback
    litellm_params:
      model: ollama/llama3.2:3b
      api_base: http://host.docker.internal:11434

router_settings:
  routing_strategy: least-busy
  fallbacks:
    - {"cheap": ["cheap-fallback"]}
    - {"smart": ["smart-fallback"]}
  num_retries: 3
  retry_after: 5

litellm_settings:
  drop_params: true
  success_callback: []
  failure_callback: []
```

### 3. Update ai-orchestrator to use LiteLLM proxy

In `services/ai-orchestrator/app/main.py`, add an env var `LITELLM_BASE_URL` (default `http://litellm:4000`). When set, use it as the `api_base` for the OpenAI client instead of `https://openrouter.ai/api/v1`. The model name becomes the LiteLLM route name (`free`, `cheap`, `smart`).

```python
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")  # empty = use OpenRouter directly (backward compat)

if LITELLM_BASE_URL:
    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        base_url=LITELLM_BASE_URL,
    )
    # model name = tier name: "free", "cheap", "smart"
    model_name = model_tier  # pass tier directly as model
else:
    # existing OpenRouter path unchanged
    ...
```

### 4. Add env vars to `.env.example`

```text
# LiteLLM fallback gateway
LITELLM_BASE_URL=http://litellm:4000
LITELLM_MASTER_KEY=sk-local-dev-key
```

### 5. Update SYSTEM.md

Update the `Integrations` table to include:

```text
| LiteLLM proxy | litellm:4000 (fallback gateway — Ollama → OpenRouter → Gemini) |
```

Update the tier routing line to:

```text
- Tier routing: free (Ollama) → cheap (OpenRouter via LiteLLM) → smart (Gemini Flash via LiteLLM) → premium (Claude, human approval)
- LiteLLM handles automatic failover: if OpenRouter hits rate limits, falls back to Ollama locally
```

---

## Do

- Use LiteLLM as an optional sidecar — if `LITELLM_BASE_URL` is unset, existing OpenRouter path still works
- Keep Ollama as the ultimate local fallback (zero cost, always available when installed)
- Keep all env var secrets out of `litellm_config.yaml` — use `os.environ/VAR_NAME` syntax
- Add `LITELLM_MASTER_KEY` as the auth key for LiteLLM proxy (any string, internal only)
- Test: with `LITELLM_BASE_URL` set, a `POST /ai/complete` with `model_tier: free` should resolve to Ollama

## Do Not

- Do not remove the existing OpenRouter fallback path — LiteLLM is additive, not a replacement
- Do not add premium tier to LiteLLM config — premium stays manually approved
- Do not expose LiteLLM port publicly — keep it internal Docker network only
- Do not add paid providers (Anthropic, OpenAI paid) to the fallback chain without human approval

---

## Outputs

- `ai-microservice/litellm_config.yaml` — LiteLLM router config
- `ai-microservice/docker-compose.yml` — `litellm` service added
- `ai-microservice/services/ai-orchestrator/app/main.py` — LiteLLM branch added
- `ai-microservice/.env.example` — `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY` added
- `ai-microservice/SYSTEM.md` — integrations table updated

---

## Verify

```bash
# Check config file exists
ls ai-microservice/litellm_config.yaml

# Check docker-compose has litellm service
grep -A 5 "litellm:" ai-microservice/docker-compose.yml

# Check .env.example has new vars
grep "LITELLM" ai-microservice/.env.example

# Check main.py has LITELLM_BASE_URL branch
grep "LITELLM_BASE_URL" ai-microservice/services/ai-orchestrator/app/main.py
```
