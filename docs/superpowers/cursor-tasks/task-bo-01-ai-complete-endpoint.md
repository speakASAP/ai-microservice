# Cursor Task: Add `/ai/complete` endpoint to AI Orchestrator

**Project:** `/home/ssf/Documents/Github/ai-microservice/services/ai-orchestrator`
**File to edit:** `app/main.py`

---

## Context

`business-orchestrator` (NestJS service) calls `POST /ai/complete` on the AI orchestrator (port 3380) for LLM inference. This endpoint does not exist yet.

---

## What to add

### 1. Pydantic models (add near the top of `app/main.py`, after existing imports/models)

```python
class AiCompleteRequest(BaseModel):
    model_tier: str = "free"          # "free" | "cheap" | "smart"
    system_prompt: str
    user_prompt: str
    output_schema: Optional[dict] = None
    max_tokens: Optional[int] = 1000
    correlation_id: Optional[str] = None
```

### 2. Model tier mapping (add as a module-level constant)

```python
MODEL_TIER_MAP = {
    "free":  "gemini-2.5-flash-lite",
    "cheap": "gemini-2.5-flash",
    "smart": "gemini-2.5-pro",
}
```

### 3. Endpoint (add after the `/models` route)

```python
@app.post("/ai/complete")
async def ai_complete(request: AiCompleteRequest, req: Request):
    """
    Generic LLM completion endpoint for business-orchestrator.
    Accepts model_tier (free|cheap|smart), system_prompt, user_prompt.
    Returns the raw JSON-parsed LLM response, or {"text": "..."} for non-JSON responses.
    """
    import httpx, json as _json

    model_name = MODEL_TIER_MAP.get(request.model_tier, "gemini-2.5-flash-lite")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured")

    contents = []
    if request.system_prompt:
        contents.append({"role": "user", "parts": [{"text": request.system_prompt}]})
        contents.append({"role": "model", "parts": [{"text": "Understood. I will follow those instructions."}]})
    contents.append({"role": "user", "parts": [{"text": request.user_prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": request.max_tokens or 1000,
            "temperature": 0.2,
            "responseMimeType": "application/json" if request.output_schema else "text/plain",
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    # Try to parse as JSON; fall back to text envelope
    try:
        return _json.loads(raw_text)
    except (_json.JSONDecodeError, ValueError):
        return {"text": raw_text, "model_used": model_name}
```

### 4. Add `httpx` to requirements if not present

Check `requirements.txt` in the ai-orchestrator service directory. If `httpx` is not listed, add:
```
httpx>=0.27.0
```

---

## Verify

```bash
# From the host (service runs on port 3380):
curl -s -X POST http://localhost:3380/ai/complete \
  -H "Authorization: Bearer <service-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"model_tier":"free","system_prompt":"You are a helpful assistant.","user_prompt":"Say hello in JSON: {\"greeting\": \"...\"}","max_tokens":50}' | python3 -m json.tool
```

Expected: JSON response with a greeting field or `{"text": "..."}`.

After verifying, restart the ai-orchestrator container:
```bash
cd /home/ssf/Documents/Github/ai-microservice && docker compose -f docker-compose.green.yml restart ai-orchestrator-green
# or rebuild if requirements changed:
docker compose -f docker-compose.green.yml up -d --build ai-orchestrator-green
```
