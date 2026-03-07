#!/usr/bin/env python3
"""
Test script for AI microservice: health checks and LLM accessibility.

Makes test requests to all AI agents (orchestrator, free-ai-service, nlp, asr,
document-ai, prototype-generator, template-repository, ai-workers, gemini-ai-service,
data-viz-service), verifies responses, and checks OpenRouter accessibility via
free-ai-service /models and optional /analyze.

Usage:
  From host (ports mapped): ./scripts/test-ai-services.py
  With custom base: AI_SERVICE_HOST=localhost python3 scripts/test-ai-services.py
  From repo root: python3 scripts/test-ai-services.py

Loads .env from ai-microservice project root for port and host overrides.
Exit code: 0 if all checks pass, 1 otherwise.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# Timeout per request (do not increase; check logs if something hangs)
REQUEST_TIMEOUT = 15

# Project root = parent of scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def load_dotenv() -> None:
    """Load .env from project root (key=value only; no export)."""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"").strip()
                if key and value and not key.startswith("#"):
                    os.environ.setdefault(key, value)


def get_port(key: str, default: str) -> str:
    """Get port from env with default."""
    return os.environ.get(key, default)


def get_host() -> str:
    """Base host for services (localhost from host, or service name in Docker)."""
    return os.environ.get("AI_SERVICE_HOST", "localhost")


def url_get(url: str, timeout: int = REQUEST_TIMEOUT) -> Tuple[int, bytes, Optional[str]]:
    """
    GET request. Returns (status_code, body_bytes, error_message).
    """
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return e.code, body, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return 0, b"", str(e.reason) if e.reason else str(e)
    except OSError as e:
        return 0, b"", str(e)


def url_post_json(url: str, data: dict, timeout: int = REQUEST_TIMEOUT) -> Tuple[int, bytes, Optional[str]]:
    """
    POST JSON. Returns (status_code, body_bytes, error_message).
    """
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return e.code, body, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return 0, b"", str(e.reason) if e.reason else str(e)
    except OSError as e:
        return 0, b"", str(e)


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Print result and return ok."""
    if ok:
        print(f"  OK   {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    load_dotenv()
    host = get_host()

    ports = {
        "orchestrator": get_port("AI_ORCHESTRATOR_PORT", "3380"),
        "nlp": get_port("NLP_SERVICE_PORT", "3381"),
        "asr": get_port("ASR_SERVICE_PORT", "3382"),
        "document_ai": get_port("DOCUMENT_AI_PORT", "3383"),
        "prototype_generator": get_port("PROTOTYPE_GENERATOR_PORT", "3384"),
        "template_repository": get_port("TEMPLATE_REPOSITORY_PORT", "3385"),
        "free_ai": get_port("FREE_AI_SERVICE_PORT", "3386"),
        "ai_workers": get_port("AI_WORKERS_PORT", "3387"),
        "gemini_ai": get_port("GEMINI_AI_SERVICE_PORT", "3388"),
        "data_viz": get_port("DATA_VIZ_SERVICE_PORT", "3389"),
    }

    base_orchestrator = f"http://{host}:{ports['orchestrator']}"
    base_free_ai = f"http://{host}:{ports['free_ai']}"

    failed = 0

    print("AI Microservice – health and LLM accessibility tests")
    print(f"Host: {host}  Timeout: {REQUEST_TIMEOUT}s")
    print("")

    # --- Orchestrator ---
    print("[Orchestrator]")
    status, body, err = url_get(f"{base_orchestrator}/health")
    if not check("GET /health", status == 200, err or f"status={status}"):
        failed += 1
    elif body:
        try:
            j = json.loads(body.decode("utf-8"))
            if j.get("status") not in ("healthy", "degraded"):
                check("GET /health status field", False, f"status={j.get('status')}")
                failed += 1
        except json.JSONDecodeError:
            pass

    status, body, err = url_get(f"{base_orchestrator}/api/multi-agent/agents/health")
    # 200 = OK; 401 = auth required (endpoint exists, script may run without JWT)
    if not check(
        "GET /api/multi-agent/agents/health",
        status in (200, 401),
        err or f"status={status}" + (" (auth required)" if status == 401 else ""),
    ):
        failed += 1

    status, body, err = url_get(f"{base_orchestrator}/health/email-triage")
    if not check("GET /health/email-triage", status == 200, err or f"status={status}"):
        failed += 1

    print("")

    # --- Per-service health ---
    print("[Service health]")
    services = [
        ("NLP", f"http://{host}:{ports['nlp']}/health"),
        ("ASR", f"http://{host}:{ports['asr']}/health"),
        ("Document AI", f"http://{host}:{ports['document_ai']}/health"),
        ("Prototype Generator", f"http://{host}:{ports['prototype_generator']}/health"),
        ("Template Repository", f"http://{host}:{ports['template_repository']}/health"),
        ("Free AI Service", f"{base_free_ai}/health"),
        ("AI Workers", f"http://{host}:{ports['ai_workers']}/health"),
        ("Gemini AI Service", f"http://{host}:{ports['gemini_ai']}/health"),
        ("Data Viz Service", f"http://{host}:{ports['data_viz']}/health"),
    ]
    for name, url in services:
        status, _, err = url_get(url)
        if not check(f"{name} GET /health", status == 200, err or f"status={status}"):
            failed += 1

    print("")

    # --- OpenRouter via free-ai-service /models ---
    print("[OpenRouter / free-ai-service models]")
    status, body, err = url_get(f"{base_free_ai}/models")
    if not check("Free AI GET /models", status == 200, err or f"status={status}"):
        failed += 1
    else:
        try:
            j = json.loads(body.decode("utf-8"))
            models = j.get("models") or {}
            providers = j.get("providers") or {}
            openrouter_models = models.get("openrouter", [])
            openrouter_status = (providers.get("openrouter") or {}).get("status", "unknown")
            if openrouter_status == "available" and len(openrouter_models) > 0:
                check("OpenRouter accessible and provides models", True, f"models={len(openrouter_models)}")
            elif openrouter_status == "unavailable" or (providers.get("openrouter") or {}).get("reason"):
                reason = (providers.get("openrouter") or {}).get("reason", openrouter_status)
                check("OpenRouter accessible and provides models", False, reason)
                failed += 1
            else:
                check("OpenRouter accessible and provides models", len(openrouter_models) > 0, str(providers))
                if len(openrouter_models) == 0:
                    failed += 1
        except json.JSONDecodeError as e:
            check("OpenRouter /models response JSON", False, str(e))
            failed += 1

    print("")

    # --- Optional: minimal LLM request (free-ai-service /analyze) ---
    print("[LLM response check – Free AI /analyze]")
    analyze_url = f"{base_free_ai}/analyze"
    payload = {
        "text_content": "Say OK in one word.",
        "analysis_type": "business_analysis",
        "user_name": "test-script",
    }
    status, body, err = url_post_json(analyze_url, payload)
    if not check("POST /analyze", status == 200, err or f"status={status}"):
        failed += 1
    else:
        try:
            j = json.loads(body.decode("utf-8"))
            if not j.get("success", False):
                err_msg = j.get("error", "success=false")
                check("POST /analyze success=true", False, err_msg[:80] if err_msg else "success=false")
                failed += 1
            else:
                check("POST /analyze success=true", True, f"provider={j.get('provider_used', '?')}")
        except json.JSONDecodeError as e:
            check("POST /analyze response JSON", False, str(e))
            failed += 1

    print("")
    if failed:
        print(f"Result: {failed} check(s) failed.")
        return 1
    print("Result: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
