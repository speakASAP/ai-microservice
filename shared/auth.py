"""
Shared JWT verification for AI microservice.

Verifies Bearer token using same JWT_SECRET as auth-microservice and
checks for global:superadmin or internal:ai-microservice:admin.
"""

import os
from typing import Any, Dict, List

import jwt
from starlette.requests import Request
from starlette.responses import JSONResponse

# Allowed roles for AI microservice admin access (aligned with NestJS JwtRolesGuard)
ALLOWED_ROLES: List[str] = [
    "global:superadmin",
    "internal:ai-microservice:admin",
]

# Paths that do not require JWT (public)
PUBLIC_PATHS = frozenset({
    "/",
    "/health",
    "/api",
    "/api/",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _is_public_path(path: str) -> bool:
    """Return True if path is public (no JWT required)."""
    # Exact match or prefix for /api (info only)
    if path in PUBLIC_PATHS:
        return True
    # Health endpoints (including /health/email-triage for Docker healthcheck)
    if path.startswith("/health"):
        return True
    # Multi-agent agents health (for monitoring and test-ai-services script)
    if path == "/api/multi-agent/agents/health":
        return True
    # Optional: treat /models as public for catalog; protect the rest
    if path == "/models":
        return True
    # Email-triage: internal service-to-service (agentic-email-processing-system)
    if path.startswith("/api/email-triage"):
        return True
    return False


def get_secret() -> str:
    """JWT secret from env (same as auth-microservice)."""
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise ValueError("JWT_SECRET is not set")
    return secret


def verify_bearer_token(authorization: str) -> Dict[str, Any]:
    """
    Verify Authorization: Bearer <token> and check roles.
    Returns decoded payload if valid.
    Raises ValueError with message if invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Missing or invalid Authorization header")
    token = authorization[7:].strip()
    if not token:
        raise ValueError("Missing Bearer token")
    secret = get_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")
    roles = payload.get("roles") or []
    if not isinstance(roles, list):
        roles = []
    allowed = set(ALLOWED_ROLES)
    if not any(r in allowed for r in roles):
        raise ValueError("Insufficient roles: require global:superadmin or internal:ai-microservice:admin")
    return payload


async def jwt_auth_middleware(request: Request, call_next):
    """
    Starlette middleware: verify JWT for non-public paths.
    Public paths: /, /health, /api, /api/, /docs, /redoc, /openapi.json, /models.
    """
    path = request.url.path.rstrip("/") or "/"
    if _is_public_path(path):
        return await call_next(request)
    auth_header = request.headers.get("Authorization")
    try:
        payload = verify_bearer_token(auth_header or "")
        request.state.jwt_payload = payload
        return await call_next(request)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"detail": str(e)},
        )
