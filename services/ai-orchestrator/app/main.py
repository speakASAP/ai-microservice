"""
StateX AI Orchestrator Service

Central coordination hub for all AI operations.
Manages workflow, routes requests, and combines results from AI agents.
Enhanced with multi-agent workflow orchestration framework.
"""

import os
import importlib.util
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
import uvicorn
import uuid
import asyncio
import json
import httpx
from datetime import datetime
import logging
import time
import sys
import os
from contextlib import asynccontextmanager
# Add the shared directory to Python path for local development and container
# Try multiple possible paths for http_clients
from pathlib import Path
current_dir = Path(__file__).parent.resolve()
possible_paths = [
    current_dir.parent.parent.parent / "shared",  # Development: /app/../shared
    Path("/app") / "shared",  # Container: /app/shared
]

shared_path = None
for path in possible_paths:
    http_clients_file = path / "http_clients.py"
    if http_clients_file.exists():
        shared_path = str(path)
        break

if shared_path is None:
    raise FileNotFoundError("Could not find http_clients.py in any expected location. Tried: " + str(possible_paths))

if shared_path not in sys.path:
    sys.path.append(shared_path)

# Import shared JWT auth (verify Bearer, roles: global:superadmin, internal:ai-microservice:admin)
auth_spec = importlib.util.spec_from_file_location("auth", os.path.join(shared_path, "auth.py"))
auth_module = importlib.util.module_from_spec(auth_spec)
auth_spec.loader.exec_module(auth_module)

# Import NotificationServiceClient with proper path resolution
spec = importlib.util.spec_from_file_location("http_clients", os.path.join(shared_path, "http_clients.py"))
http_clients_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http_clients_module)
NotificationServiceClient = http_clients_module.NotificationServiceClient  # type: ignore

# Import new multi-agent components
from .multi_agent_orchestrator import (
    MultiAgentOrchestrator, MultiAgentRequest, WorkflowResult,
    multi_agent_orchestrator
)
from .agent_coordination import agent_coordinator
from .workflow_engine import WorkflowState, TaskStatus, WorkflowStatus
from .workflow_persistence import workflow_persistence
from .workflow_recovery import workflow_recovery_manager, RecoveryStrategy
from .error_handling import error_recovery_manager
from .project_url_generator import project_url_generator
from . import shop_assistant_agents as shop_agents
from . import email_triage_agents

# Configure logging with centralized external logging service
# Import logger from shared directory
import importlib.util
logger_spec = importlib.util.spec_from_file_location("logger", os.path.join(shared_path, "logger.py"))
logger_module = importlib.util.module_from_spec(logger_spec)
logger_spec.loader.exec_module(logger_module)
logger = logger_module.setup_logger(__name__, service_name="ai-orchestrator")
shop_agents.set_logger(logger)
email_triage_agents.set_logger(logger)

# Configure Uvicorn logging to use centralized logger
import logging

# Configure root logger to use our centralized logger
root_logger = logging.getLogger()
root_logger.handlers = []

# Configure Uvicorn loggers to use our centralized logger
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = []
uvicorn_logger.propagate = True

uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = []
uvicorn_access_logger.propagate = True

# Configure uvicorn loggers to use centralized logger format with timestamps
timestamp_format = os.getenv('LOG_TIMESTAMP_FORMAT', '%Y-%m-%d %H:%M:%S')
log_format = f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure uvicorn loggers
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    uvicorn_logger = logging.getLogger(logger_name)
    uvicorn_logger.handlers = []
    
    # Add handler with timestamp format
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format, datefmt=timestamp_format)
    handler.setFormatter(formatter)
    uvicorn_logger.addHandler(handler)
    uvicorn_logger.setLevel(logging.INFO)



async def _background_startup():
    """Run Redis connect and workflow recovery in background so server can accept requests immediately."""
    try:
        await workflow_persistence.connect()
        from .workflows.prototype_workflow import register_prototype_workflow
        register_prototype_workflow(multi_agent_orchestrator.workflow_engine)
        recovered_workflows = await workflow_recovery_manager.auto_recover_interrupted_workflows()
        if recovered_workflows:
            logger.info("Auto-recovered interrupted workflows", count=len(recovered_workflows))
        logger.info("Background startup (Redis + recovery) completed")
    except Exception as e:
        logger.error("Background startup failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown. Email-triage is ready immediately; Redis/recovery run in background."""
    try:
        shop_agents.set_logger(logger)
        email_triage_agents.set_logger(logger)
        logger.info("Multi-agent orchestrator ready (email-triage and health accept requests)")
        asyncio.create_task(_background_startup())
    except Exception as e:
        logger.error("Lifespan startup failed", error=str(e))

    yield
    
    # Shutdown
    try:
        logger.info("Multi-agent orchestrator shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="StateX AI Orchestrator",
    description="Central coordination hub for AI operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Initialize notification service client
notification_client = NotificationServiceClient()

# CORS middleware — same-domain (aeps.alfares.cz → ai.alfares.cz) and statex/localhost; no 403 for alfares.cz
_cors_origins = [
    "https://statex.cz",
    "https://www.statex.cz",
    "https://aeps.alfares.cz",
    "http://aeps.alfares.cz",
    "https://agentic-email-processing-system.alfares.cz",
    "http://agentic-email-processing-system.alfares.cz",
    "https://ai.alfares.cz",
    "http://ai.alfares.cz",
    os.getenv("CORS_ORIGIN", f"http://localhost:{os.getenv('FRONTEND_PORT', '3602')}"),
    f"https://localhost:{os.getenv('FRONTEND_PORT', '3602')}",
    f"http://127.0.0.1:{os.getenv('FRONTEND_PORT', '3602')}",
    f"https://127.0.0.1:{os.getenv('FRONTEND_PORT', '3602')}",
]
_cors_extra = os.getenv("CORS_ORIGINS_EXTRA", "")
if _cors_extra:
    _cors_origins.extend(o.strip() for o in _cors_extra.split(",") if o.strip())
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://([a-z0-9-]+\.)?alfares\.cz$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Verify JWT and roles (global:superadmin, internal:ai-microservice:admin) for non-public paths."""

    async def dispatch(self, request, call_next):
        path = request.url.path.rstrip("/") or "/"
        if auth_module._is_public_path(path):
            return await call_next(request)
        auth_header = request.headers.get("Authorization")
        try:
            payload = auth_module.verify_bearer_token(auth_header or "")
            request.state.jwt_payload = payload
            return await call_next(request)
        except ValueError as e:
            from starlette.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": str(e)})


app.add_middleware(JWTAuthMiddleware)

# Service URLs
# Service URLs - use environment variables with defaults from .env
free_ai_port = os.getenv("FREE_AI_SERVICE_PORT", "3386")
gemini_ai_port = os.getenv("GEMINI_AI_SERVICE_PORT", "3388")
asr_port = os.getenv("ASR_SERVICE_PORT", "3382")
document_ai_port = os.getenv("DOCUMENT_AI_PORT", "3383")
prototype_gen_port = os.getenv("PROTOTYPE_GENERATOR_PORT", "3384")
free_ai_host = os.getenv("FREE_AI_SERVICE_HOST", "free-ai-service")
gemini_ai_host = os.getenv("GEMINI_AI_SERVICE_HOST", "gemini-ai-service")
asr_host = os.getenv("ASR_SERVICE_HOST", "asr-service")
document_ai_host = os.getenv("DOCUMENT_AI_HOST", "document-ai")
prototype_gen_host = os.getenv("PROTOTYPE_GENERATOR_HOST", "prototype-generator")

NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", f"http://{free_ai_host}:{free_ai_port}")  # Use free-ai-service
FREE_AI_SERVICE_URL = (os.getenv("FREE_AI_SERVICE_URL") or NLP_SERVICE_URL).rstrip("/")
# Optional: use LLM (free-ai-service OpenRouter) for email-triage Classifier and Decider; when false or on failure, use rule-based agents
EMAIL_TRIAGE_LLM_CLASSIFIER = os.getenv("EMAIL_TRIAGE_LLM_CLASSIFIER", "").strip().lower() in ("true", "1", "yes")
EMAIL_TRIAGE_LLM_DECIDER = os.getenv("EMAIL_TRIAGE_LLM_DECIDER", "").strip().lower() in ("true", "1", "yes")
GEMINI_AI_SERVICE_URL = os.getenv("GEMINI_AI_SERVICE_URL", f"http://{gemini_ai_host}:{gemini_ai_port}")
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", f"http://{asr_host}:{asr_port}")
DOCUMENT_AI_URL = os.getenv("DOCUMENT_AI_URL", f"http://{document_ai_host}:{document_ai_port}")
PROTOTYPE_GENERATOR_URL = os.getenv("PROTOTYPE_GENERATOR_URL", f"http://{prototype_gen_host}:{prototype_gen_port}")
# TEMPLATE_REPOSITORY_URL removed - Template Repository functionality disabled
NOTIFICATION_SERVICE_URL = os.getenv("NOTIFICATION_SERVICE_URL", "https://notifications.statex.cz")

# In-memory storage for demo (replace with database in production)
submissions_db: Dict[str, Dict[str, Any]] = {}
workflows_db: Dict[str, Dict[str, Any]] = {}

class SubmissionType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    FILE = "file"
    MIXED = "mixed"

class SubmissionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    AI_ANALYSIS = "ai_analysis"
    COMPLETED = "completed"
    FAILED = "failed"

class SubmissionRequest(BaseModel):
    submission_id: Optional[str] = None
    user_id: str
    session_id: Optional[str] = None  # Add session_id field
    submission_type: SubmissionType
    text_content: Optional[str] = None
    voice_file_url: Optional[str] = None
    file_urls: Optional[List[str]] = None
    requirements: Optional[str] = None
    contact_info: Dict[str, Any] = {}

# Shop-assistant agents: request/response models
class ShopTranscribeRequest(BaseModel):
    voice_file_url: str

class ShopRefineQueryRequest(BaseModel):
    user_text: str
    previous_params: Optional[Dict[str, Any]] = None
    role: str = "default"
    prompt_content: Optional[str] = None
    model: Optional[str] = None

class ShopSearchRequest(BaseModel):
    query_text: str
    limit: int = 20

class ShopFormatPresentationRequest(BaseModel):
    results: List[Dict[str, Any]]
    query_text: str
    role: str = "default"
    prompt_content: Optional[str] = None
    model: Optional[str] = None


class ShopCompareRequest(BaseModel):
    results: List[Dict[str, Any]]
    query_text: str
    role: str = "default"
    prompt_content: Optional[str] = None
    model: Optional[str] = None
    priority_order: Optional[List[str]] = None


class ShopLocationRequest(BaseModel):
    user_text: str
    query_text: str
    role: str = "default"
    prompt_content: Optional[str] = None
    model: Optional[str] = None
    priority_order: Optional[List[str]] = None


class EmailTriageClassifyRequest(BaseModel):
    """Payload for classifier: normalized email or raw fields."""
    payload: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    tenant_id: Optional[str] = None
    subject: Optional[str] = None
    body_plain: Optional[str] = None
    body_html: Optional[str] = None


class AIAnalysisResult(BaseModel):
    service: str
    status: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float

class WorkflowStep(BaseModel):
    step_id: str
    service: str
    status: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

class SubmissionResponse(BaseModel):
    submission_id: str
    status: SubmissionStatus
    message: str
    estimated_completion_time: Optional[str] = None

class SubmissionStatusResponse(BaseModel):
    submission_id: str
    status: SubmissionStatus
    progress: float
    current_step: str
    workflow_steps: List[WorkflowStep]
    results: Optional[Dict[str, Any]] = None
    prototype_id: Optional[str] = None
    results_page_url: Optional[str] = None
    created_at: str
    updated_at: str


@app.get("/")
async def root():
    """Root endpoint - Service information"""
    return {
        "service": "ai-microservice",
        "description": "Centralized AI processing service for business automation",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "api": "/api",
            "processSubmission": "POST /api/process-submission",
            "getStatus": "GET /api/status/{submission_id}",
            "getResults": "GET /api/results/{submission_id}",
            "multiAgentProcess": "POST /api/multi-agent/process",
            "getWorkflowStatus": "GET /api/multi-agent/workflow/{workflow_id}",
            "getAgentsHealth": "GET /api/multi-agent/agents/health",
            "shopAssistantTranscribe": "POST /api/shop-assistant/transcribe",
            "shopAssistantRefineQuery": "POST /api/shop-assistant/refine-query",
            "shopAssistantSearch": "POST /api/shop-assistant/search",
            "shopAssistantFormatPresentation": "POST /api/shop-assistant/format-presentation",
            "shopAssistantComparePrices": "POST /api/shop-assistant/compare-prices",
            "shopAssistantExtractLocation": "POST /api/shop-assistant/extract-location",
            "emailTriageIngest": "POST /api/email-triage/ingest",
            "emailTriageClassify": "POST /api/email-triage/classify",
            "emailTriageExtract": "POST /api/email-triage/extract",
            "emailTriageDecide": "POST /api/email-triage/decide",
        },
        "documentation": {
            "healthCheck": "GET /health - Check service health status with multi-agent system status",
            "processSubmission": "POST /api/process-submission - Process a new user submission",
            "getStatus": "GET /api/status/{submission_id} - Get the status of a submission",
            "getResults": "GET /api/results/{submission_id} - Get the final results of a submission",
            "multiAgentProcess": "POST /api/multi-agent/process - Process request through multi-agent workflow system",
            "getWorkflowStatus": "GET /api/multi-agent/workflow/{workflow_id} - Get status of a multi-agent workflow",
            "getAgentsHealth": "GET /api/multi-agent/agents/health - Get health status of all agents",
        },
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api")
@app.get("/api/")
async def api_info():
    """API documentation endpoint"""
    return {
        "success": True,
        "service": "ai-microservice",
        "apiVersion": "2.0.0",
        "endpoints": [
            {
                "method": "GET",
                "path": "/health",
                "description": "Health check endpoint with multi-agent system status",
                "response": {
                    "status": "healthy|degraded",
                    "service": "ai-orchestrator",
                    "timestamp": "ISO 8601 string",
                    "version": "2.0.0",
                    "multi_agent_system": "object with system health information",
                },
            },
            {
                "method": "POST",
                "path": "/api/process-submission",
                "description": "Process a new user submission",
                "contentType": "application/json",
                "requestBody": {
                    "user_id": "string (required)",
                    "submission_type": "text|voice|file|mixed (required)",
                    "text_content": "string (optional)",
                    "voice_file_url": "string (optional)",
                    "file_urls": "array of strings (optional)",
                    "requirements": "string (optional)",
                    "contact_info": "object (optional)",
                },
                "response": {
                    "submission_id": "string",
                    "status": "pending|processing|completed|failed",
                    "message": "string",
                    "estimated_completion_time": "string",
                },
            },
            {
                "method": "GET",
                "path": "/api/status/{submission_id}",
                "description": "Get the status of a submission",
                "pathParameters": {
                    "submission_id": "string (required) - submission UUID",
                },
                "response": {
                    "submission_id": "string",
                    "status": "pending|processing|completed|failed",
                    "progress": "number (0-100)",
                    "current_step": "string",
                    "workflow_steps": "array of workflow step objects",
                    "results": "object (if completed)",
                },
            },
            {
                "method": "GET",
                "path": "/api/results/{submission_id}",
                "description": "Get the final results of a submission",
                "pathParameters": {
                    "submission_id": "string (required) - submission UUID",
                },
                "response": {
                    "submission_id": "string",
                    "status": "completed",
                    "results": "object with AI analysis results",
                    "workflow_steps": "array of workflow step objects",
                },
            },
            {
                "method": "POST",
                "path": "/api/multi-agent/process",
                "description": "Process request through multi-agent workflow system",
                "contentType": "application/json",
                "requestBody": {
                    "submission_id": "string (required)",
                    "user_id": "string (required)",
                    "workflow_type": "business_analysis|document_processing|voice_analysis|prototype_generation (required)",
                    "text_content": "string (optional)",
                    "voice_file_url": "string (optional)",
                    "file_urls": "array of strings (optional)",
                    "contact_info": "object (optional)",
                },
                "response": {
                    "submission_id": "string",
                    "status": "processing",
                    "message": "string",
                    "workflow_type": "string",
                    "estimated_completion_time": "string",
                },
            },
            {
                "method": "GET",
                "path": "/api/multi-agent/workflow/{workflow_id}",
                "description": "Get status of a multi-agent workflow",
                "pathParameters": {
                    "workflow_id": "string (required) - workflow UUID",
                },
                "response": {
                    "workflow_id": "string",
                    "submission_id": "string",
                    "status": "pending|running|completed|failed|cancelled",
                    "progress": "number (0-100)",
                    "current_step": "string",
                    "agent_results": "object with agent results",
                },
            },
            {
                "method": "GET",
                "path": "/api/multi-agent/agents/health",
                "description": "Get health status of all agents",
                "response": {
                    "system_health": "object with overall system health",
                    "agent_metrics": "object with metrics for each agent",
                },
            },
            {
                "method": "POST",
                "path": "/api/email-triage/ingest",
                "description": "Email-triage ingest: validate and normalize payload per email-schema",
                "contentType": "application/json",
                "response": "{ success, payload } or { success: false, error, escalation_reason }",
            },
            {
                "method": "POST",
                "path": "/api/email-triage/classify",
                "description": "Email-triage classifier: intent + confidence per intent-taxonomy",
                "contentType": "application/json",
                "requestBody": "{ payload?: normalized email, or message_id, tenant_id, subject, body_plain, body_html }",
                "response": "{ success, intent, confidence, raw_scores }",
            },
        ],
        "timestamp": datetime.now().isoformat(),
    }

# --- Shop-assistant agents: all AI tasks run in ai-microservice ---
@app.post("/api/shop-assistant/transcribe")
async def shop_assistant_transcribe(req: ShopTranscribeRequest):
    """ASR agent: transcribe audio URL to text."""
    try:
        result = await shop_agents.agent_transcribe(req.voice_file_url)
        return result
    except Exception as e:
        logger.error("Shop-assistant transcribe failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shop-assistant/refine-query")
async def shop_assistant_refine_query(req: ShopRefineQueryRequest):
    """COMMUNICATION agent: refine user text into a search query."""
    try:
        result = await shop_agents.agent_refine_query(
            user_text=req.user_text,
            previous_params=req.previous_params,
            role=req.role,
            prompt_content=req.prompt_content,
            model=req.model,
        )
        return result
    except Exception as e:
        logger.error("Shop-assistant refine-query failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Refine-query failed: {str(e)}")

@app.post("/api/shop-assistant/search")
async def shop_assistant_search(req: ShopSearchRequest):
    """SEARCH agent: run external search (Serper) and return items."""
    try:
        result = await shop_agents.agent_search(query_text=req.query_text, limit=req.limit)
        return result
    except Exception as e:
        logger.error("Shop-assistant search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/shop-assistant/format-presentation")
async def shop_assistant_format_presentation(req: ShopFormatPresentationRequest):
    """PRESENTATION agent: format search results for user chat."""
    try:
        result = await shop_agents.agent_format_presentation(
            results=req.results,
            query_text=req.query_text,
            role=req.role,
            prompt_content=req.prompt_content,
            model=req.model,
        )
        return result
    except Exception as e:
        logger.error("Shop-assistant format-presentation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Format-presentation failed: {str(e)}")


@app.post("/api/shop-assistant/compare-prices")
async def shop_assistant_compare_prices(req: ShopCompareRequest):
    """COMPARISON agent: compare prices and priorities for products."""
    try:
        result = await shop_agents.agent_compare_prices(
            results=req.results,
            query_text=req.query_text,
            role=req.role,
            prompt_content=req.prompt_content,
            model=req.model,
            priority_order=req.priority_order,
        )
        return result
    except Exception as e:
        logger.error("Shop-assistant compare-prices failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Compare-prices failed: {str(e)}")


@app.post("/api/shop-assistant/extract-location")
async def shop_assistant_extract_location(req: ShopLocationRequest):
    """LOCATION agent: extract delivery region / shipping location."""
    try:
        result = await shop_agents.agent_extract_location(
            user_text=req.user_text,
            query_text=req.query_text,
            role=req.role,
            prompt_content=req.prompt_content,
            model=req.model,
            priority_order=req.priority_order,
        )
        return result
    except Exception as e:
        logger.error("Shop-assistant extract-location failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Extract-location failed: {str(e)}")


# --- Email-triage agents (agentic-email-processing-system) ---
@app.post("/api/email-triage/ingest")
async def email_triage_ingest(body: Dict[str, Any]):
    """
    Ingest: validate and normalize email payload per email-schema.
    Returns { success: true, payload } or 400 with error, escalation_reason.
    """
    logger.info("Email-triage ingest request received", message_id=body.get("message_id") if isinstance(body, dict) else None)
    if body is None or not isinstance(body, dict):
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": "Payload must be an object", "escalation_reason": "incomplete_data"},
        )
    t0 = time.perf_counter()
    # Ingest is rule-based only (no AI); run in thread so event loop is not blocked
    normalized, error, escalation_reason = await asyncio.to_thread(
        email_triage_agents.validate_and_normalize, body
    )
    duration_ms = round((time.perf_counter() - t0) * 1000)
    msg_id = (str(body.get("message_id")).strip() or None) if body.get("message_id") is not None else None
    if error is not None:
        logger.info("Email-triage ingest rejected", message_id=msg_id, duration_ms=duration_ms, error=error)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={
                "error": error,
                "escalation_reason": escalation_reason or "incomplete_data",
            },
        )
    logger.info("Email-triage ingest success", message_id=msg_id, duration_ms=duration_ms)
    return {"success": True, "payload": normalized, "duration_ms": duration_ms}


@app.post("/api/email-triage/classify")
async def email_triage_classify(body: Dict[str, Any]):
    """
    Classify: intent + confidence per intent-taxonomy.
    Body: { payload: <normalized email>, use_llm?: bool }.
    use_llm true = use LLM (OpenRouter); false = rule-based; omitted = use env EMAIL_TRIAGE_LLM_CLASSIFIER.
    """
    t0 = time.perf_counter()
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else body
    msg_id = str(payload.get("message_id")) if isinstance(payload, dict) and payload.get("message_id") is not None else None
    use_llm_request = body.get("use_llm")
    use_llm = bool(use_llm_request) if use_llm_request is not None else EMAIL_TRIAGE_LLM_CLASSIFIER
    logger.info("Email-triage classify request received", message_id=msg_id, use_llm=use_llm)
    result = None
    if use_llm and FREE_AI_SERVICE_URL:
        try:
            text_content = email_triage_agents.get_email_text_for_llm(payload)
            if not text_content.strip():
                raise ValueError("Empty email text for LLM classify")
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{FREE_AI_SERVICE_URL}/analyze",
                    json={"text_content": text_content, "analysis_type": "email_classify", "user_name": "email-triage"},
                )
                r.raise_for_status()
                data = r.json()
                if data.get("success") and isinstance(data.get("analysis"), dict):
                    a = data["analysis"]
                    model_used = data.get("model_used") or a.get("model_used")
                    result = {
                        "intent": a.get("intent"),
                        "confidence": float(a.get("confidence", 0)),
                        "raw_scores": a.get("raw_scores"),
                        "model_used": model_used,
                        "llm_output": {"intent": a.get("intent"), "confidence": a.get("confidence"), "raw_scores": a.get("raw_scores")},
                    }
                    if result["intent"] and 0 <= result["confidence"] <= 1:
                        logger.info(
                            "Email-triage classify via LLM",
                            message_id=msg_id,
                            intent=result["intent"],
                            confidence=result["confidence"],
                            model_used=model_used,
                            llm_output=result["llm_output"],
                        )
        except Exception as e:
            logger.warning("Email-triage LLM classify failed, falling back to rule-based", message_id=msg_id, error=str(e))
    if result is None:
        result = await asyncio.to_thread(email_triage_agents.classify_payload, payload)
        logger.info("Email-triage classify via rule-based", message_id=msg_id, intent=result.get("intent"))
    duration_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("Email-triage classify success", message_id=msg_id, intent=result.get("intent"), duration_ms=duration_ms)
    out = {
        "success": True,
        "intent": result["intent"],
        "confidence": result["confidence"],
        "raw_scores": result.get("raw_scores"),
    }
    if result.get("model_used") is not None:
        out["model_used"] = result["model_used"]
    if result.get("llm_output") is not None:
        out["llm_output"] = result["llm_output"]
    return out


@app.post("/api/email-triage/extract")
async def email_triage_extract(body: Dict[str, Any]):
    """
    Extractor: entities from normalized payload per extractor-contract.
    Body: { payload: <normalized email>, intent?: <classified intent> }.
    Returns { success: true, message_id, entities, summary? }.
    """
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else body
    intent = body.get("intent") if isinstance(body.get("intent"), str) else None
    result = await asyncio.to_thread(email_triage_agents.extract_payload, payload, intent)
    return {"success": True, **result}


@app.post("/api/email-triage/decide")
async def email_triage_decide(body: Dict[str, Any]):
    """
    Action/Decider: intent + confidence + optional entities -> action per routing-rules.
    Body: { intent, confidence, entities?: {...}, use_llm?: bool }. Returns { success: true, action, escalation_reason?, queue? }.
    use_llm true = LLM; false = rule-based; omitted = use env EMAIL_TRIAGE_LLM_DECIDER.
    """
    intent = body.get("intent")
    confidence = body.get("confidence")
    if intent is None or not isinstance(intent, str):
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": "intent is required", "escalation_reason": "incomplete_data"},
        )
    if confidence is None or not isinstance(confidence, (int, float)):
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": "confidence is required", "escalation_reason": "incomplete_data"},
        )
    entities = body.get("entities") if isinstance(body.get("entities"), dict) else None
    conf_float = float(confidence)
    use_llm_request = body.get("use_llm")
    use_llm = bool(use_llm_request) if use_llm_request is not None else EMAIL_TRIAGE_LLM_DECIDER
    result = None
    if use_llm and FREE_AI_SERVICE_URL:
        try:
            text_content = json.dumps({"intent": intent, "confidence": conf_float, "entities": entities or {}})
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{FREE_AI_SERVICE_URL}/analyze",
                    json={"text_content": text_content, "analysis_type": "email_decide", "user_name": "email-triage"},
                )
                r.raise_for_status()
                data = r.json()
                if data.get("success") and isinstance(data.get("analysis"), dict):
                    a = data["analysis"]
                    model_used = data.get("model_used") or a.get("model_used")
                    result = {
                        "action": a.get("action"),
                        "escalation_reason": a.get("escalation_reason"),
                        "queue": a.get("queue"),
                        "model_used": model_used,
                        "llm_output": {"action": a.get("action"), "escalation_reason": a.get("escalation_reason"), "queue": a.get("queue")},
                    }
                    if result["action"] in ("auto_respond", "route_to_queue", "escalate"):
                        logger.info(
                            "Email-triage decide via LLM",
                            intent=intent,
                            action=result["action"],
                            model_used=model_used,
                            llm_output=result["llm_output"],
                        )
        except Exception as e:
            logger.warning("Email-triage LLM decide failed, falling back to rule-based", intent=intent, error=str(e))
    if result is None:
        result = await asyncio.to_thread(
            email_triage_agents.decide_action, intent, conf_float, None, entities
        )
        logger.info("Email-triage decide via rule-based", intent=intent, action=result.get("action"))
    out = {"success": True, **result}
    if result.get("model_used") is not None:
        out["model_used"] = result["model_used"]
    if result.get("llm_output") is not None:
        out["llm_output"] = result["llm_output"]
    return out


@app.get("/health")
async def health_check():
    """Health check endpoint with multi-agent system status"""
    system_health = agent_coordinator.get_system_health_summary()

    return {
        "status": "healthy" if system_health.get("overall_health") != "critical" else "degraded",
        "service": "ai-orchestrator",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "multi_agent_system": system_health
    }


# Minimal payload used to verify email-triage ingest in healthcheck
_EMAIL_TRIAGE_HEALTH_PAYLOAD = {
    "message_id": "healthcheck",
    "tenant_id": "healthcheck",
    "timestamp": "2026-01-01T00:00:00Z",
    "sender": "healthcheck@local",
    "recipients": ["healthcheck@local"],
    "subject": "Health check",
    "body_plain": "Health check",
    "attachments": [],
}


@app.get("/api/email-triage/ready")
async def email_triage_ready():
    """Instant readiness check (no Redis, no validation). Use for connectivity; expect response in <1s."""
    return {"ready": True, "service": "email-triage"}


@app.get("/health/email-triage")
async def health_check_email_triage():
    """
    Health check that POST /api/email-triage/ingest responds correctly.
    Returns 200 if ingest validation succeeds, 503 otherwise.
    Used by Docker healthcheck so the orchestrator is marked unhealthy if email-triage is broken.
    """
    try:
        normalized, error, escalation_reason = await asyncio.to_thread(
            email_triage_agents.validate_and_normalize, _EMAIL_TRIAGE_HEALTH_PAYLOAD
        )
        if error is not None:
            logger.error("Health check email-triage: ingest validation failed", error=error, escalation_reason=escalation_reason)
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": error, "escalation_reason": escalation_reason},
            )
        return {"status": "ok", "message_id": normalized.get("message_id")}
    except Exception as e:
        logger.error("Health check email-triage: exception", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/models")
async def get_available_models(
    free_only: bool = False,
    context_min: Optional[int] = None,
    limit: Optional[int] = None,
    provider: Optional[str] = None
):
    """Get unified list of all available AI models from all services (fetched from source APIs)
    
    Query Parameters:
        free_only: If True, return only free models (default: False)
        context_min: Minimum context window size in tokens (optional)
        limit: Maximum number of models to return per provider (optional)
        provider: Filter by specific provider (e.g., "openrouter", "gemini") (optional)
    """
    try:
        models_by_provider = {}
        providers = {}
        model_list = []
        
        # Build query parameters for free-ai-service
        free_ai_params = {}
        if free_only:
            free_ai_params["free_only"] = "true"
        if context_min is not None:
            free_ai_params["context_min"] = str(context_min)
        if limit is not None:
            free_ai_params["limit"] = str(limit)
        
        # Fetch models from free-ai-service (which fetches from OpenRouter API)
        if not provider or provider == "openrouter":
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    free_ai_response = await client.get(
                        f"{NLP_SERVICE_URL}/models",
                        params=free_ai_params
                    )
                    if free_ai_response.status_code == 200:
                        free_ai_data = free_ai_response.json()
                        free_ai_models = free_ai_data.get("models", {})
                        free_ai_providers = free_ai_data.get("providers", {})
                    
                        # Process free-ai-service models (from OpenRouter API)
                        # Format: {"openrouter": [{"id": "...", "name": "...", "description": "...", "context_length": ..., "pricing": {...}}, ...]}
                        for prov, provider_models in free_ai_models.items():
                            if prov not in models_by_provider:
                                models_by_provider[prov] = []
                            if isinstance(provider_models, list):
                                for model_item in provider_models:
                                    if isinstance(model_item, dict):
                                        model_id = model_item.get("id") or model_item.get("name", "")
                                        model_name = model_item.get("name", model_id)
                                        description = model_item.get("description", "")
                                        context_length = model_item.get("context_length", 0)
                                        pricing = model_item.get("pricing", {})
                                        capabilities = model_item.get("capabilities", [])
                                        
                                        if model_id:
                                            models_by_provider[prov].append({
                                                "id": model_id,
                                                "name": model_name,
                                                "description": description,
                                                "context_length": context_length,
                                                "pricing": pricing,
                                                "capabilities": capabilities
                                            })
                                            model_list.append({
                                                "provider": prov,
                                                "name": model_id,
                                                "description": description,
                                                "context_length": context_length,
                                                "pricing": pricing
                                            })
                            elif isinstance(provider_models, dict):
                                # Handle dict format (fallback)
                                for model_id, model_info in provider_models.items():
                                    if isinstance(model_info, dict):
                                        model_name = model_info.get("name", model_id)
                                        description = model_info.get("description", "")
                                        context_length = model_info.get("context_length", 0)
                                        pricing = model_info.get("pricing", {})
                                        capabilities = model_info.get("capabilities", [])
                                    else:
                                        model_name = str(model_id)
                                        description = ""
                                        context_length = 0
                                        pricing = {}
                                        capabilities = []
                                    models_by_provider[prov].append({
                                        "id": model_id,
                                        "name": model_name,
                                        "description": description,
                                        "context_length": context_length,
                                        "pricing": pricing,
                                        "capabilities": capabilities
                                    })
                                    model_list.append({
                                        "provider": prov,
                                        "name": model_id,
                                        "description": description,
                                        "context_length": context_length,
                                        "pricing": pricing
                                    })
                        
                        # Add provider status
                        providers.update(free_ai_providers)
                        logger.info(f"Fetched {len(model_list)} models from free-ai-service (OpenRouter)")
            except httpx.TimeoutException as e:
                logger.error(f"Timeout fetching models from free-ai-service (connectivity or slow execution): {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to fetch models from free-ai-service: {e}")
                raise
        
        # Fetch models from gemini-ai-service
        if not provider or provider == "gemini":
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    gemini_response = await client.get(f"{GEMINI_AI_SERVICE_URL}/models")
                    if gemini_response.status_code == 200:
                        gemini_data = gemini_response.json()
                        gemini_models = gemini_data.get("models", {})
                        
                        # Process gemini-ai-service models
                        gemini_provider = "gemini"
                        if gemini_provider not in models_by_provider:
                            models_by_provider[gemini_provider] = []
                        
                        gemini_count = 0
                        if isinstance(gemini_models, dict):
                            for model_id, model_info in gemini_models.items():
                                # Apply filters
                                if free_only:
                                    # Gemini models are generally free, but check if needed
                                    pass  # Include all Gemini models as they're typically free
                                
                                if isinstance(model_info, dict):
                                    model_name = model_info.get("name", model_id)
                                    description = model_info.get("description", "")
                                    capabilities = model_info.get("capabilities", [])
                                    context_length = model_info.get("context_length", 0)
                                    
                                    # Filter by context_min if specified
                                    if context_min is not None and context_length < context_min:
                                        continue
                                    
                                    models_by_provider[gemini_provider].append({
                                        "id": model_id,
                                        "name": model_name,
                                        "description": description,
                                        "context_length": context_length,
                                        "pricing": {"free": True},
                                        "capabilities": capabilities
                                    })
                                    model_list.append({
                                        "provider": gemini_provider,
                                        "name": model_id,
                                        "description": description,
                                        "context_length": context_length,
                                        "pricing": {"free": True}
                                    })
                                    gemini_count += 1
                                    
                                    # Apply limit if specified
                                    if limit is not None and gemini_count >= limit:
                                        break
                        
                        providers[gemini_provider] = {"status": "available"}
                        logger.info(f"Fetched {gemini_count} models from gemini-ai-service")
            except httpx.TimeoutException as e:
                logger.error(f"Timeout fetching models from gemini-ai-service (connectivity or slow execution): {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to fetch models from gemini-ai-service: {e}")
                raise
        
        return {
            "models": models_by_provider,
            "providers": providers,
            "modelList": model_list
        }
    except Exception as e:
        logger.error(f"Error fetching available models: {e}")
        raise

@app.get("/metrics")
async def metrics():
    """Metrics endpoint - disabled"""
    return {"message": "Metrics endpoint disabled", "status": "disabled"}

@app.post("/api/multi-agent/process", response_model=Dict[str, Any])
async def process_multi_agent_request(request: MultiAgentRequest, background_tasks: BackgroundTasks):
    """Process request through multi-agent workflow system"""
    try:
        # Start multi-agent processing in background
        background_tasks.add_task(execute_multi_agent_workflow, request)
        
        return {
            "submission_id": request.submission_id,
            "status": "processing",
            "message": "Multi-agent workflow started",
            "workflow_type": request.workflow_type,
            "estimated_completion_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting multi-agent workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

async def execute_multi_agent_workflow(request: MultiAgentRequest):
    """Execute multi-agent workflow in background"""
    try:
        result = await multi_agent_orchestrator.process_multi_agent_request(request)
        logger.info(f"Multi-agent workflow completed for {request.submission_id}")
        
        # Update the submission in the legacy system if it exists
        if request.submission_id in submissions_db:
            submission = submissions_db[request.submission_id]
            submission["status"] = SubmissionStatus.COMPLETED
            submission["results"]["multi_agent_analysis"] = result.dict()
            submission["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Multi-agent workflow failed for {request.submission_id}: {e}")
        
        # Update submission status to failed
        if request.submission_id in submissions_db:
            submission = submissions_db[request.submission_id]
            submission["status"] = SubmissionStatus.FAILED
            submission["updated_at"] = datetime.now().isoformat()

@app.get("/api/multi-agent/workflow/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get status of a multi-agent workflow"""
    try:
        workflow_state = await multi_agent_orchestrator.get_workflow_status(workflow_id)
        
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow_state.workflow_id,
            "submission_id": workflow_state.submission_id,
            "status": workflow_state.status.value,
            "progress": workflow_state.progress,
            "current_step": workflow_state.current_step,
            "completed_tasks": len(workflow_state.completed_tasks),
            "failed_tasks": len(workflow_state.failed_tasks),
            "total_tasks": len(workflow_state.agent_results),
            "start_time": workflow_state.start_time.isoformat(),
            "end_time": workflow_state.end_time.isoformat() if workflow_state.end_time else None,
            "agent_results": {
                task_id: {
                    "agent_name": result.agent_name,
                    "status": result.status.value,
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score
                }
                for task_id, result in workflow_state.agent_results.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multi-agent/workflow/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow"""
    try:
        success = await multi_agent_orchestrator.cancel_workflow(workflow_id)
        
        if success:
            return {"message": f"Workflow {workflow_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Workflow not found or cannot be cancelled")
            
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/multi-agent/agents/health")
async def get_agents_health():
    """Get health status of all agents"""
    try:
        metrics = agent_coordinator.get_agent_metrics()
        system_health = agent_coordinator.get_system_health_summary()
        
        return {
            "system_health": system_health,
            "agent_metrics": {
                agent_type: {
                    "agent_name": metric.agent_name,
                    "health_status": metric.health_status.value,
                    "total_tasks": metric.total_tasks,
                    "success_rate": (metric.successful_tasks / metric.total_tasks * 100) if metric.total_tasks > 0 else 0,
                    "avg_processing_time": metric.avg_processing_time,
                    "consecutive_failures": metric.consecutive_failures,
                    "last_success": metric.last_success.isoformat() if metric.last_success else None,
                    "last_failure": metric.last_failure.isoformat() if metric.last_failure else None,
                    "uptime_percentage": metric.uptime_percentage
                }
                for agent_type, metric in metrics.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting agent health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/multi-agent/agents/{agent_type}/failures")
async def get_agent_failures(agent_type: str):
    """Get failure patterns for a specific agent"""
    try:
        failure_patterns = agent_coordinator.get_failure_patterns(agent_type)
        
        if agent_type not in failure_patterns:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_type": agent_type,
            "failure_patterns": [
                {
                    "failure_type": pattern.failure_type.value,
                    "count": pattern.count,
                    "first_occurrence": pattern.first_occurrence.isoformat(),
                    "last_occurrence": pattern.last_occurrence.isoformat()
                }
                for pattern in failure_patterns[agent_type]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting agent failures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/error-handling/statistics")
async def get_error_statistics():
    """Get comprehensive error statistics"""
    try:
        error_stats = error_recovery_manager.get_error_statistics()
        workflow_stats = await workflow_persistence.get_workflow_statistics()
        
        return {
            "error_statistics": error_stats,
            "workflow_statistics": workflow_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting error statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/{workflow_id}/recover")
async def recover_workflow(workflow_id: str, strategy: Optional[str] = None):
    """Manually trigger workflow recovery"""
    try:
        recovery_strategy = None
        if strategy:
            try:
                recovery_strategy = RecoveryStrategy(strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid recovery strategy: {strategy}")
        
        recovered_state = await workflow_recovery_manager.recover_workflow(
            workflow_id, recovery_strategy
        )
        
        if recovered_state:
            return {
                "workflow_id": workflow_id,
                "recovery_status": "success",
                "new_status": recovered_state.status.value,
                "recovery_strategy": recovery_strategy.value if recovery_strategy else "auto",
                "message": "Workflow recovered successfully"
            }
        else:
            return {
                "workflow_id": workflow_id,
                "recovery_status": "failed",
                "message": "Workflow recovery failed"
            }
            
    except Exception as e:
        logger.error(f"Error recovering workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/{workflow_id}/resume")
async def resume_workflow(workflow_id: str, from_checkpoint: Optional[str] = None):
    """Resume a paused or failed workflow"""
    try:
        resumed_state = await workflow_recovery_manager.resume_workflow(
            workflow_id, from_checkpoint
        )
        
        if resumed_state:
            return {
                "workflow_id": workflow_id,
                "resume_status": "success",
                "new_status": resumed_state.status.value,
                "from_checkpoint": from_checkpoint,
                "message": "Workflow resumed successfully"
            }
        else:
            return {
                "workflow_id": workflow_id,
                "resume_status": "failed",
                "message": "Workflow resume failed"
            }
            
    except Exception as e:
        logger.error(f"Error resuming workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/{workflow_id}/recovery-status")
async def get_workflow_recovery_status(workflow_id: str):
    """Get recovery status and options for a workflow"""
    try:
        recovery_status = await workflow_recovery_manager.get_recovery_status(workflow_id)
        return recovery_status
        
    except Exception as e:
        logger.error(f"Error getting recovery status for {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/maintenance/cleanup-stale-workflows")
async def cleanup_stale_workflows(max_age_hours: int = 24, auto_recover: bool = True):
    """Clean up stale workflows with optional auto-recovery"""
    try:
        cleanup_stats = await workflow_recovery_manager.cleanup_stale_workflows(
            max_age_hours, auto_recover
        )
        
        return {
            "cleanup_completed": True,
            "statistics": cleanup_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during stale workflow cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/maintenance/auto-recover-workflows")
async def auto_recover_workflows():
    """Manually trigger auto-recovery of interrupted workflows"""
    try:
        recovered_workflows = await workflow_recovery_manager.auto_recover_interrupted_workflows()
        
        return {
            "recovery_completed": True,
            "recovered_workflows": recovered_workflows,
            "total_recovered": len(recovered_workflows),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during auto-recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-submission", response_model=SubmissionResponse)
async def process_submission(submission: SubmissionRequest, background_tasks: BackgroundTasks):
    """Process a new user submission"""
    logger.info(f"Received submission request: {submission.user_id}")
    try:
        # Use provided submission_id or generate new one if not provided
        submission_id = submission.submission_id or str(uuid.uuid4())
        
        # Create submission record
        submission_record = {
            "submission_id": submission_id,
            "user_id": submission.user_id,
            "session_id": submission.session_id,  # Add session_id to submission record
            "submission_type": submission.submission_type,
            "text_content": submission.text_content,
            "voice_file_url": submission.voice_file_url,
            "file_urls": submission.file_urls or [],
            "requirements": submission.requirements,
            "contact_info": submission.contact_info,
            "status": SubmissionStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "workflow_steps": [],
            "results": {},
            # Extract user information from contact_info for notification service
            "user_name": submission.contact_info.get("name", "Unknown User"),
            "contact_type": "telegram",  # Always use telegram for individual agent notifications
            "contact_value": submission.contact_info.get("email", "694579866")  # email field contains telegram ID
        }
        
        submissions_db[submission_id] = submission_record
        
        # Check if multi-agent processing is enabled
        use_multi_agent = os.getenv("USE_MULTI_AGENT_WORKFLOW", "false").lower() == "true"
        
        if use_multi_agent:
            # Use new multi-agent workflow system
            multi_agent_request = MultiAgentRequest(
                submission_id=submission_id,
                user_id=submission.user_id,
                session_id=submission.session_id,  # Add session_id
                text_content=submission.text_content,
                voice_file_url=submission.voice_file_url,
                file_urls=submission.file_urls or [],
                contact_info=submission.contact_info,
                workflow_type="business_analysis"
            )
            background_tasks.add_task(execute_multi_agent_workflow, multi_agent_request)
        else:
            # Use legacy workflow system
            background_tasks.add_task(process_submission_workflow, submission_id)
        
        return SubmissionResponse(
            submission_id=submission_id,
            status=SubmissionStatus.PENDING,
            message="Submission received and processing started",
            estimated_completion_time="5-10 minutes"
        )
        
    except Exception as e:
        logger.error(f"Error processing submission: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process submission: {str(e)}")

async def process_submission_workflow(submission_id: str):
    """Process submission through AI workflow"""
    try:
        submission = submissions_db[submission_id]
        submission["status"] = SubmissionStatus.PROCESSING
        submission["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Starting workflow for submission {submission_id}")
        
        # Step 1: Process voice files (if any)
        if submission["voice_file_url"]:
            await process_voice_content(submission_id)
        
        # Step 2: Process document files (if any)
        if submission["file_urls"]:
            await process_document_files(submission_id)
        
        # Step 3: Analyze text content with NLP
        if submission["text_content"] or submission["voice_file_url"] or submission["file_urls"]:
            await process_nlp_analysis(submission_id)
        
        # Step 4: Generate prototype (if applicable) - DISABLED
        # await generate_prototype(submission_id)
        
        # Step 5: Create results page
        # await create_results_page(submission_id) - DISABLED
        
        # Step 7: Mark as completed
        submission["status"] = SubmissionStatus.COMPLETED
        submission["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Workflow completed for submission {submission_id}")
        
    except Exception as e:
        logger.error(f"Error in workflow for submission {submission_id}: {e}")
        submission = submissions_db[submission_id]
        submission["status"] = SubmissionStatus.FAILED
        submission["updated_at"] = datetime.now().isoformat()

async def process_voice_content(submission_id: str):
    """Process voice content through ASR service"""
    submission = submissions_db[submission_id]
    
    step = WorkflowStep(
        step_id="asr_processing",
        service="asr-service",
        status="processing",
        input_data={"voice_file_url": submission["voice_file_url"]}
    )
    
    submission["workflow_steps"].append(step.dict())
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ASR_SERVICE_URL}/api/transcribe",
                json={"voice_file_url": submission["voice_file_url"]}
            )
            
            if response.status_code == 200:
                result = response.json()
                # Update the step in the workflow_steps list
                for i, workflow_step in enumerate(submission["workflow_steps"]):
                    if workflow_step["step_id"] == step.step_id:
                        submission["workflow_steps"][i]["status"] = "completed"
                        submission["workflow_steps"][i]["output_data"] = result
                        submission["workflow_steps"][i]["processing_time"] = result.get("processing_time", 0)
                        break
                
                # Add transcribed text to submission
                submission["text_content"] = (submission["text_content"] or "") + "\n" + result.get("transcript", "")
                
            else:
                # Update the step in the workflow_steps list
                for i, workflow_step in enumerate(submission["workflow_steps"]):
                    if workflow_step["step_id"] == step.step_id:
                        submission["workflow_steps"][i]["status"] = "failed"
                        submission["workflow_steps"][i]["error_message"] = f"ASR service error: {response.status_code}"
                        break
                
    except Exception as e:
        # Update the step in the workflow_steps list
        for i, workflow_step in enumerate(submission["workflow_steps"]):
            if workflow_step["step_id"] == step.step_id:
                submission["workflow_steps"][i]["status"] = "failed"
                submission["workflow_steps"][i]["error_message"] = str(e)
                break
        logger.error(f"ASR processing error for {submission_id}: {e}")

    # Send notification with ASR results
    try:
        async with notification_client as client:
            await client.send_notification({
                "user_id": submission["user_id"],
                "type": "asr_completion",
                "title": "🎤 Voice Transcription Completed",
                "message": f"Voice file has been transcribed successfully. Transcript length: {len(submission.get('text_content', ''))} characters.",
                "contact_type": submission["contact_type"],
                "contact_value": submission["contact_value"],
                "user_name": submission["user_name"]
            })
    except Exception as e:
        logger.error(f"Failed to send ASR notification for {submission_id}: {e}")

async def process_document_files(submission_id: str):
    """Process document files through Document AI service"""
    submission = submissions_db[submission_id]
    
    step = WorkflowStep(
        step_id="document_processing",
        service="document-ai",
        status="processing",
        input_data={"file_urls": submission["file_urls"]}
    )
    
    submission["workflow_steps"].append(step.dict())
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DOCUMENT_AI_URL}/api/analyze-documents",
                json={"file_urls": submission["file_urls"]}
            )
            
            if response.status_code == 200:
                result = response.json()
                # Update the step in the workflow_steps list
                for i, workflow_step in enumerate(submission["workflow_steps"]):
                    if workflow_step["step_id"] == step.step_id:
                        submission["workflow_steps"][i]["status"] = "completed"
                        submission["workflow_steps"][i]["output_data"] = result
                        submission["workflow_steps"][i]["processing_time"] = result.get("processing_time", 0)
                        break
                
                # Add extracted text to submission
                extracted_text = result.get("extracted_text", "")
                submission["text_content"] = (submission["text_content"] or "") + "\n" + extracted_text

            else:
                step.status = "failed"
                step.error_message = f"Document AI service error: {response.status_code}"
                
    except Exception as e:
        step.status = "failed"
        step.error_message = str(e)
        logger.error(f"Document processing error for {submission_id}: {e}")

    # Send notification with document processing results
    try:
        # Get the original file URLs for input
        file_urls = submission.get("file_urls", [])
        input_text = f"Files: {', '.join([url.split('/')[-1] for url in file_urls])}" if file_urls else "No files"
        
        # Get the extracted text for output
        extracted_text = submission.get("text_content", "")
        output_text = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        
        async with notification_client as client:
            await client.send_notification({
                "user_id": submission["user_id"],
                "type": "document_completion",
                "title": "📄 Document Analysis Completed",
                "message": f"Document files have been processed successfully. Extracted text length: {len(extracted_text)} characters.\n\n📥 **Input:** {input_text}\n📤 **Output:** {output_text}",
                "contact_type": submission["contact_type"],
                "contact_value": submission["contact_value"],
                "user_name": submission["user_name"]
            })
    except Exception as e:
        logger.error(f"Failed to send document notification for {submission_id}: {e}")

async def process_nlp_analysis(submission_id: str):
    """Process text content through NLP service"""
    submission = submissions_db[submission_id]
    
    step = WorkflowStep(
        step_id="nlp_analysis",
        service="nlp-service",
        status="processing",
        input_data={"text_content": submission["text_content"]}
    )
    
    submission["workflow_steps"].append(step.dict())
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NLP_SERVICE_URL}/analyze",
                json={
                    "analysis_type": "business_analysis",
                    "text_content": submission["text_content"],
                    "user_name": submission.get("user_name", "Unknown User")
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Update the step in the workflow_steps list
                for i, workflow_step in enumerate(submission["workflow_steps"]):
                    if workflow_step["step_id"] == step.step_id:
                        submission["workflow_steps"][i]["status"] = "completed"
                        submission["workflow_steps"][i]["output_data"] = result
                        submission["workflow_steps"][i]["processing_time"] = result.get("processing_time", 0)
                        break
                
                # Store NLP results
                submission["results"]["nlp_analysis"] = result

            else:
                step.status = "failed"
                step.error_message = f"NLP service error: {response.status_code}"
                
    except Exception as e:
        step.status = "failed"
        step.error_message = str(e)
        logger.error(f"NLP processing error for {submission_id}: {e}")

    # Send notification with NLP analysis results
    try:
        # Get the input text
        input_text = submission.get("text_content", "")
        input_preview = input_text[:150] + "..." if len(input_text) > 150 else input_text
        
        # Get the analysis results
        nlp_results = submission.get("results", {}).get("nlp_analysis", {})
        analysis_summary = nlp_results.get("summary", "Analysis completed")
        output_preview = analysis_summary[:150] + "..." if len(analysis_summary) > 150 else analysis_summary
        
        async with notification_client as client:
            await client.send_notification({
                "user_id": submission["user_id"],
                "type": "nlp_completion",
                "title": "🧠 Text Analysis Completed",
                "message": f"Text content has been analyzed successfully. Analysis results are ready for review.\n\n📥 **Input:** {input_preview}\n📤 **Output:** {output_preview}",
                "contact_type": submission["contact_type"],
                "contact_value": submission["contact_value"],
                "user_name": submission["user_name"]
            })
    except Exception as e:
        logger.error(f"Failed to send NLP notification for {submission_id}: {e}")

async def create_results_page(submission_id: str):
    """Create results page for prototype"""
    submission = submissions_db[submission_id]
    
    step = WorkflowStep(
        step_id="results_page_creation",
        service="results-storage",
        status="processing",
        input_data={"prototype_id": submission.get("prototype_id", f"proto_{submission_id}")}
    )
    
    submission["workflow_steps"].append(step.dict())
    
    try:
        # Prepare results data for storage
        results_data = {
            "prototype_id": submission.get("prototype_id", f"proto_{submission_id}"),
            "form_data": {
                "text_content": submission["text_content"],
                "voice_file_url": submission.get("voice_file_url"),
                "file_urls": submission.get("file_urls", []),
                "contact_info": submission["contact_info"],
                "requirements": submission["requirements"]
            },
            "workflow_steps": submission["workflow_steps"],
            "ai_analysis": submission["results"],
            "prototype_info": submission["results"].get("prototype", {}),
            "summary": {
                "total_steps": len(submission["workflow_steps"]),
                "completed_steps": len([s for s in submission["workflow_steps"] if s.get("status") == "completed"]),
                "total_processing_time": sum([s.get("processing_time", 0) or 0 for s in submission["workflow_steps"]]),
                "created_at": submission["created_at"],
                "updated_at": submission["updated_at"]
            }
        }
        
        # Store results (for now, we'll store in the submission record)
        # In production, this would call a dedicated results storage service
        submission["results_page_data"] = results_data
        prototype_id = submission.get('prototype_id', f"proto_{submission_id}")
        frontend_port = os.getenv('FRONTEND_PORT', '3602')
        submission["results_page_url"] = f"http://localhost:{frontend_port}/prototype-results/{prototype_id}"
        
        # Update the step in the workflow_steps list
        for i, workflow_step in enumerate(submission["workflow_steps"]):
            if workflow_step["step_id"] == step.step_id:
                submission["workflow_steps"][i]["status"] = "completed"
                submission["workflow_steps"][i]["output_data"] = {
                    "results_page_url": submission["results_page_url"],
                    "status": "created"
                }
                submission["workflow_steps"][i]["processing_time"] = 0.1
                break
        
    except Exception as e:
        # Update the step in the workflow_steps list
        for i, workflow_step in enumerate(submission["workflow_steps"]):
            if workflow_step["step_id"] == step.step_id:
                submission["workflow_steps"][i]["status"] = "failed"
                submission["workflow_steps"][i]["error_message"] = str(e)
                break
        logger.error(f"Results page creation error for {submission_id}: {e}")

async def generate_prototype(submission_id: str):
    """Generate prototype through Prototype Generator service"""
    submission = submissions_db[submission_id]
    
    step = WorkflowStep(
        step_id="prototype_generation",
        service="prototype-generator",
        status="processing",
        input_data={
            "requirements": submission["requirements"],
            "nlp_analysis": submission["results"].get("nlp_analysis", {})
        }
    )
    
    submission["workflow_steps"].append(step.dict())
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PROTOTYPE_GENERATOR_URL}/api/generate-prototype",
                json={
                    "requirements": submission["requirements"],
                    "analysis": submission["results"].get("nlp_analysis", {})
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Update the step in the workflow_steps list
                for i, workflow_step in enumerate(submission["workflow_steps"]):
                    if workflow_step["step_id"] == step.step_id:
                        submission["workflow_steps"][i]["status"] = "completed"
                        submission["workflow_steps"][i]["output_data"] = result
                        submission["workflow_steps"][i]["processing_time"] = result.get("processing_time", 0)
                        break
                
                # Store prototype results
                submission["results"]["prototype"] = result
                
                # Generate proper project URLs
                try:
                    project_urls = await project_url_generator.create_and_validate_project_urls(
                        submission_id=submission_id,
                        user_id=submission.get("user_id", "anonymous")
                    )
                    submission["prototype_id"] = project_urls.project_id
                    submission["project_urls"] = {
                        "plan_url": project_urls.plan_url,
                        "offer_url": project_urls.offer_url,
                        "is_accessible": project_urls.is_accessible,
                        "validation_results": project_urls.validation_results
                    }
                except Exception as e:
                    logger.error(f"Error generating project URLs: {e}")
                    # Fallback to simple ID
                    submission["prototype_id"] = result.get("prototype_id", f"proto_{int(time.time())}")
            else:
                step.status = "failed"
                step.error_message = f"Prototype generator error: {response.status_code}"
                
    except Exception as e:
        step.status = "failed"
        step.error_message = str(e)
        logger.error(f"Prototype generation error for {submission_id}: {e}")

@app.get("/api/status/{submission_id}", response_model=SubmissionStatusResponse)
@app.head("/api/status/{submission_id}")
async def get_submission_status(submission_id: str):
    """Get the status of a submission"""
    if submission_id not in submissions_db:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    submission = submissions_db[submission_id]
    
    # Calculate progress
    total_steps = len(submission["workflow_steps"])
    completed_steps = len([step for step in submission["workflow_steps"] if step.get("status") == "completed"])
    progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
    
    # Get current step
    current_step = "pending"
    for step in submission["workflow_steps"]:
        if step.get("status") == "processing":
            current_step = step.get("step_id", "unknown")
            break
        elif step.get("status") == "completed":
            current_step = step.get("step_id", "unknown")
    
    return SubmissionStatusResponse(
        submission_id=submission_id,
        status=submission["status"],
        progress=progress,
        current_step=current_step,
        workflow_steps=[WorkflowStep(**step) for step in submission["workflow_steps"]],
        results=submission["results"] if submission["status"] == SubmissionStatus.COMPLETED else None,
        prototype_id=submission.get("prototype_id"),
        results_page_url=submission.get("results_page_url"),
        created_at=submission["created_at"],
        updated_at=submission["updated_at"]
    )

@app.get("/api/results/{submission_id}")
@app.head("/api/results/{submission_id}")
async def get_submission_results(submission_id: str):
    """Get the final results of a submission"""
    if submission_id not in submissions_db:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    submission = submissions_db[submission_id]
    
    if submission["status"] != SubmissionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Submission not ready for results")
    
    return {
        "submission_id": submission_id,
        "status": submission["status"],
        "results": submission["results"],
        "workflow_steps": submission["workflow_steps"],
        "prototype_id": submission.get("prototype_id"),
        "results_page_url": submission.get("results_page_url"),
        "created_at": submission["created_at"],
        "updated_at": submission["updated_at"]
    }

@app.get("/api/results/prototype/{prototype_id}")
async def get_results_by_prototype_id(prototype_id: str):
    """Get results by prototype ID"""
    # Find submission by prototype_id
    matching_submission = None
    for submission_id, submission in submissions_db.items():
        if submission.get("prototype_id") == prototype_id:
            matching_submission = submission
            break
    
    if not matching_submission:
        raise HTTPException(status_code=404, detail="Prototype not found")
    
    # Return the full submission data for the results page
    return {
        "submission_id": list(submissions_db.keys())[list(submissions_db.values()).index(matching_submission)],
        "prototype_id": prototype_id,
        "status": matching_submission["status"],
        "text_content": matching_submission.get("text_content", ""),
        "voice_file_url": matching_submission.get("voice_file_url"),
        "file_urls": matching_submission.get("file_urls", []),
        "contact_info": matching_submission.get("contact_info", {}),
        "requirements": matching_submission.get("requirements", ""),
        "results": matching_submission["results"],
        "workflow_steps": matching_submission["workflow_steps"],
        "results_page_data": matching_submission.get("results_page_data"),
        "results_page_url": matching_submission.get("results_page_url"),
        "created_at": matching_submission["created_at"],
        "updated_at": matching_submission["updated_at"]
    }

@app.get("/api/submissions")
async def get_all_submissions():
    """Get all submissions (for admin panel)"""
    return {
        "submissions": list(submissions_db.values()),
        "total": len(submissions_db)
    }

class PrototypeGenerationRequest(BaseModel):
    """Request model for prototype generation"""
    user_id: str
    submission_id: str
    description: str
    analysis: Dict[str, Any] = {}
    prototype_type: str = "website"
    customization_level: str = "basic"

@app.post("/api/prototype/generate")
async def generate_prototype_endpoint(request: PrototypeGenerationRequest, background_tasks: BackgroundTasks):
    """Generate a website prototype using AI workflow"""
    try:
        logger.info(f"Starting prototype generation for user {request.user_id}")
        
        # Create workflow
        workflow = await multi_agent_orchestrator.workflow_engine.create_workflow(
            submission_id=request.submission_id,
            workflow_type="prototype_generation",
            input_data={
                "user_id": request.user_id,
                "submission_id": request.submission_id,
                "description": request.description,
                "analysis": request.analysis,
                "prototype_type": request.prototype_type,
                "customization_level": request.customization_level
            }
        )
        
        # Execute workflow in background
        background_tasks.add_task(
            multi_agent_orchestrator.workflow_engine.execute_workflow,
            workflow.workflow_id
        )
        
        return {
            "success": True,
            "workflow_id": workflow.workflow_id,
            "submission_id": request.submission_id,
            "status": "queued",
            "message": "Prototype generation started",
            "estimated_completion_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting prototype generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start prototype generation: {str(e)}")

@app.get("/api/prototype/status/{workflow_id}")
async def get_prototype_status(workflow_id: str):
    """Get status of prototype generation workflow"""
    try:
        workflow_state = await multi_agent_orchestrator.workflow_engine.get_workflow_status(workflow_id)
        
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Extract prototype results from agent results
        prototype_results = {}
        for task_id, result in workflow_state.agent_results.items():
            if result.agent_type == "workflow-coordinator" and result.status.value == "completed":
                prototype_results = result.result_data or {}
                break
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_state.status.value,
            "progress": workflow_state.progress,
            "prototype_results": prototype_results,
            "created_at": workflow_state.start_time.isoformat(),
            "updated_at": workflow_state.end_time.isoformat() if workflow_state.end_time else None
        }
        
    except Exception as e:
        logger.error(f"Error getting prototype status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prototype status: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("AI_ORCHESTRATOR_PORT", "3380"))
    # Custom logging configuration for Uvicorn
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AI_ORCHESTRATOR_PORT", "3380")), log_config=log_config)
