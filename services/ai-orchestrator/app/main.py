"""
StateX AI Orchestrator Service

Central coordination hub for all AI operations.
Manages workflow, routes requests, and combines results from AI agents.
Enhanced with multi-agent workflow orchestration framework.
"""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
import uvicorn
import uuid
import asyncio
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
    Path("/app") / "statex-platform" / "shared",  # Container: /app/statex-platform/shared
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

# Import NotificationServiceClient with proper path resolution
import importlib.util
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

# Configure logging with timestamps
# Add utils to path - try multiple possible paths
utils_possible_paths = [
    Path(__file__).parent.parent.parent.parent / "utils",  # Development
    Path("/app") / "utils",  # Container: /app/utils
]

utils_path = None
for path in utils_possible_paths:
    logger_file = path / "logger.py"
    if logger_file.exists():
        utils_path = str(path)
        break

if utils_path is None:
    raise FileNotFoundError("Could not find logger.py in any expected location. Tried: " + str(utils_possible_paths))

if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.logger import setup_logger
logger = setup_logger(__name__, service_name="ai-orchestrator")

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



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    try:
        # Connect to Redis for workflow persistence
        await workflow_persistence.connect()
        
        # Register prototype workflow
        from .workflows.prototype_workflow import register_prototype_workflow
        register_prototype_workflow(multi_agent_orchestrator.workflow_engine)
        
        # Note: Agent registration with coordinator will be handled separately
        # to avoid startup issues
        
        # Recover any interrupted workflows using the enhanced recovery manager
        recovered_workflows = await workflow_recovery_manager.auto_recover_interrupted_workflows()
        if recovered_workflows:
            logger.info(f"Auto-recovered {len(recovered_workflows)} interrupted workflows")
        
        logger.info("Multi-agent orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize multi-agent system: {e}")
    
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://statex.cz",
        "https://www.statex.cz",
        os.getenv("CORS_ORIGIN", f"http://localhost:{os.getenv('FRONTEND_PORT', '3602')}"),
        f"https://localhost:{os.getenv('FRONTEND_PORT', '3602')}",
        f"http://127.0.0.1:{os.getenv('FRONTEND_PORT', '3602')}",
        f"https://127.0.0.1:{os.getenv('FRONTEND_PORT', '3602')}"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
# Service URLs - use environment variables with defaults from .env
free_ai_port = os.getenv("FREE_AI_SERVICE_PORT", "3386")
asr_port = os.getenv("ASR_SERVICE_PORT", "3382")
document_ai_port = os.getenv("DOCUMENT_AI_PORT", "3383")
prototype_gen_port = os.getenv("PROTOTYPE_GENERATOR_PORT", "3384")
free_ai_host = os.getenv("FREE_AI_SERVICE_HOST", "free-ai-service")
asr_host = os.getenv("ASR_SERVICE_HOST", "asr-service")
document_ai_host = os.getenv("DOCUMENT_AI_HOST", "document-ai")
prototype_gen_host = os.getenv("PROTOTYPE_GENERATOR_HOST", "prototype-generator")

NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", f"http://{free_ai_host}:{free_ai_port}")  # Use free-ai-service
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
