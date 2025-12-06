"""
Multi-Agent AI Orchestrator

Enhanced orchestrator that integrates with the workflow engine to coordinate
multiple AI agents for business analysis and offer generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
import os
import sys
# Add the shared directory to Python path
from pathlib import Path
current_dir = Path(__file__).parent.resolve()
possible_paths = [
    current_dir.parent.parent.parent / "shared",  # Development: /app/../shared
    Path("/app") / "shared",  # Container: /app/shared
]

shared_dir = None
for path in possible_paths:
    http_clients_file = path / "http_clients.py"
    if http_clients_file.exists():
        shared_dir = str(path)
        break

if shared_dir is None:
    raise FileNotFoundError("Could not find http_clients.py in any expected location. Tried: " + str(possible_paths))

if shared_dir not in sys.path:
    sys.path.append(shared_dir)

# Import NotificationServiceClient with proper path resolution
import importlib.util
spec = importlib.util.spec_from_file_location("http_clients", os.path.join(shared_dir, "http_clients.py"))
http_clients_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http_clients_module)
NotificationServiceClient = http_clients_module.NotificationServiceClient  # type: ignore
from pydantic import BaseModel

from .workflow_engine import (
    WorkflowEngine, AgentTask, AgentResult, AgentInterface, 
    WorkflowState, TaskStatus, WorkflowStatus, workflow_engine
)
from .workflow_persistence import workflow_persistence
from .workflow_recovery import workflow_recovery_manager, CheckpointType
from .error_handling import (
    ErrorContext, ErrorRecoveryManager, error_recovery_manager,
    ErrorType, ErrorSeverity, RetryStrategy
)
from .business_analysis_aggregator import business_analysis_aggregator, BusinessAnalysisResult
from .project_url_generator import project_url_generator, ProjectURLs
from .offer_formatter import offer_formatter, FormattedOffer

logger = logging.getLogger(__name__)

def get_session_directory(submission_id: str, user_id: str, session_id: str = None) -> str:
    """Get the session directory path for a submission"""
    try:
        logger.info(f"🔍 [AI ORCHESTRATOR] Resolving session directory:")
        logger.info(f"  📋 Submission ID: {submission_id}")
        logger.info(f"  👤 User ID: {user_id}")
        logger.info(f"  🆔 Session ID: {session_id}")
        
        # Get upload base directory from environment variable
        upload_base = os.getenv("UPLOAD_DIR", "/srv/storagebox/statex/docker-volumes/uploads")
        base_dir = Path(upload_base)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  📂 Upload base directory: {base_dir}")
        logger.info(f"  📁 Base exists: {base_dir.exists()}")
        
        # Get user directory
        user_dir = base_dir / user_id
        if not user_dir.exists():
            logger.warning(f"User directory not found: {user_dir}")
            return None
        
        # Find session directory
        if session_id:
            session_path = user_dir / session_id
            if session_path.exists():
                logger.info(f"Using session directory: {session_path}")
                return str(session_path)
            else:
                logger.warning(f"Session directory not found: {session_path}")
                return None
        else:
            # Find the most recent session directory
            session_dirs = [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("sess_")]
            if not session_dirs:
                logger.warning(f"No session directories found for user {user_id}")
                return None
            
            # Sort by modification time and get the most recent
            latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
            logger.info(f"Using latest session directory: {latest_session}")
            return str(latest_session)
            
    except Exception as e:
        logger.error(f"Error getting session directory: {e}", exc_info=True)
        logger.error(f"Error getting session directory: {e}")
        return None

def read_file_from_session(session_path: str, filename: str, subdirectory: str = None) -> str:
    """Read a file from the session directory"""
    if subdirectory:
        file_path = os.path.join(session_path, subdirectory, filename)
    else:
        file_path = os.path.join(session_path, filename)
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    else:
        logger.warning(f"File not found: {file_path}")
        return ""

def list_files_in_subdirectory(session_path: str, subdirectory: str) -> List[str]:
    """List files in a subdirectory of the session directory"""
    subdir_path = os.path.join(session_path, subdirectory)
    if os.path.exists(subdir_path):
        try:
            return [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        except Exception as e:
            logger.error(f"Error listing files in {subdir_path}: {e}")
            return []
    else:
        logger.warning(f"Subdirectory not found: {subdir_path}")
        return []

class MultiAgentRequest(BaseModel):
    """Request for multi-agent processing"""
    submission_id: str
    user_id: str
    session_id: Optional[str] = None  # Add session_id field
    text_content: Optional[str] = None
    voice_file_url: Optional[str] = None
    file_urls: List[str] = []
    contact_info: Dict[str, Any] = {}
    workflow_type: str = "business_analysis"
    priority: int = 1

class BusinessAnalysis(BaseModel):
    """Comprehensive business analysis result"""
    project_scope: str = ""
    technology_stack: List[str] = []
    timeline_estimate: str = ""
    budget_range: str = ""
    risk_factors: List[str] = []
    market_insights: str = ""
    recommendations: List[str] = []
    confidence_score: float = 0.0

class OfferDetails(BaseModel):
    """Business offer details"""
    project_id: str
    plan_url: str
    offer_url: str
    pricing_tiers: List[Dict[str, Any]] = []
    implementation_phases: List[Dict[str, Any]] = []
    deliverables: List[str] = []
    next_steps: List[str] = []

class WorkflowResult(BaseModel):
    """Complete workflow execution result"""
    submission_id: str
    workflow_id: str
    total_agents: int
    successful_agents: int
    failed_agents: int
    processing_time: float
    business_analysis: BusinessAnalysisResult
    offer_details: OfferDetails
    agent_results: Dict[str, AgentResult] = {}

# AI Agent Implementations
class NLPAgent(AgentInterface):
    """NLP Analysis Agent"""
    
    def __init__(self):
        super().__init__("NLP Analysis Agent", "nlp")
        free_ai_port = os.getenv("FREE_AI_SERVICE_PORT", "3386")
        self.service_url = os.getenv("FREE_AI_SERVICE_URL", f"http://localhost:{free_ai_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute NLP analysis task with error handling and recovery"""
        context = ErrorContext(
            submission_id=task.input_data.get("submission_id", "unknown"),
            agent_type=self.agent_type,
            task_id=task.task_id,
            service_url=self.service_url,
            operation="nlp_analysis",
            input_data=task.input_data
        )
        
        async def nlp_operation():
            # Read from disk if requested
            if task.input_data.get("read_from_disk"):
                submission_id = task.input_data.get("submission_id")
                user_id = task.input_data.get("user_id")
                session_id = task.input_data.get("session_id")
                session_path = get_session_directory(submission_id, user_id, session_id)
                
                if session_path:
                    # Read form_data.md from session directory
                    form_data_content = read_file_from_session(session_path, "form_data.md")
                    logger.info(f"Read form_data.md from {session_path}: {len(form_data_content)} characters")
                    
                    # Use the form data content for analysis
                    nlp_request = {
                        "text_content": form_data_content,
                        "analysis_type": "business_analysis",
                        "user_name": "NLP Agent"
                    }
                else:
                    # Fallback to input data
                    nlp_request = {
                        "text_content": task.input_data.get("text_content", ""),
                        "analysis_type": "business_analysis",
                        "user_name": "NLP Agent"
                    }
            else:
                # Use input data directly
                nlp_request = {
                    "text_content": task.input_data.get("text_content", ""),
                    "analysis_type": "business_analysis",
                    "user_name": "NLP Agent"
                }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.service_url}/analyze",
                    json=nlp_request
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    analysis_data = response_data.get("analysis", {})
                    
                    # Transform free-ai-service response to expected format
                    transformed_result = {
                        "analysis": {
                            "summary": analysis_data.get("summary", "Business analysis completed"),
                            "key_insights": analysis_data.get("key_insights", []),
                            "recommendations": analysis_data.get("recommendations", []),
                            "business_type": analysis_data.get("business_type", "general"),
                            "pain_points": analysis_data.get("pain_points", []),
                            "opportunities": analysis_data.get("opportunities", []),
                            "technical_recommendations": analysis_data.get("technical_recommendations", {}),
                            "next_steps": analysis_data.get("next_steps", []),
                            "budget_estimate": analysis_data.get("budget_estimate", {})
                        },
                        "confidence": response_data.get("confidence", 0.8),
                        "processing_time": response_data.get("processing_time", 0.0),
                        "model_used": response_data.get("model_used", "unknown"),
                        "ai_service": response_data.get("provider_used", "free-ai-service")
                    }
                    
                    # Save agent result to file
                    await self._save_agent_result(task, transformed_result)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=transformed_result,
                        confidence_score=response_data.get("confidence", 0),
                        processing_time=response_data.get("processing_time", 0)
                    )
                else:
                    raise httpx.HTTPStatusError(
                        f"NLP service error: {response.status_code}",
                        request=response.request,
                        response=response
                    )
        
        try:
            # Execute the operation directly
            return await nlp_operation()
        except Exception as e:
            # If all recovery attempts fail, return failed result
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check NLP service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class ASRAgent(AgentInterface):
    """Automatic Speech Recognition Agent"""
    
    def __init__(self):
        super().__init__("ASR Agent", "asr")
        asr_port = os.getenv("ASR_SERVICE_PORT", "3382")
        self.service_url = os.getenv("ASR_SERVICE_URL", f"http://localhost:{asr_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute ASR transcription task with error handling and recovery"""
        context = ErrorContext(
            submission_id=task.input_data.get("submission_id", "unknown"),
            agent_type=self.agent_type,
            task_id=task.task_id,
            service_url=self.service_url,
            operation="asr_transcription",
            input_data=task.input_data
        )
        
        async def asr_operation():
            # Read from disk if requested
            if task.input_data.get("read_from_disk"):
                submission_id = task.input_data.get("submission_id")
                user_id = task.input_data.get("user_id")
                session_id = task.input_data.get("session_id")
                session_path = get_session_directory(submission_id, user_id, session_id)
                
                if session_path:
                    # List voice files in the files/ subdirectory
                    voice_files = list_files_in_subdirectory(session_path, "files")
                    # Filter for audio files (webm, mp3, wav, etc.)
                    audio_files = [f for f in voice_files if f.lower().endswith(('.webm', '.mp3', '.wav', '.m4a', '.ogg'))]
                    
                    if audio_files:
                        # Use the first audio file found
                        voice_file_path = os.path.join(session_path, "files", audio_files[0])
                        logger.info(f"Found voice file: {voice_file_path}")
                        
                        # Create a file URL for the ASR service
                        asr_request = {
                            "voice_file_url": f"file://{voice_file_path}",
                            "submission_id": submission_id,
                            "user_id": user_id
                        }
                    else:
                        logger.warning("No voice files found in files/ subdirectory")
                        asr_request = task.input_data
                else:
                    logger.warning("Session directory not found, using input data")
                    asr_request = task.input_data
            else:
                asr_request = task.input_data
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.service_url}/api/transcribe",
                    json=asr_request
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Save agent result to file
                    await self._save_agent_result(task, result_data)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=result_data,
                        confidence_score=result_data.get("confidence", 0.7),
                        processing_time=result_data.get("processing_time", 0)
                    )
                else:
                    raise httpx.HTTPStatusError(
                        f"ASR service error: {response.status_code}",
                        request=response.request,
                        response=response
                    )
        
        try:
            # Execute the operation directly
            return await asr_operation()
        except Exception as e:
            # If all recovery attempts fail, return failed result
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check ASR service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class DocumentAgent(AgentInterface):
    """Document Analysis Agent"""
    
    def __init__(self):
        super().__init__("Document AI Agent", "document")
        document_ai_port = os.getenv("DOCUMENT_AI_PORT", "3383")
        self.service_url = os.getenv("DOCUMENT_AI_URL", f"http://localhost:{document_ai_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute document analysis task"""
        try:
            # Read from disk if requested
            if task.input_data.get("read_from_disk"):
                submission_id = task.input_data.get("submission_id")
                user_id = task.input_data.get("user_id")
                session_id = task.input_data.get("session_id")
                session_path = get_session_directory(submission_id, user_id, session_id)
                
                if session_path:
                    # List attachment files in the files/ subdirectory
                    all_files = list_files_in_subdirectory(session_path, "files")
                    # Filter for document files (pdf, doc, docx, txt, etc.)
                    doc_files = [f for f in all_files if f.lower().endswith(('.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'))]
                    
                    if doc_files:
                        # Create file URLs for the document service
                        file_urls = [f"file://{os.path.join(session_path, 'files', f)}" for f in doc_files]
                        logger.info(f"Found document files: {doc_files}")
                        
                        doc_request = {
                            "file_urls": file_urls,
                            "submission_id": submission_id,
                            "user_id": user_id
                        }
                    else:
                        logger.warning("No document files found in files/ subdirectory")
                        doc_request = task.input_data
                else:
                    logger.warning("Session directory not found, using input data")
                    doc_request = task.input_data
            else:
                doc_request = task.input_data
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.service_url}/api/analyze-documents",
                    json=doc_request
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Save agent result to file
                    await self._save_agent_result(task, result_data)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=result_data,
                        confidence_score=result_data.get("confidence", 0.6),
                        processing_time=result_data.get("processing_time", 0)
                    )
                else:
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.FAILED,
                        error_message=f"Document AI service error: {response.status_code}"
                    )
                    
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check Document AI service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class PrototypeAgent(AgentInterface):
    """Prototype Generation Agent"""
    
    def __init__(self):
        super().__init__("Prototype Generator Agent", "prototype")
        prototype_gen_port = os.getenv("PROTOTYPE_GENERATOR_PORT", "3384")
        self.service_url = os.getenv("PROTOTYPE_GENERATOR_URL", f"http://localhost:{prototype_gen_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute prototype generation task"""
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{self.service_url}/api/generate-prototype",
                    json=task.input_data
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Save agent result to file
                    await self._save_agent_result(task, result_data)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=result_data,
                        confidence_score=result_data.get("confidence", 0.9),
                        processing_time=result_data.get("processing_time", 0)
                    )
                else:
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.FAILED,
                        error_message=f"Prototype service error: {response.status_code}"
                    )
                    
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check Prototype Generator service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class GeminiAgent(AgentInterface):
    """Google Gemini AI Agent - Advanced AI analysis using Gemini models"""
    
    def __init__(self):
        super().__init__("Gemini AI Agent", "gemini")
        gemini_port = os.getenv("GEMINI_AI_SERVICE_PORT", "3388")
        self.service_url = os.getenv("GEMINI_AI_SERVICE_URL", f"http://localhost:{gemini_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute Gemini AI analysis task with error handling and recovery"""
        context = ErrorContext(
            submission_id=task.input_data.get("submission_id", "unknown"),
            agent_type=self.agent_type,
            task_id=task.task_id,
            service_url=self.service_url,
            operation="gemini_analysis",
            input_data=task.input_data
        )
        
        async def gemini_operation():
            # Read from disk if requested
            if task.input_data.get("read_from_disk"):
                submission_id = task.input_data.get("submission_id")
                user_id = task.input_data.get("user_id")
                session_id = task.input_data.get("session_id")
                session_path = get_session_directory(submission_id, user_id, session_id)
                
                if session_path:
                    # Read form_data.md from session directory
                    form_data_content = read_file_from_session(session_path, "form_data.md")
                    logger.info(f"Read form_data.md from {session_path}: {len(form_data_content)} characters")
                    
                    # Use the form data content for analysis
                    gemini_request = {
                        "text_content": form_data_content,
                        "analysis_type": "business_analysis",
                        "user_name": "Gemini Agent"
                    }
                else:
                    # Fallback to input data
                    gemini_request = {
                        "text_content": task.input_data.get("text_content", ""),
                        "analysis_type": "business_analysis",
                        "user_name": "Gemini Agent"
                    }
            else:
                # Use input data directly
                gemini_request = {
                    "text_content": task.input_data.get("text_content", ""),
                    "analysis_type": "business_analysis",
                    "user_name": "Gemini Agent"
                }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.service_url}/analyze",
                    json=gemini_request
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    analysis_data = response_data.get("analysis", {})
                    
                    # Transform Gemini service response to expected format
                    transformed_result = {
                        "analysis": {
                            "summary": analysis_data.get("summary", "Gemini analysis completed"),
                            "key_insights": analysis_data.get("key_insights", []),
                            "recommendations": analysis_data.get("recommendations", []),
                            "business_type": analysis_data.get("business_type", "general"),
                            "pain_points": analysis_data.get("pain_points", []),
                            "opportunities": analysis_data.get("opportunities", []),
                            "technical_recommendations": analysis_data.get("technical_recommendations", {}),
                            "next_steps": analysis_data.get("next_steps", []),
                            "budget_estimate": analysis_data.get("budget_estimate", {})
                        },
                        "confidence": response_data.get("confidence", 0.8),
                        "processing_time": response_data.get("processing_time", 0.0),
                        "model_used": response_data.get("model_used", "unknown"),
                        "ai_service": response_data.get("ai_service", "Google Gemini")
                    }
                    
                    # Save agent result to file
                    await self._save_agent_result(task, transformed_result)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=transformed_result,
                        confidence_score=response_data.get("confidence", 0),
                        processing_time=response_data.get("processing_time", 0)
                    )
                else:
                    raise httpx.HTTPStatusError(
                        f"Gemini service error: {response.status_code}",
                        request=response.request,
                        response=response
                    )
        
        try:
            # Execute the operation directly
            return await gemini_operation()
        except Exception as e:
            # If all recovery attempts fail, return failed result
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check Gemini service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class DataVizAgent(AgentInterface):
    """Data Analysis/Visualization Agent - Creates data visualizations from summary data"""
    
    def __init__(self):
        super().__init__("Data Visualization Agent", "data_viz")
        data_viz_port = os.getenv("DATA_VIZ_SERVICE_PORT", "3389")
        self.service_url = os.getenv("DATA_VIZ_SERVICE_URL", f"http://localhost:{data_viz_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute data visualization task with error handling and recovery"""
        context = ErrorContext(
            submission_id=task.input_data.get("submission_id", "unknown"),
            agent_type=self.agent_type,
            task_id=task.task_id,
            service_url=self.service_url,
            operation="data_visualization",
            input_data=task.input_data
        )
        
        async def data_viz_operation():
            # Prepare request data
            data_viz_request = {
                "submission_id": task.input_data.get("submission_id", ""),
                "user_id": task.input_data.get("user_id", ""),
                "session_id": task.input_data.get("session_id", ""),
                "analysis_type": "data_analysis"
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.service_url}/analyze",
                    json=data_viz_request
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    visualization_data = response_data.get("visualization_data", {})
                    
                    # Transform DataViz service response to expected format
                    transformed_result = {
                        "visualization": {
                            "file_path": visualization_data.get("file_path", ""),
                            "content_length": visualization_data.get("content_length", 0),
                            "charts_created": response_data.get("charts_created", []),
                            "insights": response_data.get("insights", [])
                        },
                        "data_points": visualization_data.get("data_points", {}),
                        "ai_analysis": visualization_data.get("ai_analysis", {}),
                        "processing_time": response_data.get("processing_time", 0.0),
                        "ai_service": "Data Analysis/Visualization Service"
                    }
                    
                    # Save agent result to file
                    await self._save_agent_result(task, transformed_result)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=transformed_result,
                        confidence_score=0.9,  # High confidence for data visualization
                        processing_time=response_data.get("processing_time", 0)
                    )
                else:
                    raise httpx.HTTPStatusError(
                        f"DataViz service error: {response.status_code}",
                        request=response.request,
                        response=response
                    )
        
        try:
            # Execute the operation directly
            return await data_viz_operation()
        except Exception as e:
            # If all recovery attempts fail, return failed result
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check DataViz service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class SummarizerAgent(AgentInterface):
    """Summarizer Agent - Aggregates all analysis results into a comprehensive summary"""
    
    def __init__(self):
        super().__init__("Summarizer Agent", "summarizer")
        free_ai_port = os.getenv("FREE_AI_SERVICE_PORT", "3386")
        self.service_url = os.getenv("FREE_AI_SERVICE_URL", f"http://localhost:{free_ai_port}")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute summarization task"""
        try:
            # Read individual agent result files
            collected_data = await self._read_agent_result_files(task)
            
            # Create a comprehensive summary prompt
            summary_prompt = self._create_summary_prompt(collected_data)
            
            # Use the free-ai-service for summarization
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.service_url}/analyze",
                    json={
                        "text_content": summary_prompt,
                        "analysis_type": "business_analysis",
                        "user_name": "AI Summarizer"
                    }
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Extract the summary from the analysis result
                    analysis = result_data.get("analysis", {})
                    summary_text = analysis.get("summary", "No summary available")
                    
                    # Create a structured summary result
                    summary_result = {
                        "summary_text": summary_text,
                        "tokens_used": result_data.get("tokens_used", 0),
                        "model_used": result_data.get("model_used", analysis.get("model_used", "unknown")),
                        "ai_service": result_data.get("provider_used", analysis.get("ai_service", "unknown")),
                        "confidence": analysis.get("confidence", 0.8),
                        "processing_time": result_data.get("processing_time", 0),
                        "processed_files": collected_data.get("processed_files", []),  # Add processed files
                        "collected_insights": {
                            "nlp_analysis": collected_data.get("nlp_analysis", {}),
                            "asr_transcript": collected_data.get("asr_transcript", ""),
                            "document_analysis": collected_data.get("document_analysis", {})
                        }
                    }
                    
                    # Save agent result to file
                    await self._save_agent_result(task, summary_result)
                    
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.COMPLETED,
                        result_data=summary_result,
                        confidence_score=analysis.get("confidence", 0.8),
                        processing_time=result_data.get("processing_time", 0)
                    )
                else:
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status=TaskStatus.FAILED,
                        error_message=f"Summarizer service error: {response.status_code}"
                    )
                    
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def _read_agent_result_files(self, task: AgentTask) -> Dict[str, Any]:
        """Read individual agent result files from the session directory"""
        collected_data = {}
        processed_files = []  # Track which files were actually processed
        
        try:
            # Get submission ID and user ID from task context
            submission_id = task.input_data.get("submission_id")
            user_id = task.input_data.get("user_id")
            
            if not submission_id or not user_id:
                logger.warning("No submission_id or user_id found in task input_data")
                return collected_data
            
            # Get session directory
            session_id = task.input_data.get("session_id")
            session_path = get_session_directory(submission_id, user_id, session_id)
            if not session_path:
                logger.warning(f"Session directory not found for submission {submission_id}")
                return collected_data
            
            # Read each agent result file (excluding prototype.md as it's created after summary)
            agent_files = {
                "nlp": "nlp.md",
                "asr": "voicerecording.md", 
                "document": "attachments.md"
            }
            
            for agent_type, filename in agent_files.items():
                try:
                    logger.info(f"Reading {agent_type} result file from {session_path}/{filename}")
                    file_content = read_file_from_session(session_path, filename)
                    
                    if file_content:
                        collected_data[f"{agent_type}_analysis"] = file_content
                        processed_files.append(filename)  # Track successfully processed files
                        logger.info(f"Successfully read {agent_type} result file: {len(file_content)} characters")
                    else:
                        collected_data[f"{agent_type}_analysis"] = ""
                        logger.warning(f"No content found in {agent_type} result file")
                        
                except Exception as e:
                    logger.warning(f"Failed to read {agent_type} result file: {e}")
                    collected_data[f"{agent_type}_analysis"] = ""
            
            # Store processed files in collected_data for use in notifications
            collected_data["processed_files"] = processed_files
            return collected_data
            
        except Exception as e:
            logger.error(f"Error reading agent result files: {e}")
            return collected_data
    
    def _create_summary_prompt(self, collected_data: Dict[str, Any]) -> str:
        """Create a comprehensive summary prompt from collected analysis data"""
        prompt_parts = []
        prompt_parts.append("Please create a comprehensive summary of the following AI analysis results:")
        prompt_parts.append("")
        
        # Add NLP analysis
        if collected_data.get("nlp_analysis"):
            nlp_data = collected_data["nlp_analysis"]
            prompt_parts.append("## NLP Analysis Results:")
            
            # Handle both string content and dict data
            if isinstance(nlp_data, str):
                prompt_parts.append(nlp_data)
            elif isinstance(nlp_data, dict):
                prompt_parts.append(f"- Summary: {nlp_data.get('summary', 'N/A')}")
                prompt_parts.append(f"- Key Insights: {', '.join(nlp_data.get('key_insights', []))}")
                prompt_parts.append(f"- Recommendations: {', '.join(nlp_data.get('recommendations', []))}")
            prompt_parts.append("")
        
        # Add ASR transcript
        if collected_data.get("asr_analysis"):
            asr_data = collected_data["asr_analysis"]
            prompt_parts.append("## Voice Recording Transcript:")
            
            # Handle both string content and dict data
            if isinstance(asr_data, str):
                prompt_parts.append(asr_data)
            elif isinstance(asr_data, dict):
                prompt_parts.append(asr_data.get('transcript', 'N/A'))
            prompt_parts.append("")
        
        # Add document analysis
        if collected_data.get("document_analysis"):
            doc_data = collected_data["document_analysis"]
            prompt_parts.append("## Document Analysis Results:")
            
            # Handle both string content and dict data
            if isinstance(doc_data, str):
                prompt_parts.append(doc_data)
            elif isinstance(doc_data, dict):
                prompt_parts.append(f"- Extracted Text: {doc_data.get('extracted_text', 'N/A')[:500]}...")
                prompt_parts.append(f"- Analysis: {doc_data.get('analysis', 'N/A')}")
            prompt_parts.append("")
        
        prompt_parts.append("Please provide a comprehensive summary that:")
        prompt_parts.append("1. Synthesizes all the analysis results into a coherent overview")
        prompt_parts.append("2. Highlights the key findings and recommendations")
        prompt_parts.append("3. Provides actionable next steps for the user")
        prompt_parts.append("4. Maintains a professional and clear tone")
        prompt_parts.append("5. Keeps the summary concise but comprehensive (2-3 paragraphs)")
        prompt_parts.append("6. Focuses on the business requirements and technical approach for prototype development")
        
        return "\n".join(prompt_parts)
    
    async def health_check(self) -> bool:
        """Check Summarizer service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except:
            return False

class MultiAgentOrchestrator:
    """Enhanced orchestrator for multi-agent workflows"""
    
    def __init__(self):
        self.workflow_engine = workflow_engine
        self.persistence = workflow_persistence
        self.notification_service_url = os.getenv("NOTIFICATION_SERVICE_URL", "https://notifications.statex.cz")
        self.notification_client = NotificationServiceClient()
        
        # Register agents
        self._register_agents()
        
        # Register workflow definitions
        self._register_workflows()
    
    def _register_agents(self):
        """Register all AI agents with the workflow engine"""
        agents = [
            NLPAgent(),
            ASRAgent(),
            DocumentAgent(),
            PrototypeAgent(),
            SummarizerAgent(),
            GeminiAgent(),
            DataVizAgent()
        ]
        
        for agent in agents:
            self.workflow_engine.register_agent(agent)
    
    def _register_workflows(self):
        """Register workflow definitions"""
        self.workflow_engine.register_workflow("business_analysis", self._business_analysis_workflow)
        self.workflow_engine.register_workflow("document_processing", self._document_processing_workflow)
        self.workflow_engine.register_workflow("voice_analysis", self._voice_analysis_workflow)
        self.workflow_engine.register_workflow("prototype_generation", self._prototype_generation_workflow)
    
    async def process_multi_agent_request(self, request: MultiAgentRequest) -> WorkflowResult:
        """Process a multi-agent request through the workflow engine with comprehensive error handling"""
        workflow_state = None
        try:
            # Connect to persistence if not already connected
            if not self.persistence.redis_client:
                await self.persistence.connect()
            
            # Create workflow
            workflow_state = await self.workflow_engine.create_workflow(
                submission_id=request.submission_id,
                workflow_type=request.workflow_type,
                input_data=request.dict()
            )
            
            # Save initial state and create initial checkpoint
            await self.persistence.save_workflow_state(workflow_state)
            await workflow_recovery_manager.create_checkpoint(
                workflow_state,
                CheckpointType.WORKFLOW_PHASE,
                {"phase": "workflow_started", "request_type": request.workflow_type}
            )
            
            # Execute workflow with error handling
            start_time = datetime.now()
            
            try:
                completed_workflow = await self.workflow_engine.execute_workflow(workflow_state.workflow_id)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Create completion checkpoint
                await workflow_recovery_manager.create_checkpoint(
                    completed_workflow,
                    CheckpointType.WORKFLOW_PHASE,
                    {"phase": "workflow_completed", "processing_time": processing_time}
                )
                
            except Exception as workflow_error:
                # Handle workflow execution errors
                logger.error(f"Workflow execution failed for {request.submission_id}: {workflow_error}")
                
                # Try to recover the workflow
                context = ErrorContext(
                    submission_id=request.submission_id,
                    workflow_id=workflow_state.workflow_id,
                    operation="workflow_execution",
                    input_data=request.dict()
                )
                
                try:
                    # Attempt workflow recovery
                    recovered_workflow = await workflow_recovery_manager.recover_workflow(
                        workflow_state.workflow_id
                    )
                    
                    if recovered_workflow:
                        completed_workflow = recovered_workflow
                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info(f"Successfully recovered workflow {workflow_state.workflow_id}")
                    else:
                        raise workflow_error
                        
                except Exception as recovery_error:
                    logger.error(f"Workflow recovery failed: {recovery_error}")
                    raise workflow_error
            
            # Save final state
            await self.persistence.save_workflow_state(completed_workflow)
            
            # Generate business analysis and offer with error handling
            try:
                business_analysis_result = await self._generate_business_analysis(completed_workflow)
                offer_details = await self._generate_offer_details(request.submission_id, business_analysis_result)
            except Exception as analysis_error:
                logger.error(f"Business analysis generation failed: {analysis_error}")
                # Use fallback analysis
                business_analysis_result = await business_analysis_aggregator._create_fallback_analysis(completed_workflow)
                offer_details = await self._generate_simple_offer_fallback(request.submission_id, business_analysis_result)
            
            # Create result
            successful_agents = len([r for r in completed_workflow.agent_results.values() 
                                  if r.status == TaskStatus.COMPLETED])
            failed_agents = len([r for r in completed_workflow.agent_results.values() 
                               if r.status == TaskStatus.FAILED])
            
            result = WorkflowResult(
                submission_id=request.submission_id,
                workflow_id=completed_workflow.workflow_id,
                total_agents=len(completed_workflow.agent_results),
                successful_agents=successful_agents,
                failed_agents=failed_agents,
                processing_time=processing_time,
                business_analysis=business_analysis_result,
                offer_details=offer_details,
                agent_results=completed_workflow.agent_results
            )
            
            # Persist summary if summarizer agent completed successfully
            try:
                await self._persist_summary_if_available(request, completed_workflow)
            except Exception as summary_error:
                logger.error(f"Failed to persist summary: {summary_error}")
                # Don't fail the entire workflow for summary persistence errors
            
            # Send individual agent notifications
            try:
                await self._send_agent_notifications(request, completed_workflow)
            except Exception as notification_error:
                logger.error(f"Failed to send agent notifications: {notification_error}")
                # Don't fail the entire workflow for notification errors
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent processing failed for {request.submission_id}: {e}")
            
            # Mark workflow as failed if we have a workflow state
            if workflow_state:
                workflow_state.status = WorkflowStatus.FAILED
                workflow_state.end_time = datetime.now()
                await self.persistence.save_workflow_state(workflow_state)
                
                # Create failure checkpoint
                await workflow_recovery_manager.create_checkpoint(
                    workflow_state,
                    CheckpointType.ERROR_RECOVERY,
                    {"error": str(e), "failure_reason": "multi_agent_processing_failed"}
                )
            
            raise
    
    async def _business_analysis_workflow(self, input_data: Dict[str, Any]) -> List[AgentTask]:
        """Define the business analysis workflow tasks"""
        tasks = []
        
        # Extract session information for disk access
        submission_id = input_data.get("submission_id", "unknown")
        user_id = input_data.get("user_id", "unknown")
        session_id = input_data.get("session_id")
        
        # Check if files exist in session directory
        session_path = get_session_directory(submission_id, user_id, session_id)
        has_voice_files = False
        has_document_files = False
        
        if session_path:
            # Check for voice files
            voice_files = list_files_in_subdirectory(session_path, "files")
            audio_files = [f for f in voice_files if f.lower().endswith(('.webm', '.mp3', '.wav', '.m4a', '.ogg'))]
            has_voice_files = len(audio_files) > 0
            
            # Check for document files
            doc_files = [f for f in voice_files if f.lower().endswith(('.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'))]
            has_document_files = len(doc_files) > 0
            
            logger.info(f"Session {session_path}: voice_files={len(audio_files)}, doc_files={len(doc_files)}")
        
        # Task 1: Process voice if available
        if has_voice_files or input_data.get("voice_file_url"):
            asr_task = AgentTask(
                agent_type="asr",
                agent_name="ASR Agent",
                input_data={
                    "voice_file_url": input_data.get("voice_file_url", ""),
                    "submission_id": submission_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "read_from_disk": True
                },
                priority=2,
                timeout=60
            )
            tasks.append(asr_task)
        
        # Task 2: Process documents if available
        if has_document_files or input_data.get("file_urls"):
            doc_task = AgentTask(
                agent_type="document",
                agent_name="Document AI Agent",
                input_data={
                    "file_urls": input_data.get("file_urls", []),
                    "submission_id": submission_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "read_from_disk": True
                },
                priority=2,
                timeout=120
            )
            tasks.append(doc_task)
        
        # Task 3: NLP analysis (depends on ASR if voice is present)
        nlp_dependencies = []
        if has_voice_files and tasks:
            nlp_dependencies.append(tasks[0].task_id)  # ASR task
        
        nlp_task = AgentTask(
            agent_type="nlp",
            agent_name="NLP Analysis Agent",
            input_data={
                "text_content": input_data.get("text_content", ""),
                "requirements": input_data.get("requirements", ""),
                "submission_id": submission_id,
                "user_id": user_id,
                "session_id": session_id,
                "read_from_disk": True
            },
            priority=1,
            timeout=45,
            dependencies=nlp_dependencies
        )
        tasks.append(nlp_task)
        
        # Task 3.5: Gemini AI analysis (parallel to NLP)
        gemini_dependencies = []
        if has_voice_files and tasks:
            gemini_dependencies.append(tasks[0].task_id)  # ASR task
        
        gemini_task = AgentTask(
            agent_type="gemini",
            agent_name="Gemini AI Agent",
            input_data={
                "text_content": input_data.get("text_content", ""),
                "requirements": input_data.get("requirements", ""),
                "submission_id": submission_id,
                "user_id": user_id,
                "session_id": session_id,
                "read_from_disk": True
            },
            priority=1,
            timeout=45,
            dependencies=gemini_dependencies
        )
        tasks.append(gemini_task)
        
        # Task 4: Summarizer (depends on all analysis tasks)
        summarizer_dependencies = [nlp_task.task_id, gemini_task.task_id]
        if has_voice_files and tasks:
            summarizer_dependencies.append(tasks[0].task_id)  # ASR task
        if has_document_files and len(tasks) > 1:
            summarizer_dependencies.append(tasks[1].task_id)  # Document task
        
        summarizer_task = AgentTask(
            agent_type="summarizer",
            agent_name="Summarizer Agent",
            input_data={
                "collected_data": {
                    "nlp_analysis": {},  # Will be filled from NLP results
                    "asr_transcript": "",  # Will be filled from ASR results
                    "document_analysis": {},  # Will be filled from Document results
                },
                "submission_id": submission_id,
                "user_id": user_id,
                "session_id": session_id,
                "read_from_disk": True
            },
            priority=1,
            timeout=120,
            dependencies=summarizer_dependencies
        )
        tasks.append(summarizer_task)
        
        # Task 5: Data Visualization (depends on Summarizer)
        data_viz_task = AgentTask(
            agent_type="data_viz",
            agent_name="Data Visualization Agent",
            input_data={
                "submission_id": submission_id,
                "user_id": user_id,
                "session_id": session_id,
                "read_from_disk": True
            },
            priority=2,
            timeout=60,
            dependencies=[summarizer_task.task_id]
        )
        tasks.append(data_viz_task)
        
        # Task 6: Prototype generation (depends on Summarizer) - DISABLED UNTIL SUMMARIZER IS TESTED
        # prototype_task = AgentTask(
        #     agent_type="prototype",
        #     agent_name="Prototype Generator Agent",
        #     input_data={
        #         "requirements": input_data.get("requirements", ""),
        #         "analysis": {},  # Will be filled from NLP results
        #         "summary": {}  # Will be filled from Summarizer results
        #     },
        #     priority=1,
        #     timeout=180,
        #     dependencies=[summarizer_task.task_id]
        # )
        # tasks.append(prototype_task)
        
        return tasks
    
    async def _document_processing_workflow(self, input_data: Dict[str, Any]) -> List[AgentTask]:
        """Define document-focused workflow"""
        tasks = []
        
        if input_data.get("file_urls"):
            doc_task = AgentTask(
                agent_type="document",
                agent_name="Document AI Agent",
                input_data={"file_urls": input_data["file_urls"]},
                priority=1,
                timeout=120
            )
            tasks.append(doc_task)
            
            # Follow up with NLP analysis of extracted text
            nlp_task = AgentTask(
                agent_type="nlp",
                agent_name="NLP Analysis Agent",
                input_data={
                    "text_content": "",  # Will be filled from document results
                    "requirements": input_data.get("requirements", "")
                },
                priority=1,
                timeout=45,
                dependencies=[doc_task.task_id]
            )
            tasks.append(nlp_task)
        
        return tasks
    
    async def _voice_analysis_workflow(self, input_data: Dict[str, Any]) -> List[AgentTask]:
        """Define voice-focused workflow"""
        tasks = []
        
        if input_data.get("voice_file_url"):
            asr_task = AgentTask(
                agent_type="asr",
                agent_name="ASR Agent",
                input_data={"voice_file_url": input_data["voice_file_url"]},
                priority=1,
                timeout=60
            )
            tasks.append(asr_task)
            
            # Follow up with NLP analysis of transcription
            nlp_task = AgentTask(
                agent_type="nlp",
                agent_name="NLP Analysis Agent",
                input_data={
                    "text_content": "",  # Will be filled from ASR results
                    "requirements": input_data.get("requirements", "")
                },
                priority=1,
                timeout=45,
                dependencies=[asr_task.task_id]
            )
            tasks.append(nlp_task)
        
        return tasks
    
    async def _prototype_generation_workflow(self, input_data: Dict[str, Any]) -> List[AgentTask]:
        """Define prototype generation workflow tasks"""
        tasks = []
        
        # Task 1: Generate prototype using prototype generator
        prototype_task = AgentTask(
            agent_type="prototype_generator",
            agent_name="Prototype Generator Agent",
            input_data={
                "project_id": input_data.get("project_id"),
                "description": input_data.get("description"),
                "requirements": input_data.get("requirements"),
                "ai_analysis": input_data.get("ai_analysis", {}),
                "user_id": input_data.get("user_id"),
                "submission_id": input_data.get("submission_id")
            },
            priority=1,
            timeout=300  # 5 minutes for prototype generation
        )
        tasks.append(prototype_task)
        
        return tasks
    
    async def _generate_business_analysis(self, workflow_state: WorkflowState) -> BusinessAnalysisResult:
        """Generate comprehensive business analysis from agent results using the aggregator"""
        try:
            # Use the comprehensive business analysis aggregator
            analysis_result = await business_analysis_aggregator.aggregate_business_analysis(workflow_state)
            return analysis_result
        except Exception as e:
            logger.error(f"Failed to generate business analysis: {e}")
            # Fallback to basic analysis
            return await business_analysis_aggregator._create_fallback_analysis(workflow_state)
    
    async def _generate_offer_details(self, submission_id: str, analysis: BusinessAnalysisResult) -> OfferDetails:
        """Generate detailed business offer with comprehensive formatting"""
        try:
            # Generate unique project ID and URLs
            project_urls = await project_url_generator.create_and_validate_project_urls(
                submission_id=submission_id,
                user_id=submission_id  # Use submission_id as user_id for now
            )
            
            # Create comprehensive formatted offer
            formatted_offer = offer_formatter.format_comprehensive_offer(
                analysis=analysis,
                project_urls={
                    "plan_url": project_urls.plan_url,
                    "offer_url": project_urls.offer_url
                }
            )
            
            # Convert formatted offer to OfferDetails for compatibility
            pricing_tiers = []
            for tier in formatted_offer.pricing_tiers:
                pricing_tiers.append({
                    "name": tier.name,
                    "price": tier.price_range,
                    "features": tier.features,
                    "description": tier.description,
                    "timeline": tier.timeline,
                    "support_level": tier.support_level,
                    "recommended": tier.recommended
                })
            
            # Convert deliverables to simple list for compatibility
            deliverables = []
            for category in formatted_offer.deliverables:
                deliverables.extend(category.items)
            
            # Convert next steps to simple list
            next_steps = [step.action for step in formatted_offer.next_steps]
            
            offer = OfferDetails(
                project_id=project_urls.project_id,
                plan_url=project_urls.plan_url,
                offer_url=project_urls.offer_url,
                pricing_tiers=pricing_tiers,
                implementation_phases=formatted_offer.implementation_phases,
                deliverables=deliverables,
                next_steps=next_steps
            )
            
            return offer
        except Exception as e:
            logger.error(f"Error generating comprehensive offer: {e}")
            # Fallback to simple offer generation
            return await self._generate_simple_offer_fallback(submission_id, analysis)
    
    async def _generate_simple_offer_fallback(self, submission_id: str, analysis: BusinessAnalysisResult) -> OfferDetails:
        """Generate simple offer as fallback when comprehensive formatting fails"""
        try:
            project_urls = await project_url_generator.create_and_validate_project_urls(
                submission_id=submission_id,
                user_id=submission_id
            )
            
            return OfferDetails(
                project_id=project_urls.project_id,
                plan_url=project_urls.plan_url,
                offer_url=project_urls.offer_url,
                pricing_tiers=[{
                    "name": "Standard",
                    "price": analysis.budget_range,
                    "features": ["Complete solution", "Full support"],
                    "recommended": True
                }],
                implementation_phases=[{
                    "phase": "Development",
                    "duration": analysis.timeline_estimate,
                    "deliverables": ["Working software", "Documentation"]
                }],
                deliverables=[
                    "Complete source code",
                    "Technical documentation",
                    "3 months support"
                ],
                next_steps=[
                    "Schedule consultation",
                    "Review requirements",
                    "Begin development"
                ]
            )
        except Exception as e:
            logger.error(f"Error in fallback offer generation: {e}")
            raise
            
    
    async def _send_agent_notifications(self, request: MultiAgentRequest, workflow_state: WorkflowState):
        """Send individual notifications for each agent result"""
        try:
            for task_id, result in workflow_state.agent_results.items():
                # Create basic input data from request
                task_input_data = {
                    "submission_id": request.submission_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "text_content": request.text_content,
                    "voice_file_url": request.voice_file_url,
                    "file_urls": request.file_urls,
                    "read_from_disk": True
                }
                await self._send_single_agent_notification(request, result, task_input_data)
        except Exception as e:
            logger.error(f"Failed to send agent notifications: {e}")
    
    async def _send_single_agent_notification(self, request: MultiAgentRequest, result: AgentResult, task_input_data: Dict[str, Any] = None):
        """Send comprehensive notification for a single agent result"""
        try:
            message_parts = []
            message_parts.append(f"🤖 **{result.agent_name} Report**")
            message_parts.append(f"📋 Service: {result.agent_type}")
            message_parts.append(f"⏱️ Processing Time: {result.processing_time:.2f}s")
            message_parts.append(f"✅ Status: {result.status.value.upper()}")
            message_parts.append(f"🎯 Confidence: {result.confidence_score:.1%}")
            message_parts.append("")
            
            # Add input data information with agent-specific filtering
            message_parts.append("📥 **Input Data:**")
            if task_input_data:
                # Define allowed keys per agent type
                allowed_keys = {
                    "document": ["file_urls"],  # Only files
                    "nlp": ["text_content", "requirements"],  # Only text
                    "asr": ["voice_file_url"],  # Only voice
                    "gemini": ["text_content", "requirements"],  # Only text
                    "data_viz": [],  # Shows nothing (reads from disk)
                    "summarizer": [],  # Shows nothing (reads from disk)
                    "prototype": []  # Shows nothing (reads from disk)
                }
                
                # Filter input data based on agent type
                agent_allowed_keys = allowed_keys.get(result.agent_type, task_input_data.keys())
                
                for key, value in task_input_data.items():
                    if key in ["submission_id", "user_id", "read_from_disk"]:
                        continue  # Skip internal fields
                    elif agent_allowed_keys and key not in agent_allowed_keys:
                        continue  # Skip non-relevant fields for this agent
                    elif key == "file_urls" and result.agent_type == "document":
                        # For document agent, show original filenames from submission service
                        logger.info(f"Processing file_urls for document agent: {value}")
                        if isinstance(value, list):
                            # Try to get original filenames from submission service
                            session_id = task_input_data.get("session_id")
                            logger.info(f"Getting original filenames for session_id: {session_id}")
                            original_filenames = await self._get_original_filenames(session_id)
                            if original_filenames:
                                # Filter to only show document files (PDF, DOC, DOCX, etc.)
                                doc_filenames = [f for f in original_filenames if f.lower().endswith(('.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'))]
                                if doc_filenames:
                                    logger.info(f"Using filtered document filenames: {doc_filenames}")
                                    message_parts.append(f"• files: {', '.join(doc_filenames)}")
                                else:
                                    logger.warning("No document files found in original filenames")
                                    message_parts.append(f"• files: No document files found")
                            else:
                                logger.warning("No original filenames found, filtering from file URLs")
                                # Fallback to extracting from file URLs and filtering
                                doc_filenames = []
                                for file_url in value:
                                    if isinstance(file_url, str):
                                        filename = file_url.split("/")[-1]
                                        if filename and filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt')):
                                            doc_filenames.append(filename)
                                if doc_filenames:
                                    logger.info(f"Using filtered fallback filenames: {doc_filenames}")
                                    message_parts.append(f"• files: {', '.join(doc_filenames)}")
                                else:
                                    logger.warning(f"No document files found in file_urls: {value}")
                                    message_parts.append(f"• files: No document files found")
                        else:
                            message_parts.append(f"• {key}: {value}")
                    elif key == "voice_file_url" and result.agent_type == "asr":
                        # For ASR agent, show actual voice file info if processed from disk
                        if task_input_data.get("read_from_disk") and not value:
                            # Voice file was processed from disk, show that info
                            submission_id = task_input_data.get("submission_id")
                            user_id = task_input_data.get("user_id")
                            session_id = task_input_data.get("session_id")
                            session_path = get_session_directory(submission_id, user_id, session_id)
                            
                            if session_path:
                                # List voice files in the files/ subdirectory
                                voice_files = list_files_in_subdirectory(session_path, "files")
                                audio_files = [f for f in voice_files if f.lower().endswith(('.webm', '.mp3', '.wav', '.m4a', '.ogg'))]
                                
                                if audio_files:
                                    message_parts.append(f"• voice_file_url: Processed from disk: {audio_files[0]}")
                                else:
                                    message_parts.append(f"• voice_file_url: None (no voice files found on disk)")
                            else:
                                message_parts.append(f"• voice_file_url: None (session directory not found)")
                        else:
                            message_parts.append(f"• voice_file_url: {value}")
                    elif isinstance(value, str) and len(value) > 200:
                        message_parts.append(f"• {key}: {value[:200]}...")
                    elif isinstance(value, list) and len(value) > 0:
                        message_parts.append(f"• {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        message_parts.append(f"• {key}: {len(value)} fields")
                    else:
                        message_parts.append(f"• {key}: {value}")
            else:
                message_parts.append("• No input data available")
            message_parts.append("")
            
            # Add comprehensive output data
            if result.status == TaskStatus.COMPLETED and result.result_data:
                message_parts.append("📤 **Results:**")
                
                # Special handling for different agent types
                if result.agent_type == "nlp":
                    self._format_nlp_results(message_parts, result.result_data)
                elif result.agent_type == "asr":
                    self._format_asr_results(message_parts, result.result_data)
                elif result.agent_type == "document":
                    self._format_document_results(message_parts, result.result_data)
                elif result.agent_type == "gemini":
                    self._format_gemini_results(message_parts, result.result_data)
                elif result.agent_type == "data_viz":
                    self._format_data_viz_results(message_parts, result.result_data)
                elif result.agent_type == "summarizer":
                    self._format_summarizer_results(message_parts, result.result_data)
                elif result.agent_type == "prototype":
                    self._format_prototype_results(message_parts, result.result_data)
                else:
                    # Default formatting
                    for key, value in result.result_data.items():
                        if isinstance(value, str) and len(value) > 100:
                            message_parts.append(f"• {key}: {value[:100]}...")
                        elif isinstance(value, dict):
                            message_parts.append(f"• {key}: {len(value)} items")
                        elif isinstance(value, list):
                            message_parts.append(f"• {key}: {len(value)} items")
                        else:
                            message_parts.append(f"• {key}: {value}")
            
            if result.error_message:
                message_parts.append(f"❌ **Error:** {result.error_message}")
            
            notification_message = "\n".join(message_parts)
            
            # Send notification via notification service
            async with self.notification_client as client:
                await client.send_notification({
                    "user_id": request.user_id,
                    "type": "agent_completion",
                    "title": f"🤖 {result.agent_name} Completed",
                    "message": notification_message,
                    "contact_type": "telegram",
                    "contact_value": request.contact_info.get("email", "694579866"),
                    "user_name": request.contact_info.get("name", "User"),
                    "parse_mode": "Markdown"
                })
                
        except Exception as e:
            logger.error(f"Failed to send notification for {result.agent_name}: {e}")
    
    def _format_nlp_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format NLP agent results for notification"""
        for key, value in result_data.items():
            if key == "analysis" and isinstance(value, dict):
                # Show summary prominently
                if "summary" in value and value["summary"]:
                    summary = value["summary"]
                    if len(summary) > 200:
                        message_parts.append(f"• **Summary:** {summary[:200]}...")
                    else:
                        message_parts.append(f"• **Summary:** {summary}")
                
                # Show key insights
                if "key_insights" in value and value["key_insights"]:
                    insights = value["key_insights"]
                    if isinstance(insights, list) and insights:
                        message_parts.append(f"• **Key Insights:** {len(insights)} items")
                        for i, insight in enumerate(insights[:3]):  # Show first 3 insights
                            if isinstance(insight, str) and len(insight) > 80:
                                message_parts.append(f"  - {insight[:80]}...")
                            else:
                                message_parts.append(f"  - {insight}")
                    elif isinstance(insights, str):
                        message_parts.append(f"• **Key Insights:** {insights[:100]}...")
                
                # Show recommendations
                if "recommendations" in value and value["recommendations"]:
                    recommendations = value["recommendations"]
                    if isinstance(recommendations, list) and recommendations:
                        message_parts.append(f"• **Recommendations:** {len(recommendations)} items")
                        for i, rec in enumerate(recommendations[:2]):  # Show first 2 recommendations
                            if isinstance(rec, str) and len(rec) > 80:
                                message_parts.append(f"  - {rec[:80]}...")
                            else:
                                message_parts.append(f"  - {rec}")
                    elif isinstance(recommendations, str):
                        message_parts.append(f"• **Recommendations:** {recommendations[:100]}...")
                
                # Show business type
                if "business_type" in value and value["business_type"]:
                    message_parts.append(f"• **Business Type:** {value['business_type']}")
                
                # Show other important fields
                for sub_key in ["pain_points", "opportunities", "next_steps"]:
                    if sub_key in value and value[sub_key]:
                        sub_value = value[sub_key]
                        if isinstance(sub_value, list) and sub_value:
                            message_parts.append(f"• **{sub_key.replace('_', ' ').title()}:** {len(sub_value)} items")
                        elif isinstance(sub_value, str):
                            message_parts.append(f"• **{sub_key.replace('_', ' ').title()}:** {sub_value[:50]}...")
            elif key in ["tokens_used", "model_used", "ai_service", "confidence", "processing_time"]:
                message_parts.append(f"• {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    message_parts.append(f"• {key}: {value[:100]}...")
                else:
                    message_parts.append(f"• {key}: {value}")
    
    def _format_asr_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format ASR agent results for notification"""
        for key, value in result_data.items():
            if key == "results" and isinstance(value, dict):
                message_parts.append(f"• {key}: {len(value)} fields")
                for sub_key, sub_value in value.items():
                    if sub_key == "transcript" and isinstance(sub_value, str):
                        message_parts.append(f"  - {sub_key}: {sub_value[:150]}...")
                    elif sub_key in ["confidence", "language", "duration"]:
                        message_parts.append(f"  - {sub_key}: {sub_value}")
                    else:
                        message_parts.append(f"  - {sub_key}: {str(sub_value)[:50]}...")
            elif key in ["tokens_used", "model_used", "ai_service", "confidence", "processing_time"]:
                message_parts.append(f"• {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    message_parts.append(f"• {key}: {value[:100]}...")
                else:
                    message_parts.append(f"• {key}: {value}")
    
    def _format_document_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format Document agent results for notification"""
        for key, value in result_data.items():
            if key == "results" and isinstance(value, dict):
                message_parts.append(f"• {key}: {len(value)} fields")
                for sub_key, sub_value in value.items():
                    if sub_key == "extracted_text" and isinstance(sub_value, str):
                        message_parts.append(f"  - {sub_key}: {sub_value[:150]}...")
                    elif sub_key in ["document_type", "page_count", "word_count"]:
                        message_parts.append(f"  - {sub_key}: {sub_value}")
                    elif sub_key == "documents" and isinstance(sub_value, list):
                        # Show original filenames from submission service
                        # Note: This would need submission_id context to work properly
                        # For now, show stored filenames as fallback
                        filenames = []
                        for doc in sub_value:
                            if isinstance(doc, dict):
                                file_url = doc.get("file_url", "")
                                # Extract filename from URL or use file_path
                                if file_url:
                                    filename = file_url.split("/")[-1].split("?")[0]  # Remove query params
                                    if filename:
                                        filenames.append(filename)
                                elif doc.get("file_path"):
                                    filename = doc["file_path"].split("/")[-1]
                                    filenames.append(filename)
                        if filenames:
                            message_parts.append(f"  - processed_files: {', '.join(filenames)}")
                        else:
                            message_parts.append(f"  - {sub_key}: {len(sub_value)} documents")
                    else:
                        message_parts.append(f"  - {sub_key}: {str(sub_value)[:50]}...")
            elif key in ["tokens_used", "model_used", "ai_service", "confidence", "processing_time"]:
                message_parts.append(f"• {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    message_parts.append(f"• {key}: {value[:100]}...")
                else:
                    message_parts.append(f"• {key}: {value}")
    
    def _format_gemini_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format Gemini agent results for notification"""
        for key, value in result_data.items():
            if key == "analysis" and isinstance(value, dict):
                # Show summary prominently
                if "summary" in value and value["summary"]:
                    summary = value["summary"]
                    if len(summary) > 200:
                        message_parts.append(f"• **Summary:** {summary[:200]}...")
                    else:
                        message_parts.append(f"• **Summary:** {summary}")
                
                # Show key insights
                if "key_insights" in value and value["key_insights"]:
                    insights = value["key_insights"]
                    if isinstance(insights, list) and insights:
                        message_parts.append(f"• **Key Insights:** {len(insights)} items")
                        for i, insight in enumerate(insights[:3]):  # Show first 3 insights
                            if isinstance(insight, str) and len(insight) > 80:
                                message_parts.append(f"  - {insight[:80]}...")
                            else:
                                message_parts.append(f"  - {insight}")
                    elif isinstance(insights, str):
                        message_parts.append(f"• **Key Insights:** {insights[:100]}...")
                
                # Show recommendations
                if "recommendations" in value and value["recommendations"]:
                    recommendations = value["recommendations"]
                    if isinstance(recommendations, list) and recommendations:
                        message_parts.append(f"• **Recommendations:** {len(recommendations)} items")
                        for i, rec in enumerate(recommendations[:2]):  # Show first 2 recommendations
                            if isinstance(rec, str) and len(rec) > 80:
                                message_parts.append(f"  - {rec[:80]}...")
                            else:
                                message_parts.append(f"  - {rec}")
                    elif isinstance(recommendations, str):
                        message_parts.append(f"• **Recommendations:** {recommendations[:100]}...")
                
                # Show business type
                if "business_type" in value and value["business_type"]:
                    message_parts.append(f"• **Business Type:** {value['business_type']}")
                
                # Show other important fields
                for sub_key in ["pain_points", "opportunities", "next_steps"]:
                    if sub_key in value and value[sub_key]:
                        sub_value = value[sub_key]
                        if isinstance(sub_value, list) and sub_value:
                            message_parts.append(f"• **{sub_key.replace('_', ' ').title()}:** {len(sub_value)} items")
                        elif isinstance(sub_value, str):
                            message_parts.append(f"• **{sub_key.replace('_', ' ').title()}:** {sub_value[:50]}...")
            elif key in ["tokens_used", "model_used", "ai_service", "confidence", "processing_time"]:
                message_parts.append(f"• {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    message_parts.append(f"• {key}: {value[:100]}...")
                else:
                    message_parts.append(f"• {key}: {value}")
    
    def _format_data_viz_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format DataViz agent results for notification"""
        for key, value in result_data.items():
            if key == "visualization" and isinstance(value, dict):
                # Show file path
                if "file_path" in value and value["file_path"]:
                    file_path = value["file_path"]
                    filename = file_path.split("/")[-1] if "/" in file_path else file_path
                    message_parts.append(f"• **Visualization File:** {filename}")
                
                # Show content length
                if "content_length" in value:
                    message_parts.append(f"• **Content Length:** {value['content_length']} characters")
                
                # Show charts created
                if "charts_created" in value and value["charts_created"]:
                    charts = value["charts_created"]
                    if isinstance(charts, list) and charts:
                        message_parts.append(f"• **Charts Created:** {len(charts)} charts")
                        for i, chart in enumerate(charts[:3]):  # Show first 3 charts
                            message_parts.append(f"  - {chart}")
                    elif isinstance(charts, str):
                        message_parts.append(f"• **Charts Created:** {charts}")
                
                # Show insights
                if "insights" in value and value["insights"]:
                    insights = value["insights"]
                    if isinstance(insights, list) and insights:
                        message_parts.append(f"• **Data Insights:** {len(insights)} insights")
                        for i, insight in enumerate(insights[:2]):  # Show first 2 insights
                            if isinstance(insight, str) and len(insight) > 80:
                                message_parts.append(f"  - {insight[:80]}...")
                            else:
                                message_parts.append(f"  - {insight}")
                    elif isinstance(insights, str):
                        message_parts.append(f"• **Data Insights:** {insights[:100]}...")
            
            elif key == "data_points" and isinstance(value, dict):
                # Show data points summary
                total_points = sum(len(v) for v in value.values() if isinstance(v, list))
                message_parts.append(f"• **Data Points Analyzed:** {total_points}")
                
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and sub_value:
                        message_parts.append(f"  - {sub_key.replace('_', ' ').title()}: {len(sub_value)} items")
            
            elif key == "ai_analysis" and isinstance(value, dict):
                # Show AI analysis summary
                if "data_insights" in value and value["data_insights"]:
                    insights = value["data_insights"]
                    if isinstance(insights, list):
                        message_parts.append(f"• **AI Data Insights:** {len(insights)} insights")
                    else:
                        message_parts.append(f"• **AI Data Insights:** {insights[:50]}...")
                
                if "visualization_suggestions" in value and value["visualization_suggestions"]:
                    suggestions = value["visualization_suggestions"]
                    if isinstance(suggestions, list):
                        message_parts.append(f"• **Visualization Suggestions:** {len(suggestions)} types")
                    else:
                        message_parts.append(f"• **Visualization Suggestions:** {suggestions[:50]}...")
            
            elif key in ["processing_time", "ai_service"]:
                message_parts.append(f"• {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    message_parts.append(f"• {key}: {value[:100]}...")
                else:
                    message_parts.append(f"• {key}: {value}")
    
    def _format_summarizer_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format Summarizer agent results for notification"""
        # Show the actual files that were processed by the summarizer
        processed_files = result_data.get("processed_files", [])
        if processed_files:
            message_parts.append(f"• processed_files: {', '.join(processed_files)}")
        else:
            message_parts.append("• processed_files: No files processed")
        
        # Show AI service information
        for key, value in result_data.items():
            if key in ["tokens_used", "model_used", "ai_service", "confidence", "processing_time"]:
                message_parts.append(f"• {key}: {value}")
            elif key == "summary_text" and isinstance(value, str):
                message_parts.append(f"• {key}: {value[:200]}...")
            elif key == "collected_insights" and isinstance(value, dict):
                message_parts.append(f"• {key}: {len(value)} sources")
                for sub_key, sub_value in value.items():
                    if sub_value:
                        message_parts.append(f"  - {sub_key}: {len(str(sub_value))} chars")
    
    def _format_prototype_results(self, message_parts: List[str], result_data: Dict[str, Any]):
        """Format Prototype agent results for notification"""
        for key, value in result_data.items():
            if key == "results" and isinstance(value, dict):
                message_parts.append(f"• {key}: {len(value)} fields")
                for sub_key, sub_value in value.items():
                    if sub_key == "deployment" and isinstance(sub_value, dict):
                        message_parts.append(f"  - {sub_key}: {{'url': '[View Prototype]', 'status': 'ready', 'deploymenttime': '2-3 minutes'}}")
                    elif isinstance(sub_value, str) and len(sub_value) > 50:
                        message_parts.append(f"  - {sub_key}: {sub_value[:50]}...")
                    elif isinstance(sub_value, list):
                        message_parts.append(f"  - {sub_key}: {len(sub_value)} items")
                    else:
                        message_parts.append(f"  - {sub_key}: {sub_value}")
            elif key in ["tokens_used", "model_used", "ai_service", "confidence", "processing_time"]:
                message_parts.append(f"• {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    message_parts.append(f"• {key}: {value[:100]}...")
                else:
                    message_parts.append(f"• {key}: {value}")
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow status"""
        return await self.workflow_engine.get_workflow_status(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        return await self.workflow_engine.cancel_workflow(workflow_id)
    
    async def _get_original_filenames(self, session_id: str) -> List[str]:
        """Get original filenames from submission service for a given session_id"""
        try:
            logger.info(f"Attempting to get original filenames for session_id: {session_id}")
            if not session_id:
                logger.warning("No session_id provided for filename mapping")
                return []
            
            submission_port = os.getenv("SUBMISSION_SERVICE_INTERNAL_PORT", "8000")
            submission_service_url = os.getenv("SUBMISSION_SERVICE_URL", f"http://localhost:{submission_port}")
            logger.info(f"Using submission service URL: {submission_service_url}")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{submission_service_url}/api/sessions/{session_id}/filename-mappings"
                logger.info(f"Making request to: {url}")
                
                response = await client.get(url)
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Response data: {data}")
                    mappings = data.get("mappings", [])
                    original_filenames = [mapping.get("original_name", "") for mapping in mappings if mapping.get("original_name")]
                    logger.info(f"Extracted original filenames: {original_filenames}")
                    return original_filenames
                else:
                    logger.warning(f"Failed to get filename mappings for session {session_id}: {response.status_code} - {response.text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting original filenames for session {session_id}: {e}")
            return []

    async def _persist_summary_if_available(self, request: MultiAgentRequest, workflow_state: WorkflowState):
        """Persist summary to submission service if summarizer agent completed successfully"""
        # Find summarizer agent result
        summarizer_result = None
        for task_id, result in workflow_state.agent_results.items():
            if result.agent_type == "summarizer" and result.status == TaskStatus.COMPLETED:
                summarizer_result = result
                break
        
        if not summarizer_result:
            logger.info(f"No summarizer result found for submission {request.submission_id}")
            return
        
        # Extract summary text and model information from the result
        summary_text = summarizer_result.result_data.get("summary_text", "")
        if not summary_text:
            logger.warning(f"No summary text found in summarizer result for submission {request.submission_id}")
            return
        
        model_used = summarizer_result.result_data.get("model_used", "unknown")
        tokens_used = summarizer_result.result_data.get("tokens_used", 0)
        processing_time = summarizer_result.result_data.get("processing_time", 0)
        
        # Call submission service to persist the summary
        submission_port = os.getenv("SUBMISSION_SERVICE_INTERNAL_PORT", "8000")
        submission_service_url = os.getenv("SUBMISSION_SERVICE_URL", f"http://localhost:{submission_port}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{submission_service_url}/api/submissions/{request.submission_id}/summary",
                    json={
                        "summary": summary_text,
                        "model_used": model_used,
                        "tokens_used": tokens_used,
                        "processing_time": processing_time
                    }
                )
                response.raise_for_status()
                logger.info(f"Successfully persisted summary for submission {request.submission_id} using model {model_used}")
        except Exception as e:
            logger.error(f"Failed to persist summary for submission {request.submission_id}: {e}")
            raise

# Global orchestrator instance
multi_agent_orchestrator = MultiAgentOrchestrator()