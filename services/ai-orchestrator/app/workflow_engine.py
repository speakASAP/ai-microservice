"""
Multi-Agent Workflow Engine

Implements workflow orchestration using LangChain and asyncio for parallel task execution.
Provides base classes for agent coordination and workflow state management.
"""

import asyncio
import importlib.util
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from pydantic import BaseModel, Field
import json
import time
import httpx

# Centralized logger for all logs to logging-microservice
_workflow_shared = Path(__file__).resolve().parent.parent.parent.parent / "shared"
if not _workflow_shared.exists():
    _workflow_shared = Path("/app") / "shared"
_logger_spec = importlib.util.spec_from_file_location("logger", str(_workflow_shared / "logger.py"))
_logger_module = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(_logger_module)
logger = _logger_module.setup_logger(__name__, service_name="ai-orchestrator")

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class AgentTask(BaseModel):
    """Represents a task to be executed by an AI agent"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: str
    agent_name: str
    input_data: Dict[str, Any]
    priority: int = Field(default=1, ge=1, le=10)
    timeout: int = Field(default=30, ge=1)  # seconds
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=2, ge=0)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)  # task_ids this task depends on

class AgentResult(BaseModel):
    """Represents the result of an agent task execution"""
    task_id: str
    agent_type: str
    agent_name: str
    status: TaskStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class WorkflowState(BaseModel):
    """Manages the state of a workflow execution"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str
    workflow_type: str
    status: WorkflowStatus = WorkflowStatus.CREATED
    current_step: str = ""
    completed_steps: List[str] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    running_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    input_data: Dict[str, Any] = Field(default_factory=dict)

class AgentInterface(ABC):
    """Abstract base class for AI agents"""
    
    def __init__(self, agent_name: str, agent_type: str):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.is_healthy = True
        self.last_health_check = datetime.now()
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a task and return the result"""
        pass
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and available"""
        try:
            # Default implementation - can be overridden by specific agents
            self.last_health_check = datetime.now()
            return True
        except Exception as e:
            logger.error("Workflow engine agent health check failed", agent_name=self.agent_name, agent_type=self.agent_type, error=str(e))
            self.is_healthy = False
            return False
    
    async def _save_agent_result(self, task: AgentTask, result_data: Dict[str, Any]) -> None:
        """Save agent result to submission service"""
        try:
            submission_id = task.input_data.get("submission_id")
            if not submission_id:
                logger.warning("Workflow engine _save_agent_result: no submission_id", agent_type=self.agent_type, task_id=task.task_id)
                return
            
            # Get submission service URL
            submission_host = os.getenv("SUBMISSION_SERVICE_HOST", "submission-service")
            submission_port = os.getenv("SUBMISSION_SERVICE_INTERNAL_PORT", os.getenv("SUBMISSION_SERVICE_PORT", "8000"))
            submission_service_url = os.getenv("SUBMISSION_SERVICE_URL", f"http://{submission_host}:{submission_port}")
            
            # Extract model information from result
            analysis = result_data.get("analysis", {})
            model_used = analysis.get("model_used", result_data.get("model_used", "unknown"))
            tokens_used = result_data.get("tokens_used", 0)
            processing_time = result_data.get("processing_time", 0)
            
            # Save to submission service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{submission_service_url}/api/submissions/{submission_id}/agent-result",
                    json={
                        "agent_type": self.agent_type,
                        "result_data": result_data,
                        "model_used": model_used,
                        "tokens_used": tokens_used,
                        "processing_time": processing_time
                    }
                )
                
                if response.status_code == 200:
                    logger.info("Workflow engine _save_agent_result success", agent_type=self.agent_type, submission_id=submission_id)
                else:
                    logger.warning("Workflow engine _save_agent_result failed", agent_type=self.agent_type, submission_id=submission_id, status_code=response.status_code)

        except Exception as e:
            logger.error("Workflow engine _save_agent_result error", agent_type=self.agent_type, submission_id=task.input_data.get("submission_id"), error=str(e))

class WorkflowEngine:
    """Main workflow engine for orchestrating multi-agent tasks"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.agents: Dict[str, AgentInterface] = {}
        self.workflows: Dict[str, WorkflowState] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.workflow_definitions: Dict[str, Callable] = {}
        self._shutdown = False
        
    def register_agent(self, agent: AgentInterface):
        """Register an AI agent with the workflow engine"""
        self.agents[agent.agent_type] = agent
        logger.info("Workflow engine registered agent", agent_name=agent.agent_name, agent_type=agent.agent_type)
    
    def register_workflow(self, workflow_type: str, workflow_func: Callable):
        """Register a workflow definition"""
        self.workflow_definitions[workflow_type] = workflow_func
        logger.info("Workflow engine registered workflow", workflow_type=workflow_type)
    
    async def create_workflow(self, submission_id: str, workflow_type: str, 
                            input_data: Dict[str, Any]) -> WorkflowState:
        """Create a new workflow instance"""
        workflow_state = WorkflowState(
            submission_id=submission_id,
            workflow_type=workflow_type,
            metadata=input_data
        )
        
        self.workflows[workflow_state.workflow_id] = workflow_state
        logger.info("Workflow engine create_workflow", workflow_id=workflow_state.workflow_id, submission_id=submission_id, workflow_type=workflow_type)

        return workflow_state
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowState:
        """Execute a workflow by its ID"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow_state = self.workflows[workflow_id]
        workflow_func = self.workflow_definitions.get(workflow_state.workflow_type)

        if not workflow_func:
            raise ValueError(f"Workflow type {workflow_state.workflow_type} not registered")

        logger.info("Workflow engine execute_workflow started", workflow_id=workflow_id, submission_id=workflow_state.submission_id, workflow_type=workflow_state.workflow_type)
        start_sec = time.time()
        try:
            workflow_state.status = WorkflowStatus.RUNNING
            workflow_state.start_time = datetime.now()

            # Execute the workflow function
            tasks = await workflow_func(workflow_state.metadata)
            logger.info("Workflow engine workflow_func returned tasks", workflow_id=workflow_id, task_count=len(tasks), task_agent_types=[t.agent_type for t in tasks])

            # Execute tasks in parallel with dependency management
            await self._execute_tasks_with_dependencies(workflow_id, tasks)

            # Update workflow status
            workflow_state.status = WorkflowStatus.COMPLETED
            workflow_state.end_time = datetime.now()
            workflow_state.progress = 1.0
            duration_sec = time.time() - start_sec
            logger.info("Workflow engine execute_workflow completed", workflow_id=workflow_id, submission_id=workflow_state.submission_id, status="COMPLETED", duration_sec=round(duration_sec, 2), completed_tasks=len(workflow_state.completed_tasks), failed_tasks=len(workflow_state.failed_tasks))

        except Exception as e:
            duration_sec = time.time() - start_sec
            logger.error("Workflow engine execute_workflow failed", workflow_id=workflow_id, submission_id=workflow_state.submission_id, error=str(e), duration_sec=round(duration_sec, 2))
            workflow_state.status = WorkflowStatus.FAILED
            workflow_state.end_time = datetime.now()

        return workflow_state
    
    async def _execute_tasks_with_dependencies(self, workflow_id: str, tasks: List[AgentTask]):
        """Execute tasks in parallel while respecting dependencies"""
        workflow_state = self.workflows[workflow_id]
        task_map = {task.task_id: task for task in tasks}
        
        # Track task completion
        completed_tasks = set()
        running_tasks = {}
        
        while len(completed_tasks) < len(tasks):
            # Find tasks that can be executed (dependencies satisfied)
            ready_tasks = []
            for task in tasks:
                if (task.task_id not in completed_tasks and 
                    task.task_id not in running_tasks and
                    all(dep_id in completed_tasks for dep_id in task.dependencies)):
                    ready_tasks.append(task)
            
            # Start new tasks up to the concurrency limit
            while (ready_tasks and 
                   len(running_tasks) < self.max_concurrent_tasks):
                task = ready_tasks.pop(0)
                
                # Start the task
                task_coroutine = self._execute_single_task(workflow_id, task)
                running_task = asyncio.create_task(task_coroutine)
                running_tasks[task.task_id] = running_task

                workflow_state.running_tasks.append(task.task_id)
                logger.info("Workflow engine task started", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=workflow_state.submission_id)
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task_future in done:
                    # Find which task completed
                    completed_task_id = None
                    for task_id, future in running_tasks.items():
                        if future == task_future:
                            completed_task_id = task_id
                            break
                    
                    if completed_task_id:
                        completed_tasks.add(completed_task_id)
                        del running_tasks[completed_task_id]
                        
                        # Update workflow state
                        workflow_state.running_tasks.remove(completed_task_id)
                        workflow_state.completed_tasks.append(completed_task_id)
                        workflow_state.progress = len(completed_tasks) / len(tasks)

                        logger.info("Workflow engine task completed", workflow_id=workflow_id, task_id=completed_task_id, submission_id=workflow_state.submission_id, progress_pct=round(100 * len(completed_tasks) / len(tasks), 1))
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
    
    async def _execute_single_task(self, workflow_id: str, task: AgentTask) -> AgentResult:
        """Execute a single agent task with timeout and retry logic"""
        workflow_state = self.workflows[workflow_id]
        submission_id = workflow_state.submission_id
        logger.info("Workflow engine _execute_single_task started", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=submission_id, timeout=task.timeout)
        agent = self.agents.get(task.agent_type)

        if not agent:
            error_msg = f"Agent type {task.agent_type} not found"
            logger.error("Workflow engine _execute_single_task agent not found", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=submission_id)
            result = AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                agent_name=task.agent_name,
                status=TaskStatus.FAILED,
                error_message=error_msg
            )
            workflow_state.agent_results[task.task_id] = result
            workflow_state.failed_tasks.append(task.task_id)
            return result
        
        # Check agent health
        if not await agent.health_check():
            error_msg = f"Agent {task.agent_name} is not healthy"
            logger.warning("Workflow engine _execute_single_task agent unhealthy", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=submission_id)
            result = AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                agent_name=task.agent_name,
                status=TaskStatus.FAILED,
                error_message=error_msg
            )
            workflow_state.agent_results[task.task_id] = result
            workflow_state.failed_tasks.append(task.task_id)
            return result
        
        # Execute task with retry logic
        for attempt in range(task.max_retries + 1):
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                task.retry_count = attempt
                
                start_time = time.time()
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.execute_task(task),
                    timeout=task.timeout
                )
                
                result.processing_time = time.time() - start_time
                result.status = TaskStatus.COMPLETED
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                workflow_state.agent_results[task.task_id] = result
                duration_sec = time.time() - start_time
                logger.info("Workflow engine _execute_single_task success", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=submission_id, duration_sec=round(duration_sec, 2))
                return result

            except asyncio.TimeoutError:
                error_msg = f"Task {task.task_id} timed out after {task.timeout}s"
                logger.error("Workflow engine _execute_single_task timeout", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=submission_id, attempt=attempt + 1, max_retries=task.max_retries + 1, timeout=task.timeout)
                
                if attempt == task.max_retries:
                    result = AgentResult(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        agent_name=task.agent_name,
                        status=TaskStatus.TIMEOUT,
                        error_message=error_msg,
                        processing_time=task.timeout
                    )
                    task.status = TaskStatus.TIMEOUT
                    workflow_state.agent_results[task.task_id] = result
                    workflow_state.failed_tasks.append(task.task_id)
                    return result
                
            except Exception as e:
                error_msg = f"Task {task.task_id} failed: {str(e)}"
                logger.error("Workflow engine _execute_single_task exception", workflow_id=workflow_id, task_id=task.task_id, agent_type=task.agent_type, submission_id=submission_id, error=str(e), attempt=attempt + 1, max_retries=task.max_retries + 1)
                
                if attempt == task.max_retries:
                    result = AgentResult(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        agent_name=task.agent_name,
                        status=TaskStatus.FAILED,
                        error_message=error_msg,
                        processing_time=time.time() - start_time if 'start_time' in locals() else 0
                    )
                    task.status = TaskStatus.FAILED
                    workflow_state.agent_results[task.task_id] = result
                    workflow_state.failed_tasks.append(task.task_id)
                    return result
            
            # Wait before retry
            if attempt < task.max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get the current status of a workflow"""
        return self.workflows.get(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id not in self.workflows:
            return False
        
        workflow_state = self.workflows[workflow_id]
        
        if workflow_state.status == WorkflowStatus.RUNNING:
            workflow_state.status = WorkflowStatus.CANCELLED
            workflow_state.end_time = datetime.now()
            
            # Cancel running tasks
            for task_id in workflow_state.running_tasks:
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
            
            logger.info("Workflow engine cancel_workflow", workflow_id=workflow_id)
            return True
        
        return False
    
    async def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        workflows_to_remove = []
        for workflow_id, workflow_state in self.workflows.items():
            if (workflow_state.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                workflow_state.end_time and workflow_state.end_time < cutoff_time):
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.workflows[workflow_id]
            logger.info("Workflow engine cleanup_completed_workflows removed", workflow_id=workflow_id)
        
        return len(workflows_to_remove)

# Global workflow engine instance
workflow_engine = WorkflowEngine()