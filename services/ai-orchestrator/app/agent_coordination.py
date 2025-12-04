"""
Agent Task Coordination System

Implements failure detection and graceful degradation when agents fail.
Provides comprehensive coordination for multi-agent workflows.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pydantic import BaseModel
import time
import statistics

from .workflow_engine import AgentTask, AgentResult, AgentInterface, TaskStatus

logger = logging.getLogger(__name__)

class AgentHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class FailureType(str, Enum):
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_ERROR = "service_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"

class AgentMetrics(BaseModel):
    """Metrics for agent performance tracking"""
    agent_type: str
    agent_name: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    avg_processing_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    health_status: AgentHealth = AgentHealth.HEALTHY
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    response_times: List[float] = []
    error_messages: List[str] = []

class FailurePattern(BaseModel):
    """Pattern of failures for analysis"""
    failure_type: FailureType
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    affected_tasks: List[str] = []

class AgentCoordinator:
    """Coordinates agent tasks with failure detection and handling"""
    
    def __init__(self, health_check_interval: int = 30, max_consecutive_failures: int = 3):
        self.agents: Dict[str, AgentInterface] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.failure_patterns: Dict[str, List[FailurePattern]] = {}
        self.health_check_interval = health_check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.degradation_strategies: Dict[str, callable] = {}
        self._shutdown = False
        
        # Register default degradation strategies
        self._register_default_strategies()
    
    def register_agent(self, agent: AgentInterface):
        """Register an agent for coordination"""
        self.agents[agent.agent_type] = agent
        self.agent_metrics[agent.agent_type] = AgentMetrics(
            agent_type=agent.agent_type,
            agent_name=agent.agent_name
        )
        self.failure_patterns[agent.agent_type] = []
        logger.info(f"Registered agent for coordination: {agent.agent_name}")
    
    def register_degradation_strategy(self, agent_type: str, strategy: callable):
        """Register a degradation strategy for when an agent fails"""
        self.degradation_strategies[agent_type] = strategy
        logger.info(f"Registered degradation strategy for {agent_type}")
    
    def _register_default_strategies(self):
        """Register default degradation strategies"""
        self.degradation_strategies["nlp"] = self._nlp_degradation_strategy
        self.degradation_strategies["asr"] = self._asr_degradation_strategy
        self.degradation_strategies["document"] = self._document_degradation_strategy
        self.degradation_strategies["prototype"] = self._prototype_degradation_strategy
    
    async def execute_task_with_coordination(self, task: AgentTask) -> AgentResult:
        """Execute a task with coordination and fallback handling"""
        agent = self.agents.get(task.agent_type)
        metrics = self.agent_metrics.get(task.agent_type)
        
        if not agent or not metrics:
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                agent_name=task.agent_name,
                status=TaskStatus.FAILED,
                error_message=f"Agent {task.agent_type} not registered"
            )
        
        # Check if agent is healthy enough to execute tasks
        if metrics.health_status == AgentHealth.OFFLINE:
            logger.warning(f"Agent {task.agent_type} is offline, attempting degradation")
            return await self._handle_agent_degradation(task)
        
        # Execute the task
        try:
            start_time = time.time()
            result = await agent.execute_task(task)
            processing_time = time.time() - start_time
            
            # Update metrics
            metrics.total_tasks += 1
            if result.status == TaskStatus.COMPLETED:
                metrics.successful_tasks += 1
                metrics.consecutive_failures = 0
                metrics.last_success = datetime.now()
            else:
                metrics.failed_tasks += 1
                metrics.consecutive_failures += 1
                metrics.last_failure = datetime.now()
                
                # Record failure pattern
                failure_type = self._classify_failure(result.error_message or "Unknown error")
                await self._record_failure_pattern(task.agent_type, failure_type, result.error_message or "")
            
            # Update average processing time
            if metrics.response_times:
                metrics.avg_processing_time = statistics.mean(metrics.response_times)
            
            return result
            
        except asyncio.TimeoutError:
            await self._handle_agent_timeout(task.agent_type)
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                agent_name=task.agent_name,
                status=TaskStatus.TIMEOUT,
                error_message=f"Task timed out after {task.timeout}s"
            )
        except Exception as e:
            await self._handle_agent_error(task.agent_type, str(e))
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                agent_name=task.agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    async def _handle_agent_timeout(self, agent_type: str):
        """Handle agent timeout"""
        metrics = self.agent_metrics[agent_type]
        metrics.timeout_tasks += 1
        metrics.consecutive_failures += 1
        metrics.last_failure = datetime.now()
        
        if metrics.consecutive_failures >= self.max_consecutive_failures:
            metrics.health_status = AgentHealth.OFFLINE
        
        await self._record_failure_pattern(agent_type, FailureType.TIMEOUT, "Agent timed out")
        logger.warning(f"Agent {agent_type} timed out")
    
    async def _handle_agent_error(self, agent_type: str, error_message: str):
        """Handle agent error"""
        metrics = self.agent_metrics[agent_type]
        metrics.failed_tasks += 1
        metrics.consecutive_failures += 1
        metrics.last_failure = datetime.now()
        
        if metrics.consecutive_failures >= self.max_consecutive_failures:
            metrics.health_status = AgentHealth.OFFLINE
        
        failure_type = self._classify_failure(error_message)
        await self._record_failure_pattern(agent_type, failure_type, error_message)
        logger.error(f"Agent {agent_type} error: {error_message}")
    
    def _classify_failure(self, error_message: str) -> FailureType:
        """Classify the type of failure based on error message"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower or "timed out" in error_lower:
            return FailureType.TIMEOUT
        elif "connection" in error_lower or "network" in error_lower:
            return FailureType.CONNECTION_ERROR
        elif "validation" in error_lower or "invalid" in error_lower:
            return FailureType.VALIDATION_ERROR
        elif "service" in error_lower or "server" in error_lower:
            return FailureType.SERVICE_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    async def _record_failure_pattern(self, agent_type: str, failure_type: FailureType, error_message: str):
        """Record failure pattern for analysis"""
        patterns = self.failure_patterns[agent_type]
        now = datetime.now()
        
        # Find existing pattern or create new one
        existing_pattern = None
        for pattern in patterns:
            if pattern.failure_type == failure_type:
                existing_pattern = pattern
                break
        
        if existing_pattern:
            existing_pattern.count += 1
            existing_pattern.last_occurrence = now
        else:
            new_pattern = FailurePattern(
                failure_type=failure_type,
                count=1,
                first_occurrence=now,
                last_occurrence=now
            )
            patterns.append(new_pattern)
        
        # Keep error messages (last 10)
        metrics = self.agent_metrics[agent_type]
        metrics.error_messages.append(f"{now.isoformat()}: {error_message}")
        if len(metrics.error_messages) > 10:
            metrics.error_messages = metrics.error_messages[-10:]
    
    async def _handle_agent_degradation(self, task: AgentTask) -> AgentResult:
        """Handle task execution when agent is degraded or offline"""
        strategy = self.degradation_strategies.get(task.agent_type)
        
        if strategy:
            logger.info(f"Applying degradation strategy for {task.agent_type}")
            try:
                return await strategy(task)
            except Exception as e:
                logger.error(f"Degradation strategy failed for {task.agent_type}: {e}")
        
        # Default fallback: return a failed result with degradation message
        return AgentResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            agent_name=task.agent_name,
            status=TaskStatus.FAILED,
            error_message=f"Agent {task.agent_type} is offline and no degradation strategy available",
            metadata={"degraded": True}
        )
    
    # Default degradation strategies
    async def _nlp_degradation_strategy(self, task: AgentTask) -> AgentResult:
        """Degradation strategy for NLP agent"""
        # Provide basic text analysis without AI
        text_content = task.input_data.get("text_content", "")
        
        basic_analysis = {
            "text_summary": text_content[:200] + "..." if len(text_content) > 200 else text_content,
            "word_count": len(text_content.split()),
            "key_insights": ["Basic text analysis available", "Full AI analysis unavailable"],
            "sentiment_analysis": {"overall_sentiment": "neutral"},
            "topic_categorization": ["general"]
        }
        
        return AgentResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            agent_name="NLP Agent (Degraded)",
            status=TaskStatus.COMPLETED,
            result_data={"results": basic_analysis, "status": "degraded"},
            confidence_score=0.3,
            metadata={"degraded": True, "strategy": "basic_text_analysis"}
        )
    
    async def _asr_degradation_strategy(self, task: AgentTask) -> AgentResult:
        """Degradation strategy for ASR agent"""
        # Skip voice processing when ASR is unavailable
        return AgentResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            agent_name="ASR Agent (Degraded)",
            status=TaskStatus.COMPLETED,
            result_data={
                "transcript": "[Voice processing unavailable]",
                "confidence": 0.0,
                "status": "skipped"
            },
            confidence_score=0.0,
            metadata={"degraded": True, "strategy": "skip_voice_processing"}
        )
    
    async def _document_degradation_strategy(self, task: AgentTask) -> AgentResult:
        """Degradation strategy for Document agent"""
        # Provide basic file information without processing
        file_urls = task.input_data.get("file_urls", [])
        
        basic_info = {
            "file_count": len(file_urls),
            "files": file_urls,
            "extracted_text": "[Document processing unavailable]",
            "status": "degraded"
        }
        
        return AgentResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            agent_name="Document Agent (Degraded)",
            status=TaskStatus.COMPLETED,
            result_data={"results": basic_info},
            confidence_score=0.1,
            metadata={"degraded": True, "strategy": "basic_file_info"}
        )
    
    async def _prototype_degradation_strategy(self, task: AgentTask) -> AgentResult:
        """Degradation strategy for Prototype agent"""
        # Provide basic project template without AI generation
        basic_prototype = {
            "prototype_id": f"basic_{int(time.time())}",
            "technologies": ["React", "Node.js", "PostgreSQL"],
            "timeline": "2-4 weeks",
            "budget": "$5,000 - $15,000",
            "status": "template_based"
        }
        
        return AgentResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            agent_name="Prototype Agent (Degraded)",
            status=TaskStatus.COMPLETED,
            result_data={"results": basic_prototype},
            confidence_score=0.4,
            metadata={"degraded": True, "strategy": "basic_template"}
        )
    
    def get_agent_metrics(self, agent_type: str = None) -> Dict[str, AgentMetrics]:
        """Get metrics for specific agent or all agents"""
        if agent_type:
            return {agent_type: self.agent_metrics.get(agent_type)}
        return self.agent_metrics.copy()
    
    def get_failure_patterns(self, agent_type: str = None) -> Dict[str, List[FailurePattern]]:
        """Get failure patterns for specific agent or all agents"""
        if agent_type:
            return {agent_type: self.failure_patterns.get(agent_type, [])}
        return self.failure_patterns.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        total_agents = len(self.agent_metrics)
        healthy_agents = sum(1 for m in self.agent_metrics.values() 
                           if m.health_status == AgentHealth.HEALTHY)
        degraded_agents = sum(1 for m in self.agent_metrics.values() 
                            if m.health_status == AgentHealth.DEGRADED)
        offline_agents = sum(1 for m in self.agent_metrics.values() 
                           if m.health_status == AgentHealth.OFFLINE)
        
        total_tasks = sum(m.total_tasks for m in self.agent_metrics.values())
        successful_tasks = sum(m.successful_tasks for m in self.agent_metrics.values())
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "degraded_agents": degraded_agents,
            "offline_agents": offline_agents,
            "overall_health": "healthy" if offline_agents == 0 else "degraded" if degraded_agents > 0 else "critical",
            "total_tasks_processed": total_tasks,
            "success_rate": (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

# Global coordinator instance
agent_coordinator = AgentCoordinator()