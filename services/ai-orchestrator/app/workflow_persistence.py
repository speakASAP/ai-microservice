"""
Workflow State Persistence

Manages workflow state storage and recovery using Redis for crash recovery
and state persistence across service restarts.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis
import os
from pydantic import BaseModel

from .workflow_engine import WorkflowState, AgentResult, TaskStatus, WorkflowStatus

logger = logging.getLogger(__name__)

class WorkflowPersistence:
    """Handles workflow state persistence in Redis"""
    
    def __init__(self, redis_url: str = None):
        # Port configured in database-server/.env: REDIS_SERVER_PORT (default: 6379)
        self.redis_url = redis_url or os.getenv("REDIS_URL", f"redis://localhost:{os.getenv('REDIS_SERVER_PORT', '6379')}")
        self.redis_client = None
        self.key_prefix = "workflow:"
        self.state_key_prefix = f"{self.key_prefix}state:"
        self.index_key = f"{self.key_prefix}index"
        self.cleanup_key = f"{self.key_prefix}cleanup"
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self._ping()
            logger.info("Connected to Redis for workflow persistence")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _ping(self):
        """Test Redis connection"""
        if self.redis_client:
            try:
                # Use sync ping for redis-py
                self.redis_client.ping()
                return True
            except Exception as e:
                logger.error(f"Redis ping failed: {e}")
                return False
        return False
    
    async def save_workflow_state(self, workflow_state: WorkflowState) -> bool:
        """Save workflow state to Redis"""
        if not self.redis_client:
            logger.warning("Redis not connected, cannot save workflow state")
            return False
        
        try:
            # Convert workflow state to JSON
            state_data = {
                "workflow_id": workflow_state.workflow_id,
                "submission_id": workflow_state.submission_id,
                "workflow_type": workflow_state.workflow_type,
                "status": workflow_state.status.value,
                "current_step": workflow_state.current_step,
                "completed_steps": workflow_state.completed_steps,
                "pending_tasks": workflow_state.pending_tasks,
                "running_tasks": workflow_state.running_tasks,
                "completed_tasks": workflow_state.completed_tasks,
                "failed_tasks": workflow_state.failed_tasks,
                "agent_results": {
                    task_id: {
                        "task_id": result.task_id,
                        "agent_name": result.agent_name,
                        "agent_type": result.agent_type,
                        "status": result.status.value,
                        "result_data": result.result_data,
                        "processing_time": result.processing_time,
                        "confidence_score": result.confidence_score,
                        "created_at": result.created_at.isoformat()
                    } for task_id, result in workflow_state.agent_results.items()
                },
                "start_time": workflow_state.start_time.isoformat(),
                "end_time": workflow_state.end_time.isoformat() if workflow_state.end_time else None,
                "estimated_completion": workflow_state.estimated_completion.isoformat() if workflow_state.estimated_completion else None,
                "progress": workflow_state.progress,
                "metadata": workflow_state.metadata,
                "input_data": workflow_state.input_data,
                "saved_at": datetime.now().isoformat()
            }
            
            # Save to Redis
            key = f"{self.state_key_prefix}{workflow_state.workflow_id}"
            self.redis_client.setex(
                key,
                timedelta(days=7),  # Keep for 7 days
                json.dumps(state_data)
            )
            
            # Add to index
            self.redis_client.sadd(self.index_key, workflow_state.workflow_id)
            
            logger.debug(f"Saved workflow state {workflow_state.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save workflow state {workflow_state.workflow_id}: {e}")
            return False
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state from Redis"""
        if not self.redis_client:
            logger.warning("Redis not connected, cannot load workflow state")
            return None
        
        try:
            key = f"{self.state_key_prefix}{workflow_id}"
            state_json = self.redis_client.get(key)
            
            if not state_json:
                logger.debug(f"Workflow state {workflow_id} not found in Redis")
                return None
            
            state_data = json.loads(state_json)
            
            # Reconstruct AgentResult objects
            agent_results = {}
            for task_id, result_data in state_data.get("agent_results", {}).items():
                agent_results[task_id] = AgentResult(**result_data)
            
            # Reconstruct WorkflowState
            workflow_state = WorkflowState(
                workflow_id=state_data["workflow_id"],
                submission_id=state_data["submission_id"],
                workflow_type=state_data["workflow_type"],
                status=WorkflowStatus(state_data["status"]),
                current_step=state_data["current_step"],
                completed_steps=state_data["completed_steps"],
                pending_tasks=state_data["pending_tasks"],
                running_tasks=state_data["running_tasks"],
                completed_tasks=state_data["completed_tasks"],
                failed_tasks=state_data["failed_tasks"],
                agent_results=agent_results,
                start_time=datetime.fromisoformat(state_data["start_time"]),
                end_time=datetime.fromisoformat(state_data["end_time"]) if state_data["end_time"] else None,
                estimated_completion=datetime.fromisoformat(state_data["estimated_completion"]) if state_data["estimated_completion"] else None,
                progress=state_data["progress"],
                metadata=state_data["metadata"],
                input_data=state_data.get("input_data", {})
            )
            
            logger.debug(f"Loaded workflow state {workflow_id}")
            return workflow_state
            
        except Exception as e:
            logger.error(f"Failed to load workflow state {workflow_id}: {e}")
            return None
    
    async def delete_workflow_state(self, workflow_id: str) -> bool:
        """Delete workflow state from Redis"""
        if not self.redis_client:
            return False
        
        try:
            key = f"{self.state_key_prefix}{workflow_id}"
            deleted = self.redis_client.delete(key)
            
            # Remove from index
            self.redis_client.srem(self.index_key, workflow_id)
            
            logger.debug(f"Deleted workflow state {workflow_id}")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Failed to delete workflow state {workflow_id}: {e}")
            return False
    
    async def list_workflow_ids(self) -> List[str]:
        """List all workflow IDs in Redis"""
        if not self.redis_client:
            return []
        
        try:
            workflow_ids = self.redis_client.smembers(self.index_key)
            return list(workflow_ids)
        except Exception as e:
            logger.error(f"Failed to list workflow IDs: {e}")
            return []
    
    async def get_workflows_by_status(self, status: WorkflowStatus) -> List[str]:
        """Get workflow IDs by status"""
        workflow_ids = await self.list_workflow_ids()
        matching_workflows = []
        
        for workflow_id in workflow_ids:
            workflow_state = await self.load_workflow_state(workflow_id)
            if workflow_state and workflow_state.status == status:
                matching_workflows.append(workflow_id)
        
        return matching_workflows
    
    async def get_stale_workflows(self, max_age_hours: int = 24) -> List[str]:
        """Get workflows that are older than max_age_hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        workflow_ids = await self.list_workflow_ids()
        stale_workflows = []
        
        for workflow_id in workflow_ids:
            workflow_state = await self.load_workflow_state(workflow_id)
            if workflow_state and workflow_state.start_time < cutoff_time:
                stale_workflows.append(workflow_id)
        
        return stale_workflows
    
    async def cleanup_stale_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up workflows older than max_age_hours"""
        stale_workflows = await self.get_stale_workflows(max_age_hours)
        cleaned_count = 0
        
        for workflow_id in stale_workflows:
            if await self.delete_workflow_state(workflow_id):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} stale workflows")
        return cleaned_count
    
    async def recover_interrupted_workflows(self) -> List[WorkflowState]:
        """Recover workflows that were interrupted (status = RUNNING)"""
        if not self.redis_client:
            return []
        
        interrupted_workflows = []
        running_workflow_ids = await self.get_workflows_by_status(WorkflowStatus.RUNNING)
        
        for workflow_id in running_workflow_ids:
            workflow_state = await self.load_workflow_state(workflow_id)
            if workflow_state:
                # Mark running tasks as failed since they were interrupted
                for task_id in workflow_state.running_tasks:
                    if task_id in workflow_state.agent_results:
                        workflow_state.agent_results[task_id].status = TaskStatus.FAILED
                        workflow_state.agent_results[task_id].error_message = "Task interrupted by service restart"
                
                # Move running tasks to failed tasks
                workflow_state.failed_tasks.extend(workflow_state.running_tasks)
                workflow_state.running_tasks = []
                
                # Update status
                workflow_state.status = WorkflowStatus.FAILED
                workflow_state.end_time = datetime.now()
                
                # Save updated state
                await self.save_workflow_state(workflow_state)
                interrupted_workflows.append(workflow_state)
        
        logger.info(f"Recovered {len(interrupted_workflows)} interrupted workflows")
        return interrupted_workflows
    
    async def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get statistics about workflows in Redis"""
        if not self.redis_client:
            return {}
        
        try:
            workflow_ids = await self.list_workflow_ids()
            stats = {
                "total_workflows": len(workflow_ids),
                "status_counts": {status.value: 0 for status in WorkflowStatus},
                "oldest_workflow": None,
                "newest_workflow": None
            }
            
            oldest_time = None
            newest_time = None
            
            for workflow_id in workflow_ids:
                workflow_state = await self.load_workflow_state(workflow_id)
                if workflow_state:
                    # Count by status
                    stats["status_counts"][workflow_state.status.value] += 1
                    
                    # Track oldest and newest
                    if oldest_time is None or workflow_state.start_time < oldest_time:
                        oldest_time = workflow_state.start_time
                        stats["oldest_workflow"] = {
                            "workflow_id": workflow_id,
                            "start_time": workflow_state.start_time.isoformat(),
                            "status": workflow_state.status.value
                        }
                    
                    if newest_time is None or workflow_state.start_time > newest_time:
                        newest_time = workflow_state.start_time
                        stats["newest_workflow"] = {
                            "workflow_id": workflow_id,
                            "start_time": workflow_state.start_time.isoformat(),
                            "status": workflow_state.status.value
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get workflow statistics: {e}")
            return {}

# Global persistence instance
workflow_persistence = WorkflowPersistence()