"""
Workflow Recovery System

Enhanced workflow recovery capabilities including automatic recovery,
checkpoint management, and workflow resumption after interruptions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from .workflow_engine import WorkflowState, AgentResult, TaskStatus, WorkflowStatus
from .workflow_persistence import workflow_persistence

logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Different recovery strategies for interrupted workflows"""
    RESTART_FROM_BEGINNING = "restart_from_beginning"
    RESUME_FROM_CHECKPOINT = "resume_from_checkpoint"
    RESUME_FROM_LAST_SUCCESS = "resume_from_last_success"
    SKIP_FAILED_CONTINUE = "skip_failed_continue"
    MANUAL_INTERVENTION = "manual_intervention"

class CheckpointType(Enum):
    """Types of workflow checkpoints"""
    TASK_COMPLETION = "task_completion"
    AGENT_SUCCESS = "agent_success"
    WORKFLOW_PHASE = "workflow_phase"
    ERROR_RECOVERY = "error_recovery"
    MANUAL_CHECKPOINT = "manual_checkpoint"

class WorkflowCheckpoint:
    """Represents a workflow checkpoint for recovery"""
    
    def __init__(
        self,
        workflow_id: str,
        checkpoint_id: str,
        checkpoint_type: CheckpointType,
        timestamp: datetime,
        workflow_state: WorkflowState,
        metadata: Dict[str, Any] = None
    ):
        self.workflow_id = workflow_id
        self.checkpoint_id = checkpoint_id
        self.checkpoint_type = checkpoint_type
        self.timestamp = timestamp
        self.workflow_state = workflow_state
        self.metadata = metadata or {}

class WorkflowRecoveryManager:
    """Manages workflow recovery and resumption"""
    
    def __init__(self):
        self.persistence = workflow_persistence
        self.checkpoints: Dict[str, List[WorkflowCheckpoint]] = {}
        self.recovery_strategies = {
            WorkflowStatus.FAILED: RecoveryStrategy.RESUME_FROM_LAST_SUCCESS,
            WorkflowStatus.RUNNING: RecoveryStrategy.RESUME_FROM_CHECKPOINT,
            WorkflowStatus.PAUSED: RecoveryStrategy.RESUME_FROM_CHECKPOINT,
        }
        self.max_recovery_attempts = 3
        self.recovery_timeout_minutes = 30
    
    async def create_checkpoint(
        self,
        workflow_state: WorkflowState,
        checkpoint_type: CheckpointType,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a checkpoint for workflow recovery"""
        try:
            checkpoint_id = f"{workflow_state.workflow_id}_{checkpoint_type.value}_{int(datetime.now().timestamp())}"
            
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_state.workflow_id,
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                timestamp=datetime.now(),
                workflow_state=workflow_state,
                metadata=metadata or {}
            )
            
            # Store checkpoint
            if workflow_state.workflow_id not in self.checkpoints:
                self.checkpoints[workflow_state.workflow_id] = []
            
            self.checkpoints[workflow_state.workflow_id].append(checkpoint)
            
            logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_state.workflow_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for workflow {workflow_state.workflow_id}: {e}")
            return ""
    
    async def recover_workflow(
        self,
        workflow_id: str,
        strategy: Optional[RecoveryStrategy] = None
    ) -> Optional[WorkflowState]:
        """Recover an interrupted or failed workflow"""
        try:
            # Load current workflow state
            current_state = await self.persistence.load_workflow_state(workflow_id)
            if not current_state:
                logger.error(f"Cannot recover workflow {workflow_id}: state not found")
                return None
            
            # Determine recovery strategy
            if not strategy:
                strategy = self.recovery_strategies.get(
                    current_state.status, 
                    RecoveryStrategy.RESTART_FROM_BEGINNING
                )
            
            logger.info(f"Recovering workflow {workflow_id} using strategy {strategy.value}")
            
            # Execute recovery based on strategy
            if strategy == RecoveryStrategy.RESTART_FROM_BEGINNING:
                return await self._restart_workflow_from_beginning(current_state)
            elif strategy == RecoveryStrategy.RESUME_FROM_LAST_SUCCESS:
                return await self._resume_from_last_success(current_state)
            else:
                logger.error(f"Recovery strategy {strategy} not implemented")
                return None
                
        except Exception as e:
            logger.error(f"Failed to recover workflow {workflow_id}: {e}")
            return None
    
    async def auto_recover_interrupted_workflows(self) -> List[str]:
        """Automatically recover all interrupted workflows"""
        recovered_workflows = []
        
        try:
            # Get interrupted workflows from persistence
            interrupted_workflows = await self.persistence.recover_interrupted_workflows()
            
            for workflow_state in interrupted_workflows:
                try:
                    # Attempt recovery
                    recovered_state = await self.recover_workflow(
                        workflow_state.workflow_id,
                        RecoveryStrategy.RESUME_FROM_LAST_SUCCESS
                    )
                    
                    if recovered_state:
                        recovered_workflows.append(workflow_state.workflow_id)
                        logger.info(f"Successfully recovered workflow {workflow_state.workflow_id}")
                    else:
                        logger.warning(f"Failed to recover workflow {workflow_state.workflow_id}")
                        
                except Exception as e:
                    logger.error(f"Error recovering workflow {workflow_state.workflow_id}: {e}")
            
            logger.info(f"Auto-recovery completed: {len(recovered_workflows)} workflows recovered")
            return recovered_workflows
            
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
            return []
    
    async def get_recovery_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get recovery status and options for a workflow"""
        try:
            workflow_state = await self.persistence.load_workflow_state(workflow_id)
            if not workflow_state:
                return {"error": "Workflow not found"}
            
            # Get available checkpoints
            checkpoints = self.checkpoints.get(workflow_id, [])
            
            # Analyze recovery options
            recovery_options = []
            
            if workflow_state.status == WorkflowStatus.FAILED:
                recovery_options.extend([
                    RecoveryStrategy.RESUME_FROM_LAST_SUCCESS,
                    RecoveryStrategy.RESTART_FROM_BEGINNING
                ])
            
            if workflow_state.status == WorkflowStatus.RUNNING:
                recovery_options.extend([
                    RecoveryStrategy.RESUME_FROM_CHECKPOINT,
                    RecoveryStrategy.RESTART_FROM_BEGINNING
                ])
            
            return {
                "workflow_id": workflow_id,
                "current_status": workflow_state.status.value,
                "recovery_options": [option.value for option in recovery_options],
                "available_checkpoints": len(checkpoints),
                "last_checkpoint": checkpoints[-1].timestamp.isoformat() if checkpoints else None,
                "workflow_age_hours": (datetime.now() - workflow_state.start_time).total_seconds() / 3600,
                "can_auto_recover": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get recovery status for {workflow_id}: {e}")
            return {"error": str(e)}
    
    async def _restart_workflow_from_beginning(self, workflow_state: WorkflowState) -> WorkflowState:
        """Restart workflow from the beginning"""
        # Reset workflow state
        workflow_state.status = WorkflowStatus.PENDING
        workflow_state.current_step = "restarting"
        workflow_state.completed_steps = []
        workflow_state.running_tasks = []
        workflow_state.completed_tasks = []
        workflow_state.failed_tasks = []
        workflow_state.agent_results = {}
        workflow_state.progress = 0.0
        workflow_state.end_time = None
        
        # Save updated state
        await self.persistence.save_workflow_state(workflow_state)
        
        return workflow_state
    
    async def _resume_from_last_success(self, workflow_state: WorkflowState) -> WorkflowState:
        """Resume workflow from the last successful task"""
        # Reset running tasks
        workflow_state.running_tasks = []
        
        # Update status
        workflow_state.status = WorkflowStatus.RUNNING
        workflow_state.current_step = "resumed_from_last_success"
        
        await self.persistence.save_workflow_state(workflow_state)
        return workflow_state

# Global workflow recovery manager instance
workflow_recovery_manager = WorkflowRecoveryManager()