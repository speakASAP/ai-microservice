"""
Queue management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_queue.queue_manager import QueueManager, PrototypeJob

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prototype/queue", tags=["prototype-queue"])

# Initialize queue manager
queue_manager = QueueManager()

class JobSubmissionRequest(BaseModel):
    """Request model for job submission."""
    user_id: str
    submission_id: str
    requirements: str

class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    created_at: str
    updated_at: str
    expires_at: str
    error_message: Optional[str] = None
    generated_files: Optional[dict] = None

@router.post("/submit")
async def submit_job(request: JobSubmissionRequest) -> dict:
    """Submit a new prototype generation job."""
    try:
        job_id = queue_manager.enqueue_job(
            user_id=request.user_id,
            submission_id=request.submission_id,
            requirements=request.requirements
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Job submitted successfully",
            "estimated_completion_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get job status by ID."""
    try:
        logger.info(f"Getting job status for ID: {job_id}")
        job = queue_manager.get_job(job_id)
        logger.info(f"Job found: {job is not None}")
        
        if not job:
            logger.warning(f"Job {job_id} not found")
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            expires_at=job.expires_at.isoformat(),
            error_message=job.error_message,
            generated_files=job.generated_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/list")
async def list_jobs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status")
) -> List[JobStatusResponse]:
    """List jobs with optional filtering."""
    try:
        jobs = queue_manager.list_jobs(user_id=user_id, status=status)
        
        return [
            JobStatusResponse(
                job_id=job.id,
                status=job.status,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                expires_at=job.expires_at.isoformat(),
                error_message=job.error_message,
                generated_files=job.generated_files
            )
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@router.delete("/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel a job (only if pending)."""
    try:
        job = queue_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "pending":
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job in status: {job.status}"
            )
        
        # Remove from queue
        queue_manager.redis_client.lrem(queue_manager.queue_name, 0, job_id)
        
        # Update status to cancelled
        queue_manager.update_job_status(job_id, "cancelled")
        
        return {
            "success": True,
            "message": f"Job {job_id} cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/stats")
async def get_queue_stats() -> dict:
    """Get queue statistics."""
    try:
        stats = queue_manager.get_queue_stats()
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@router.post("/cleanup")
async def cleanup_expired_jobs() -> dict:
    """Clean up expired jobs."""
    try:
        cleaned_count = queue_manager.cleanup_expired_jobs()
        
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} expired jobs"
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup jobs: {str(e)}")
