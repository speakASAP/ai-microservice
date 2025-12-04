"""
Redis-based queue manager for prototype generation jobs.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PrototypeJob(BaseModel):
    """Prototype generation job model."""
    id: str
    user_id: str
    submission_id: str
    status: str  # pending, processing, completed, failed
    requirements: str
    generated_files: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    error_message: Optional[str] = None

class QueueManager:
    """Manages prototype generation job queue using Redis."""
    
    def __init__(self, redis_url: str = None):
        """Initialize queue manager with Redis connection."""
        if redis_url is None:
            import os
            # Port configured in database-server/.env: REDIS_SERVER_PORT (default: 6379)
            redis_url = os.getenv('REDIS_URL', f'redis://localhost:{os.getenv("REDIS_SERVER_PORT", "6379")}/1')
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.queue_name = "prototype_generation"
        self.job_prefix = "prototype_job:"
        self.expiry_days = 30
        
    def enqueue_job(self, user_id: str, submission_id: str, requirements: str) -> str:
        """Enqueue a new prototype generation job."""
        job_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(days=self.expiry_days)
        
        job = PrototypeJob(
            id=job_id,
            user_id=user_id,
            submission_id=submission_id,
            status="pending",
            requirements=requirements,
            created_at=now,
            updated_at=now,
            expires_at=expires_at
        )
        
        # Store job data
        job_key = f"{self.job_prefix}{job_id}"
        job_data = job.model_dump()
        # Convert datetime objects to ISO strings for Redis storage
        job_data['created_at'] = job_data['created_at'].isoformat()
        job_data['updated_at'] = job_data['updated_at'].isoformat()
        job_data['expires_at'] = job_data['expires_at'].isoformat()
        # Filter out None values for Redis
        job_data = {k: v for k, v in job_data.items() if v is not None}
        
        # Debug: log the data being stored
        logger.info(f"Storing job data: {job_data}")
        
        # Convert all values to strings for Redis
        job_data = {k: str(v) for k, v in job_data.items()}
        
        self.redis_client.hset(job_key, mapping=job_data)
        
        # Add to queue
        self.redis_client.lpush(self.queue_name, job_id)
        
        logger.info(f"Enqueued prototype job {job_id} for user {user_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[PrototypeJob]:
        """Get job by ID."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis_client.hgetall(job_key)
        
        if not job_data:
            return None
            
        try:
            # Decode binary data to strings
            job_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
            
            # Convert datetime strings back to datetime objects
            job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
            job_data['updated_at'] = datetime.fromisoformat(job_data['updated_at'])
            job_data['expires_at'] = datetime.fromisoformat(job_data['expires_at'])
            
            # Parse JSON strings back to objects
            if 'generated_files' in job_data and job_data['generated_files']:
                job_data['generated_files'] = json.loads(job_data['generated_files'])
            
            return PrototypeJob(**job_data)
        except Exception as e:
            logger.error(f"Error parsing job {job_id}: {e}")
            return None
    
    def update_job_status(self, job_id: str, status: str, error_message: str = None, generated_files: Dict = None) -> bool:
        """Update job status and metadata."""
        job_key = f"{self.job_prefix}{job_id}"
        
        if not self.redis_client.exists(job_key):
            return False
            
        updates = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        if error_message:
            updates["error_message"] = error_message
            
        if generated_files:
            updates["generated_files"] = json.dumps(generated_files)
            
        self.redis_client.hset(job_key, mapping=updates)
        
        logger.info(f"Updated job {job_id} status to {status}")
        return True
    
    def get_next_job(self) -> Optional[PrototypeJob]:
        """Get next job from queue (blocking)."""
        # Block for up to 10 seconds waiting for a job
        result = self.redis_client.brpop(self.queue_name, timeout=10)
        
        if not result:
            return None
            
        job_id = result[1]
        return self.get_job(job_id)
    
    def list_jobs(self, user_id: str = None, status: str = None) -> List[PrototypeJob]:
        """List jobs with optional filtering."""
        jobs = []
        
        # Get all job keys
        job_keys = self.redis_client.keys(f"{self.job_prefix}*")
        
        for job_key in job_keys:
            job_data = self.redis_client.hgetall(job_key)
            
            if not job_data:
                continue
                
            try:
                # Convert datetime strings back to datetime objects
                job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
                job_data['updated_at'] = datetime.fromisoformat(job_data['updated_at'])
                job_data['expires_at'] = datetime.fromisoformat(job_data['expires_at'])
                
                job = PrototypeJob(**job_data)
                
                # Apply filters
                if user_id and job.user_id != user_id:
                    continue
                if status and job.status != status:
                    continue
                    
                jobs.append(job)
                
            except Exception as e:
                logger.error(f"Error parsing job from {job_key}: {e}")
                continue
                
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs and return count of cleaned jobs."""
        cleaned_count = 0
        now = datetime.now()
        
        # Get all job keys
        job_keys = self.redis_client.keys(f"{self.job_prefix}*")
        
        for job_key in job_keys:
            job_data = self.redis_client.hgetall(job_key)
            
            if not job_data:
                continue
                
            try:
                expires_at = datetime.fromisoformat(job_data['expires_at'])
                
                if now > expires_at:
                    # Remove from queue if still there
                    job_id = job_key.replace(self.job_prefix, "")
                    self.redis_client.lrem(self.queue_name, 0, job_id)
                    
                    # Delete job data
                    self.redis_client.delete(job_key)
                    
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired job {job_id}")
                    
            except Exception as e:
                logger.error(f"Error checking expiry for {job_key}: {e}")
                continue
                
        return cleaned_count
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_length = self.redis_client.llen(self.queue_name)
        total_jobs = len(self.redis_client.keys(f"{self.job_prefix}*"))
        
        # Count jobs by status
        status_counts = {}
        job_keys = self.redis_client.keys(f"{self.job_prefix}*")
        
        for job_key in job_keys:
            job_data = self.redis_client.hgetall(job_key)
            if job_data and 'status' in job_data:
                status = job_data['status']
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "queue_length": queue_length,
            "total_jobs": total_jobs,
            "status_counts": status_counts
        }
