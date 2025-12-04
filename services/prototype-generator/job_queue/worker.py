"""
Queue worker for processing prototype generation jobs.
"""

import asyncio
import logging
import time
from typing import Optional
from datetime import datetime

from .queue_manager import QueueManager, PrototypeJob

logger = logging.getLogger(__name__)

class QueueWorker:
    """Worker that processes prototype generation jobs from the queue."""
    
    def __init__(self, queue_manager: QueueManager, max_retries: int = 3, retry_delay: int = 60):
        """Initialize queue worker."""
        self.queue_manager = queue_manager
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.running = False
        self.current_job: Optional[PrototypeJob] = None
        
    async def start(self):
        """Start the worker."""
        self.running = True
        logger.info("Queue worker started")
        
        while self.running:
            try:
                # Get next job from queue
                job = self.queue_manager.get_next_job()
                
                if not job:
                    # No jobs available, wait a bit
                    await asyncio.sleep(1)
                    continue
                    
                self.current_job = job
                logger.info(f"Processing job {job.id} for user {job.user_id}")
                
                # Update job status to processing
                self.queue_manager.update_job_status(job.id, "processing")
                
                # Process the job
                success = await self.process_job(job)
                
                if success:
                    self.queue_manager.update_job_status(
                        job.id, 
                        "completed",
                        generated_files={"status": "generated"}
                    )
                    logger.info(f"Successfully completed job {job.id}")
                else:
                    # Handle retry logic
                    retry_count = getattr(job, 'retry_count', 0)
                    
                    if retry_count < self.max_retries:
                        retry_count += 1
                        logger.warning(f"Job {job.id} failed, retrying ({retry_count}/{self.max_retries})")
                        
                        # Update retry count and re-queue
                        self.queue_manager.update_job_status(
                            job.id, 
                            "pending",
                            error_message=f"Retry {retry_count}/{self.max_retries}"
                        )
                        
                        # Re-queue the job
                        self.queue_manager.redis_client.lpush(
                            self.queue_manager.queue_name, 
                            job.id
                        )
                        
                        # Wait before retry
                        await asyncio.sleep(self.retry_delay)
                    else:
                        # Max retries exceeded, mark as failed
                        self.queue_manager.update_job_status(
                            job.id, 
                            "failed",
                            error_message="Max retries exceeded"
                        )
                        logger.error(f"Job {job.id} failed after {self.max_retries} retries")
                
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                if self.current_job:
                    self.queue_manager.update_job_status(
                        self.current_job.id, 
                        "failed",
                        error_message=str(e)
                    )
                    self.current_job = None
                await asyncio.sleep(5)  # Wait before continuing
    
    async def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info("Queue worker stopped")
    
    async def process_job(self, job: PrototypeJob) -> bool:
        """Process a single prototype generation job."""
        try:
            start_time = time.time()
            
            # TODO: Implement actual prototype generation
            # This is a placeholder that simulates the generation process
            await self.simulate_prototype_generation(job)
            
            generation_time = time.time() - start_time
            logger.info(f"Job {job.id} processed in {generation_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing job {job.id}: {e}")
            return False
    
    async def simulate_prototype_generation(self, job: PrototypeJob):
        """Simulate prototype generation process."""
        # Simulate different stages of generation
        stages = [
            "Analyzing requirements...",
            "Generating HTML structure...",
            "Creating CSS styles...",
            "Adding JavaScript functionality...",
            "Integrating content...",
            "Finalizing prototype..."
        ]
        
        for i, stage in enumerate(stages):
            logger.info(f"Job {job.id}: {stage}")
            
            # Simulate processing time (1-3 seconds per stage)
            await asyncio.sleep(1 + (i * 0.5))
            
            # Update progress (this would be real progress in actual implementation)
            progress = int((i + 1) / len(stages) * 100)
            logger.info(f"Job {job.id}: {progress}% complete")
    
    def get_status(self) -> dict:
        """Get worker status."""
        return {
            "running": self.running,
            "current_job": self.current_job.id if self.current_job else None,
            "queue_stats": self.queue_manager.get_queue_stats()
        }
