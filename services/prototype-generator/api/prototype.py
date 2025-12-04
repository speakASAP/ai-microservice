"""
Prototype management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
import logging
import os
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_queue.queue_manager import QueueManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prototype", tags=["prototype"])

# Initialize queue manager
queue_manager = QueueManager()

class PrototypeInfo(BaseModel):
    """Prototype information model."""
    project_id: str
    title: str
    description: str
    created_at: str
    expires_at: str
    file_count: int
    total_size: int

class PrototypeFile(BaseModel):
    """Prototype file model."""
    filename: str
    size: int
    type: str
    path: str

@router.get("/{project_id}")
async def get_prototype_info(project_id: str) -> PrototypeInfo:
    """Get prototype information by project ID."""
    try:
        # Find job by project_id (assuming project_id is derived from submission_id)
        jobs = queue_manager.list_jobs()
        job = None
        
        for j in jobs:
            if j.submission_id == project_id or j.id == project_id:
                job = j
                break
        
        if not job:
            raise HTTPException(status_code=404, detail="Prototype not found")
        
        # Get prototype directory
        prototype_dir = f"/app/public/prototypes/{project_id}"
        
        if not os.path.exists(prototype_dir):
            raise HTTPException(status_code=404, detail="Prototype files not found")
        
        # Calculate file count and total size
        file_count = 0
        total_size = 0
        
        for root, dirs, files in os.walk(prototype_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_count += 1
                    total_size += os.path.getsize(file_path)
        
        return PrototypeInfo(
            project_id=project_id,
            title=f"Prototype {project_id}",
            description=job.requirements[:100] + "..." if len(job.requirements) > 100 else job.requirements,
            created_at=job.created_at.isoformat(),
            expires_at=job.expires_at.isoformat(),
            file_count=file_count,
            total_size=total_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prototype info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prototype info: {str(e)}")

@router.get("/{project_id}/files")
async def list_prototype_files(project_id: str) -> List[PrototypeFile]:
    """List prototype files by project ID."""
    try:
        prototype_dir = f"/app/public/prototypes/{project_id}"
        
        if not os.path.exists(prototype_dir):
            raise HTTPException(status_code=404, detail="Prototype files not found")
        
        files = []
        
        for root, dirs, filenames in os.walk(prototype_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    file_type = "unknown"
                    
                    if filename.endswith('.html'):
                        file_type = "html"
                    elif filename.endswith('.css'):
                        file_type = "css"
                    elif filename.endswith('.js'):
                        file_type = "javascript"
                    elif filename.endswith('.json'):
                        file_type = "json"
                    elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                        file_type = "image"
                    
                    files.append(PrototypeFile(
                        filename=filename,
                        size=size,
                        type=file_type,
                        path=file_path
                    ))
        
        return files
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing prototype files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prototype files: {str(e)}")

@router.get("/{project_id}/download")
async def download_prototype(project_id: str) -> dict:
    """Get download information for prototype."""
    try:
        prototype_dir = f"/app/public/prototypes/{project_id}"
        
        if not os.path.exists(prototype_dir):
            raise HTTPException(status_code=404, detail="Prototype files not found")
        
        # Create a zip file path (this would be implemented in actual system)
        zip_path = f"/tmp/prototype_{project_id}.zip"
        
        return {
            "success": True,
            "download_url": f"/api/prototype/{project_id}/download/zip",
            "message": "Download ready"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preparing download: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare download: {str(e)}")

@router.delete("/{project_id}")
async def delete_prototype(project_id: str) -> dict:
    """Delete prototype and associated files."""
    try:
        import shutil
        
        prototype_dir = f"/app/public/prototypes/{project_id}"
        
        if os.path.exists(prototype_dir):
            shutil.rmtree(prototype_dir)
        
        # Also remove from queue if exists
        jobs = queue_manager.list_jobs()
        for job in jobs:
            if job.submission_id == project_id or job.id == project_id:
                queue_manager.redis_client.delete(f"prototype_job:{job.id}")
                break
        
        return {
            "success": True,
            "message": f"Prototype {project_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting prototype: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prototype: {str(e)}")

@router.get("/")
async def list_prototypes(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[PrototypeInfo]:
    """List all prototypes with pagination."""
    try:
        jobs = queue_manager.list_jobs()
        prototypes = []
        
        for job in jobs:
            if job.status == "completed":
                project_id = job.submission_id
                prototype_dir = f"/app/public/prototypes/{project_id}"
                
                if os.path.exists(prototype_dir):
                    # Calculate file count and total size
                    file_count = 0
                    total_size = 0
                    
                    for root, dirs, files in os.walk(prototype_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                file_count += 1
                                total_size += os.path.getsize(file_path)
                    
                    prototypes.append(PrototypeInfo(
                        project_id=project_id,
                        title=f"Prototype {project_id}",
                        description=job.requirements[:100] + "..." if len(job.requirements) > 100 else job.requirements,
                        created_at=job.created_at.isoformat(),
                        expires_at=job.expires_at.isoformat(),
                        file_count=file_count,
                        total_size=total_size
                    ))
        
        # Apply pagination
        start = offset
        end = offset + limit
        paginated_prototypes = prototypes[start:end]
        
        return paginated_prototypes
        
    except Exception as e:
        logger.error(f"Error listing prototypes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prototypes: {str(e)}")
