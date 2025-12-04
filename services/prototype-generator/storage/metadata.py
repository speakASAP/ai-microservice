"""
Metadata management for prototype generation.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class MetadataManager:
    """Manages metadata for generated prototypes."""
    
    def __init__(self, base_path: str = "./prototypes"):
        """Initialize metadata manager."""
        self.base_path = Path(base_path)
        
    def create_metadata(self, project_id: str, prototype_type: str, requirements: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a new prototype."""
        metadata = {
            "project_id": project_id,
            "prototype_type": prototype_type,
            "requirements": requirements,
            "analysis": analysis,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "generating",
            "ai_generated": True,
            "file_count": 0,
            "total_size": 0,
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        return metadata
    
    def save_metadata(self, project_id: str, metadata: Dict[str, Any]) -> bool:
        """Save metadata to file."""
        try:
            metadata_file = self.base_path / project_id / "metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metadata saved for project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata for {project_id}: {e}")
            return False
    
    def load_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata from file."""
        try:
            metadata_file = self.base_path / project_id / "metadata.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata for {project_id}: {e}")
            return None
    
    def update_metadata(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata with new values."""
        try:
            metadata = self.load_metadata(project_id)
            if not metadata:
                return False
            
            metadata.update(updates)
            metadata["updated_at"] = datetime.now().isoformat()
            
            return self.save_metadata(project_id, metadata)
            
        except Exception as e:
            logger.error(f"Error updating metadata for {project_id}: {e}")
            return False
    
    def get_expired_projects(self, expiry_days: int = 30) -> List[str]:
        """Get list of expired project IDs."""
        expired_projects = []
        cutoff_date = datetime.now() - timedelta(days=expiry_days)
        
        for project_dir in self.base_path.iterdir():
            if not project_dir.is_dir():
                continue
            
            metadata = self.load_metadata(project_dir.name)
            if metadata:
                created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                if created_at < cutoff_date:
                    expired_projects.append(project_dir.name)
        
        return expired_projects
    
    def get_project_stats(self) -> Dict[str, Any]:
        """Get statistics about all projects."""
        stats = {
            "total_projects": 0,
            "active_projects": 0,
            "expired_projects": 0,
            "total_size": 0,
            "prototype_types": {}
        }
        
        for project_dir in self.base_path.iterdir():
            if not project_dir.is_dir():
                continue
            
            metadata = self.load_metadata(project_dir.name)
            if metadata:
                stats["total_projects"] += 1
                
                # Check if expired
                created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                if created_at > datetime.now() - timedelta(days=30):
                    stats["active_projects"] += 1
                else:
                    stats["expired_projects"] += 1
                
                # Count by type
                prototype_type = metadata.get("prototype_type", "unknown")
                stats["prototype_types"][prototype_type] = stats["prototype_types"].get(prototype_type, 0) + 1
                
                # Add size
                stats["total_size"] += metadata.get("total_size", 0)
        
        return stats
