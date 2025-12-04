"""
File management for prototype generation.
"""

import os
import shutil
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file storage for generated prototypes."""
    
    def __init__(self, base_path: str = "./prototypes"):
        """Initialize file manager."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def create_prototype_directory(self, project_id: str) -> Path:
        """Create directory for a new prototype."""
        prototype_dir = self.base_path / project_id
        prototype_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (prototype_dir / "assets").mkdir(exist_ok=True)
        (prototype_dir / "assets" / "images").mkdir(exist_ok=True)
        (prototype_dir / "assets" / "fonts").mkdir(exist_ok=True)
        
        logger.info(f"Created prototype directory: {prototype_dir}")
        return prototype_dir
    
    def save_prototype_files(self, project_id: str, files: Dict[str, str]) -> Dict[str, str]:
        """Save generated files to prototype directory."""
        prototype_dir = self.create_prototype_directory(project_id)
        saved_files = {}
        
        for filename, content in files.items():
            file_path = prototype_dir / filename
            
            try:
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                saved_files[filename] = str(file_path)
                logger.info(f"Saved file: {filename}")
                
            except Exception as e:
                logger.error(f"Error saving file {filename}: {e}")
                continue
        
        return saved_files
    
    def get_prototype_files(self, project_id: str) -> Dict[str, str]:
        """Get all files for a prototype."""
        prototype_dir = self.base_path / project_id
        
        if not prototype_dir.exists():
            return {}
        
        files = {}
        
        for file_path in prototype_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(prototype_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files[str(relative_path)] = f.read()
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    continue
        
        return files
    
    def get_file_content(self, project_id: str, filename: str) -> Optional[str]:
        """Get content of a specific file."""
        file_path = self.base_path / project_id / filename
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def update_file_content(self, project_id: str, filename: str, content: str) -> bool:
        """Update content of a specific file."""
        file_path = self.base_path / project_id / filename
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Updated file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating file {file_path}: {e}")
            return False
    
    def delete_prototype(self, project_id: str) -> bool:
        """Delete entire prototype directory."""
        prototype_dir = self.base_path / project_id
        
        if not prototype_dir.exists():
            return False
        
        try:
            shutil.rmtree(prototype_dir)
            logger.info(f"Deleted prototype: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting prototype {project_id}: {e}")
            return False
    
    def list_prototypes(self) -> List[Dict[str, Any]]:
        """List all prototypes with metadata."""
        prototypes = []
        
        for project_dir in self.base_path.iterdir():
            if project_dir.is_dir():
                metadata_file = project_dir / "metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        prototypes.append({
                            "project_id": project_dir.name,
                            "metadata": metadata,
                            "created_at": metadata.get("created_at"),
                            "file_count": len(list(project_dir.rglob("*")))
                        })
                    except Exception as e:
                        logger.error(f"Error reading metadata for {project_dir.name}: {e}")
                        continue
        
        return sorted(prototypes, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def get_prototype_size(self, project_id: str) -> int:
        """Get total size of prototype files in bytes."""
        prototype_dir = self.base_path / project_id
        
        if not prototype_dir.exists():
            return 0
        
        total_size = 0
        
        for file_path in prototype_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def cleanup_expired_prototypes(self, expiry_days: int = 30) -> List[str]:
        """Clean up expired prototypes."""
        cleaned_prototypes = []
        cutoff_date = datetime.now() - timedelta(days=expiry_days)
        
        for project_dir in self.base_path.iterdir():
            if not project_dir.is_dir():
                continue
            
            metadata_file = project_dir / "metadata.json"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                    
                    if created_at < cutoff_date:
                        if self.delete_prototype(project_dir.name):
                            cleaned_prototypes.append(project_dir.name)
                            
                except Exception as e:
                    logger.error(f"Error checking expiry for {project_dir.name}: {e}")
                    continue
            else:
                # If no metadata, check directory modification time
                dir_mtime = datetime.fromtimestamp(project_dir.stat().st_mtime)
                if dir_mtime < cutoff_date:
                    if self.delete_prototype(project_dir.name):
                        cleaned_prototypes.append(project_dir.name)
        
        logger.info(f"Cleaned up {len(cleaned_prototypes)} expired prototypes")
        return cleaned_prototypes
    
    def create_zip_archive(self, project_id: str) -> Optional[str]:
        """Create a zip archive of the prototype."""
        import zipfile
        
        prototype_dir = self.base_path / project_id
        
        if not prototype_dir.exists():
            return None
        
        zip_path = self.base_path / f"{project_id}.zip"
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in prototype_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(prototype_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Created zip archive: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Error creating zip archive for {project_id}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_prototypes = 0
        total_size = 0
        total_files = 0
        
        for project_dir in self.base_path.iterdir():
            if project_dir.is_dir():
                total_prototypes += 1
                
                for file_path in project_dir.rglob("*"):
                    if file_path.is_file():
                        total_files += 1
                        total_size += file_path.stat().st_size
        
        return {
            "total_prototypes": total_prototypes,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_path": str(self.base_path)
        }
    
    def validate_prototype_structure(self, project_id: str) -> Dict[str, Any]:
        """Validate prototype file structure."""
        prototype_dir = self.base_path / project_id
        
        if not prototype_dir.exists():
            return {"valid": False, "error": "Prototype directory not found"}
        
        required_files = ["index.html"]
        optional_files = ["styles.css", "script.js", "metadata.json"]
        
        validation_result = {
            "valid": True,
            "missing_files": [],
            "present_files": [],
            "errors": []
        }
        
        # Check required files
        for filename in required_files:
            file_path = prototype_dir / filename
            if file_path.exists():
                validation_result["present_files"].append(filename)
            else:
                validation_result["missing_files"].append(filename)
                validation_result["valid"] = False
        
        # Check optional files
        for filename in optional_files:
            file_path = prototype_dir / filename
            if file_path.exists():
                validation_result["present_files"].append(filename)
        
        # Check for index.html content
        index_file = prototype_dir / "index.html"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content.strip()) < 100:
                        validation_result["errors"].append("index.html appears to be empty or too short")
                        validation_result["valid"] = False
            except Exception as e:
                validation_result["errors"].append(f"Error reading index.html: {e}")
                validation_result["valid"] = False
        
        return validation_result
