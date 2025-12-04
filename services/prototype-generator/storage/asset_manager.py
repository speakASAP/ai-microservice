"""
Asset management for prototype generation.
"""

import os
import shutil
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AssetManager:
    """Manages assets for generated prototypes."""
    
    def __init__(self, base_path: str = "./prototypes"):
        """Initialize asset manager."""
        self.base_path = Path(base_path)
        
    def create_asset_directory(self, project_id: str) -> Path:
        """Create asset directory for a project."""
        asset_dir = self.base_path / project_id / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (asset_dir / "images").mkdir(exist_ok=True)
        (asset_dir / "fonts").mkdir(exist_ok=True)
        (asset_dir / "icons").mkdir(exist_ok=True)
        
        return asset_dir
    
    def save_asset(self, project_id: str, filename: str, content: bytes) -> bool:
        """Save an asset file."""
        try:
            asset_dir = self.create_asset_directory(project_id)
            file_path = asset_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Asset saved: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving asset {filename}: {e}")
            return False
    
    def get_asset(self, project_id: str, filename: str) -> Optional[bytes]:
        """Get an asset file."""
        try:
            asset_path = self.base_path / project_id / "assets" / filename
            
            if not asset_path.exists():
                return None
            
            with open(asset_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error getting asset {filename}: {e}")
            return None
    
    def list_assets(self, project_id: str) -> List[Dict[str, Any]]:
        """List all assets for a project."""
        assets = []
        asset_dir = self.base_path / project_id / "assets"
        
        if not asset_dir.exists():
            return assets
        
        for file_path in asset_dir.rglob("*"):
            if file_path.is_file():
                assets.append({
                    "filename": file_path.name,
                    "path": str(file_path.relative_to(asset_dir)),
                    "size": file_path.stat().st_size,
                    "type": file_path.suffix
                })
        
        return assets
    
    def delete_asset(self, project_id: str, filename: str) -> bool:
        """Delete an asset file."""
        try:
            asset_path = self.base_path / project_id / "assets" / filename
            
            if asset_path.exists():
                asset_path.unlink()
                logger.info(f"Asset deleted: {filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting asset {filename}: {e}")
            return False
    
    def cleanup_assets(self, project_id: str) -> bool:
        """Clean up all assets for a project."""
        try:
            asset_dir = self.base_path / project_id / "assets"
            
            if asset_dir.exists():
                shutil.rmtree(asset_dir)
                logger.info(f"Assets cleaned up for project {project_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cleaning up assets for {project_id}: {e}")
            return False
