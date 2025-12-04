"""
API module for prototype generation service.
"""

from .queue import router as queue_router
from .prototype import router as prototype_router

__all__ = ["queue_router", "prototype_router"]
