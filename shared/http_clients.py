"""
StateX AI HTTP Clients

Shared HTTP clients for AI services to communicate with other StateX services.
Provides unified interface for service-to-service communication.
Supports both Docker container and localhost environments.
"""

import httpx
import asyncio
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class BaseServiceClient:
    """Base class for service HTTP clients"""
    
    def __init__(self, base_url: str, timeout: int = 30, retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retries = retries
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.warning(f"HTTP request failed (attempt {attempt + 1}/{self.retries}): {e}")
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("All retry attempts failed")

class NotificationServiceClient(BaseServiceClient):
    """Client for communicating with StateX Notification Service"""
    
    def __init__(self):
        # Support both Docker and localhost environments
        notification_base_url = os.getenv("NOTIFICATION_SERVICE_URL", "https://notifications.statex.cz")
        super().__init__(notification_base_url)
    
    async def send_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification via external notification service"""
        return await self._make_request("POST", "/api/notifications", json=notification_data)
    
    async def send_ai_analysis_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send AI analysis notification with comprehensive results"""
        return await self._make_request("POST", "/api/notifications/ai-analysis", json=notification_data)
    
    async def get_notification_status(self, notification_id: str) -> Dict[str, Any]:
        """Get notification status"""
        return await self._make_request("GET", f"/api/notifications/{notification_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check notification service health"""
        return await self._make_request("GET", "/health")

