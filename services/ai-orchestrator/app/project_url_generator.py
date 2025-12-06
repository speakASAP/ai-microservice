"""
Project Prototype URL Generation

This module creates unique project IDs for each submission and generates
project plan URLs and offer detail URLs with proper validation and accessibility testing.

Requirements: 6.8, 6.9
"""

import os
import logging
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel
import httpx
import asyncio

logger = logging.getLogger(__name__)

class ProjectURLs(BaseModel):
    """Project URL details"""
    project_id: str
    plan_url: str
    offer_url: str
    created_at: str
    is_accessible: bool = False
    validation_results: Dict[str, Any] = {}

class ProjectURLGenerator:
    """Generates and validates project prototype URLs"""
    
    def __init__(self):
        frontend_port = os.getenv("FRONTEND_PORT", "3602")
        self.base_domain = f"localhost:{frontend_port}"  # Configured via environment
        self.url_prefix = "project-proto"
        self.validation_timeout = 10  # seconds
        
    def generate_unique_project_id(self, submission_id: str, user_id: str = None, timestamp: datetime = None) -> str:
        """
        Generate a unique project ID for each submission
        
        Args:
            submission_id: The submission identifier
            user_id: Optional user identifier
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            str: Unique project ID
        """
        try:
            # Use current timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now()
            
            # Create a unique string combining multiple factors
            unique_string = f"{submission_id}_{user_id or 'anonymous'}_{timestamp.isoformat()}_{time.time()}"
            
            # Generate hash for uniqueness and brevity
            hash_object = hashlib.md5(unique_string.encode())
            hash_hex = hash_object.hexdigest()
            
            # Create readable project ID with timestamp component
            timestamp_component = timestamp.strftime("%Y%m%d_%H%M%S")
            project_id = f"{timestamp_component}_{hash_hex[:8]}"
            
            logger.info(f"Generated project ID: {project_id} for submission: {submission_id}")
            return project_id
            
        except Exception as e:
            logger.error(f"Error generating project ID: {e}")
            # Fallback to UUID-based ID
            fallback_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
            logger.warning(f"Using fallback project ID: {fallback_id}")
            return fallback_id
    
    def generate_project_urls(self, project_id: str, custom_domain: str = None) -> ProjectURLs:
        """
        Generate project plan and offer URLs
        
        Args:
            project_id: Unique project identifier
            custom_domain: Optional custom domain (defaults to localhost:{frontend_port})
            
        Returns:
            ProjectURLs: Generated URLs with metadata
        """
        try:
            domain = custom_domain or self.base_domain
            
            # Generate the URLs according to the specification
            plan_url = f"http://{self.url_prefix}_{project_id}.{domain}/plan"
            offer_url = f"http://{self.url_prefix}_{project_id}.{domain}/offer"
            
            urls = ProjectURLs(
                project_id=project_id,
                plan_url=plan_url,
                offer_url=offer_url,
                created_at=datetime.now().isoformat(),
                is_accessible=False,
                validation_results={}
            )
            
            logger.info(f"Generated URLs for project {project_id}: plan={plan_url}, offer={offer_url}")
            return urls
            
        except Exception as e:
            logger.error(f"Error generating project URLs: {e}")
            raise
    
    async def validate_url_accessibility(self, url: str) -> Dict[str, Any]:
        """
        Validate that a URL is accessible
        
        Args:
            url: URL to validate
            
        Returns:
            Dict: Validation results
        """
        validation_result = {
            "url": url,
            "is_accessible": False,
            "status_code": None,
            "response_time": None,
            "error_message": None,
            "validated_at": datetime.now().isoformat()
        }
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=self.validation_timeout) as client:
                response = await client.get(url)
                
                response_time = time.time() - start_time
                validation_result.update({
                    "is_accessible": response.status_code < 400,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content_length": len(response.content) if response.content else 0
                })
                
                if response.status_code >= 400:
                    validation_result["error_message"] = f"HTTP {response.status_code}: {response.reason_phrase}"
                
        except httpx.TimeoutException:
            validation_result["error_message"] = f"Request timeout after {self.validation_timeout} seconds"
        except httpx.ConnectError:
            validation_result["error_message"] = "Connection error - service may not be running"
        except Exception as e:
            validation_result["error_message"] = str(e)
        
        logger.info(f"URL validation for {url}: accessible={validation_result['is_accessible']}")
        return validation_result
    
    async def validate_project_urls(self, project_urls: ProjectURLs) -> ProjectURLs:
        """
        Validate accessibility of both plan and offer URLs
        
        Args:
            project_urls: ProjectURLs object to validate
            
        Returns:
            ProjectURLs: Updated object with validation results
        """
        try:
            # Validate both URLs concurrently
            plan_validation, offer_validation = await asyncio.gather(
                self.validate_url_accessibility(project_urls.plan_url),
                self.validate_url_accessibility(project_urls.offer_url),
                return_exceptions=True
            )
            
            # Handle validation results
            if isinstance(plan_validation, Exception):
                plan_validation = {
                    "url": project_urls.plan_url,
                    "is_accessible": False,
                    "error_message": str(plan_validation)
                }
            
            if isinstance(offer_validation, Exception):
                offer_validation = {
                    "url": project_urls.offer_url,
                    "is_accessible": False,
                    "error_message": str(offer_validation)
                }
            
            # Update project URLs with validation results
            project_urls.validation_results = {
                "plan_validation": plan_validation,
                "offer_validation": offer_validation,
                "overall_accessible": plan_validation.get("is_accessible", False) and offer_validation.get("is_accessible", False)
            }
            
            project_urls.is_accessible = project_urls.validation_results["overall_accessible"]
            
            logger.info(f"Project {project_urls.project_id} URL validation completed. Accessible: {project_urls.is_accessible}")
            return project_urls
            
        except Exception as e:
            logger.error(f"Error validating project URLs: {e}")
            project_urls.validation_results = {
                "error": str(e),
                "overall_accessible": False
            }
            project_urls.is_accessible = False
            return project_urls
    
    async def create_and_validate_project_urls(self, submission_id: str, user_id: str = None, custom_domain: str = None) -> ProjectURLs:
        """
        Complete workflow: generate project ID, create URLs, and validate accessibility
        
        Args:
            submission_id: Submission identifier
            user_id: Optional user identifier
            custom_domain: Optional custom domain
            
        Returns:
            ProjectURLs: Complete project URL information with validation
        """
        try:
            # Generate unique project ID
            project_id = self.generate_unique_project_id(submission_id, user_id)
            
            # Generate URLs
            project_urls = self.generate_project_urls(project_id, custom_domain)
            
            # Validate accessibility (optional - can be disabled for development)
            validated_urls = await self.validate_project_urls(project_urls)
            
            return validated_urls
            
        except Exception as e:
            logger.error(f"Error in complete URL generation workflow: {e}")
            raise
    
    def get_project_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract project ID from a generated URL
        
        Args:
            url: Project URL
            
        Returns:
            Optional[str]: Extracted project ID or None if not found
        """
        try:
            # Parse URL to extract project ID
            # Expected format: http://project-proto_{project_id}.localhost:{frontend_port}/plan
            if self.url_prefix in url:
                # Extract the subdomain part
                start_marker = f"{self.url_prefix}_"
                end_marker = f".{self.base_domain}"
                
                start_idx = url.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = url.find(end_marker, start_idx)
                    if end_idx != -1:
                        project_id = url[start_idx:end_idx]
                        return project_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting project ID from URL {url}: {e}")
            return None
    
    def is_valid_project_url(self, url: str) -> bool:
        """
        Check if a URL matches the expected project URL format
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL format is valid
        """
        try:
            project_id = self.get_project_id_from_url(url)
            return project_id is not None and len(project_id) > 0
        except:
            return False
    
    async def batch_validate_urls(self, urls: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple URLs concurrently
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            Dict: Validation results for each URL
        """
        try:
            # Create validation tasks
            validation_tasks = [self.validate_url_accessibility(url) for url in urls]
            
            # Execute all validations concurrently
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            validation_results = {}
            for i, result in enumerate(results):
                url = urls[i]
                if isinstance(result, Exception):
                    validation_results[url] = {
                        "url": url,
                        "is_accessible": False,
                        "error_message": str(result)
                    }
                else:
                    validation_results[url] = result
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in batch URL validation: {e}")
            return {url: {"url": url, "is_accessible": False, "error_message": str(e)} for url in urls}

# Global URL generator instance
project_url_generator = ProjectURLGenerator()