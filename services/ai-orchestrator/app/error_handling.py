"""
Comprehensive Error Handling and Recovery System

This module provides robust error classification, retry strategies, graceful degradation,
and fallback mechanisms for the multi-agent AI orchestration workflow.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel
import httpx
import json

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of different error types"""
    VALIDATION_ERROR = "validation_error"
    AI_SERVICE_ERROR = "ai_service_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    WORKFLOW_ERROR = "workflow_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RetryStrategy(Enum):
    """Different retry strategies"""
    NO_RETRY = "no_retry"
    LINEAR_RETRY = "linear_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    submission_id: str
    workflow_id: Optional[str] = None
    agent_type: Optional[str] = None
    task_id: Optional[str] = None
    service_url: Optional[str] = None
    operation: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    attempt_number: int = 1
    max_attempts: int = 3

class ErrorClassification(BaseModel):
    """Classified error information"""
    error_type: ErrorType
    severity: ErrorSeverity
    is_retryable: bool
    retry_strategy: RetryStrategy
    max_retries: int
    base_delay: float
    max_delay: float
    backoff_multiplier: float = 2.0
    jitter: bool = True
    fallback_available: bool = False
    requires_immediate_attention: bool = False

class ErrorRecord(BaseModel):
    """Record of an error occurrence"""
    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    retry_count: int = 0
    final_outcome: Optional[str] = None

class ErrorClassifier:
    """Classifies errors based on exception type and context"""
    
    @staticmethod
    def classify_error(exception: Exception, context: ErrorContext) -> ErrorClassification:
        """Classify an error and return appropriate handling configuration"""
        error_type = ErrorClassifier._determine_error_type(exception, context)
        return ErrorClassifier._get_config(error_type)
    
    @staticmethod
    def _determine_error_type(exception: Exception, context: ErrorContext) -> ErrorType:
        """Determine the error type based on exception and context"""
        error_message = str(exception).lower()
        
        # Network-related errors
        if isinstance(exception, (httpx.ConnectError, httpx.NetworkError)):
            return ErrorType.NETWORK_ERROR
        
        # Timeout errors
        if isinstance(exception, (httpx.TimeoutException, asyncio.TimeoutError)):
            return ErrorType.TIMEOUT_ERROR
        
        # HTTP status code based classification
        if isinstance(exception, httpx.HTTPStatusError):
            status_code = exception.response.status_code
            if status_code == 401:
                return ErrorType.AUTHENTICATION_ERROR
            elif status_code == 429:
                return ErrorType.RATE_LIMIT_ERROR
            elif status_code in [502, 503, 504]:
                return ErrorType.SERVICE_UNAVAILABLE
            elif 400 <= status_code < 500:
                return ErrorType.VALIDATION_ERROR
            elif 500 <= status_code < 600:
                return ErrorType.AI_SERVICE_ERROR
        
        # Validation errors
        if "validation" in error_message or "invalid" in error_message:
            return ErrorType.VALIDATION_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    @staticmethod
    def _get_config(error_type: ErrorType) -> ErrorClassification:
        """Get retry configuration for specific error type"""
        configs = {
            ErrorType.NETWORK_ERROR: ErrorClassification(
                error_type=error_type,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0,
                fallback_available=True
            ),
            ErrorType.TIMEOUT_ERROR: ErrorClassification(
                error_type=error_type,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                retry_strategy=RetryStrategy.LINEAR_RETRY,
                max_retries=2,
                base_delay=5.0,
                max_delay=15.0,
                fallback_available=True
            ),
            ErrorType.AI_SERVICE_ERROR: ErrorClassification(
                error_type=error_type,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                base_delay=2.0,
                max_delay=60.0,
                fallback_available=True
            ),
        }
        
        return configs.get(error_type, ErrorClassification(
            error_type=ErrorType.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=2,
            base_delay=2.0,
            max_delay=30.0,
            fallback_available=False
        ))

class GracefulDegradation:
    """Handles graceful degradation when services are unavailable"""
    
    async def handle_agent_failure(
        self, 
        agent_type: str, 
        context: ErrorContext, 
        original_error: Exception
    ) -> Dict[str, Any]:
        """Handle agent failure with graceful degradation"""
        logger.warning(f"Applying graceful degradation for {agent_type} agent in {context.submission_id}")
        
        if agent_type == "nlp":
            return await self._nlp_fallback(context, original_error)
        elif agent_type == "asr":
            return await self._asr_fallback(context, original_error)
        elif agent_type == "document":
            return await self._document_fallback(context, original_error)
        elif agent_type == "prototype":
            return await self._prototype_fallback(context, original_error)
        else:
            return self._create_minimal_fallback_result(agent_type, context, original_error)
    
    async def _nlp_fallback(self, context: ErrorContext, error: Exception) -> Dict[str, Any]:
        """Fallback strategy for NLP agent"""
        text_content = context.input_data.get("text_content", "")
        word_count = len(text_content.split()) if text_content else 0
        
        return {
            "status": "degraded",
            "processing_time": 0.1,
            "confidence": 0.3,
            "results": {
                "text_summary": f"Text analysis unavailable. Content contains {word_count} words.",
                "key_insights": ["Basic text processing"],
                "sentiment_analysis": {"overall_sentiment": "neutral", "confidence": 0.3},
                "topic_categorization": ["general"],
                "business_requirements": {
                    "identified_needs": ["Custom solution required"],
                    "complexity_estimate": "medium" if word_count > 100 else "low",
                    "technology_suggestions": ["Web-based solution"]
                }
            },
            "fallback_reason": f"NLP service unavailable: {str(error)[:100]}"
        }
    
    async def _asr_fallback(self, context: ErrorContext, error: Exception) -> Dict[str, Any]:
        """Fallback strategy for ASR agent"""
        return {
            "status": "degraded",
            "processing_time": 0.1,
            "confidence": 0.0,
            "results": {
                "transcript": "Voice transcription unavailable - please provide text description",
                "confidence_score": 0.0,
                "language": "unknown",
                "duration": 0
            },
            "fallback_reason": f"ASR service unavailable: {str(error)[:100]}"
        }
    
    async def _document_fallback(self, context: ErrorContext, error: Exception) -> Dict[str, Any]:
        """Fallback strategy for Document AI agent"""
        file_urls = context.input_data.get("file_urls", [])
        
        return {
            "status": "degraded",
            "processing_time": 0.1,
            "confidence": 0.0,
            "results": {
                "extracted_text": f"Document processing unavailable for {len(file_urls)} files",
                "document_type": "unknown",
                "page_count": 0,
                "file_analysis": [{"filename": url.split("/")[-1], "status": "not_processed"} 
                                for url in file_urls]
            },
            "fallback_reason": f"Document AI service unavailable: {str(error)[:100]}"
        }
    
    async def _prototype_fallback(self, context: ErrorContext, error: Exception) -> Dict[str, Any]:
        """Fallback strategy for Prototype Generator agent"""
        return {
            "status": "degraded",
            "processing_time": 0.1,
            "confidence": 0.4,
            "results": {
                "type": "web_application",
                "complexity": "medium",
                "estimated_time": "4-8 weeks",
                "tech_stack": ["React", "Node.js", "PostgreSQL"],
                "features": ["User interface design", "Basic functionality implementation"],
                "deployment": {
                    "url": "Prototype generation unavailable",
                    "status": "pending",
                    "deployment_time": "Manual review required"
                }
            },
            "fallback_reason": f"Prototype service unavailable: {str(error)[:100]}"
        }
    
    def _create_minimal_fallback_result(
        self, 
        agent_type: str, 
        context: ErrorContext, 
        error: Exception
    ) -> Dict[str, Any]:
        """Create minimal fallback result when specific fallback fails"""
        return {
            "status": "failed",
            "processing_time": 0.0,
            "confidence": 0.0,
            "results": {
                "error": f"{agent_type} agent unavailable",
                "message": "Service temporarily unavailable, please try again later"
            },
            "fallback_reason": f"All fallback strategies failed: {str(error)[:100]}"
        }

class ErrorRecoveryManager:
    """Manages error recovery and tracks error patterns"""
    
    def __init__(self):
        self.error_history: List[ErrorRecord] = []
        self.classifier = ErrorClassifier()
        self.graceful_degradation = GracefulDegradation()
    
    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        operation: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main error handling entry point"""
        # Classify the error
        classification = self.classifier.classify_error(exception, context)
        
        # Record the error
        error_record = self._create_error_record(exception, context, classification)
        self.error_history.append(error_record)
        
        # Log the error
        self._log_error(error_record, classification)
        
        # Handle based on classification
        try:
            if classification.fallback_available and context.agent_type:
                # Use graceful degradation
                return await self.graceful_degradation.handle_agent_failure(
                    context.agent_type, context, exception
                )
            else:
                # Re-raise if no recovery possible
                raise exception
        
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            error_record.final_outcome = "recovery_failed"
            raise recovery_error
    
    def _create_error_record(
        self,
        exception: Exception,
        context: ErrorContext,
        classification: ErrorClassification
    ) -> ErrorRecord:
        """Create error record for tracking"""
        return ErrorRecord(
            error_id=f"{context.submission_id}_{int(time.time())}",
            error_type=classification.error_type,
            severity=classification.severity,
            message=str(exception),
            context={
                "submission_id": context.submission_id,
                "workflow_id": context.workflow_id,
                "agent_type": context.agent_type,
                "task_id": context.task_id,
                "service_url": context.service_url,
                "operation": context.operation,
                "attempt_number": context.attempt_number
            },
            timestamp=datetime.now()
        )
    
    def _log_error(self, error_record: ErrorRecord, classification: ErrorClassification):
        """Log error with appropriate level"""
        log_message = (
            f"Error {error_record.error_id}: {error_record.error_type.value} "
            f"({error_record.severity.value}) - {error_record.message}"
        )
        
        if classification.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif classification.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif classification.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        severity_counts = {}
        recent_errors = []
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for error in self.error_history:
            # Count by type
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Recent errors
            if error.timestamp > cutoff_time:
                recent_errors.append({
                    "error_id": error.error_id,
                    "error_type": error.error_type.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp.isoformat()
                })
        
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_type": error_counts,
            "error_counts_by_severity": severity_counts,
            "recent_errors_24h": len(recent_errors),
            "recent_error_details": recent_errors[-10:]  # Last 10 recent errors
        }

# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()