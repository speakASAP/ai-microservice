#!/usr/bin/env python3
"""
StateX Gemini AI Service

This service provides access to Google Gemini AI models:
1. Gemini 2.5 Flash - Fast and efficient model
2. Gemini 2.5 Pro - Most powerful model
3. Gemini 2.5 Flash-Lite - Fastest and most cost-efficient

Port: 3388 (configured in ai-microservice/.env)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import json
import ssl
import time
import logging
from datetime import datetime
from enum import Enum
import os
import sys
import pathlib
from pathlib import Path

# Import logger from shared directory
import importlib.util
shared_path = "/app/shared"
logger_spec = importlib.util.spec_from_file_location("logger", os.path.join(shared_path, "logger.py"))
logger_module = importlib.util.module_from_spec(logger_spec)
logger_spec.loader.exec_module(logger_module)
logger = logger_module.setup_logger(__name__, service_name="gemini-ai-service")

# Configure Uvicorn logging to use centralized logger
import logging

# Configure root logger to use our centralized logger
root_logger = logging.getLogger()
root_logger.handlers = []

# Configure Uvicorn loggers to use our centralized logger
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = []
uvicorn_logger.propagate = True

uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = []
uvicorn_access_logger.propagate = True

# Configure uvicorn loggers to use centralized logger format with timestamps
timestamp_format = os.getenv('LOG_TIMESTAMP_FORMAT', '%Y-%m-%d %H:%M:%S')
log_format = f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure uvicorn loggers
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    uvicorn_logger = logging.getLogger(logger_name)
    uvicorn_logger.handlers = []
    
    # Add handler with timestamp format
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format, datefmt=timestamp_format)
    handler.setFormatter(formatter)
    uvicorn_logger.addHandler(handler)
    uvicorn_logger.setLevel(logging.INFO)

app = FastAPI(
    title="StateX Gemini AI Service",
    description="Google Gemini AI models service using Google AI SDK",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
env_path = pathlib.Path(__file__).parent.parent.parent.parent.parent / '.env'
logger.debug(f"Loading .env from: {env_path}")
logger.debug(f".env file exists: {env_path.exists()}")
load_dotenv(env_path, override=True)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Debug: Print environment variables
logger.debug("Environment variables after loading .env:")
api_key_value = os.getenv('GEMINI_API_KEY')
logger.info(f"  GEMINI_API_KEY: {'SET' if api_key_value else 'NOT SET'}")
if api_key_value:
    logger.info(f"  GEMINI_API_KEY (first 10 chars): {api_key_value[:10]}...")

class AnalysisType(str, Enum):
    BUSINESS_ANALYSIS = "business_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    CONTENT_GENERATION = "content_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class GeminiModel(str, Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"

class GeminiAnalysisRequest(BaseModel):
    text_content: str
    analysis_type: AnalysisType = AnalysisType.BUSINESS_ANALYSIS
    user_name: str = "User"
    model: Optional[GeminiModel] = None  # Auto-select if not specified

class GeminiAnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    model_used: str
    processing_time: float
    confidence: float
    error: Optional[str] = None

class GeminiModelInfo(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    status: str  # available, unavailable, loading

class GeminiAIService:
    def __init__(self):
        self.available_models = {}
        self.model_preferences = {
            AnalysisType.BUSINESS_ANALYSIS: [
                GeminiModel.GEMINI_2_5_FLASH,
                GeminiModel.GEMINI_2_5_PRO,
                GeminiModel.GEMINI_2_5_FLASH_LITE
            ],
            AnalysisType.TECHNICAL_ANALYSIS: [
                GeminiModel.GEMINI_2_5_PRO,
                GeminiModel.GEMINI_2_5_FLASH,
                GeminiModel.GEMINI_2_5_FLASH_LITE
            ],
            AnalysisType.CONTENT_GENERATION: [
                GeminiModel.GEMINI_2_5_FLASH,
                GeminiModel.GEMINI_2_5_PRO,
                GeminiModel.GEMINI_2_5_FLASH_LITE
            ],
            AnalysisType.SENTIMENT_ANALYSIS: [
                GeminiModel.GEMINI_2_5_FLASH,
                GeminiModel.GEMINI_2_5_PRO,
                GeminiModel.GEMINI_2_5_FLASH_LITE
            ]
        }
        
        # Initialize available models
        self.available_models = {
            "gemini-2.5-flash": {
                "name": "Gemini 2.5 Flash",
                "description": "Most balanced model with 1M token context window",
                "capabilities": ["text_generation", "reasoning", "analysis"],
                "status": "available"
            },
            "gemini-2.5-pro": {
                "name": "Gemini 2.5 Pro",
                "description": "Most powerful model with advanced reasoning",
                "capabilities": ["text_generation", "reasoning", "analysis", "complex_tasks"],
                "status": "available"
            },
            "gemini-2.5-flash-lite": {
                "name": "Gemini 2.5 Flash Lite",
                "description": "Fastest and most cost-efficient model",
                "capabilities": ["text_generation", "quick_analysis"],
                "status": "available"
            }
        }
        
    def get_best_model(self, analysis_type: AnalysisType, model: str = None) -> str:
        """Get the best available model for the given analysis type"""
        
        logger.info(f"🔍 get_best_model called with analysis_type={analysis_type}, model={model}")
        
        if model:
            # Use specific model if provided
            if model in self.available_models:
                logger.info(f"✅ Selected specific model: {model}")
                return model
            else:
                logger.warning(f"⚠️ Model {model} not found, falling back to auto-selection")
        
        # Auto-select best model for analysis type
        preferred_models = self.model_preferences.get(analysis_type, [])
        
        for preferred_model in preferred_models:
            model_name = preferred_model.value
            if model_name in self.available_models:
                logger.info(f"✅ Selected auto model: {model_name}")
                return model_name
        
        # Fallback to first available model
        if self.available_models:
            fallback_model = list(self.available_models.keys())[0]
            logger.info(f"✅ Selected fallback model: {fallback_model}")
            return fallback_model
        
        raise HTTPException(
            status_code=503, 
            detail="No Gemini models are currently available"
        )
    
    async def analyze_with_gemini(self, request: GeminiAnalysisRequest) -> Dict[str, Any]:
        """Analyze using Google Gemini API"""
        
        # Determine model to use
        model = self.get_best_model(request.analysis_type, request.model.value if request.model else None)
        
        logger.info(f"🤖 Analyzing with Gemini: {model}")
        
        # Create a comprehensive prompt based on analysis type
        if request.analysis_type == AnalysisType.BUSINESS_ANALYSIS:
            prompt = f"""Analyze this business request and provide a comprehensive business analysis:

User: {request.user_name}
Request: {request.text_content}

Please provide a JSON response with these fields:
- business_type: The type of business
- pain_points: Array of current pain points
- opportunities: Array of business opportunities with name, description, potential, timeline
- technical_recommendations: Object with frontend, backend, integrations arrays
- next_steps: Array of action items with action, priority, timeline
- budget_estimate: Object with development, infrastructure, maintenance costs
- confidence: Float between 0 and 1
- summary: String summary of the analysis
"""
        elif request.analysis_type == AnalysisType.TECHNICAL_ANALYSIS:
            prompt = f"""Provide a technical analysis for this request:

User: {request.user_name}
Request: {request.text_content}

Please provide a JSON response with:
- technical_requirements: Object with frontend, backend, database, infrastructure
- architecture_recommendations: Object with patterns, technologies, scalability
- implementation_phases: Array of phases with name, description, timeline
- technology_stack: Object with recommended technologies
- confidence: Float between 0 and 1
- summary: String summary
"""
        else:
            prompt = f"""Analyze this request and provide insights:

User: {request.user_name}
Request: {request.text_content}

Please provide a JSON response with:
- key_insights: Array of insights
- recommendations: Array of recommendations
- next_steps: Array of next steps
- confidence: Float between 0 and 1
- summary: String summary
"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Create SSL context that doesn't verify certificates for testing
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 2000
                    }
                }
                
                url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={GEMINI_API_KEY}"
                
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract the generated content
                        if "candidates" in result and len(result["candidates"]) > 0:
                            candidate = result["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                ai_response = candidate["content"]["parts"][0].get("text", "")
                            else:
                                ai_response = ""
                        else:
                            ai_response = ""
                        
                        # Try to parse JSON response
                        try:
                            json_start = ai_response.find('{')
                            json_end = ai_response.rfind('}') + 1
                            if json_start != -1 and json_end != -1:
                                json_str = ai_response[json_start:json_end]
                                analysis = json.loads(json_str)
                            else:
                                analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                        except:
                            analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                        
                        analysis["ai_service"] = "Google Gemini"
                        analysis["model_used"] = model
                        return analysis
                    else:
                        error_text = await response.text()
                        raise Exception(f"Gemini API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise e
    
    def _parse_text_response(self, text: str, user_name: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Parse text response from Gemini into structured format"""
        return {
            "summary": f"Gemini Analysis for {user_name}: {text[:200]}...",
            "key_insights": ["AI-generated insight 1", "AI-generated insight 2"],
            "recommendations": ["AI-generated recommendation 1", "AI-generated recommendation 2"],
            "confidence": 0.8,
            "ai_service": "Google Gemini",
            "model_used": "gemini-2.5-flash"
        }

# Initialize service
gemini_ai_service = GeminiAIService()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("🚀 Gemini AI Service starting up...")
    if GEMINI_API_KEY:
        logger.info("✅ Gemini API key found - service ready")
    else:
        logger.warning("⚠️ No Gemini API key found - service will not function")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if GEMINI_API_KEY else "degraded",
        "service": "gemini-ai-service",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "api_key_configured": bool(GEMINI_API_KEY),
        "available_models": list(gemini_ai_service.available_models.keys())
    }

@app.get("/models")
async def get_available_models():
    """Get list of available Gemini models"""
    return {
        "models": gemini_ai_service.available_models,
        "service": "gemini-ai-service"
    }

@app.post("/analyze", response_model=GeminiAnalysisResponse)
async def analyze_text(request: GeminiAnalysisRequest):
    """Analyze text using Gemini AI"""
    start_time = time.time()
    
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=503, 
                detail="Gemini API key not configured"
            )
        
        analysis = await gemini_ai_service.analyze_with_gemini(request)
        processing_time = time.time() - start_time
        
        return GeminiAnalysisResponse(
            success=True,
            analysis=analysis,
            model_used=analysis.get("model_used", "unknown"),
            processing_time=processing_time,
            confidence=analysis.get("confidence", 0.8)
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        processing_time = time.time() - start_time
        
        return GeminiAnalysisResponse(
            success=False,
            analysis={},
            model_used="none",
            processing_time=processing_time,
            confidence=0.0,
            error=str(e)
        )

@app.post("/api/analyze-text")
async def analyze_text_compatibility(request: GeminiAnalysisRequest):
    """Compatibility endpoint for NLP Agent - proxies to /analyze with business analysis"""
    # Override analysis type to business analysis for NLP compatibility
    request.analysis_type = AnalysisType.BUSINESS_ANALYSIS
    return await analyze_text(request)

@app.get("/api/status/{submission_id}")
async def get_submission_status(submission_id: str):
    """Get status of a submission"""
    # For now, return a mock status since we don't have persistent storage
    return {
        "submission_id": submission_id,
        "status": "completed",
        "progress": 100,
        "current_step": "analysis_complete",
        "steps_completed": [
            "requirements_analysis",
            "technology_recommendation", 
            "prototype_generation"
        ],
        "timestamp": datetime.now().isoformat(),
        "message": "Analysis completed successfully"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "StateX Gemini AI Service",
        "version": "1.0.0",
        "description": "Google Gemini AI models service using Google AI SDK",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "analyze": "/analyze",
            "analyze_text": "/api/analyze-text",
            "status": "/api/status/{submission_id}",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GEMINI_AI_SERVICE_PORT", "3388"))
    
    # Custom logging configuration for Uvicorn
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("GEMINI_AI_SERVICE_PORT", "3388")), log_config=log_config)
