#!/usr/bin/env python3
"""
StateX Free AI Service

This service provides access to free AI models:
1. OpenRouter API - Google Gemini, Claude, GPT models
2. Ollama (Local LLM) - Llama 2, Mistral, CodeLlama
3. Hugging Face Inference API - Various open-source models

Port: 3386 (configured in ai-microservice/.env)
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
import os
from datetime import datetime
from enum import Enum

# Configure logging with timestamps
import sys
from pathlib import Path
# Add utils to path
utils_path = Path(__file__).parent.parent.parent.parent.parent / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from logger import setup_logger
logger = setup_logger(__name__, service_name="free-ai-service")
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
    title="StateX Free AI Service",
    description="Free AI models service using OpenRouter, Ollama, and Hugging Face",
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
import os
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
# Note: load_dotenv with override=True will override existing environment variables
import pathlib
env_path = pathlib.Path(__file__).parent.parent.parent.parent.parent / '.env'
logger.debug(r"Loading .env from: {env_path}")
logger.debug(r".env file exists: {env_path.exists()}")
load_dotenv(env_path, override=True)

# Debug: Print environment variables
logger.debug(r"Environment variables after loading .env:")
api_key_value = os.getenv('OPENROUTER_API_KEY')
logger.info(r"  OPENROUTER_API_KEY: {'SET' if api_key_value else 'NOT SET'}")
if api_key_value:
    logger.info(r"  OPENROUTER_API_KEY (first 10 chars): {api_key_value[:10]}...")
logger.info(r"  OPENROUTER_API_BASE: {os.getenv('OPENROUTER_API_BASE', 'NOT SET')}")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HUGGINGFACE_URL = "https://api-inference.huggingface.co/models"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

# Debug: Print loaded values
logger.debug(r"Loaded values:")
logger.info(r"  OPENROUTER_API_KEY: {'SET' if OPENROUTER_API_KEY else 'NOT SET'}")
logger.info(r"  OPENROUTER_API_BASE: {OPENROUTER_API_BASE}")
logger.info(r"  OPENROUTER_MODEL: {OPENROUTER_MODEL}")


class AIProvider(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"

class AnalysisType(str, Enum):
    BUSINESS_ANALYSIS = "business_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    CONTENT_GENERATION = "content_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class AIAnalysisRequest(BaseModel):
    text_content: str
    analysis_type: AnalysisType = AnalysisType.BUSINESS_ANALYSIS
    user_name: str = "User"
    provider: Optional[AIProvider] = None  # Auto-detect if not specified
    model: Optional[str] = None  # Specific model to use

class AIAnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    provider_used: str
    model_used: str
    processing_time: float
    confidence: float
    error: Optional[str] = None

class AIModelInfo(BaseModel):
    name: str
    provider: str
    description: str
    capabilities: List[str]
    status: str  # available, unavailable, loading

class FreeAIService:
    def __init__(self):
        self.available_models = {}
        self.provider_status = {}
        self.model_preferences = {
            AnalysisType.BUSINESS_ANALYSIS: {
                "openrouter": ["google/gemini-2.0-flash-exp:free", "openai/gpt-oss-20b:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"],
                "ollama": ["llama2:7b", "mistral:7b"],
                "huggingface": ["microsoft/DialoGPT-medium", "gpt2", "facebook/blenderbot-400M-distill"]
            },
            AnalysisType.TECHNICAL_ANALYSIS: {
                "openrouter": ["google/gemini-2.0-flash-exp:free", "openai/gpt-oss-20b:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"],
                "ollama": ["codellama:7b", "mistral:7b", "llama2:7b"],
                "huggingface": ["microsoft/CodeBERT-base", "gpt2", "microsoft/DialoGPT-medium"]
            },
            AnalysisType.CONTENT_GENERATION: {
                "openrouter": ["google/gemini-2.0-flash-exp:free", "openai/gpt-oss-20b:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"],
                "ollama": ["llama2:7b", "mistral:7b"],
                "huggingface": ["gpt2", "microsoft/DialoGPT-medium"]
            },
            AnalysisType.SENTIMENT_ANALYSIS: {
                "openrouter": ["google/gemini-2.0-flash-exp:free", "openai/gpt-oss-20b:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
                "ollama": ["llama2:7b"],
                "huggingface": ["cardiffnlp/twitter-roberta-base-sentiment-latest", "distilbert-base-uncased"]
            }
        }
        
    async def check_providers(self):
        """Check which AI providers are available"""
        logger.info("🔍 Checking AI providers availability...")
        
        # Skip Ollama and Hugging Face checks - disabled by user request
        logger.info("⏭️ Skipping Ollama check (disabled)")
        self.provider_status["ollama"] = "disabled"
        
        logger.info("⏭️ Skipping Hugging Face check (disabled)")
        self.provider_status["huggingface"] = "disabled"
        
        # Check OpenRouter
        try:
            logger.info(f"🔍 Checking OpenRouter API key: {'SET' if OPENROUTER_API_KEY else 'NOT SET'}")
            logger.info(f"🔍 OpenRouter API Base: {OPENROUTER_API_BASE}")
            
            if OPENROUTER_API_KEY:
                # Test with a simple request to check if API key is valid
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                # Create SSL context that doesn't verify certificates for testing
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    logger.info(f"🔍 Testing OpenRouter API connection to {OPENROUTER_API_BASE}/models")
                    async with session.get(
                        f"{OPENROUTER_API_BASE}/models",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        logger.info(f"🔍 OpenRouter API response status: {response.status}")
                        if response.status == 200:
                            self.provider_status["openrouter"] = "available"
                            self.available_models["openrouter"] = [
                                {"name": "google/gemini-2.0-flash-exp:free", "description": "Google: Gemini 2.0 Flash Experimental (free)"},
                                {"name": "openai/gpt-oss-20b:free", "description": "OpenAI: gpt-oss-20b (free)"},   
                                {"name": "anthropic/claude-3.5-sonnet", "description": "Claude 3.5 Sonnet - Advanced reasoning"},
                                {"name": "openai/gpt-4o", "description": "GPT-4o - Multimodal AI model"},
                                {"name": "meta-llama/llama-3.1-70b-instruct", "description": "Llama 3.1 70B - Large language model"}
                            ]
                            logger.info("✅ OpenRouter API is available")
                        else:
                            response_text = await response.text()
                            logger.warning(f"❌ OpenRouter API returned status {response.status}: {response_text}")
                            self.provider_status["openrouter"] = "unavailable"
            else:
                self.provider_status["openrouter"] = "unavailable"
                logger.error("❌ OpenRouter API key not provided - AI service will not function")
        except Exception as e:
            logger.error(f"❌ OpenRouter API not available: {e}")
            self.provider_status["openrouter"] = "unavailable"
        
        # No Mock AI - only real AI providers
    
    def get_best_model(self, analysis_type: AnalysisType, provider: str = None) -> tuple[str, str]:
        """Get the best available model for the given analysis type and provider"""
        
        logger.info(f"🔍 get_best_model called with analysis_type={analysis_type}, provider={provider}")
        logger.info(f"🔍 Provider status: {self.provider_status}")
        logger.info(f"🔍 Available models: {self.available_models}")
        
        # Define provider priority
        provider_priority = ["openrouter", "ollama", "huggingface"]
        
        if provider and provider in self.provider_status and self.provider_status[provider] == "available":
            # Use specific provider
            preferred_models = self.model_preferences.get(analysis_type, {}).get(provider, [])
            available_models = [m["name"] for m in self.available_models.get(provider, [])]
            
            logger.info(f"🔍 Specific provider {provider}: preferred={preferred_models}, available={available_models}")
            
            for model in preferred_models:
                if model in available_models or provider == "ollama":  # Ollama models might not be in the list yet
                    logger.info(f"✅ Selected specific provider {provider} with model {model}")
                    return provider, model
            
            # Fallback to first available model for this provider
            if available_models:
                logger.info(f"✅ Selected specific provider {provider} with first available model {available_models[0]}")
                return provider, available_models[0]
        else:
            # Auto-select best provider and model
            for prov in provider_priority:
                if self.provider_status.get(prov) == "available":
                    preferred_models = self.model_preferences.get(analysis_type, {}).get(prov, [])
                    available_models = [m["name"] for m in self.available_models.get(prov, [])]
                    logger.info(f"🔍 Auto-selecting {prov}: preferred={preferred_models}, available={available_models}")
                    for model in preferred_models:
                        if model in available_models or prov == "ollama":
                            logger.info(f"✅ Selected auto provider {prov} with model {model}")
                            return prov, model
                    # Fallback to first available model
                    if available_models:
                        logger.info(f"✅ Selected auto provider {prov} with first available model {available_models[0]}")
                        return prov, available_models[0]
            
            # No available providers - raise exception
            raise HTTPException(
                status_code=503, 
                detail="No AI providers are currently available. Please check service status."
            )
    
    async def analyze_with_fallback(self, request: AIAnalysisRequest) -> Dict[str, Any]:
        """Analyze with automatic fallback between providers"""
        
        # Determine provider and model
        provider, model = self.get_best_model(request.analysis_type, request.provider)
        
        # Override model if specified in request
        if request.model:
            model = request.model
        
        logger.info(f"🎯 Selected provider: {provider}, model: {model}")
        
        # Try primary provider
        try:
            if provider == "openrouter":
                request.model = model
                return await self.analyze_with_openrouter(request)
            elif provider == "ollama":
                request.model = model
                return await self.analyze_with_ollama(request)
            elif provider == "huggingface":
                request.model = model
                return await self.analyze_with_huggingface(request)
            else:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Unsupported provider: {provider}. Available providers: openrouter, ollama, huggingface"
                )
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.warning(f"⚠️ Primary provider {provider} failed: {e}")
            
            # Try fallback providers
            fallback_providers = ["openrouter", "ollama", "huggingface"]
            fallback_providers = [p for p in fallback_providers if p != provider]
            
            for fallback_provider in fallback_providers:
                if self.provider_status.get(fallback_provider) == "available":
                    try:
                        logger.info(f"🔄 Trying fallback provider: {fallback_provider}")
                        fallback_provider_name, fallback_model = self.get_best_model(request.analysis_type, fallback_provider)
                        
                        if fallback_provider == "openrouter":
                            request.model = fallback_model
                            return await self.analyze_with_openrouter(request)
                        elif fallback_provider == "ollama":
                            request.model = fallback_model
                            return await self.analyze_with_ollama(request)
                        elif fallback_provider == "huggingface":
                            request.model = fallback_model
                            return await self.analyze_with_huggingface(request)
                    except Exception as fallback_error:
                        logger.warning(f"⚠️ Fallback provider {fallback_provider} also failed: {fallback_error}")
                        continue
            
            # No providers available - raise exception
            logger.error("❌ All AI providers failed or are unavailable")
            raise HTTPException(
                status_code=503, 
                detail="All AI providers are currently unavailable. Please check service status and try again later."
            )
    
    async def analyze_with_ollama(self, request: AIAnalysisRequest) -> Dict[str, Any]:
        """Analyze using Ollama (Local LLM)"""
        logger.info(f"🤖 Analyzing with Ollama: {request.model or 'llama2:7b'}")
        
        model = request.model or "llama2:7b"
        
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
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1000
                    }
                }
                
                async with session.post(
                    f"{OLLAMA_URL}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get("response", "")
                        
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
                        
                        analysis["ai_service"] = "Ollama"
                        analysis["model_used"] = model
                        return analysis
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            raise e
    
    async def analyze_with_huggingface(self, request: AIAnalysisRequest) -> Dict[str, Any]:
        """Analyze using Hugging Face API"""
        logger.info(f"🤖 Analyzing with Hugging Face: {request.model or 'gpt2'}")
        
        model = request.model or "gpt2"
        
        try:
            headers = {"Content-Type": "application/json"}
            if HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
            
            # Create a focused prompt for business analysis
            prompt = f"Business Analysis Request from {request.user_name}: {request.text_content[:400]}"
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "temperature": 0.7,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                async with session.post(
                    f"{HUGGINGFACE_URL}/{model}",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Handle different response formats
                        ai_response = ""
                        if isinstance(result, list) and len(result) > 0:
                            if "generated_text" in result[0]:
                                ai_response = result[0]["generated_text"]
                            elif "text" in result[0]:
                                ai_response = result[0]["text"]
                        elif isinstance(result, dict):
                            ai_response = result.get("generated_text", result.get("text", ""))
                        
                        analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                        analysis["ai_service"] = "Hugging Face"
                        analysis["model_used"] = model
                        return analysis
                    elif response.status == 503:
                        # Model is loading, wait and retry once
                        logger.info("⏳ Model is loading, waiting 10 seconds...")
                        await asyncio.sleep(10)
                        
                        async with session.post(
                            f"{HUGGINGFACE_URL}/{model}",
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as retry_response:
                            if retry_response.status == 200:
                                result = await retry_response.json()
                                ai_response = result[0].get("generated_text", "") if isinstance(result, list) else ""
                                analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                                analysis["ai_service"] = "Hugging Face"
                                analysis["model_used"] = model
                                return analysis
                            else:
                                error_text = await retry_response.text()
                                raise Exception(f"Hugging Face API error after retry: {retry_response.status} - {error_text}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Hugging Face API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Hugging Face analysis failed: {e}")
            raise e
    
    async def analyze_with_openrouter(self, request: AIAnalysisRequest) -> Dict[str, Any]:
        """Analyze using OpenRouter API"""
        logger.info(f"🤖 Analyzing with OpenRouter: {request.model or OPENROUTER_MODEL}")
        
        model = request.model or OPENROUTER_MODEL
        
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
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://statex.cz",
                "X-Title": "StateX AI Platform"
            }
            
            # Create SSL context that doesn't verify certificates for testing
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
                
                async with session.post(
                    f"{OPENROUTER_API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
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
                        
                        analysis["ai_service"] = "OpenRouter"
                        analysis["model_used"] = model
                        return analysis
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"OpenRouter analysis failed: {e}")
            raise e
    
    # Mock AI methods removed - only real AI providers are supported
    
    def _parse_text_response(self, text: str, user_name: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Parse text response from AI into structured format"""
        return {
            "summary": f"AI Analysis for {user_name}: {text[:200]}...",
            "key_insights": ["AI-generated insight 1", "AI-generated insight 2"],
            "recommendations": ["AI-generated recommendation 1", "AI-generated recommendation 2"],
            "confidence": 0.8,
            "ai_service": "Text Parser",
            "model_used": "text-parser"
        }

# Initialize service
free_ai_service = FreeAIService()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    await free_ai_service.check_providers()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "free-ai-service",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "providers": free_ai_service.provider_status
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint - disabled"""
    return {"message": "Metrics collection is disabled", "status": "disabled"}

@app.get("/models")
async def get_available_models():
    """Get list of available AI models"""
    return {
        "models": free_ai_service.available_models,
        "providers": free_ai_service.provider_status
    }

@app.post("/analyze", response_model=AIAnalysisResponse)
async def analyze_text(request: AIAnalysisRequest):
    """Analyze text using free AI services"""
    start_time = time.time()
    
    try:
        # Use the new fallback analysis method
        analysis = await free_ai_service.analyze_with_fallback(request)
        provider = analysis.get("ai_service", "unknown").lower()
        
        processing_time = time.time() - start_time
        
        # Update metrics
        provider_name = provider if isinstance(provider, str) else provider.value
        
        return AIAnalysisResponse(
            success=True,
            analysis=analysis,
            provider_used=analysis.get("ai_service", "unknown"),
            model_used=analysis.get("model_used", "unknown"),
            processing_time=processing_time,
            confidence=analysis.get("confidence", 0.8)
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        processing_time = time.time() - start_time
        
        # Update metrics for error case
        
        return AIAnalysisResponse(
            success=False,
            analysis={},
            provider_used="error",
            model_used="none",
            processing_time=processing_time,
            confidence=0.0,
            error=str(e)
        )

@app.post("/api/analyze-text")
async def analyze_text_compatibility(request: AIAnalysisRequest):
    """Compatibility endpoint for NLP Agent - proxies to /analyze with business analysis"""
    # Override analysis type to business analysis for NLP compatibility
    request.analysis_type = AnalysisType.BUSINESS_ANALYSIS
    return await analyze_text(request)

@app.get("/api/status/{submission_id}")
async def get_submission_status(submission_id: str):
    """Get status of a submission"""
    # For now, return a mock status since we don't have persistent storage
    # In a real implementation, this would check a database or cache
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
        "service": "StateX Free AI Service",
        "version": "1.0.0",
        "description": "Free AI models service using OpenRouter, Ollama, and Hugging Face",
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
    port = int(os.getenv("FREE_AI_SERVICE_PORT", "8000"))
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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FREE_AI_SERVICE_PORT", "3386")), log_config=log_config)
