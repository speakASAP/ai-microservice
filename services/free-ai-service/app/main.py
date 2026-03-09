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
# Import logger from shared directory
import importlib.util
shared_path = "/app/shared"
logger_spec = importlib.util.spec_from_file_location("logger", os.path.join(shared_path, "logger.py"))
logger_module = importlib.util.module_from_spec(logger_spec)
logger_spec.loader.exec_module(logger_module)
logger = logger_module.setup_logger(__name__, service_name="free-ai-service")
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
    # Email-triage (agentic-email-processing-system): Classifier and Decider via LLM
    EMAIL_CLASSIFY = "email_classify"
    EMAIL_DECIDE = "email_decide"

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
            },
            # EMAIL_*: openrouter/free router first (auto-selects free model); then specific free/paid
            AnalysisType.EMAIL_CLASSIFY: {
                "openrouter": ["openrouter/free", "openai/gpt-oss-20b:free", "google/gemini-2.0-flash-exp:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"],
                "ollama": ["llama2:7b", "mistral:7b"],
                "huggingface": ["microsoft/DialoGPT-medium", "gpt2"]
            },
            AnalysisType.EMAIL_DECIDE: {
                "openrouter": ["openrouter/free", "openai/gpt-oss-20b:free", "google/gemini-2.0-flash-exp:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"],
                "ollama": ["llama2:7b", "mistral:7b"],
                "huggingface": ["microsoft/DialoGPT-medium", "gpt2"]
            }
        }
        
    async def check_providers(self):
        """Check which AI providers are available"""
        logger.info("🔍 Checking AI providers availability...")
        
        # Ollama: check if OLLAMA_URL is set and reachable (fallback when OpenRouter fails)
        ollama_url = (os.getenv("OLLAMA_URL") or "").strip()
        if ollama_url and ollama_url.lower() != "disabled":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{ollama_url.rstrip('/')}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            self.provider_status["ollama"] = "available"
                            data = await response.json()
                            models = [m.get("name", "") for m in data.get("models", [])]
                            self.available_models["ollama"] = [{"name": n, "description": n} for n in models] if models else [{"name": "llama2:7b", "description": "Llama 2 7B"}]
                            logger.info("✅ Ollama is available")
                        else:
                            self.provider_status["ollama"] = "unavailable"
                            logger.info("⏭️ Ollama not reachable (status %s)", response.status)
            except Exception as e:
                self.provider_status["ollama"] = "unavailable"
                logger.info("⏭️ Ollama check skipped or failed: %s", e)
        else:
            self.provider_status["ollama"] = "disabled"
            logger.info("⏭️ Ollama disabled (OLLAMA_URL not set)")
        
        # Hugging Face: disabled by user request
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
                                {"name": "openrouter/free", "description": "OpenRouter free router (auto-selects free model)"},
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
    
    def _is_openrouter_model_unavailable(self, exc: Exception) -> bool:
        """True if error indicates we should try next model (404, 402 credits, 429 rate limit)."""
        msg = str(exc).lower()
        return "404" in msg or "no endpoints found" in msg or "402" in msg or "credits" in msg or "429" in msg or "rate limit" in msg

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
                # Try preferred OpenRouter models in order; on 404 (model unavailable) try next
                preferred = self.model_preferences.get(request.analysis_type, {}).get("openrouter", [])
                if not preferred:
                    preferred = [model]
                last_err = None
                for m in preferred:
                    request.model = m
                    try:
                        return await self.analyze_with_openrouter(request)
                    except Exception as e:
                        last_err = e
                        if self._is_openrouter_model_unavailable(e):
                            logger.warning(
                                "OpenRouter model unavailable (404/402/429), trying next preferred model",
                                model=m,
                                error=str(e)[:200],
                            )
                            continue
                        raise
                logger.error("All OpenRouter preferred models failed (e.g. 404)", last_error=str(last_err)[:200] if last_err else None)
                raise last_err or HTTPException(status_code=503, detail="OpenRouter models unavailable")
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
        elif request.analysis_type == AnalysisType.EMAIL_CLASSIFY:
            prompt = f"""Classify this business email into exactly one intent. Return only valid JSON, no other text.

Email content:
{request.text_content}

Intent must be one of: support, sales, contract, technical, billing, spam, unknown, multi_intent.
Return JSON with: "intent", "confidence" (0-1), "raw_scores" (optional object). Example: {{"intent": "support", "confidence": 0.92, "raw_scores": {{"support": 0.92, "sales": 0.1, "contract": 0.05, "technical": 0.2, "billing": 0.15, "spam": 0.02}}}}
"""
        elif request.analysis_type == AnalysisType.EMAIL_DECIDE:
            prompt = f"""Given triage inputs, decide the action. Return only valid JSON. Input: {request.text_content}
Actions: auto_respond, route_to_queue, escalate. Queues: support, sales, technical, billing, spam_review.
Return JSON: "action", "escalation_reason" (null or string), "queue" (null or queue name). Example: {{"action": "route_to_queue", "escalation_reason": null, "queue": "support"}}
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
                        except Exception:
                            analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                        if request.analysis_type == AnalysisType.EMAIL_CLASSIFY:
                            analysis = self._normalize_email_classify(analysis)
                        elif request.analysis_type == AnalysisType.EMAIL_DECIDE:
                            analysis = self._normalize_email_decide(analysis)
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
            
            # Email-triage: use same prompts as OpenRouter for JSON output
            if request.analysis_type == AnalysisType.EMAIL_CLASSIFY:
                prompt = f"""Classify this email. Return only JSON with intent, confidence, raw_scores. Intents: support, sales, contract, technical, billing, spam, unknown, multi_intent.\n\nEmail:\n{request.text_content[:2000]}"""
            elif request.analysis_type == AnalysisType.EMAIL_DECIDE:
                prompt = f"""Decide action. Return only JSON with action (auto_respond|route_to_queue|escalate), escalation_reason, queue.\n\nInput:\n{request.text_content[:1000]}"""
            else:
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
                        analysis = self._parse_response_for_type(ai_response, request)
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
                                analysis = self._parse_response_for_type(ai_response, request)
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
        elif request.analysis_type == AnalysisType.EMAIL_CLASSIFY:
            prompt = f"""Classify this business email into exactly one intent. Return only valid JSON, no other text.

Email content:
{request.text_content}

Intent must be one of: support, sales, contract, technical, billing, spam, unknown, multi_intent.
- support: general customer support
- sales: product/offer/pricing inquiry
- contract: contract question or change
- technical: technical problem or question
- billing: billing or payment issue
- spam: spam or irrelevant
- unknown: cannot determine
- multi_intent: clearly multiple intents

Return JSON with these keys only:
- "intent": one of the values above
- "confidence": number between 0 and 1
- "raw_scores": object with keys support, sales, contract, technical, billing, spam and number values 0-1 for each (optional; if missing we use intent and confidence only)

Example: {{"intent": "support", "confidence": 0.92, "raw_scores": {{"support": 0.92, "sales": 0.1, "contract": 0.05, "technical": 0.2, "billing": 0.15, "spam": 0.02}}}}
"""
        elif request.analysis_type == AnalysisType.EMAIL_DECIDE:
            prompt = f"""Given triage inputs, decide the action. Return only valid JSON, no other text.

Input (JSON):
{request.text_content}

Actions: auto_respond (only for high-confidence support when safe), route_to_queue (assign to a queue), escalate (human review).
Queues: support, sales, technical, billing, spam_review. Always escalate for intent unknown, multi_intent, or contract.

Return JSON with:
- "action": one of auto_respond, route_to_queue, escalate
- "escalation_reason": null or string (e.g. ambiguous_intent, low_confidence, contract_change) only when action is escalate
- "queue": null or one of support, sales, technical, billing, spam_review (when action is route_to_queue)

Example: {{"action": "route_to_queue", "escalation_reason": null, "queue": "support"}}
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
            # Use lower max_tokens for email-triage so free/limited credits stay within quota (e.g. 402)
            max_tokens = 512 if request.analysis_type in (AnalysisType.EMAIL_CLASSIFY, AnalysisType.EMAIL_DECIDE) else 2000
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
                    "max_tokens": max_tokens
                }
                
                async with session.post(
                    f"{OPENROUTER_API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        effective_model = result.get("model") or model
                        choices = result.get("choices") or []
                        first = choices[0] if choices else None
                        msg = (first.get("message") if isinstance(first, dict) else None) or {}
                        ai_response = (msg.get("content") if isinstance(msg, dict) else None) or ""
                        
                        # Try to parse JSON response
                        try:
                            json_start = ai_response.find('{')
                            json_end = ai_response.rfind('}') + 1
                            if json_start != -1 and json_end != -1:
                                json_str = ai_response[json_start:json_end]
                                analysis = json.loads(json_str)
                            else:
                                analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                        except Exception:
                            analysis = self._parse_text_response(ai_response, request.user_name, request.analysis_type)
                        # Normalize email-triage responses to contract shape
                        if request.analysis_type == AnalysisType.EMAIL_CLASSIFY:
                            analysis = self._normalize_email_classify(analysis)
                        elif request.analysis_type == AnalysisType.EMAIL_DECIDE:
                            analysis = self._normalize_email_decide(analysis)
                        analysis["ai_service"] = "OpenRouter"
                        display_model = model
                        if effective_model and effective_model != model:
                            display_model = f"{model} -> {effective_model}"
                        analysis["model_used"] = display_model
                        analysis["openrouter_model"] = effective_model
                        analysis["requested_model"] = model
                        return analysis
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"OpenRouter analysis failed: {e}")
            raise e
    
    # Mock AI methods removed - only real AI providers are supported

    _EMAIL_INTENTS = frozenset({"support", "sales", "contract", "technical", "billing", "spam", "unknown", "multi_intent"})
    _EMAIL_ACTIONS = frozenset({"auto_respond", "route_to_queue", "escalate"})
    _EMAIL_QUEUES = frozenset({"support", "sales", "technical", "billing", "spam_review"})

    def _normalize_email_classify(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize LLM output to classifier contract. Defaults to unknown if invalid."""
        intent = (raw.get("intent") or "").strip().lower()
        if intent not in self._EMAIL_INTENTS:
            intent = "unknown"
        try:
            confidence = float(raw.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        raw_scores = raw.get("raw_scores")
        if not isinstance(raw_scores, dict):
            raw_scores = {k: 0.2 for k in ["support", "sales", "contract", "technical", "billing", "spam"]}
        return {"intent": intent, "confidence": confidence, "raw_scores": raw_scores}

    def _normalize_email_decide(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize LLM output to decider contract. Defaults to escalate if invalid."""
        action = (raw.get("action") or "").strip().lower()
        if action not in self._EMAIL_ACTIONS:
            action = "escalate"
            raw = dict(raw)
            raw["escalation_reason"] = raw.get("escalation_reason") or "llm_invalid_or_empty_action"
        escalation_reason = raw.get("escalation_reason")
        if escalation_reason is not None and not isinstance(escalation_reason, str):
            escalation_reason = str(escalation_reason)
        queue = raw.get("queue")
        if queue is not None:
            queue = str(queue).strip().lower()
            if queue not in self._EMAIL_QUEUES:
                queue = "support"
        return {"action": action, "escalation_reason": escalation_reason, "queue": queue}

    def _parse_response_for_type(self, ai_response: str, request: AIAnalysisRequest) -> Dict[str, Any]:
        """Parse AI response; for email-triage types try JSON and normalize, else generic parse."""
        if request.analysis_type in (AnalysisType.EMAIL_CLASSIFY, AnalysisType.EMAIL_DECIDE):
            try:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    analysis = json.loads(ai_response[json_start:json_end])
                    if request.analysis_type == AnalysisType.EMAIL_CLASSIFY:
                        return self._normalize_email_classify(analysis)
                    return self._normalize_email_decide(analysis)
            except (json.JSONDecodeError, ValueError):
                pass
        return self._parse_text_response(ai_response, request.user_name, request.analysis_type)
    
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
async def get_available_models(free_only: bool = False, context_min: Optional[int] = None, limit: Optional[int] = None):
    """Get list of available AI models from source APIs
    
    Args:
        free_only: If True, return only free models
        context_min: Minimum context window size in tokens
        limit: Maximum number of models to return per provider
    """
    try:
        models_by_provider = {}
        providers = {}
        
        # Fetch models from OpenRouter API (from source)
        if OPENROUTER_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        f"{OPENROUTER_API_BASE}/models",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        if response.status == 200:
                            openrouter_data = await response.json()
                            openrouter_models = openrouter_data.get("data", [])
                            
                            provider = "openrouter"
                            models_by_provider[provider] = []
                            
                            for model_data in openrouter_models:
                                model_id = model_data.get("id", "")
                                if not model_id:
                                    continue
                                
                                # Extract pricing info
                                pricing = model_data.get("pricing", {})
                                prompt_price = pricing.get("prompt", "0")
                                completion_price = pricing.get("completion", "0")
                                is_free = prompt_price == "0" and completion_price == "0"
                                
                                # Filter by free_only if specified
                                if free_only and not is_free:
                                    continue
                                
                                # Extract context length
                                context_length = model_data.get("context_length", 0)
                                
                                # Filter by context_min if specified
                                if context_min is not None and context_length < context_min:
                                    continue
                                
                                # Extract model info
                                model_name = model_data.get("name", model_id)
                                description = model_data.get("description", "")
                                architecture = model_data.get("architecture", {})
                                top_provider = model_data.get("top_provider", {})
                                
                                model_info = {
                                    "id": model_id,
                                    "name": model_name,
                                    "description": description,
                                    "context_length": context_length,
                                    "pricing": {
                                        "prompt": prompt_price,
                                        "completion": completion_price,
                                        "free": is_free
                                    },
                                    "architecture": architecture,
                                    "top_provider": top_provider.get("name", ""),
                                    "capabilities": []
                                }
                                
                                models_by_provider[provider].append(model_info)
                            
                            # Apply limit if specified
                            if limit is not None and limit > 0:
                                models_by_provider[provider] = models_by_provider[provider][:limit]
                            
                            providers[provider] = {"status": "available"}
                            logger.info(f"✅ Fetched {len(models_by_provider[provider])} models from OpenRouter API")
                        else:
                            logger.warning(f"OpenRouter API returned status {response.status}")
                            providers["openrouter"] = {"status": "unavailable"}
            except Exception as e:
                logger.error(f"Failed to fetch models from OpenRouter API: {e}")
                providers["openrouter"] = {"status": "error"}
        else:
            providers["openrouter"] = {"status": "unavailable", "reason": "API key not set"}
        
        # Return models from source APIs
        return {
            "models": models_by_provider,
            "providers": providers
        }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {
            "models": {},
            "providers": {}
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
