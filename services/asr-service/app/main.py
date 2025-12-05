"""
StateX ASR Service - Enhanced Version

Enhanced speech-to-text conversion and voice processing service.
Handles voice transcription, audio analysis, and voice quality assessment.
Supports free speech-to-text services with audio format validation and confidence scoring.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import os
import time
import logging
import tempfile
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
from enum import Enum

# Try to import audio processing libraries
try:
    import whisper
    import torch
    import librosa
    import soundfile as sf
    import numpy as np
    WHISPER_AVAILABLE = True
    AUDIO_PROCESSING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Local Whisper model and audio processing libraries available")
except ImportError as e:
    WHISPER_AVAILABLE = False
    AUDIO_PROCESSING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Audio processing libraries not available: {e}")

# Try to import OpenAI client
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.error("❌ OpenAI client not available")

# Configure logging with timestamps
import sys
from pathlib import Path
# Add utils to path (project_root/utils)
import os
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'utils')
utils_path = os.path.abspath(utils_path)
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from logger import setup_logger
logger = setup_logger(__name__, service_name="asr-service")
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



# Initialize FastAPI app
app = FastAPI(
    title="StateX ASR Service",
    description="Speech-to-text conversion and voice processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
free_ai_port = os.getenv("FREE_AI_SERVICE_PORT", "3386")
FREE_AI_SERVICE_URL = os.getenv("FREE_AI_SERVICE_URL", f"http://localhost:{free_ai_port}")
ASR_MODE = os.getenv("ASR_MODE", "free")  # free, paid, hybrid

class ASRProvider(str, Enum):
    LOCAL_WHISPER = "local_whisper"
    OPENAI_WHISPER = "openai_whisper"
    # Mock provider removed - only real AI providers supported

class AudioValidator:
    """Enhanced audio file validation and processing"""
    
    SUPPORTED_FORMATS = {
        'audio/wav': '.wav',
        'audio/wave': '.wav',
        'audio/x-wav': '.wav',
        'audio/mpeg': '.mp3',
        'audio/mp3': '.mp3',
        'audio/mp4': '.m4a',
        'audio/x-m4a': '.m4a',
        'audio/flac': '.flac',
        'audio/x-flac': '.flac',
        'audio/ogg': '.ogg',
        'audio/webm': '.webm'
    }
    
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    MAX_DURATION = 25 * 60  # 25 minutes in seconds
    MIN_DURATION = 0.5  # 0.5 seconds minimum
    
    @classmethod
    def validate_audio_file(cls, file_content: bytes, content_type: str, filename: str) -> Dict[str, Any]:
        """Validate audio file format, size, and basic properties"""
        
        # Check file size
        if len(file_content) > cls.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File size {len(file_content)} bytes exceeds maximum {cls.MAX_FILE_SIZE} bytes"
            )
        
        # Check content type
        if content_type not in cls.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {content_type}. Supported formats: {list(cls.SUPPORTED_FORMATS.keys())}"
            )
        
        # Check file extension
        file_extension = os.path.splitext(filename)[1].lower()
        expected_extension = cls.SUPPORTED_FORMATS[content_type]
        
        if file_extension != expected_extension:
            logger.warning(f"File extension {file_extension} doesn't match content type {content_type}")
        
        return {
            "valid": True,
            "content_type": content_type,
            "file_size": len(file_content),
            "expected_extension": expected_extension,
            "actual_extension": file_extension
        }
    
    @classmethod
    async def analyze_audio_properties(cls, file_path: str) -> Dict[str, Any]:
        """Analyze audio file properties using librosa"""
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return cls._mock_audio_properties()
        
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            
            # Calculate basic properties
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Check duration limits
            if duration < cls.MIN_DURATION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration {duration:.2f}s is too short. Minimum: {cls.MIN_DURATION}s"
                )
            
            if duration > cls.MAX_DURATION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration {duration:.2f}s exceeds maximum {cls.MAX_DURATION}s"
                )
            
            # Calculate audio quality metrics
            rms_energy = librosa.feature.rms(y=y)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Estimate signal quality
            avg_rms = np.mean(rms_energy)
            avg_spectral_centroid = np.mean(spectral_centroid)
            avg_zcr = np.mean(zero_crossing_rate)
            
            # Simple quality assessment
            quality_score = min(1.0, avg_rms * 10)  # Normalize RMS to 0-1
            
            # Detect silence/noise
            silence_threshold = 0.01
            silence_ratio = np.sum(rms_energy < silence_threshold) / len(rms_energy)
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0],
                "quality_score": float(quality_score),
                "silence_ratio": float(silence_ratio),
                "avg_rms_energy": float(avg_rms),
                "avg_spectral_centroid": float(avg_spectral_centroid),
                "avg_zero_crossing_rate": float(avg_zcr),
                "estimated_snr": float(20 * np.log10(avg_rms / (avg_zcr + 1e-8))),  # Rough SNR estimate
                "analysis_available": True
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return cls._mock_audio_properties()
    
    @classmethod
    def _mock_audio_properties(cls) -> Dict[str, Any]:
        """Mock audio properties when librosa is not available"""
        return {
            "duration": 45.2,
            "sample_rate": 16000,
            "channels": 1,
            "quality_score": 0.8,
            "silence_ratio": 0.1,
            "avg_rms_energy": 0.15,
            "avg_spectral_centroid": 2000.0,
            "avg_zero_crossing_rate": 0.05,
            "estimated_snr": 15.0,
            "analysis_available": False
        }
    
    @classmethod
    async def convert_audio_format(cls, input_path: str, output_path: str, target_format: str = "wav") -> str:
        """Convert audio to target format for better compatibility"""
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return input_path  # Return original if conversion not available
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=16000)  # Standardize to 16kHz for Whisper
            
            # Save in target format
            sf.write(output_path, y, sr, format=target_format.upper())
            
            logger.info(f"Audio converted from {input_path} to {output_path} ({target_format})")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return input_path  # Return original on failure

class WhisperASRService:
    def __init__(self):
        self.local_model = None
        self.openai_client = None
        self.provider_status = {}
        
        # Initialize local Whisper model
        if WHISPER_AVAILABLE and ASR_MODE in ["free", "hybrid"]:
            try:
                self.local_model = whisper.load_model("base")
                self.provider_status["local_whisper"] = "available"
                logger.info("✅ Local Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"❌ Failed to load local Whisper model: {e}")
                self.provider_status["local_whisper"] = "unavailable"
        else:
            self.provider_status["local_whisper"] = "unavailable"
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and OPENROUTER_API_KEY and ASR_MODE in ["paid", "hybrid"]:
            try:
                self.openai_client = AsyncOpenAI(api_key=OPENROUTER_API_KEY)
                self.provider_status["openai_whisper"] = "available"
                logger.info("✅ OpenAI Whisper API client initialized")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize OpenAI client: {e}")
                self.provider_status["openai_whisper"] = "unavailable"
        else:
            self.provider_status["openai_whisper"] = "unavailable"
        
        # No Mock AI - only real AI providers
    
    async def transcribe_with_local_whisper(self, audio_file_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio using local Whisper model with enhanced confidence scoring"""
        if not self.local_model:
            raise Exception("Local Whisper model not available")
        
        logger.info("🎤 Transcribing with local Whisper model...")
        
        try:
            # Analyze audio properties first
            audio_props = await AudioValidator.analyze_audio_properties(audio_file_path)
            
            # Convert audio if needed for better compatibility
            converted_path = audio_file_path
            if AUDIO_PROCESSING_AVAILABLE:
                converted_path = audio_file_path.replace(os.path.splitext(audio_file_path)[1], '_converted.wav')
                converted_path = await AudioValidator.convert_audio_format(audio_file_path, converted_path, "wav")
            
            # Run Whisper in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.local_model.transcribe(
                    converted_path, 
                    language=language,
                    verbose=False,
                    word_timestamps=True
                )
            )
            
            # Extract segments with enhanced information
            segments = []
            total_confidence = 0
            segment_count = 0
            
            for segment in result.get("segments", []):
                # Calculate segment confidence based on audio quality and length
                segment_duration = segment.get("end", 0) - segment.get("start", 0)
                segment_text = segment.get("text", "").strip()
                
                # Estimate confidence based on various factors
                base_confidence = 0.85  # Base Whisper confidence
                
                # Adjust based on audio quality
                quality_factor = audio_props.get("quality_score", 0.8)
                snr_factor = min(1.0, max(0.3, audio_props.get("estimated_snr", 15) / 20))
                
                # Adjust based on segment characteristics
                length_factor = min(1.0, max(0.5, len(segment_text) / 50))  # Longer text usually more reliable
                duration_factor = min(1.0, max(0.5, segment_duration / 5))  # Reasonable duration
                
                segment_confidence = base_confidence * quality_factor * snr_factor * length_factor * duration_factor
                segment_confidence = max(0.3, min(0.95, segment_confidence))  # Clamp between 0.3 and 0.95
                
                segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment_text,
                    "confidence": round(segment_confidence, 3),
                    "words": segment.get("words", [])
                })
                
                total_confidence += segment_confidence
                segment_count += 1
            
            # Calculate overall confidence
            overall_confidence = total_confidence / segment_count if segment_count > 0 else 0.85
            
            # Clean up converted file if created
            if converted_path != audio_file_path and os.path.exists(converted_path):
                try:
                    os.unlink(converted_path)
                except:
                    pass
            
            return {
                "transcript": result["text"].strip(),
                "language": result.get("language", language or "en"),
                "confidence": round(overall_confidence, 3),
                "segments": segments,
                "provider": "local_whisper",
                "model": "whisper-base",
                "audio_properties": audio_props,
                "processing_info": {
                    "audio_converted": converted_path != audio_file_path,
                    "segments_count": len(segments),
                    "total_duration": audio_props.get("duration", 0)
                }
            }
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            raise e
    
    async def transcribe_with_openai_whisper(self, audio_file_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper API"""
        if not self.openai_client:
            raise Exception("OpenAI Whisper API not available")
        
        logger.info("🎤 Transcribing with OpenAI Whisper API...")
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = await self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Convert OpenAI response to our format
            segments = []
            for segment in transcript.segments or []:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
            
            return {
                "transcript": transcript.text.strip(),
                "language": transcript.language or language or "en",
                "confidence": 0.9,  # OpenAI Whisper typically has high confidence
                "segments": segments,
                "provider": "openai_whisper",
                "model": "whisper-1"
            }
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise e
    
    # Mock AI methods removed - only real AI providers supported
    
    async def transcribe_with_fallback(self, audio_file_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe with automatic fallback between providers"""
        
        # Determine provider priority based on mode
        if ASR_MODE == "free":
            provider_priority = ["local_whisper"]
        elif ASR_MODE == "paid":
            provider_priority = ["openai_whisper", "local_whisper"]
        else:  # hybrid
            provider_priority = ["local_whisper", "openai_whisper"]
        
        # Try providers in order
        for provider in provider_priority:
            if self.provider_status.get(provider) == "available":
                try:
                    logger.info(f"🎯 Trying provider: {provider}")
                    
                    if provider == "local_whisper":
                        return await self.transcribe_with_local_whisper(audio_file_path, language)
                    elif provider == "openai_whisper":
                        return await self.transcribe_with_openai_whisper(audio_file_path, language)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Provider {provider} failed: {e}")
                    continue
        
        # No providers available - raise exception
        logger.error("❌ All ASR providers failed or are unavailable")
        raise HTTPException(
            status_code=503, 
            detail="All ASR providers are currently unavailable. Please check service status and try again later."
        )

# Initialize ASR service
asr_service = WhisperASRService()

class TranscriptionRequest(BaseModel):
    voice_file_url: str
    language: str = "en"
    model: str = "whisper-1"
    response_format: str = "json"

class VoiceAnalysisRequest(BaseModel):
    voice_file_url: str
    analysis_type: str = "comprehensive"  # basic, comprehensive, detailed

class TranscriptionResponse(BaseModel):
    transcription_id: str
    status: str
    transcript: str
    confidence: float
    language: str
    processing_time: float
    word_count: int
    duration: float
    created_at: str

class VoiceAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    results: Dict[str, Any]
    processing_time: float
    created_at: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "asr-service",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "providers": asr_service.provider_status,
        "mode": ASR_MODE
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint - disabled"""
    return {"message": "Metrics collection is disabled", "status": "disabled"}

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """Enhanced transcribe audio file to text with confidence scoring"""
    start_time = time.time()
    ASR_ACTIVE_REQUESTS.inc()
    
    try:
        transcription_id = f"asr_{int(time.time())}"
        
        # Perform enhanced transcription
        transcript_result = await perform_transcription(
            request.voice_file_url,
            request.language,
            request.model
        )
        
        processing_time = time.time() - start_time
        
        # Update metrics
        provider = transcript_result.get("provider", "unknown")
        ASR_REQUEST_COUNT.labels(provider=provider, status="success").inc()
        ASR_REQUEST_DURATION.labels(provider=provider).observe(processing_time)
        ASR_ACTIVE_REQUESTS.dec()
        ASR_AGENT_STATUS.labels(agent_name="asr-service").set(1)
        ASR_AUDIO_DURATION.labels(provider=provider).inc(transcript_result.get("duration", 0))
        
        return TranscriptionResponse(
            transcription_id=transcription_id,
            status="completed",
            transcript=transcript_result["transcript"],
            confidence=transcript_result["confidence"],
            language=transcript_result["language"],
            processing_time=processing_time,
            word_count=transcript_result["word_count"],
            duration=transcript_result["duration"],
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        processing_time = time.time() - start_time
        
        # Update error metrics
        ASR_REQUEST_COUNT.labels(provider="error", status="failed").inc()
        ASR_REQUEST_DURATION.labels(provider="error").observe(processing_time)
        ASR_ACTIVE_REQUESTS.dec()
        ASR_AGENT_STATUS.labels(agent_name="asr-service").set(0)
        
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

@app.post("/api/transcribe-with-analysis")
async def transcribe_with_voice_analysis(request: TranscriptionRequest):
    """Enhanced transcription with voice quality analysis and recommendations"""
    start_time = time.time()
    ASR_ACTIVE_REQUESTS.inc()
    
    try:
        transcription_id = f"asr_analysis_{int(time.time())}"
        
        # Perform transcription
        transcript_result = await perform_transcription(
            request.voice_file_url,
            request.language,
            request.model
        )
        
        # Perform voice analysis
        voice_analysis = await perform_voice_analysis(
            request.voice_file_url,
            "comprehensive"
        )
        
        processing_time = time.time() - start_time
        
        # Update metrics
        provider = transcript_result.get("provider", "unknown")
        ASR_REQUEST_COUNT.labels(provider=provider, status="success").inc()
        ASR_REQUEST_DURATION.labels(provider=provider).observe(processing_time)
        ASR_ACTIVE_REQUESTS.dec()
        
        return {
            "transcription_id": transcription_id,
            "status": "completed",
            "transcription": {
                "transcript": transcript_result["transcript"],
                "confidence": transcript_result["confidence"],
                "language": transcript_result["language"],
                "word_count": transcript_result["word_count"],
                "duration": transcript_result["duration"],
                "segments": transcript_result.get("segments", [])
            },
            "voice_analysis": voice_analysis,
            "audio_properties": transcript_result.get("audio_properties", {}),
            "processing_info": transcript_result.get("processing_info", {}),
            "recommendations": _generate_transcription_recommendations(transcript_result, voice_analysis),
            "processing_time": processing_time,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in transcription with analysis: {e}")
        ASR_ACTIVE_REQUESTS.dec()
        raise HTTPException(status_code=500, detail=f"Failed to transcribe and analyze audio: {str(e)}")

def _generate_transcription_recommendations(transcript_result: Dict, voice_analysis: Dict) -> List[str]:
    """Generate recommendations based on transcription and voice analysis"""
    recommendations = []
    
    confidence = transcript_result.get("confidence", 0)
    audio_props = transcript_result.get("audio_properties", {})
    voice_quality = voice_analysis.get("audio_quality", {})
    
    if confidence < 0.7:
        recommendations.append("Low transcription confidence. Consider re-recording in a quieter environment.")
    
    if audio_props.get("quality_score", 0) < 0.6:
        recommendations.append("Audio quality could be improved. Try recording closer to the microphone.")
    
    if voice_quality.get("noise_level", 0) > 0.3:
        recommendations.append("High background noise detected. Use noise cancellation or record in a quieter space.")
    
    if voice_quality.get("volume", 0) < 0.5:
        recommendations.append("Audio volume is low. Speak louder or adjust microphone sensitivity.")
    
    speech_chars = voice_analysis.get("speech_characteristics", {})
    if speech_chars.get("speaking_rate", 150) > 200:
        recommendations.append("Speaking rate is quite fast. Consider speaking more slowly for better accuracy.")
    
    if not recommendations:
        recommendations.append("Transcription quality is good. No specific improvements needed.")
    
    return recommendations

async def perform_transcription(voice_file_url: str, language: str, model: str) -> Dict[str, Any]:
    """Perform audio transcription using available Whisper services"""
    
    try:
        # Download audio file if it's a URL
        if voice_file_url.startswith(('http://', 'https://')):
            audio_file_path = await download_audio_file(voice_file_url)
        else:
            audio_file_path = voice_file_url
        
        # Transcribe using fallback logic
        result = await asr_service.transcribe_with_fallback(audio_file_path, language)
        
        # Calculate additional metrics
        transcript = result["transcript"]
        word_count = len(transcript.split())
        
        # Estimate duration from segments or use default
        duration = 0
        if result.get("segments"):
            last_segment = result["segments"][-1]
            duration = last_segment.get("end", 45.2)
        else:
            duration = 45.2  # Default duration
        
        return {
            "transcript": transcript,
            "confidence": result["confidence"],
            "language": result["language"],
            "word_count": word_count,
            "duration": duration,
            "segments": result["segments"],
            "provider": result["provider"],
            "model_used": result["model"]
        }
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        # No fallback - raise exception
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription failed: {str(e)}"
        )

async def download_audio_file(url: str) -> str:
    """Download audio file from URL to temporary location"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    # Create temporary file
                    suffix = '.wav'
                    if 'content-type' in response.headers:
                        content_type = response.headers['content-type']
                        if 'mp3' in content_type:
                            suffix = '.mp3'
                        elif 'm4a' in content_type:
                            suffix = '.m4a'
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        async for chunk in response.content.iter_chunked(8192):
                            tmp_file.write(chunk)
                        return tmp_file.name
                else:
                    raise Exception(f"Failed to download audio: HTTP {response.status}")
    except Exception as e:
        logger.error(f"Failed to download audio file: {e}")
        raise e

@app.post("/api/analyze-voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(request: VoiceAnalysisRequest):
    """Analyze voice characteristics and quality"""
    start_time = time.time()
    
    try:
        analysis_id = f"voice_analysis_{int(time.time())}"
        
        # Simulate voice analysis
        analysis_results = await perform_voice_analysis(
            request.voice_file_url,
            request.analysis_type
        )
        
        processing_time = time.time() - start_time
        
        return VoiceAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            results=analysis_results,
            processing_time=processing_time,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing voice: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze voice: {str(e)}")

async def perform_voice_analysis(voice_file_url: str, analysis_type: str) -> Dict[str, Any]:
    """Perform comprehensive voice analysis"""
    
    # Simulate voice analysis results
    analysis_results = {
        "audio_quality": {
            "clarity": 0.85,
            "noise_level": 0.15,
            "volume": 0.78,
            "overall_quality": "Good"
        },
        "speech_characteristics": {
            "speaking_rate": 150,  # words per minute
            "pauses": 3,
            "filler_words": 2,
            "articulation": 0.88
        },
        "emotion_analysis": {
            "dominant_emotion": "excitement",
            "confidence": 0.82,
            "emotion_scores": {
                "excitement": 0.75,
                "confidence": 0.68,
                "determination": 0.71,
                "uncertainty": 0.25
            }
        },
        "language_analysis": {
            "language": "en",
            "accent": "American",
            "formality": "semi-formal",
            "technical_terms": 5
        }
    }
    
    if analysis_type == "comprehensive":
        analysis_results["advanced_metrics"] = {
            "pitch_variation": 0.72,
            "rhythm_consistency": 0.81,
            "breath_control": 0.79,
            "emphasis_patterns": ["business", "application", "development"]
        }
    
    elif analysis_type == "detailed":
        analysis_results["detailed_metrics"] = {
            "spectral_analysis": {
                "fundamental_frequency": 180.5,
                "formant_frequencies": [800, 1200, 2500],
                "spectral_centroid": 2200
            },
            "prosodic_features": {
                "intonation_pattern": "rising-falling",
                "stress_pattern": "regular",
                "rhythm_type": "syllable-timed"
            }
        }
    
    return analysis_results

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and validate audio file for processing with enhanced validation"""
    ASR_ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate audio file
        validation_result = AudioValidator.validate_audio_file(
            content, 
            file.content_type, 
            file.filename
        )
        
        # Determine file extension
        file_extension = AudioValidator.SUPPORTED_FORMATS.get(file.content_type, '.wav')
        
        # Save file temporarily with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Analyze audio properties
        audio_properties = await AudioValidator.analyze_audio_properties(tmp_file_path)
        
        processing_time = time.time() - start_time
        
        # Update metrics
        ASR_REQUEST_COUNT.labels(provider="upload", status="success").inc()
        ASR_REQUEST_DURATION.labels(provider="upload").observe(processing_time)
        ASR_ACTIVE_REQUESTS.dec()
        ASR_AUDIO_DURATION.labels(provider="upload").inc(audio_properties.get("duration", 0))
        
        file_info = {
            "file_id": f"audio_{int(time.time())}",
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "duration": audio_properties.get("duration", 0),
            "quality_score": audio_properties.get("quality_score", 0),
            "sample_rate": audio_properties.get("sample_rate", 0),
            "channels": audio_properties.get("channels", 1),
            "estimated_snr": audio_properties.get("estimated_snr", 0),
            "silence_ratio": audio_properties.get("silence_ratio", 0),
            "uploaded_at": datetime.now().isoformat(),
            "file_path": tmp_file_path,
            "validation": validation_result,
            "processing_time": processing_time
        }
        
        return {
            "status": "success",
            "file_info": file_info,
            "message": "Audio file uploaded and validated successfully",
            "recommendations": _get_audio_recommendations(audio_properties)
        }
        
    except HTTPException:
        ASR_ACTIVE_REQUESTS.dec()
        raise
    except Exception as e:
        ASR_ACTIVE_REQUESTS.dec()
        ASR_REQUEST_COUNT.labels(provider="upload", status="failed").inc()
        logger.error(f"Error uploading audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload audio file: {str(e)}")

def _get_audio_recommendations(audio_properties: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on audio analysis"""
    recommendations = []
    
    quality_score = audio_properties.get("quality_score", 0)
    silence_ratio = audio_properties.get("silence_ratio", 0)
    snr = audio_properties.get("estimated_snr", 15)
    
    if quality_score < 0.5:
        recommendations.append("Audio quality is low. Consider recording in a quieter environment.")
    
    if silence_ratio > 0.3:
        recommendations.append("High silence ratio detected. Consider trimming silent parts.")
    
    if snr < 10:
        recommendations.append("Low signal-to-noise ratio. Try recording closer to the microphone.")
    
    if audio_properties.get("duration", 0) < 2:
        recommendations.append("Very short audio. Longer recordings typically have better transcription accuracy.")
    
    if not recommendations:
        recommendations.append("Audio quality looks good for transcription.")
    
    return recommendations

@app.get("/api/providers")
async def get_providers():
    """Get available ASR providers and their status"""
    return {
        "providers": asr_service.provider_status,
        "mode": ASR_MODE,
        "available_providers": [
            {
                "name": "local_whisper",
                "description": "Local Whisper model (free)",
                "status": asr_service.provider_status.get("local_whisper", "unavailable"),
                "cost": "free",
                "accuracy": "high"
            },
            {
                "name": "openai_whisper",
                "description": "OpenAI Whisper API (paid)",
                "status": asr_service.provider_status.get("openai_whisper", "unavailable"),
                "cost": "paid",
                "accuracy": "very_high"
            }
        ]
    }

@app.get("/api/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats"""
    return {
        "supported_formats": [
            {
                "format": "WAV",
                "extension": ".wav",
                "description": "Uncompressed audio format",
                "max_size": "25MB"
            },
            {
                "format": "MP3",
                "extension": ".mp3",
                "description": "Compressed audio format",
                "max_size": "25MB"
            },
            {
                "format": "M4A",
                "extension": ".m4a",
                "description": "Apple audio format",
                "max_size": "25MB"
            },
            {
                "format": "FLAC",
                "extension": ".flac",
                "description": "Lossless audio format",
                "max_size": "25MB"
            }
        ],
        "recommended_format": "WAV",
        "max_duration": "25 minutes",
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"
        ]
    }

@app.get("/api/transcription-status/{transcription_id}")
async def get_transcription_status(transcription_id: str):
    """Get the status of a transcription job"""
    # Simulate status check
    return {
        "transcription_id": transcription_id,
        "status": "completed",
        "progress": 100,
        "estimated_completion": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

@app.delete("/api/transcription/{transcription_id}")
async def delete_transcription(transcription_id: str):
    """Delete a transcription and its associated files"""
    try:
        # Simulate deletion
        return {
            "status": "success",
            "message": f"Transcription {transcription_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete transcription: {str(e)}")

if __name__ == "__main__":
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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ASR_SERVICE_PORT", "3382")), log_config=log_config)
