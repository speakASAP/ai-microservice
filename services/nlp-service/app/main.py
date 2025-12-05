"""
StateX NLP Service
Natural Language Processing service for business requirement analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
import os

# Configure logging with timestamps
import sys
from pathlib import Path
# Add workspace root to path - try multiple possible paths
workspace_possible_paths = [
    Path(__file__).parent.parent.parent.parent.parent,  # Development: workspace root
    Path("/app").parent,  # Container: workspace root at /app/..
]

workspace_root = None
for path in workspace_possible_paths:
    logger_file = path / "statex" / "utils" / "logger.py"
    if logger_file.exists():
        workspace_root = str(path)
        break

if workspace_root is None:
    raise FileNotFoundError("Could not find logger.py in any expected location. Tried: " + str(workspace_possible_paths))

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from statex.utils.logger import setup_logger  # type: ignore
logger = setup_logger(__name__, service_name="nlp-service")
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
    title="StateX NLP Service",
    description="Natural Language Processing service for business requirement analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    sentiment: str
    topics: List[str]
    industry: str
    requirements: List[str]
    confidence: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nlp-service"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text content for business requirements"""
    try:
        text = request.text.lower()
        
        # Simple sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "problem", "issue", "difficult"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract topics
        topics = []
        topic_keywords = {
            "business": ["business", "company", "enterprise", "organization"],
            "technology": ["technology", "tech", "software", "digital", "ai", "automation"],
            "website": ["website", "web", "site", "online", "internet"],
            "ecommerce": ["ecommerce", "e-commerce", "shop", "store", "selling", "buy", "sell"],
            "mobile": ["mobile", "app", "smartphone", "ios", "android"],
            "data": ["data", "analytics", "database", "information"],
            "marketing": ["marketing", "advertising", "promotion", "brand", "social media"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        # Detect industry
        industry_keywords = {
            "automotive": ["car", "auto", "vehicle", "repair", "garage", "mechanic"],
            "healthcare": ["health", "medical", "clinic", "doctor", "patient", "hospital"],
            "retail": ["store", "shop", "retail", "ecommerce", "inventory", "sales"],
            "education": ["school", "education", "learning", "course", "training", "student"],
            "finance": ["bank", "finance", "money", "investment", "trading", "loan"],
            "technology": ["tech", "software", "digital", "ai", "automation", "development"]
        }
        
        industry = "general"
        for ind, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                industry = ind
                break
        
        # Extract requirements
        requirements = []
        requirement_keywords = {
            "website": ["website", "web", "site", "online"],
            "mobile_app": ["mobile", "app", "smartphone"],
            "ecommerce": ["ecommerce", "shop", "store", "selling"],
            "database": ["database", "data", "storage"],
            "api": ["api", "integration", "connect"],
            "security": ["security", "secure", "protection"],
            "scalability": ["scalable", "scale", "growth"],
            "performance": ["fast", "speed", "performance", "optimization"]
        }
        
        for req, keywords in requirement_keywords.items():
            if any(keyword in text for keyword in keywords):
                requirements.append(req)
        
        # Calculate confidence
        confidence = min(0.9, 0.5 + (len(topics) * 0.1) + (len(requirements) * 0.05))
        
        return AnalysisResponse(
            sentiment=sentiment,
            topics=topics,
            industry=industry,
            requirements=requirements,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("NLP_SERVICE_PORT", "3381")), log_config=log_config)
