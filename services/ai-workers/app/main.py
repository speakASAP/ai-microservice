"""
StateX ai-workers Service

User management and authentication for the StateX platform.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# Import logger from shared directory
import importlib.util
shared_path = "/app/shared"
logger_spec = importlib.util.spec_from_file_location("logger", os.path.join(shared_path, "logger.py"))
logger_module = importlib.util.module_from_spec(logger_spec)
logger_spec.loader.exec_module(logger_module)
logger = logger_module.setup_logger(__name__, service_name="ai-workers")

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
    title="StateX ai-workers",
    description="User management and authentication for the StateX platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://statex.cz", "https://www.statex.cz", f"http://localhost:{os.getenv('FRONTEND_PORT', '3000')}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-workers",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {
        "status": "ready",
        "service": "ai-workers",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "StateX ai-workers Service",
        "version": "1.0.0",
        "status": "running"
    }

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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AI_WORKERS_PORT", "3387")), log_config=log_config)
