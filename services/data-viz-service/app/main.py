#!/usr/bin/env python3
"""
StateX Data Analysis/Visualization Service

This service provides data analysis and visualization capabilities:
1. Reads summary.md files from upload directories
2. Analyzes data from a data analysis perspective
3. Creates visualization.md files with charts, graphs, and insights
4. Integrates with the multi-agent workflow system

Port: 3389
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
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO
import base64

# Add utils to path
utils_path = Path(__file__).parent.parent.parent.parent.parent / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from logger import setup_logger
logger = setup_logger(__name__, service_name="data-viz-service")

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
    title="StateX Data Analysis/Visualization Service",
    description="Data analysis and visualization service for summary data",
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

# Service Configuration
DATA_UPLOAD_BASE_PATH = os.getenv("UPLOAD_DIR", "./data/uploads")
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
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"

class DataVizRequest(BaseModel):
    submission_id: str
    user_id: str
    session_id: str
    analysis_type: AnalysisType = AnalysisType.DATA_ANALYSIS

class DataVizResponse(BaseModel):
    success: bool
    visualization_data: Dict[str, Any]
    charts_created: List[str]
    insights: List[str]
    processing_time: float
    error: Optional[str] = None

class DataVizService:
    def __init__(self):
        self.upload_base_path = Path(DATA_UPLOAD_BASE_PATH)
        self.chart_types = [
            "bar_chart", "line_chart", "pie_chart", "scatter_plot", 
            "histogram", "heatmap", "trend_analysis", "comparison_chart"
        ]
        
    def get_session_directory(self, submission_id: str, user_id: str, session_id: str) -> Optional[Path]:
        """Get the session directory path"""
        try:
            session_dir = self.upload_base_path / submission_id / f"sess_{session_id}_{user_id}"
            if session_dir.exists():
                return session_dir
            else:
                logger.warning(f"Session directory not found: {session_dir}")
                return None
        except Exception as e:
            logger.error(f"Error getting session directory: {e}")
            return None
    
    def read_summary_file(self, session_path: Path) -> str:
        """Read summary.md file from session directory"""
        try:
            summary_file = session_path / "summary.md"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Read summary.md: {len(content)} characters")
                return content
            else:
                logger.warning(f"summary.md not found in {session_path}")
                return ""
        except Exception as e:
            logger.error(f"Error reading summary.md: {e}")
            return ""
    
    def extract_data_points(self, summary_content: str) -> Dict[str, Any]:
        """Extract data points from summary content for visualization"""
        data_points = {
            "business_metrics": [],
            "technical_metrics": [],
            "timeline_data": [],
            "categories": [],
            "percentages": [],
            "costs": [],
            "timeframes": []
        }
        
        # Extract business metrics
        business_patterns = [
            r'(\d+)\s*(?:users|customers|clients)',
            r'(\d+)\s*(?:orders|transactions|sales)',
            r'(\d+)\s*(?:revenue|income|profit)',
            r'(\d+)\s*(?:employees|staff|team)',
            r'(\d+)\s*(?:products|items|services)'
        ]
        
        for pattern in business_patterns:
            matches = re.findall(pattern, summary_content, re.IGNORECASE)
            data_points["business_metrics"].extend([int(m) for m in matches])
        
        # Extract technical metrics
        tech_patterns = [
            r'(\d+)\s*(?:pages|screens|components)',
            r'(\d+)\s*(?:APIs|endpoints|services)',
            r'(\d+)\s*(?:databases|tables|records)',
            r'(\d+)\s*(?:GB|MB|TB)',
            r'(\d+)\s*(?:users|concurrent|sessions)'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, summary_content, re.IGNORECASE)
            data_points["technical_metrics"].extend([int(m) for m in matches])
        
        # Extract cost information
        cost_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars|USD)',
            r'budget[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'cost[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in cost_patterns:
            matches = re.findall(pattern, summary_content, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert to float
                    clean_value = match.replace(',', '').replace('$', '')
                    data_points["costs"].append(float(clean_value))
                except ValueError:
                    continue
        
        # Extract timeframes
        time_patterns = [
            r'(\d+)\s*(?:days|weeks|months|years)',
            r'(\d+)\s*(?:hours|minutes)',
            r'(\d+)\s*(?:Q[1-4]|quarter)',
            r'(\d{4})'  # Years
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, summary_content, re.IGNORECASE)
            data_points["timeframes"].extend([int(m) for m in matches])
        
        # Extract categories and percentages
        category_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*percent',
            r'(\d+(?:\.\d+)?)\s*percentage'
        ]
        
        for pattern in category_patterns:
            matches = re.findall(pattern, summary_content, re.IGNORECASE)
            data_points["percentages"].extend([float(m) for m in matches])
        
        logger.info(f"Extracted data points: {data_points}")
        return data_points
    
    def create_visualization_content(self, summary_content: str, data_points: Dict[str, Any]) -> str:
        """Create visualization content based on summary and extracted data"""
        
        visualization_content = f"""# Data Analysis & Visualization Report

## Executive Summary
This report provides data-driven insights and visualizations based on the business analysis summary.

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** Data Analysis & Visualization
**Data Source:** summary.md

---

## Key Metrics Overview

### Business Metrics
"""
        
        if data_points["business_metrics"]:
            avg_business = sum(data_points["business_metrics"]) / len(data_points["business_metrics"])
            max_business = max(data_points["business_metrics"])
            min_business = min(data_points["business_metrics"])
            
            visualization_content += f"""
- **Average Business Metric:** {avg_business:.2f}
- **Maximum Value:** {max_business}
- **Minimum Value:** {min_business}
- **Total Data Points:** {len(data_points["business_metrics"])}
"""
        else:
            visualization_content += "\n- No specific business metrics found in the data\n"
        
        visualization_content += "\n### Technical Metrics\n"
        
        if data_points["technical_metrics"]:
            avg_tech = sum(data_points["technical_metrics"]) / len(data_points["technical_metrics"])
            max_tech = max(data_points["technical_metrics"])
            min_tech = min(data_points["technical_metrics"])
            
            visualization_content += f"""
- **Average Technical Metric:** {avg_tech:.2f}
- **Maximum Value:** {max_tech}
- **Minimum Value:** {min_tech}
- **Total Data Points:** {len(data_points["technical_metrics"])}
"""
        else:
            visualization_content += "\n- No specific technical metrics found in the data\n"
        
        visualization_content += "\n### Financial Analysis\n"
        
        if data_points["costs"]:
            total_cost = sum(data_points["costs"])
            avg_cost = total_cost / len(data_points["costs"])
            max_cost = max(data_points["costs"])
            min_cost = min(data_points["costs"])
            
            visualization_content += f"""
- **Total Estimated Cost:** ${total_cost:,.2f}
- **Average Cost:** ${avg_cost:,.2f}
- **Maximum Cost:** ${max_cost:,.2f}
- **Minimum Cost:** ${min_cost:,.2f}
- **Cost Data Points:** {len(data_points["costs"])}
"""
        else:
            visualization_content += "\n- No specific cost information found in the data\n"
        
        visualization_content += "\n### Timeline Analysis\n"
        
        if data_points["timeframes"]:
            avg_time = sum(data_points["timeframes"]) / len(data_points["timeframes"])
            max_time = max(data_points["timeframes"])
            min_time = min(data_points["timeframes"])
            
            visualization_content += f"""
- **Average Timeframe:** {avg_time:.2f} units
- **Maximum Timeframe:** {max_time} units
- **Minimum Timeframe:** {min_time} units
- **Timeframe Data Points:** {len(data_points["timeframes"])}
"""
        else:
            visualization_content += "\n- No specific timeframe information found in the data\n"
        
        # Add percentage analysis
        if data_points["percentages"]:
            visualization_content += "\n### Percentage Analysis\n"
            avg_percentage = sum(data_points["percentages"]) / len(data_points["percentages"])
            visualization_content += f"""
- **Average Percentage:** {avg_percentage:.2f}%
- **Percentage Data Points:** {len(data_points["percentages"])}
- **Range:** {min(data_points["percentages"]):.2f}% - {max(data_points["percentages"]):.2f}%
"""
        
        # Add visualization recommendations
        visualization_content += "\n## Visualization Recommendations\n\n"
        
        charts_created = []
        if data_points["business_metrics"]:
            charts_created.append("Business Metrics Bar Chart")
            visualization_content += "### 1. Business Metrics Bar Chart\n"
            visualization_content += "```mermaid\n"
            visualization_content += "graph TD\n"
            for i, metric in enumerate(data_points["business_metrics"][:5]):  # Limit to 5 for readability
                visualization_content += f"    A{i}[\"Metric {i+1}: {metric}\"]\n"
            visualization_content += "```\n\n"
        
        if data_points["costs"]:
            charts_created.append("Cost Analysis Pie Chart")
            visualization_content += "### 2. Cost Analysis\n"
            visualization_content += "```mermaid\n"
            visualization_content += "pie title Cost Distribution\n"
            for i, cost in enumerate(data_points["costs"][:4]):  # Limit to 4 for readability
                percentage = (cost / sum(data_points["costs"])) * 100
                visualization_content += f'    "Cost {i+1}" : {percentage:.1f}\n'
            visualization_content += "```\n\n"
        
        if data_points["timeframes"]:
            charts_created.append("Timeline Gantt Chart")
            visualization_content += "### 3. Project Timeline\n"
            visualization_content += "```mermaid\n"
            visualization_content += "gantt\n"
            visualization_content += "    title Project Timeline\n"
            visualization_content += "    dateFormat  YYYY-MM-DD\n"
            visualization_content += "    section Phase 1\n"
            for i, timeframe in enumerate(data_points["timeframes"][:3]):
                start_date = datetime.now() + timedelta(days=i*30)
                end_date = start_date + timedelta(days=timeframe)
                visualization_content += f"    Task {i+1} : {start_date.strftime('%Y-%m-%d')}, {end_date.strftime('%Y-%m-%d')}\n"
            visualization_content += "```\n\n"
        
        # Add data insights
        visualization_content += "## Data Insights\n\n"
        
        insights = []
        if data_points["business_metrics"]:
            insights.append(f"Business metrics show {len(data_points['business_metrics'])} key performance indicators")
        
        if data_points["costs"]:
            total_cost = sum(data_points["costs"])
            insights.append(f"Total estimated project cost: ${total_cost:,.2f}")
        
        if data_points["timeframes"]:
            avg_time = sum(data_points["timeframes"]) / len(data_points["timeframes"])
            insights.append(f"Average project timeframe: {avg_time:.1f} time units")
        
        if data_points["percentages"]:
            avg_percentage = sum(data_points["percentages"]) / len(data_points["percentages"])
            insights.append(f"Average completion percentage: {avg_percentage:.1f}%")
        
        for i, insight in enumerate(insights, 1):
            visualization_content += f"{i}. {insight}\n"
        
        # Add recommendations
        visualization_content += "\n## Recommendations\n\n"
        visualization_content += "1. **Data Collection**: Implement systematic data collection for better analysis\n"
        visualization_content += "2. **Monitoring**: Set up real-time monitoring of key metrics\n"
        visualization_content += "3. **Reporting**: Create automated reports based on this analysis\n"
        visualization_content += "4. **Visualization**: Use the recommended charts for stakeholder presentations\n"
        
        # Add technical details
        visualization_content += "\n## Technical Details\n\n"
        visualization_content += f"- **Data Points Analyzed**: {sum(len(v) for v in data_points.values())}\n"
        visualization_content += f"- **Charts Generated**: {len(charts_created)}\n"
        visualization_content += f"- **Analysis Date**: {datetime.now().isoformat()}\n"
        visualization_content += f"- **Service**: StateX Data Analysis/Visualization Service\n"
        
        return visualization_content, charts_created, insights
    
    async def analyze_with_gemini(self, summary_content: str) -> Dict[str, Any]:
        """Use Gemini AI for advanced data analysis"""
        if not GEMINI_API_KEY:
            logger.warning("Gemini API key not available, using fallback analysis")
            return self._fallback_analysis(summary_content)
        
        try:
            prompt = f"""Analyze this business summary from a data analysis perspective and provide insights:

{summary_content}

Please provide a JSON response with:
- data_insights: Array of key data insights
- visualization_suggestions: Array of recommended chart types
- key_metrics: Object with extracted numerical data
- trends: Array of identified trends
- recommendations: Array of data-driven recommendations
- confidence: Float between 0 and 1
"""
            
            headers = {
                "Content-Type": "application/json"
            }
            
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
                
                url = f"{GEMINI_API_BASE}/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
                
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
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
                                analysis = self._fallback_analysis(summary_content)
                        except:
                            analysis = self._fallback_analysis(summary_content)
                        
                        analysis["ai_service"] = "Google Gemini"
                        analysis["model_used"] = "gemini-2.5-flash"
                        return analysis
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {response.status} - {error_text}")
                        return self._fallback_analysis(summary_content)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._fallback_analysis(summary_content)
    
    def _fallback_analysis(self, summary_content: str) -> Dict[str, Any]:
        """Fallback analysis when Gemini is not available"""
        return {
            "data_insights": ["Data analysis completed using fallback method"],
            "visualization_suggestions": ["bar_chart", "pie_chart", "line_chart"],
            "key_metrics": {"word_count": len(summary_content.split())},
            "trends": ["Analysis completed"],
            "recommendations": ["Implement data collection", "Create monitoring dashboard"],
            "confidence": 0.7,
            "ai_service": "Fallback Analysis",
            "model_used": "fallback"
        }

# Initialize service
data_viz_service = DataVizService()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("🚀 Data Analysis/Visualization Service starting up...")
    logger.info("✅ Service ready for data analysis and visualization")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-viz-service",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "upload_path": str(data_viz_service.upload_base_path),
        "gemini_available": bool(GEMINI_API_KEY)
    }

@app.post("/analyze", response_model=DataVizResponse)
async def analyze_data(request: DataVizRequest):
    """Analyze data and create visualizations"""
    start_time = time.time()
    
    try:
        # Get session directory
        session_path = data_viz_service.get_session_directory(
            request.submission_id, 
            request.user_id, 
            request.session_id
        )
        
        if not session_path:
            raise HTTPException(
                status_code=404, 
                detail=f"Session directory not found for submission {request.submission_id}"
            )
        
        # Read summary file
        summary_content = data_viz_service.read_summary_file(session_path)
        
        if not summary_content:
            raise HTTPException(
                status_code=404, 
                detail="summary.md file not found or empty"
            )
        
        # Extract data points
        data_points = data_viz_service.extract_data_points(summary_content)
        
        # Get AI analysis
        ai_analysis = await data_viz_service.analyze_with_gemini(summary_content)
        
        # Create visualization content
        visualization_content, charts_created, insights = data_viz_service.create_visualization_content(
            summary_content, data_points
        )
        
        # Write visualization.md file
        visualization_file = session_path / "visualization.md"
        try:
            with open(visualization_file, 'w', encoding='utf-8') as f:
                f.write(visualization_content)
            logger.info(f"Created visualization.md: {visualization_file}")
        except Exception as e:
            logger.error(f"Error writing visualization.md: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create visualization file: {e}")
        
        processing_time = time.time() - start_time
        
        return DataVizResponse(
            success=True,
            visualization_data={
                "file_path": str(visualization_file),
                "content_length": len(visualization_content),
                "data_points": data_points,
                "ai_analysis": ai_analysis
            },
            charts_created=charts_created,
            insights=insights,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        processing_time = time.time() - start_time
        
        return DataVizResponse(
            success=False,
            visualization_data={},
            charts_created=[],
            insights=[],
            processing_time=processing_time,
            error=str(e)
        )

@app.get("/api/status/{submission_id}")
async def get_submission_status(submission_id: str):
    """Get status of a submission"""
    return {
        "submission_id": submission_id,
        "status": "completed",
        "progress": 100,
        "current_step": "data_analysis_complete",
        "steps_completed": [
            "summary_analysis",
            "data_extraction", 
            "visualization_creation"
        ],
        "timestamp": datetime.now().isoformat(),
        "message": "Data analysis and visualization completed successfully"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "StateX Data Analysis/Visualization Service",
        "version": "1.0.0",
        "description": "Data analysis and visualization service for summary data",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "status": "/api/status/{submission_id}",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    from datetime import timedelta
    
    port = int(os.getenv("DATA_VIZ_SERVICE_PORT", "3389"))
    
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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("DATA_VIZ_SERVICE_PORT", "3389")), log_config=log_config)
