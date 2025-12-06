"""
StateX Template Repository Service

Template management and optimization service.
Handles template storage, search, matching, and optimization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import os
import time
import logging
from datetime import datetime

# Configure logging with timestamps
import sys
from pathlib import Path
# Import logger from shared directory
import importlib.util
shared_path = "/app/shared"
logger_spec = importlib.util.spec_from_file_location("logger", os.path.join(shared_path, "logger.py"))
logger_module = importlib.util.module_from_spec(logger_spec)
logger_spec.loader.exec_module(logger_module)
logger = logger_module.setup_logger(__name__, service_name="template-repository")
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
    title="StateX Template Repository Service",
    description="Template management and optimization",
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

class TemplateSearchRequest(BaseModel):
    requirements: str
    analysis: Dict[str, Any] = {}
    template_type: Optional[str] = None
    industry: Optional[str] = None
    features: Optional[List[str]] = None

class TemplateMatchResponse(BaseModel):
    search_id: str
    status: str
    matches: List[Dict[str, Any]]
    processing_time: float
    created_at: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "template-repository",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/find-templates", response_model=TemplateMatchResponse)
async def find_templates(request: TemplateSearchRequest):
    """Find matching templates based on requirements"""
    start_time = time.time()
    
    try:
        search_id = f"template_search_{int(time.time())}"
        
        # Simulate template matching
        matches = await find_matching_templates(
            request.requirements,
            request.analysis,
            request.template_type,
            request.industry,
            request.features
        )
        
        processing_time = time.time() - start_time
        
        return TemplateMatchResponse(
            search_id=search_id,
            status="completed",
            matches=matches,
            processing_time=processing_time,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error finding templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find templates: {str(e)}")

async def find_matching_templates(
    requirements: str,
    analysis: Dict[str, Any],
    template_type: Optional[str],
    industry: Optional[str],
    features: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """Find templates that match the requirements"""
    
    # Simulate template matching algorithm
    templates = [
        {
            "template_id": "business-website-001",
            "name": "Professional Business Website",
            "type": "website",
            "industry": "general",
            "description": "Clean, professional website template for businesses",
            "features": ["hero", "about", "services", "contact", "blog"],
            "tech_stack": ["Next.js", "Tailwind CSS", "TypeScript"],
            "match_score": 0.92,
            "popularity": 0.85,
            "last_updated": "2024-01-15",
            "preview_url": "https://templates.statex.cz/business-website-001",
            "repository_url": "https://github.com/statex/templates/business-website-001",
            "customization_level": "medium",
            "estimated_setup_time": "2-3 hours"
        },
        {
            "template_id": "ecommerce-store-002",
            "name": "Modern E-commerce Store",
            "type": "ecommerce",
            "industry": "retail",
            "description": "Complete e-commerce solution with modern design",
            "features": ["product_catalog", "shopping_cart", "checkout", "user_account", "admin_panel"],
            "tech_stack": ["Next.js", "Stripe", "PostgreSQL", "Prisma"],
            "match_score": 0.88,
            "popularity": 0.78,
            "last_updated": "2024-01-10",
            "preview_url": "https://templates.statex.cz/ecommerce-store-002",
            "repository_url": "https://github.com/statex/templates/ecommerce-store-002",
            "customization_level": "advanced",
            "estimated_setup_time": "4-6 hours"
        },
        {
            "template_id": "saas-dashboard-003",
            "name": "SaaS Application Dashboard",
            "type": "app",
            "industry": "technology",
            "description": "Modern SaaS application with dashboard and user management",
            "features": ["dashboard", "user_management", "billing", "api", "analytics"],
            "tech_stack": ["Next.js", "Prisma", "PostgreSQL", "NextAuth.js"],
            "match_score": 0.85,
            "popularity": 0.82,
            "last_updated": "2024-01-12",
            "preview_url": "https://templates.statex.cz/saas-dashboard-003",
            "repository_url": "https://github.com/statex/templates/saas-dashboard-003",
            "customization_level": "advanced",
            "estimated_setup_time": "6-8 hours"
        },
        {
            "template_id": "landing-page-004",
            "name": "High-Converting Landing Page",
            "type": "landing_page",
            "industry": "marketing",
            "description": "Single page landing site optimized for conversions",
            "features": ["hero", "features", "testimonials", "pricing", "contact"],
            "tech_stack": ["Next.js", "Tailwind CSS", "Framer Motion"],
            "match_score": 0.90,
            "popularity": 0.88,
            "last_updated": "2024-01-08",
            "preview_url": "https://templates.statex.cz/landing-page-004",
            "repository_url": "https://github.com/statex/templates/landing-page-004",
            "customization_level": "basic",
            "estimated_setup_time": "1-2 hours"
        },
        {
            "template_id": "portfolio-site-005",
            "name": "Creative Portfolio Website",
            "type": "website",
            "industry": "creative",
            "description": "Portfolio website for creative professionals",
            "features": ["portfolio", "about", "services", "contact", "blog"],
            "tech_stack": ["Next.js", "Tailwind CSS", "MDX"],
            "match_score": 0.75,
            "popularity": 0.70,
            "last_updated": "2024-01-05",
            "preview_url": "https://templates.statex.cz/portfolio-site-005",
            "repository_url": "https://github.com/statex/templates/portfolio-site-005",
            "customization_level": "medium",
            "estimated_setup_time": "2-3 hours"
        }
    ]
    
    # Filter templates based on criteria
    filtered_templates = []
    
    for template in templates:
        score = calculate_match_score(template, requirements, analysis, template_type, industry, features)
        if score > 0.5:  # Only include templates with decent match
            template["match_score"] = score
            filtered_templates.append(template)
    
    # Sort by match score
    filtered_templates.sort(key=lambda x: x["match_score"], reverse=True)
    
    return filtered_templates[:5]  # Return top 5 matches

def calculate_match_score(
    template: Dict[str, Any],
    requirements: str,
    analysis: Dict[str, Any],
    template_type: Optional[str],
    industry: Optional[str],
    features: Optional[List[str]]
) -> float:
    """Calculate how well a template matches the requirements"""
    
    score = 0.0
    
    # Type matching
    if template_type and template["type"] == template_type:
        score += 0.3
    elif not template_type:
        score += 0.1  # Neutral score if no type specified
    
    # Industry matching
    if industry and template["industry"] == industry:
        score += 0.2
    elif not industry:
        score += 0.1  # Neutral score if no industry specified
    
    # Feature matching
    if features:
        template_features = template["features"]
        matching_features = set(features) & set(template_features)
        feature_score = len(matching_features) / len(features) if features else 0
        score += feature_score * 0.3
    
    # Requirements keyword matching
    requirements_lower = requirements.lower()
    template_name_lower = template["name"].lower()
    template_desc_lower = template["description"].lower()
    
    keyword_matches = 0
    keywords = ["business", "website", "ecommerce", "app", "dashboard", "landing", "portfolio"]
    
    for keyword in keywords:
        if keyword in requirements_lower and keyword in (template_name_lower + " " + template_desc_lower):
            keyword_matches += 1
    
    if keywords:
        keyword_score = keyword_matches / len(keywords)
        score += keyword_score * 0.2
    
    return min(score, 1.0)  # Cap at 1.0

@app.get("/api/templates")
async def get_all_templates():
    """Get all available templates"""
    return {
        "templates": [
            {
                "template_id": "business-website-001",
                "name": "Professional Business Website",
                "type": "website",
                "industry": "general",
                "description": "Clean, professional website template for businesses",
                "features": ["hero", "about", "services", "contact", "blog"],
                "tech_stack": ["Next.js", "Tailwind CSS", "TypeScript"],
                "popularity": 0.85,
                "last_updated": "2024-01-15"
            },
            {
                "template_id": "ecommerce-store-002",
                "name": "Modern E-commerce Store",
                "type": "ecommerce",
                "industry": "retail",
                "description": "Complete e-commerce solution with modern design",
                "features": ["product_catalog", "shopping_cart", "checkout", "user_account"],
                "tech_stack": ["Next.js", "Stripe", "PostgreSQL", "Prisma"],
                "popularity": 0.78,
                "last_updated": "2024-01-10"
            },
            {
                "template_id": "saas-dashboard-003",
                "name": "SaaS Application Dashboard",
                "type": "app",
                "industry": "technology",
                "description": "Modern SaaS application with dashboard and user management",
                "features": ["dashboard", "user_management", "billing", "api", "analytics"],
                "tech_stack": ["Next.js", "Prisma", "PostgreSQL", "NextAuth.js"],
                "popularity": 0.82,
                "last_updated": "2024-01-12"
            }
        ],
        "total": 3
    }

@app.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    """Get specific template details"""
    # Simulate template retrieval
    return {
        "template_id": template_id,
        "name": "Professional Business Website",
        "type": "website",
        "description": "Clean, professional website template for businesses",
        "features": ["hero", "about", "services", "contact", "blog"],
        "tech_stack": ["Next.js", "Tailwind CSS", "TypeScript"],
        "preview_url": f"https://templates.statex.cz/{template_id}",
        "repository_url": f"https://github.com/statex/templates/{template_id}",
        "documentation_url": f"https://docs.statex.cz/templates/{template_id}",
        "setup_instructions": [
            "Clone the repository",
            "Install dependencies with npm install",
            "Configure environment variables",
            "Run development server with npm run dev"
        ],
        "customization_guide": {
            "colors": "Update Tailwind CSS color variables",
            "content": "Replace placeholder content in components",
            "images": "Add your own images to the public folder",
            "styling": "Modify Tailwind classes for custom styling"
        }
    }

@app.get("/api/templates/categories")
async def get_template_categories():
    """Get template categories and types"""
    return {
        "categories": [
            {
                "id": "website",
                "name": "Website Templates",
                "description": "Complete website solutions",
                "count": 15
            },
            {
                "id": "ecommerce",
                "name": "E-commerce Templates",
                "description": "Online store solutions",
                "count": 8
            },
            {
                "id": "app",
                "name": "Application Templates",
                "description": "Web application solutions",
                "count": 12
            },
            {
                "id": "landing_page",
                "name": "Landing Page Templates",
                "description": "Single page solutions",
                "count": 20
            }
        ],
        "industries": [
            "general", "technology", "retail", "healthcare", 
            "finance", "education", "creative", "marketing"
        ]
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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("TEMPLATE_REPOSITORY_PORT", "3385")), log_config=log_config)
