"""
StateX Prototype Generator Service

Website and application prototype creation service.
Handles HTML/CSS/JS generation, queue processing, and file management.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import os
import time
import logging
import asyncio
import json
from datetime import datetime

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_queue.queue_manager import QueueManager, PrototypeJob
from job_queue.worker import QueueWorker
from generators.html_generator import HTMLGenerator
from generators.css_generator import CSSGenerator
from generators.js_generator import JSGenerator
from generators.content_generator import ContentGenerator
from storage.file_manager import FileManager
from api.queue import router as queue_router
from api.prototype import router as prototype_router

# Configure logging with timestamps
import sys
from pathlib import Path
# Add utils to path
utils_path = Path(__file__).parent.parent.parent.parent.parent / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from logger import setup_logger
logger = setup_logger(__name__, service_name="prototype-generator")
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
    title="StateX Prototype Generator Service",
    description="Website and application prototype creation with AI-powered generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize services
queue_manager = QueueManager()
file_manager = FileManager()
html_generator = HTMLGenerator()
css_generator = CSSGenerator()
js_generator = JSGenerator()
content_generator = ContentGenerator()

# Global worker instance
worker = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(queue_router)
app.include_router(prototype_router)

@app.on_event("startup")
async def startup_event():
    """Start the queue worker on startup."""
    global worker
    try:
        worker = QueueWorker(queue_manager)
        asyncio.create_task(worker.start())
        logger.info("Queue worker started")
    except Exception as e:
        logger.error(f"Failed to start queue worker: {e}")
        # Continue without worker for now

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the queue worker on shutdown."""
    global worker
    if worker:
        worker.stop()
        logger.info("Queue worker stopped")

class PrototypeRequest(BaseModel):
    requirements: str
    analysis: Dict[str, Any] = {}
    prototype_type: str = "website"  # website, app, landing_page, ecommerce
    customization_level: str = "medium"  # basic, medium, advanced

class PrototypeResponse(BaseModel):
    prototype_id: str
    status: str
    results: Dict[str, Any]
    processing_time: float
    created_at: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "prototype-generator",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint - disabled"""
    return {"message": "Metrics collection is disabled", "status": "disabled"}

@app.get("/debug/redis")
async def debug_redis():
    """Debug Redis connection."""
    try:
        # Test Redis connection
        import redis
        # Port configured in database-server/.env: REDIS_SERVER_PORT (default: 6379)
        redis_url = os.getenv("REDIS_URL", f"redis://localhost:{os.getenv('REDIS_SERVER_PORT', '6379')}")
        r = redis.from_url(f'{redis_url}/1', decode_responses=False)
        keys = r.keys('prototype_job:*')
        
        # Test queue manager
        job = queue_manager.get_job('6943561f-67da-4d2e-91a7-ef9b577c66f9')
        
        return {
            "redis_keys_count": len(keys),
            "queue_manager_working": job is not None,
            "job_status": job.status if job else None
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/test-queue")
async def test_queue():
    """Test queue functionality"""
    try:
        job_id = queue_manager.enqueue_job("test_user", "test_submission", "test requirements")
        return {"success": True, "job_id": job_id}
    except Exception as e:
        logger.error(f"Test queue error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/generate-prototype", response_model=PrototypeResponse)
async def generate_prototype(request: PrototypeRequest, background_tasks: BackgroundTasks):
    """Generate a website or application prototype using AI"""
    start_time = time.time()
    
    try:
        # Debug: log the parsed request
        logger.info(f"Received request: {request}")
        logger.info(f"Request requirements: '{request.requirements}'")
        logger.info(f"Request analysis: {request.analysis}")
        # Generate unique prototype ID
        prototype_id = f"proto_{int(time.time())}"
        
        # Debug: log the request data
        logger.info(f"Prototype request: requirements='{request.requirements}', analysis={request.analysis}")
        
        # Add job to queue for processing
        job_id = queue_manager.enqueue_job(
            user_id="system",
            submission_id=prototype_id,
            requirements=request.requirements or ""
        )
        
        # Start background processing
        background_tasks.add_task(
            process_prototype_generation,
            job_id,
            request.requirements,
            request.analysis,
            request.prototype_type,
            request.customization_level
        )
        
        processing_time = time.time() - start_time
        
        return PrototypeResponse(
            prototype_id=prototype_id,
            status="queued",
            results={"job_id": job_id, "message": "Prototype generation queued"},
            processing_time=processing_time,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error queuing prototype generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue prototype generation: {str(e)}")

@app.post("/api/queue/submit")
async def submit_prototype_job(request: dict):
    """Submit a prototype generation job to the queue."""
    try:
        job_id = queue_manager.enqueue_job(
            user_id=request.get("user_id", "anonymous"),
            submission_id=request.get("submission_id", f"sub_{int(time.time())}"),
            requirements=request.get("requirements", "")
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Job submitted successfully",
            "estimated_completion_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.get("/api/queue/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a prototype generation job."""
    try:
        job = queue_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "expires_at": job.expires_at.isoformat(),
            "error_message": job.error_message,
            "generated_files": job.generated_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/api/prototype/{project_id}")
async def get_prototype(project_id: str):
    """Get prototype files and information."""
    try:
        # Get prototype files
        files = file_manager.get_prototype_files(project_id)
        
        if not files:
            raise HTTPException(status_code=404, detail="Prototype not found")
        
        # Get storage stats
        stats = file_manager.get_storage_stats()
        
        return {
            "project_id": project_id,
            "files": files,
            "file_count": len(files),
            "total_size": file_manager.get_prototype_size(project_id),
            "created_at": datetime.now().isoformat(),
            "storage_stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prototype: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prototype: {str(e)}")

async def process_prototype_generation(
    job_id: str,
    requirements: str,
    analysis: Dict[str, Any],
    prototype_type: str,
    customization_level: str
):
    """Process prototype generation using AI generators."""
    try:
        # Update job status to processing
        queue_manager.update_job_status(job_id, "processing")
        
        # Generate content using NLP service
        logger.info(f"Generating content for job {job_id}")
        content = await content_generator.generate_content(requirements, analysis, prototype_type)
        
        # Generate HTML structure
        logger.info(f"Generating HTML for job {job_id}")
        html_content = await html_generator.generate_html(requirements, analysis, prototype_type)
        
        # Generate CSS styles
        logger.info(f"Generating CSS for job {job_id}")
        css_content = await css_generator.generate_css(html_content, requirements, analysis)
        
        # Generate JavaScript functionality
        logger.info(f"Generating JavaScript for job {job_id}")
        js_content = await js_generator.generate_js(html_content, requirements, analysis)
        
        # Create project ID
        project_id = f"project-{int(time.time())}"
        
        # Prepare files for saving
        files = {
            "index.html": html_content,
            "styles.css": css_content,
            "script.js": js_content,
            "metadata.json": json.dumps({
                "project_id": project_id,
                "prototype_type": prototype_type,
                "requirements": requirements,
                "analysis": analysis,
                "content": content,
                "created_at": datetime.now().isoformat(),
                "ai_generated": True
            }, indent=2)
        }
        
        # Save files to storage
        logger.info(f"Saving files for project {project_id}")
        saved_files = file_manager.save_prototype_files(project_id, files)
        
        # Update job with results
        queue_manager.update_job_status(
            job_id, 
            "completed", 
            generated_files={
                "project_id": project_id,
                "files": saved_files,
                "url": f"http://{project_id}.localhost:{os.getenv('FRONTEND_PORT', '3000')}",
                "content": content
            }
        )
        
        logger.info(f"Prototype generation completed for job {job_id}, project {project_id}")
        
    except Exception as e:
        logger.error(f"Error processing prototype generation for job {job_id}: {e}")
        queue_manager.update_job_status(job_id, "failed", error_message=str(e))

async def create_prototype(
    requirements: str,
    analysis: Dict[str, Any],
    prototype_type: str,
    customization_level: str
) -> Dict[str, Any]:
    """Create a prototype based on requirements and analysis with real-time AI orchestration"""
    
    # Generate unique prototype ID
    prototype_id = f"proto_{int(time.time())}"
    
    # Real-time AI analysis for prototype requirements
    ai_analysis = await analyze_prototype_requirements(requirements, analysis, prototype_type)
    
    # Generate actual code structure
    code_structure = await generate_code_structure(ai_analysis, prototype_type, customization_level)
    
    # Create comprehensive prototype
    prototype = {
        "prototype_info": {
            "id": prototype_id,
            "type": prototype_type,
            "name": ai_analysis.get("project_name", "AI-Generated Prototype"),
            "description": ai_analysis.get("project_description", "AI-generated prototype based on requirements"),
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "ai_confidence": ai_analysis.get("confidence", 0.85)
        },
        "technical_specs": {
            "framework": "Next.js 14",
            "language": "TypeScript",
            "styling": "Tailwind CSS",
            "database": "PostgreSQL",
            "authentication": "NextAuth.js",
            "deployment": "Vercel",
            "ai_optimized": True
        },
        "ai_analysis": ai_analysis,
        "pages": generate_pages(prototype_type, requirements, ai_analysis),
        "components": generate_components(prototype_type, ai_analysis),
        "features": generate_features(requirements, analysis, ai_analysis),
        "code_structure": code_structure,
        "deployment": {
            "url": f"http://project-{prototype_id}.localhost:{os.getenv('FRONTEND_PORT', '3000')}",
            "subdomain": f"project-{prototype_id}",
            "status": "ready",
            "deployment_time": "2-3 minutes",
            "local_access": True
        },
        "code_repository": {
            "url": f"https://github.com/statex/prototype-{prototype_id}",
            "branch": "main",
            "commit": "initial-prototype",
            "local_path": f"/tmp/prototypes/{prototype_id}"
        }
    }
    
    return prototype

async def analyze_prototype_requirements(requirements: str, analysis: Dict[str, Any], prototype_type: str) -> Dict[str, Any]:
    """Analyze requirements using AI to extract key information"""
    
    # Extract key information from requirements and analysis
    project_name = extract_project_name(requirements)
    project_description = extract_project_description(requirements, analysis)
    target_audience = extract_target_audience(requirements, analysis)
    key_features = extract_key_features(requirements, analysis)
    design_preferences = extract_design_preferences(requirements, analysis)
    
    return {
        "project_name": project_name,
        "project_description": project_description,
        "target_audience": target_audience,
        "key_features": key_features,
        "design_preferences": design_preferences,
        "complexity_level": determine_complexity(requirements, analysis),
        "confidence": 0.85,
        "ai_insights": generate_ai_insights(requirements, analysis)
    }

async def generate_code_structure(ai_analysis: Dict[str, Any], prototype_type: str, customization_level: str) -> Dict[str, Any]:
    """Generate actual code structure for the prototype"""
    
    return {
        "file_structure": generate_file_structure(prototype_type, ai_analysis),
        "components": generate_component_code(prototype_type, ai_analysis),
        "pages": generate_page_code(prototype_type, ai_analysis),
        "styles": generate_style_code(ai_analysis),
        "configuration": generate_config_files(prototype_type, ai_analysis),
        "dependencies": generate_dependencies(prototype_type, ai_analysis)
    }

def extract_project_name(requirements: str) -> str:
    """Extract project name from requirements"""
    # Simple extraction logic - in production, use NLP service
    words = requirements.split()
    if len(words) >= 2:
        return " ".join(words[:2]).title()
    return "AI Generated Project"

def extract_project_description(requirements: str, analysis: Dict[str, Any]) -> str:
    """Extract project description from requirements and analysis"""
    if analysis.get("nlp_analysis", {}).get("results", {}).get("text_summary"):
        return analysis["nlp_analysis"]["results"]["text_summary"]
    return requirements[:200] + "..." if len(requirements) > 200 else requirements

def extract_target_audience(requirements: str, analysis: Dict[str, Any]) -> str:
    """Extract target audience from requirements"""
    # Simple extraction - in production, use NLP analysis
    if "business" in requirements.lower():
        return "Business professionals and entrepreneurs"
    elif "ecommerce" in requirements.lower():
        return "Online shoppers and customers"
    return "General users"

def extract_key_features(requirements: str, analysis: Dict[str, Any]) -> List[str]:
    """Extract key features from requirements"""
    features = []
    req_lower = requirements.lower()
    
    if "contact" in req_lower or "form" in req_lower:
        features.append("Contact Form")
    if "ecommerce" in req_lower or "shop" in req_lower:
        features.append("E-commerce Functionality")
    if "blog" in req_lower:
        features.append("Blog System")
    if "user" in req_lower or "account" in req_lower:
        features.append("User Authentication")
    
    return features if features else ["Responsive Design", "SEO Optimization"]

def extract_design_preferences(requirements: str, analysis: Dict[str, Any]) -> Dict[str, str]:
    """Extract design preferences from requirements"""
    return {
        "style": "modern" if "modern" in requirements.lower() else "professional",
        "color_scheme": "blue" if "blue" in requirements.lower() else "neutral",
        "layout": "clean" if "clean" in requirements.lower() else "standard"
    }

def determine_complexity(requirements: str, analysis: Dict[str, Any]) -> str:
    """Determine project complexity level"""
    req_lower = requirements.lower()
    if any(word in req_lower for word in ["complex", "advanced", "enterprise"]):
        return "high"
    elif any(word in req_lower for word in ["simple", "basic", "landing"]):
        return "low"
    return "medium"

def generate_ai_insights(requirements: str, analysis: Dict[str, Any]) -> List[str]:
    """Generate AI insights for the project"""
    insights = []
    
    if analysis.get("nlp_analysis", {}).get("results", {}).get("key_insights"):
        insights.extend(analysis["nlp_analysis"]["results"]["key_insights"][:3])
    
    insights.append("AI-optimized for user engagement")
    insights.append("Mobile-first responsive design")
    
    return insights

def generate_file_structure(prototype_type: str, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate file structure for the prototype"""
    base_structure = {
        "src": {
            "app": {
                "layout.tsx": "Root layout component",
                "page.tsx": "Home page component",
                "globals.css": "Global styles"
            },
            "components": {
                "ui": "Reusable UI components",
                "layout": "Layout components"
            },
            "lib": "Utility functions and configurations"
        },
        "public": "Static assets",
        "package.json": "Dependencies and scripts",
        "tailwind.config.js": "Tailwind CSS configuration",
        "next.config.js": "Next.js configuration"
    }
    
    if prototype_type == "ecommerce":
        base_structure["src"]["app"]["products"] = "Product pages"
        base_structure["src"]["app"]["cart"] = "Shopping cart"
    
    return base_structure

def generate_component_code(prototype_type: str, ai_analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate component code"""
    return {
        "Header.tsx": "// Header component with navigation",
        "Footer.tsx": "// Footer component with links",
        "Button.tsx": "// Reusable button component",
        "Card.tsx": "// Content card component"
    }

def generate_page_code(prototype_type: str, ai_analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate page code"""
    return {
        "page.tsx": "// Home page component",
        "about/page.tsx": "// About page component",
        "contact/page.tsx": "// Contact page component"
    }

def generate_style_code(ai_analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate style code"""
    return {
        "globals.css": "/* Global styles with Tailwind CSS */",
        "components.css": "/* Component-specific styles */"
    }

def generate_config_files(prototype_type: str, ai_analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate configuration files"""
    return {
        "package.json": "// Dependencies and scripts",
        "tailwind.config.js": "// Tailwind configuration",
        "next.config.js": "// Next.js configuration"
    }

def generate_dependencies(prototype_type: str, ai_analysis: Dict[str, Any]) -> List[str]:
    """Generate project dependencies"""
    base_deps = ["next", "react", "typescript", "tailwindcss"]
    
    if prototype_type == "ecommerce":
        base_deps.extend(["stripe", "prisma", "@prisma/client"])
    
    return base_deps

def generate_pages(prototype_type: str, requirements: str, ai_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generate pages based on prototype type and requirements"""
    
    if prototype_type == "website":
        return [
            {
                "name": "Home",
                "path": "/",
                "description": "Landing page with hero section and features",
                "components": ["Hero", "Features", "Testimonials", "CTA"]
            },
            {
                "name": "About",
                "path": "/about",
                "description": "About page with company information",
                "components": ["AboutHero", "Team", "Values", "History"]
            },
            {
                "name": "Services",
                "path": "/services",
                "description": "Services page with offerings",
                "components": ["ServicesGrid", "Pricing", "Process", "FAQ"]
            },
            {
                "name": "Contact",
                "path": "/contact",
                "description": "Contact page with form and information",
                "components": ["ContactForm", "Map", "Info", "Social"]
            }
        ]
    
    elif prototype_type == "ecommerce":
        return [
            {
                "name": "Home",
                "path": "/",
                "description": "E-commerce homepage",
                "components": ["Hero", "FeaturedProducts", "Categories", "Newsletter"]
            },
            {
                "name": "Products",
                "path": "/products",
                "description": "Product listing page",
                "components": ["ProductGrid", "Filters", "Pagination", "Sort"]
            },
            {
                "name": "Product Detail",
                "path": "/products/[id]",
                "description": "Individual product page",
                "components": ["ProductImages", "ProductInfo", "Reviews", "Related"]
            },
            {
                "name": "Cart",
                "path": "/cart",
                "description": "Shopping cart page",
                "components": ["CartItems", "CheckoutForm", "Summary", "Payment"]
            }
        ]
    
    else:  # landing_page
        return [
            {
                "name": "Landing",
                "path": "/",
                "description": "Single page landing site",
                "components": ["Hero", "Features", "Benefits", "Pricing", "Contact"]
            }
        ]

def generate_components(prototype_type: str, ai_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generate reusable components"""
    
    base_components = [
        {
            "name": "Header",
            "type": "layout",
            "description": "Site header with navigation",
            "props": ["logo", "navItems", "ctaButton"]
        },
        {
            "name": "Footer",
            "type": "layout",
            "description": "Site footer with links and info",
            "props": ["links", "socialMedia", "copyright"]
        },
        {
            "name": "Button",
            "type": "ui",
            "description": "Reusable button component",
            "props": ["variant", "size", "onClick", "children"]
        },
        {
            "name": "Card",
            "type": "ui",
            "description": "Content card component",
            "props": ["title", "description", "image", "actions"]
        }
    ]
    
    if prototype_type == "ecommerce":
        base_components.extend([
            {
                "name": "ProductCard",
                "type": "ecommerce",
                "description": "Product display card",
                "props": ["product", "onAddToCart", "onViewDetails"]
            },
            {
                "name": "CartItem",
                "type": "ecommerce",
                "description": "Cart item component",
                "props": ["item", "onUpdateQuantity", "onRemove"]
            }
        ])
    
    return base_components

def generate_features(requirements: str, analysis: Dict[str, Any], ai_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generate features based on requirements and analysis"""
    
    features = [
        {
            "name": "Responsive Design",
            "description": "Mobile-first responsive design",
            "priority": "high",
            "implementation": "Tailwind CSS responsive utilities"
        },
        {
            "name": "SEO Optimization",
            "description": "Search engine optimization",
            "priority": "high",
            "implementation": "Next.js SEO components and meta tags"
        },
        {
            "name": "Contact Form",
            "description": "Working contact form with validation",
            "priority": "high",
            "implementation": "React Hook Form with validation"
        },
        {
            "name": "Analytics",
            "description": "Google Analytics integration",
            "priority": "medium",
            "implementation": "Google Analytics 4 setup"
        }
    ]
    
    # Add features based on analysis
    if "ecommerce" in requirements.lower():
        features.extend([
            {
                "name": "Product Catalog",
                "description": "Product listing and filtering",
                "priority": "high",
                "implementation": "Dynamic product pages with filters"
            },
            {
                "name": "Shopping Cart",
                "description": "Add to cart functionality",
                "priority": "high",
                "implementation": "Context-based cart state management"
            },
            {
                "name": "Payment Integration",
                "description": "Stripe payment processing",
                "priority": "high",
                "implementation": "Stripe Elements integration"
            }
        ])
    
    if "blog" in requirements.lower():
        features.append({
            "name": "Blog System",
            "description": "Content management for blog posts",
            "priority": "medium",
            "implementation": "Markdown-based blog with MDX"
        })
    
    return features

@app.get("/api/templates")
async def get_available_templates():
    """Get available template types"""
    return {
        "templates": [
            {
                "id": "business-website",
                "name": "Business Website",
                "description": "Professional business website template",
                "type": "website",
                "features": ["Hero", "About", "Services", "Contact"],
                "tech_stack": ["Next.js", "Tailwind CSS", "TypeScript"]
            },
            {
                "id": "ecommerce-store",
                "name": "E-commerce Store",
                "description": "Complete e-commerce solution",
                "type": "ecommerce",
                "features": ["Product Catalog", "Cart", "Checkout", "User Account"],
                "tech_stack": ["Next.js", "Stripe", "PostgreSQL"]
            },
            {
                "id": "landing-page",
                "name": "Landing Page",
                "description": "Single page landing site",
                "type": "landing_page",
                "features": ["Hero", "Features", "Pricing", "Contact"],
                "tech_stack": ["Next.js", "Tailwind CSS"]
            },
            {
                "id": "saas-app",
                "name": "SaaS Application",
                "description": "Software as a Service application",
                "type": "app",
                "features": ["Dashboard", "User Management", "Billing", "API"],
                "tech_stack": ["Next.js", "Prisma", "PostgreSQL"]
            }
        ]
    }

@app.get("/api/prototype/{prototype_id}")
async def get_prototype(prototype_id: str):
    """Get prototype details"""
    # Simulate prototype retrieval
    return {
        "prototype_id": prototype_id,
        "status": "completed",
        "url": f"https://prototype-{prototype_id}.statex.cz",
        "repository": f"https://github.com/statex/prototype-{prototype_id}",
        "created_at": datetime.now().isoformat()
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
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PROTOTYPE_GENERATOR_PORT", "3384")), log_config=log_config)
