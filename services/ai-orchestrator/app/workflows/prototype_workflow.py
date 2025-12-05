"""
Prototype Generation Workflow for AI Orchestrator

This workflow coordinates the generation of website prototypes using multiple AI agents.
It integrates with the prototype generator service and manages the complete generation pipeline.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
import json

from app.workflow_engine import WorkflowEngine, AgentTask, AgentResult, WorkflowState
from app.agent_coordination import AgentInterface

logger = logging.getLogger(__name__)

class PrototypeWorkflowAgent(AgentInterface):
    """AI Agent for prototype generation workflow coordination."""
    
    def __init__(self):
        super().__init__("prototype-workflow", "workflow-coordinator")
        prototype_host = os.getenv("PROTOTYPE_SERVICE_HOST", "prototype-service")
        prototype_port = os.getenv("PROTOTYPE_SERVICE_PORT", os.getenv("PROTOTYPE_GENERATOR_PORT", "3384"))
        self.prototype_service_url = os.getenv("PROTOTYPE_SERVICE_URL", f"http://{prototype_host}:{prototype_port}")
        
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute prototype generation workflow task."""
        try:
            logger.info(f"Executing prototype workflow task: {task.task_id}")
            
            # Extract task data
            submission_data = task.input_data
            user_id = submission_data.get("user_id", "system")
            submission_id = submission_data.get("submission_id", f"sub_{int(datetime.now().timestamp())}")
            requirements = submission_data.get("description", "")
            analysis = submission_data.get("analysis", {})
            
            # Determine prototype type based on requirements
            prototype_type = self._determine_prototype_type(requirements, analysis)
            
            # Call prototype generator service
            prototype_result = await self._call_prototype_generator(
                user_id, submission_id, requirements, analysis, prototype_type
            )
            
            if prototype_result.get("success"):
                # Extract prototype information
                prototype_id = prototype_result.get("prototype_id")
                job_id = prototype_result.get("results", {}).get("job_id")
                
                # Wait for generation to complete
                final_result = await self._wait_for_completion(job_id)
                
                if final_result.get("status") == "completed":
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status="completed",
                        result_data={
                            "prototype_id": prototype_id,
                            "job_id": job_id,
                            "status": "completed",
                            "generated_files": final_result.get("generated_files", {}),
                            "url": final_result.get("generated_files", {}).get("url"),
                            "content": final_result.get("generated_files", {}).get("content", {}),
                            "message": "Prototype generated successfully"
                        },
                        confidence_score=1.0,
                        metadata={
                            "prototype_type": prototype_type,
                            "generation_time": final_result.get("processing_time", 0),
                            "created_at": datetime.now().isoformat()
                        }
                    )
                else:
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        agent_name=self.agent_name,
                        status="failed",
                        error_message=f"Prototype generation failed: {final_result.get('error_message', 'Unknown error')}",
                        result_data=final_result
                    )
            else:
                return AgentResult(
                    task_id=task.task_id,
                    agent_type=self.agent_type,
                    agent_name=self.agent_name,
                    status="failed",
                    error_message=f"Failed to submit prototype generation: {prototype_result.get('error', 'Unknown error')}",
                    result_data=prototype_result
                )
                
        except Exception as e:
            logger.error(f"Error in prototype workflow task {task.task_id}: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                status="failed",
                error_message=str(e)
            )
    
    def _determine_prototype_type(self, requirements: str, analysis: Dict[str, Any]) -> str:
        """Determine the appropriate prototype type based on requirements."""
        requirements_lower = requirements.lower()
        
        # Check for e-commerce keywords
        if any(keyword in requirements_lower for keyword in ["shop", "store", "ecommerce", "e-commerce", "sell", "product", "cart", "payment"]):
            return "ecommerce"
        
        # Check for landing page keywords
        if any(keyword in requirements_lower for keyword in ["landing", "marketing", "promotion", "campaign", "lead"]):
            return "landing_page"
        
        # Check for SaaS keywords
        if any(keyword in requirements_lower for keyword in ["saas", "software", "app", "platform", "dashboard", "admin"]):
            return "saas"
        
        # Check for blog keywords
        if any(keyword in requirements_lower for keyword in ["blog", "news", "article", "content", "publishing"]):
            return "blog"
        
        # Default to general website
        return "website"
    
    async def _call_prototype_generator(self, user_id: str, submission_id: str, requirements: str, 
                                      analysis: Dict[str, Any], prototype_type: str) -> Dict[str, Any]:
        """Call the prototype generator service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.prototype_service_url}/api/generate-prototype",
                    json={
                        "requirements": requirements,
                        "analysis": analysis,
                        "prototype_type": prototype_type,
                        "customization_level": "basic"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        **response.json()
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Prototype generator service timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error calling prototype generator: {str(e)}"
            }
    
    async def _wait_for_completion(self, job_id: str, max_wait_time: int = 300) -> Dict[str, Any]:
        """Wait for prototype generation to complete."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.prototype_service_url}/api/queue/status/{job_id}",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        job_data = response.json()
                        status = job_data.get("status")
                        
                        if status == "completed":
                            return {
                                "status": "completed",
                                "generated_files": job_data.get("generated_files", {}),
                                "processing_time": (datetime.now() - start_time).total_seconds()
                            }
                        elif status == "failed":
                            return {
                                "status": "failed",
                                "error_message": job_data.get("error_message", "Generation failed")
                            }
                        else:
                            # Still processing, wait a bit
                            await asyncio.sleep(5)
                    else:
                        logger.warning(f"Failed to get job status: {response.status_code}")
                        await asyncio.sleep(5)
                        
            except Exception as e:
                logger.warning(f"Error checking job status: {e}")
                await asyncio.sleep(5)
        
        return {
            "status": "timeout",
            "error_message": f"Generation timed out after {max_wait_time} seconds"
        }

def create_prototype_workflow(submission_data: Dict[str, Any]) -> List[AgentTask]:
    """Create a prototype generation workflow."""
    
    # Create the main prototype generation task
    prototype_task = AgentTask(
        agent_type="workflow-coordinator",
        agent_name="prototype-workflow",
        input_data=submission_data,
        priority=1,
        timeout=300,  # 5 minutes
        max_retries=1
    )
    
    return [prototype_task]

# Register the workflow with the engine
def register_prototype_workflow(workflow_engine: WorkflowEngine):
    """Register the prototype workflow with the workflow engine."""
    
    # Create and register the prototype workflow agent
    prototype_agent = PrototypeWorkflowAgent()
    workflow_engine.register_agent(prototype_agent)
    
    # Register the workflow definition
    workflow_engine.register_workflow("prototype_generation", create_prototype_workflow)
    
    logger.info("Prototype workflow registered successfully")

# Example usage
async def run_prototype_workflow_example():
    """Example of how to run the prototype workflow."""
    
    # Create workflow engine
    engine = WorkflowEngine()
    
    # Register the prototype workflow
    register_prototype_workflow(engine)
    
    # Create a sample submission
    submission_data = {
        "user_id": "test_user",
        "submission_id": "test_submission_123",
        "description": "Create a modern e-commerce website for selling tech gadgets",
        "analysis": {
            "project_name": "TechGadgets Store",
            "industry": "e-commerce",
            "target_audience": "tech enthusiasts"
        }
    }
    
    # Create workflow
    workflow = await engine.create_workflow(
        submission_id="test_submission_123",
        workflow_type="prototype_generation",
        input_data=submission_data
    )
    
    # Execute workflow
    result = await engine.execute_workflow(workflow.workflow_id)
    
    print(f"Workflow completed with status: {result.status}")
    print(f"Results: {result.agent_results}")

if __name__ == "__main__":
    # Run example
    asyncio.run(run_prototype_workflow_example())
