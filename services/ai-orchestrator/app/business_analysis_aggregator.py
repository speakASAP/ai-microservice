"""
Business Analysis Aggregation Logic

This module combines results from all AI agents into comprehensive business analysis,
generates project scope, timeline, and budget estimates, creates technology stack
recommendations and implementation strategies, and adds market insights and risk
factor identification.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel
import re
import json

from .workflow_engine import WorkflowState, TaskStatus, AgentResult

logger = logging.getLogger(__name__)

class ProjectComplexity(BaseModel):
    """Project complexity assessment"""
    level: str  # "simple", "moderate", "complex", "enterprise"
    score: float  # 0.0 to 1.0
    factors: List[str]

class TechnologyRecommendation(BaseModel):
    """Technology stack recommendation"""
    category: str  # "frontend", "backend", "database", "infrastructure"
    technology: str
    reason: str
    confidence: float

class RiskFactor(BaseModel):
    """Risk factor identification"""
    category: str  # "technical", "timeline", "budget", "market"
    description: str
    impact: str  # "low", "medium", "high"
    mitigation: str

class MarketInsight(BaseModel):
    """Market analysis insight"""
    category: str  # "demand", "competition", "trends", "opportunities"
    insight: str
    relevance: float

class ImplementationPhase(BaseModel):
    """Implementation phase details"""
    name: str
    duration: str
    description: str
    deliverables: List[str]
    dependencies: List[str]

class BusinessAnalysisResult(BaseModel):
    """Comprehensive business analysis result"""
    project_scope: str
    complexity_assessment: ProjectComplexity
    technology_stack: List[TechnologyRecommendation]
    timeline_estimate: str
    budget_range: str
    risk_factors: List[RiskFactor]
    market_insights: List[MarketInsight]
    recommendations: List[str]
    implementation_strategy: List[ImplementationPhase]
    confidence_score: float
    analysis_metadata: Dict[str, Any]
class BusinessAnalysisAggregator:
    """Aggregates results from multiple AI agents into comprehensive business analysis"""

    def __init__(self):
        self.complexity_keywords = {
            "simple": ["basic", "simple", "straightforward", "minimal", "standard"],
            "moderate": ["moderate", "medium", "intermediate", "custom", "integration"],
            "complex": ["complex", "advanced", "sophisticated", "enterprise", "scalable"],
            "enterprise": ["enterprise", "large-scale", "distributed", "microservices", "high-availability"]
        }

        self.technology_patterns = {
            "web_app": ["website", "web app", "portal", "dashboard", "interface"],
            "mobile_app": ["mobile", "app", "ios", "android", "react native"],
            "api": ["api", "service", "backend", "microservice", "rest"],
            "database": ["database", "data", "storage", "sql", "nosql"],
            "ai_ml": ["ai", "machine learning", "ml", "artificial intelligence", "nlp"],
            "ecommerce": ["shop", "store", "ecommerce", "payment", "cart"],
            "crm": ["crm", "customer", "management", "sales", "leads"],
            "analytics": ["analytics", "reporting", "dashboard", "metrics", "kpi"]
        }

        self.risk_patterns = {
            "technical": ["integration", "performance", "scalability", "security", "compatibility"],
            "timeline": ["deadline", "timeline", "schedule", "urgent", "fast"],
            "budget": ["budget", "cost", "expensive", "cheap", "price"],
            "market": ["competition", "market", "demand", "trends", "adoption"]
        }

    async def aggregate_business_analysis(self, workflow_state: WorkflowState) -> BusinessAnalysisResult:
        """
        Combine results from all AI agents into comprehensive analysis

        Args:
            workflow_state: Complete workflow state with agent results

        Returns:
            BusinessAnalysisResult: Comprehensive business analysis
        """
        try:
            logger.info(f"Starting business analysis aggregation for workflow {workflow_state.workflow_id}")

            # Extract and combine all agent data
            combined_data = await self._extract_agent_data(workflow_state)

            # Analyze project complexity
            complexity = await self._assess_project_complexity(combined_data)

            # Generate project scope
            project_scope = await self._generate_project_scope(combined_data, complexity)

            # Recommend technology stack
            tech_stack = await self._recommend_technology_stack(combined_data, complexity)

            # Estimate timeline and budget
            timeline, budget = await self._estimate_timeline_and_budget(complexity, tech_stack)

            # Identify risk factors
            risks = await self._identify_risk_factors(combined_data, complexity)

            # Generate market insights
            market_insights = await self._generate_market_insights(combined_data, tech_stack)

            # Create recommendations
            recommendations = await self._generate_recommendations(combined_data, complexity, risks)

            # Design implementation strategy
            implementation_strategy = await self._design_implementation_strategy(complexity, tech_stack)

            # Calculate overall confidence
            confidence = await self._calculate_confidence_score(workflow_state, combined_data)

            # Create metadata
            metadata = {
                "analysis_timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_state.workflow_id,
                "agents_used": list(workflow_state.agent_results.keys()),
                "successful_agents": len([r for r in workflow_state.agent_results.values() 
                                        if r.status == TaskStatus.COMPLETED]),
                "total_processing_time": sum([r.processing_time for r in workflow_state.agent_results.values()]),
                "data_sources": list(combined_data.keys())
            }

            result = BusinessAnalysisResult(
                project_scope=project_scope,
                complexity_assessment=complexity,
                technology_stack=tech_stack,
                timeline_estimate=timeline,
                budget_range=budget,
                risk_factors=risks,
                market_insights=market_insights,
                recommendations=recommendations,
                implementation_strategy=implementation_strategy,
                confidence_score=confidence,
                analysis_metadata=metadata
            )

            logger.info(f"Business analysis aggregation completed with confidence {confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error in business analysis aggregation: {e}")
            # Return minimal analysis on error
            return await self._create_fallback_analysis(workflow_state)
    async def _extract_agent_data(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Extract and normalize data from all agent results"""
        combined_data = {
            "text_content": "",
            "extracted_text": "",
            "transcribed_text": "",
            "nlp_insights": {},
            "document_analysis": {},
            "voice_analysis": {},
            "prototype_info": {},
            "requirements": "",
            "user_input": workflow_state.input_data
        }

        # Extract original text content
        if workflow_state.input_data.get("text_content"):
            combined_data["text_content"] = workflow_state.input_data["text_content"]

        if workflow_state.input_data.get("requirements"):
            combined_data["requirements"] = workflow_state.input_data["requirements"]

        # Process each agent result
        for task_id, result in workflow_state.agent_results.items():
            if result.status == TaskStatus.COMPLETED and result.result_data:

                if result.agent_type == "nlp":
                    combined_data["nlp_insights"] = result.result_data.get("results", {})

                elif result.agent_type == "asr":
                    asr_results = result.result_data.get("results", {})
                    if isinstance(asr_results, dict):
                        combined_data["transcribed_text"] = asr_results.get("transcript", "")
                        combined_data["voice_analysis"] = asr_results
                    elif isinstance(asr_results, str):
                        combined_data["transcribed_text"] = asr_results

                elif result.agent_type == "document":
                    doc_results = result.result_data.get("results", {})
                    if isinstance(doc_results, dict):
                        combined_data["extracted_text"] = doc_results.get("extracted_text", "")
                        combined_data["document_analysis"] = doc_results
                    elif isinstance(doc_results, str):
                        combined_data["extracted_text"] = doc_results

                elif result.agent_type == "prototype":
                    combined_data["prototype_info"] = result.result_data.get("results", {})

        # Combine all text content
        all_text = " ".join([
            combined_data["text_content"],
            combined_data["extracted_text"],
            combined_data["transcribed_text"],
            combined_data["requirements"]
        ]).strip()

        combined_data["all_text"] = all_text

        return combined_data

    async def _assess_project_complexity(self, combined_data: Dict[str, Any]) -> ProjectComplexity:
        """Assess project complexity based on combined data"""
        all_text = combined_data.get("all_text", "").lower()
        complexity_scores = {"simple": 0, "moderate": 0, "complex": 0, "enterprise": 0}
        factors = []

        # Analyze text for complexity indicators
        for level, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                count = all_text.count(keyword)
                complexity_scores[level] += count
                if count > 0:
                    factors.append(f"Contains '{keyword}' ({count} times)")

        # Additional complexity factors
        if len(all_text.split()) > 500:
            complexity_scores["complex"] += 2
            factors.append("Detailed requirements (500+ words)")

        if combined_data.get("document_analysis"):
            complexity_scores["moderate"] += 1
            factors.append("Document analysis required")

        if combined_data.get("voice_analysis"):
            complexity_scores["moderate"] += 1
            factors.append("Voice input processing")

        # Check for enterprise indicators
        enterprise_indicators = ["microservices", "scalability", "high availability", "enterprise", "large scale"]
        for indicator in enterprise_indicators:
            if indicator in all_text:
                complexity_scores["enterprise"] += 2
                factors.append(f"Enterprise requirement: {indicator}")

        # Determine final complexity level
        max_score = max(complexity_scores.values())
        if max_score == 0:
            level = "simple"
            score = 0.3
        else:
            level = max(complexity_scores, key=complexity_scores.get)
            score = min(max_score / 10.0, 1.0)  # Normalize to 0-1

        return ProjectComplexity(
            level=level,
            score=score,
            factors=factors[:10]  # Limit to top 10 factors
        ) 
    async def _generate_project_scope(self, combined_data: Dict[str, Any], complexity: ProjectComplexity) -> str:
        """Generate comprehensive project scope description"""
        scope_parts = []

        # Start with base description
        if combined_data.get("nlp_insights", {}).get("text_summary"):
            scope_parts.append(combined_data["nlp_insights"]["text_summary"])
        elif combined_data.get("all_text"):
            # Create summary from available text
            text = combined_data["all_text"][:1000]  # First 1000 chars
            scope_parts.append(f"Project based on requirements: {text}")
        else:
            scope_parts.append("Custom software development project")

        # Add complexity-based scope details
        if complexity.level == "simple":
            scope_parts.append("This is a straightforward project with standard functionality and minimal customization requirements.")
        elif complexity.level == "moderate":
            scope_parts.append("This project involves moderate complexity with custom features and integration requirements.")
        elif complexity.level == "complex":
            scope_parts.append("This is a complex project requiring advanced features, custom architecture, and sophisticated integrations.")
        else:  # enterprise
            scope_parts.append("This is an enterprise-level project requiring scalable architecture, high availability, and comprehensive feature set.")

        # Add specific features based on detected patterns
        features = []
        all_text = combined_data.get("all_text", "").lower()

        for pattern_type, keywords in self.technology_patterns.items():
            if any(keyword in all_text for keyword in keywords):
                if pattern_type == "web_app":
                    features.append("Web application development")
                elif pattern_type == "mobile_app":
                    features.append("Mobile application development")
                elif pattern_type == "api":
                    features.append("API development and integration")
                elif pattern_type == "database":
                    features.append("Database design and implementation")
                elif pattern_type == "ai_ml":
                    features.append("AI/ML integration")
                elif pattern_type == "ecommerce":
                    features.append("E-commerce functionality")
                elif pattern_type == "crm":
                    features.append("CRM system development")
                elif pattern_type == "analytics":
                    features.append("Analytics and reporting")

        if features:
            scope_parts.append(f"Key features include: {', '.join(features)}")

        # Add document/voice specific scope if available
        if combined_data.get("document_analysis"):
            scope_parts.append("Project includes document processing and analysis capabilities.")

        if combined_data.get("voice_analysis"):
            scope_parts.append("Project includes voice/audio processing functionality.")

        return " ".join(scope_parts)

    async def _recommend_technology_stack(self, combined_data: Dict[str, Any], complexity: ProjectComplexity) -> List[TechnologyRecommendation]:
        """Recommend technology stack based on requirements and complexity"""
        recommendations = []
        all_text = combined_data.get("all_text", "").lower()

        # Frontend recommendations
        if any(keyword in all_text for keyword in ["web", "frontend", "ui", "interface", "dashboard"]):
            if complexity.level in ["complex", "enterprise"]:
                recommendations.append(TechnologyRecommendation(
                    category="frontend",
                    technology="React with TypeScript",
                    reason="Scalable, type-safe frontend for complex applications",
                    confidence=0.9
                ))
            else:
                recommendations.append(TechnologyRecommendation(
                    category="frontend",
                    technology="React",
                    reason="Modern, flexible frontend framework",
                    confidence=0.8
                ))

        # Backend recommendations
        if any(keyword in all_text for keyword in ["api", "backend", "server", "service"]):
            if "python" in all_text or "ai" in all_text or "ml" in all_text:
                recommendations.append(TechnologyRecommendation(
                    category="backend",
                    technology="FastAPI (Python)",
                    reason="Excellent for AI/ML integration and rapid development",
                    confidence=0.9
                ))
            elif complexity.level in ["complex", "enterprise"]:
                recommendations.append(TechnologyRecommendation(
                    category="backend",
                    technology="Node.js with Express",
                    reason="Scalable JavaScript backend for complex applications",
                    confidence=0.8
                ))
            else:
                recommendations.append(TechnologyRecommendation(
                    category="backend",
                    technology="Node.js",
                    reason="Fast development with JavaScript ecosystem",
                    confidence=0.7
                ))

        # Database recommendations
        if any(keyword in all_text for keyword in ["data", "database", "storage", "records"]):
            if complexity.level in ["complex", "enterprise"] or "analytics" in all_text:
                recommendations.append(TechnologyRecommendation(
                    category="database",
                    technology="PostgreSQL",
                    reason="Robust relational database with advanced features",
                    confidence=0.9
                ))
            elif "nosql" in all_text or "document" in all_text:
                recommendations.append(TechnologyRecommendation(
                    category="database",
                    technology="MongoDB",
                    reason="Flexible document database for unstructured data",
                    confidence=0.8
                ))
            else:
                recommendations.append(TechnologyRecommendation(
                    category="database",
                    technology="PostgreSQL",
                    reason="Reliable and feature-rich database",
                    confidence=0.7
                ))

        # Infrastructure recommendations
        if complexity.level in ["complex", "enterprise"]:
            recommendations.append(TechnologyRecommendation(
                category="infrastructure",
                technology="Docker + Docker Compose",
                reason="Containerized deployment for scalability",
                confidence=0.8
            ))
        else:
            recommendations.append(TechnologyRecommendation(
                category="infrastructure",
                technology="Docker",
                reason="Consistent deployment environment",
                confidence=0.7
            ))

        # AI/ML specific recommendations
        if any(keyword in all_text for keyword in ["ai", "ml", "machine learning", "nlp", "analysis"]):
            recommendations.append(TechnologyRecommendation(
                category="ai_ml",
                technology="Python with scikit-learn/TensorFlow",
                reason="Comprehensive AI/ML ecosystem",
                confidence=0.9
            ))

        # Mobile recommendations
        if any(keyword in all_text for keyword in ["mobile", "app", "ios", "android"]):
            recommendations.append(TechnologyRecommendation(
                category="mobile",
                technology="React Native",
                reason="Cross-platform mobile development",
                confidence=0.8
            ))

        return recommendations 
    async def _estimate_timeline_and_budget(self, complexity: ProjectComplexity, tech_stack: List[TechnologyRecommendation]) -> Tuple[str, str]:
        """Estimate project timeline and budget based on complexity and technology stack"""

        # Base estimates by complexity
        timeline_estimates = {
            "simple": {"min": 1, "max": 2, "unit": "weeks"},
            "moderate": {"min": 2, "max": 4, "unit": "weeks"},
            "complex": {"min": 4, "max": 8, "unit": "weeks"},
            "enterprise": {"min": 8, "max": 16, "unit": "weeks"}
        }

        budget_estimates = {
            "simple": {"min": 3000, "max": 8000},
            "moderate": {"min": 8000, "max": 15000},
            "complex": {"min": 15000, "max": 30000},
            "enterprise": {"min": 30000, "max": 60000}
        }

        # Adjust for technology stack complexity
        tech_multiplier = 1.0
        for tech in tech_stack:
            if tech.category == "ai_ml":
                tech_multiplier += 0.3
            elif tech.category == "mobile":
                tech_multiplier += 0.2
            elif "docker compose" in tech.technology.lower():
                tech_multiplier += 0.4

        # Calculate final estimates
        base_timeline = timeline_estimates[complexity.level]
        timeline_min = int(base_timeline["min"] * tech_multiplier)
        timeline_max = int(base_timeline["max"] * tech_multiplier)

        if timeline_max > 12:  # Convert to months if > 12 weeks
            timeline_min_months = max(1, timeline_min // 4)
            timeline_max_months = timeline_max // 4
            timeline = f"{timeline_min_months}-{timeline_max_months} months"
        else:
            timeline = f"{timeline_min}-{timeline_max} {base_timeline['unit']}"

        base_budget = budget_estimates[complexity.level]
        budget_min = int(base_budget["min"] * tech_multiplier)
        budget_max = int(base_budget["max"] * tech_multiplier)
        budget = f"${budget_min:,} - ${budget_max:,}"

        return timeline, budget

    async def _identify_risk_factors(self, combined_data: Dict[str, Any], complexity: ProjectComplexity) -> List[RiskFactor]:
        """Identify potential risk factors based on project analysis"""
        risks = []
        all_text = combined_data.get("all_text", "").lower()

        # Technical risks
        if complexity.level in ["complex", "enterprise"]:
            risks.append(RiskFactor(
                category="technical",
                description="High technical complexity may require additional architecture planning",
                impact="medium",
                mitigation="Conduct thorough technical discovery and create detailed architecture documentation"
            ))

        if any(keyword in all_text for keyword in ["integration", "api", "third-party"]):
            risks.append(RiskFactor(
                category="technical",
                description="Third-party integrations may have dependencies and limitations",
                impact="medium",
                mitigation="Validate integration capabilities early and have fallback options"
            ))

        # Timeline risks
        if any(keyword in all_text for keyword in ["urgent", "asap", "quickly", "fast"]):
            risks.append(RiskFactor(
                category="timeline",
                description="Aggressive timeline may impact quality or require additional resources",
                impact="high",
                mitigation="Prioritize core features and plan phased delivery approach"
            ))

        # Budget risks
        if complexity.level == "enterprise":
            risks.append(RiskFactor(
                category="budget",
                description="Enterprise-level features may require additional investment",
                impact="medium",
                mitigation="Define clear scope boundaries and plan for iterative development"
            ))

        # Market risks
        if any(keyword in all_text for keyword in ["competitive", "market", "first-to-market"]):
            risks.append(RiskFactor(
                category="market",
                description="Market competition may require faster delivery or unique features",
                impact="medium",
                mitigation="Focus on unique value proposition and rapid MVP development"
            ))

        # Add default risks if none identified
        if not risks:
            risks.append(RiskFactor(
                category="technical",
                description="Standard development risks including scope creep and technical challenges",
                impact="low",
                mitigation="Regular communication and agile development methodology"
            ))

        return risks

    async def _generate_market_insights(self, combined_data: Dict[str, Any], tech_stack: List[TechnologyRecommendation]) -> List[MarketInsight]:
        """Generate market insights based on project type and technology stack"""
        insights = []
        all_text = combined_data.get("all_text", "").lower()

        # Technology trend insights
        modern_techs = ["react", "node.js", "python", "docker", "docker compose"]
        used_modern_techs = [tech.technology for tech in tech_stack if any(mt in tech.technology.lower() for mt in modern_techs)]

        if used_modern_techs:
            insights.append(MarketInsight(
                category="trends",
                insight=f"Using modern technologies ({', '.join(used_modern_techs[:3])}) aligns with current market trends and developer preferences",
                relevance=0.8
            ))

        # Industry-specific insights
        if any(keyword in all_text for keyword in ["ecommerce", "shop", "store", "payment"]):
            insights.append(MarketInsight(
                category="demand",
                insight="E-commerce solutions continue to see strong market demand, especially with mobile-first approaches",
                relevance=0.9
            ))

        if any(keyword in all_text for keyword in ["ai", "ml", "machine learning", "automation"]):
            insights.append(MarketInsight(
                category="opportunities",
                insight="AI/ML integration provides competitive advantage and is increasingly expected by users",
                relevance=0.9
            ))

        if any(keyword in all_text for keyword in ["mobile", "app", "ios", "android"]):
            insights.append(MarketInsight(
                category="trends",
                insight="Mobile-first approach is essential as mobile usage continues to dominate web traffic",
                relevance=0.8
            ))

        # General market insights
        insights.append(MarketInsight(
            category="demand",
            insight="Custom software development market shows consistent growth with increasing digitization needs",
            relevance=0.7
        ))

        return insights  
    async def _generate_recommendations(self, combined_data: Dict[str, Any], complexity: ProjectComplexity, risks: List[RiskFactor]) -> List[str]:
        """Generate strategic recommendations based on analysis"""
        recommendations = []

        # Complexity-based recommendations
        if complexity.level == "simple":
            recommendations.append("Start with MVP approach to validate core functionality quickly")
            recommendations.append("Use proven technologies to minimize development risk")
        elif complexity.level == "moderate":
            recommendations.append("Plan development in phases to manage complexity")
            recommendations.append("Invest in good architecture foundation for future scalability")
        elif complexity.level in ["complex", "enterprise"]:
            recommendations.append("Conduct thorough discovery phase before development")
            recommendations.append("Consider microservices architecture for better maintainability")
            recommendations.append("Plan for comprehensive testing and quality assurance")

        # Risk-based recommendations
        high_risk_categories = [risk.category for risk in risks if risk.impact == "high"]
        if "timeline" in high_risk_categories:
            recommendations.append("Prioritize core features and plan phased delivery")
        if "technical" in high_risk_categories:
            recommendations.append("Allocate extra time for technical spikes and proof of concepts")

        # Technology-specific recommendations
        all_text = combined_data.get("all_text", "").lower()
        if "ai" in all_text or "ml" in all_text:
            recommendations.append("Start with pre-trained models before building custom AI solutions")

        if "mobile" in all_text:
            recommendations.append("Design API-first to support both web and mobile clients")

        # General best practices
        recommendations.append("Implement continuous integration and deployment from the start")
        recommendations.append("Plan for monitoring and analytics to track user engagement")
        recommendations.append("Consider security requirements throughout the development process")

        return recommendations[:8]  # Limit to top 8 recommendations

    async def _design_implementation_strategy(self, complexity: ProjectComplexity, tech_stack: List[TechnologyRecommendation]) -> List[ImplementationPhase]:
        """Design implementation strategy with phases"""
        phases = []

        # Phase 1: Discovery and Planning
        discovery_deliverables = [
            "Detailed requirements analysis",
            "Technical architecture design",
            "Project timeline and milestones",
            "Risk assessment and mitigation plan"
        ]

        if complexity.level in ["complex", "enterprise"]:
            discovery_deliverables.extend([
                "System integration analysis",
                "Performance requirements specification",
                "Security assessment"
            ])

        phases.append(ImplementationPhase(
            name="Discovery & Planning",
            duration="3-7 days" if complexity.level in ["simple", "moderate"] else "1-2 weeks",
            description="Comprehensive analysis and planning phase",
            deliverables=discovery_deliverables,
            dependencies=[]
        ))

        # Phase 2: Core Development
        core_deliverables = [
            "Core functionality implementation",
            "Database design and setup",
            "Basic user interface",
            "Unit testing"
        ]

        if any(tech.category == "api" for tech in tech_stack):
            core_deliverables.append("API development and documentation")

        phases.append(ImplementationPhase(
            name="Core Development",
            duration=self._get_development_duration(complexity),
            description="Implementation of core features and functionality",
            deliverables=core_deliverables,
            dependencies=["Discovery & Planning"]
        ))

        # Phase 3: Integration and Enhancement (for complex projects)
        if complexity.level in ["complex", "enterprise"]:
            integration_deliverables = [
                "Third-party integrations",
                "Advanced features implementation",
                "Performance optimization",
                "Integration testing"
            ]

            phases.append(ImplementationPhase(
                name="Integration & Enhancement",
                duration="1-2 weeks",
                description="Advanced features and system integrations",
                deliverables=integration_deliverables,
                dependencies=["Core Development"]
            ))

        # Final Phase: Testing and Deployment
        deployment_deliverables = [
            "Comprehensive testing",
            "Production deployment setup",
            "Documentation and user guides",
            "Training and handover"
        ]

        if complexity.level in ["complex", "enterprise"]:
            deployment_deliverables.extend([
                "Load testing and optimization",
                "Security testing",
                "Monitoring setup"
            ])

        phases.append(ImplementationPhase(
            name="Testing & Deployment",
            duration="3-5 days" if complexity.level in ["simple", "moderate"] else "1 week",
            description="Final testing, deployment, and project handover",
            deliverables=deployment_deliverables,
            dependencies=["Integration & Enhancement"] if complexity.level in ["complex", "enterprise"] else ["Core Development"]
        ))

        return phases

    def _get_development_duration(self, complexity: ProjectComplexity) -> str:
        """Get development phase duration based on complexity"""
        durations = {
            "simple": "1-2 weeks",
            "moderate": "2-3 weeks",
            "complex": "3-5 weeks",
            "enterprise": "4-8 weeks"
        }
        return durations.get(complexity.level, "2-4 weeks")   
    async def _calculate_confidence_score(self, workflow_state: WorkflowState, combined_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis"""
        confidence_factors = []

        # Agent success rate
        successful_agents = len([r for r in workflow_state.agent_results.values() 
                               if r.status == TaskStatus.COMPLETED])
        total_agents = len(workflow_state.agent_results)
        agent_success_rate = successful_agents / total_agents if total_agents > 0 else 0
        confidence_factors.append(agent_success_rate * 0.4)  # 40% weight

        # Data quality
        text_length = len(combined_data.get("all_text", ""))
        text_quality = min(text_length / 1000, 1.0)  # Normalize to 1000 chars
        confidence_factors.append(text_quality * 0.3)  # 30% weight

        # Agent confidence scores
        agent_confidences = [r.confidence_score for r in workflow_state.agent_results.values() 
                           if r.status == TaskStatus.COMPLETED and r.confidence_score > 0]
        avg_agent_confidence = sum(agent_confidences) / len(agent_confidences) if agent_confidences else 0.5
        confidence_factors.append(avg_agent_confidence * 0.3)  # 30% weight

        return sum(confidence_factors)

    async def _create_fallback_analysis(self, workflow_state: WorkflowState) -> BusinessAnalysisResult:
        """Create minimal analysis when main aggregation fails"""
        return BusinessAnalysisResult(
            project_scope="Custom software development project based on provided requirements",
            complexity_assessment=ProjectComplexity(
                level="moderate",
                score=0.5,
                factors=["Fallback analysis due to processing error"]
            ),
            technology_stack=[
                TechnologyRecommendation(
                    category="frontend",
                    technology="React",
                    reason="Modern web framework",
                    confidence=0.7
                ),
                TechnologyRecommendation(
                    category="backend",
                    technology="Node.js",
                    reason="JavaScript backend",
                    confidence=0.7
                ),
                TechnologyRecommendation(
                    category="database",
                    technology="PostgreSQL",
                    reason="Reliable database",
                    confidence=0.7
                )
            ],
            timeline_estimate="2-4 weeks",
            budget_range="$8,000 - $15,000",
            risk_factors=[
                RiskFactor(
                    category="technical",
                    description="Standard development risks",
                    impact="medium",
                    mitigation="Regular communication and testing"
                )
            ],
            market_insights=[
                MarketInsight(
                    category="demand",
                    insight="Custom software development continues to show strong market demand",
                    relevance=0.7
                )
            ],
            recommendations=[
                "Start with MVP approach",
                "Use proven technologies",
                "Plan for iterative development"
            ],
            implementation_strategy=[
                ImplementationPhase(
                    name="Discovery & Planning",
                    duration="3-5 days",
                    description="Requirements analysis and planning",
                    deliverables=["Requirements document", "Technical plan"],
                    dependencies=[]
                ),
                ImplementationPhase(
                    name="Development",
                    duration="2-3 weeks",
                    description="Core development work",
                    deliverables=["Working application", "Testing"],
                    dependencies=["Discovery & Planning"]
                )
            ],
            confidence_score=0.3,
            analysis_metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_state.workflow_id,
                "fallback_reason": "Main analysis failed"
            }
        )

# Global aggregator instance
business_analysis_aggregator = BusinessAnalysisAggregator()