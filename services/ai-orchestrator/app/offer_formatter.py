"""
Offer Formatting and Presentation System

This module designs comprehensive offer templates for different project types,
implements pricing tier generation based on project complexity, creates
implementation phase breakdown and deliverable lists, and adds next steps
and call-to-action recommendations.

Requirements: 6.7, 6.8, 6.9
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum

from .business_analysis_aggregator import BusinessAnalysisResult, ProjectComplexity, TechnologyRecommendation

logger = logging.getLogger(__name__)

class OfferTemplate(str, Enum):
    """Available offer templates"""
    WEB_APPLICATION = "web_application"
    MOBILE_APPLICATION = "mobile_application"
    API_SERVICE = "api_service"
    ECOMMERCE_PLATFORM = "ecommerce_platform"
    CRM_SYSTEM = "crm_system"
    ANALYTICS_DASHBOARD = "analytics_dashboard"
    AI_ML_SOLUTION = "ai_ml_solution"
    ENTERPRISE_SYSTEM = "enterprise_system"
    GENERAL_SOFTWARE = "general_software"

class PricingTier(BaseModel):
    """Pricing tier details"""
    name: str
    price_range: str
    description: str
    features: List[str]
    timeline: str
    support_level: str
    recommended: bool = False

class DeliverableCategory(BaseModel):
    """Deliverable category with items"""
    category: str
    items: List[str]
    description: str

class CallToAction(BaseModel):
    """Call-to-action item"""
    action: str
    description: str
    urgency: str  # "low", "medium", "high"
    timeline: str

class FormattedOffer(BaseModel):
    """Complete formatted offer"""
    project_id: str
    template_type: OfferTemplate
    title: str
    executive_summary: str
    pricing_tiers: List[PricingTier]
    implementation_phases: List[Dict[str, Any]]
    deliverables: List[DeliverableCategory]
    next_steps: List[CallToAction]
    terms_and_conditions: List[str]
    contact_information: Dict[str, str]
    validity_period: str
    created_at: str
    urls: Dict[str, str]

class OfferFormatter:
    """Formats business analysis into comprehensive offers"""
    
    def __init__(self):
        self.template_patterns = {
            OfferTemplate.WEB_APPLICATION: ["web", "website", "portal", "dashboard", "frontend"],
            OfferTemplate.MOBILE_APPLICATION: ["mobile", "app", "ios", "android", "react native"],
            OfferTemplate.API_SERVICE: ["api", "service", "backend", "microservice", "rest"],
            OfferTemplate.ECOMMERCE_PLATFORM: ["shop", "store", "ecommerce", "payment", "cart"],
            OfferTemplate.CRM_SYSTEM: ["crm", "customer", "management", "sales", "leads"],
            OfferTemplate.ANALYTICS_DASHBOARD: ["analytics", "reporting", "dashboard", "metrics", "kpi"],
            OfferTemplate.AI_ML_SOLUTION: ["ai", "ml", "machine learning", "nlp", "artificial intelligence"],
            OfferTemplate.ENTERPRISE_SYSTEM: ["enterprise", "large-scale", "distributed", "microservices"]
        }
        
        self.base_deliverables = {
            "technical": {
                "category": "Technical Deliverables",
                "description": "Core technical components and documentation",
                "items": [
                    "Complete source code with version control",
                    "Technical architecture documentation",
                    "API documentation and specifications",
                    "Database schema and migration scripts",
                    "Deployment and configuration guides"
                ]
            },
            "design": {
                "category": "Design & User Experience",
                "description": "User interface and experience components",
                "items": [
                    "User interface mockups and prototypes",
                    "User experience flow diagrams",
                    "Brand guidelines and style guide",
                    "Responsive design for all devices",
                    "Accessibility compliance documentation"
                ]
            },
            "testing": {
                "category": "Quality Assurance",
                "description": "Testing and quality assurance deliverables",
                "items": [
                    "Comprehensive test suite (unit, integration, e2e)",
                    "Test coverage reports and documentation",
                    "Performance testing results",
                    "Security testing and vulnerability assessment",
                    "User acceptance testing procedures"
                ]
            },
            "deployment": {
                "category": "Deployment & Operations",
                "description": "Production deployment and operational support",
                "items": [
                    "Production deployment setup",
                    "CI/CD pipeline configuration",
                    "Monitoring and logging setup",
                    "Backup and disaster recovery procedures",
                    "Performance optimization and tuning"
                ]
            },
            "support": {
                "category": "Support & Maintenance",
                "description": "Ongoing support and maintenance services",
                "items": [
                    "3 months post-launch support",
                    "Bug fixes and minor enhancements",
                    "Performance monitoring and optimization",
                    "Security updates and patches",
                    "User training and documentation"
                ]
            }
        }
    
    def detect_offer_template(self, analysis: BusinessAnalysisResult) -> OfferTemplate:
        """
        Detect the most appropriate offer template based on business analysis
        
        Args:
            analysis: Business analysis result
            
        Returns:
            OfferTemplate: Most appropriate template
        """
        try:
            # Combine all text for analysis
            all_text = f"{analysis.project_scope} {' '.join(analysis.recommendations)}".lower()
            
            # Add technology stack context
            tech_text = " ".join([tech.technology.lower() for tech in analysis.technology_stack])
            all_text += f" {tech_text}"
            
            # Score each template
            template_scores = {}
            for template, keywords in self.template_patterns.items():
                score = sum(1 for keyword in keywords if keyword in all_text)
                template_scores[template] = score
            
            # Find the template with highest score
            best_template = max(template_scores, key=template_scores.get)
            
            # If no clear winner, use complexity-based fallback
            if template_scores[best_template] == 0:
                if analysis.complexity_assessment.level == "enterprise":
                    best_template = OfferTemplate.ENTERPRISE_SYSTEM
                else:
                    best_template = OfferTemplate.GENERAL_SOFTWARE
            
            logger.info(f"Detected offer template: {best_template.value} (score: {template_scores[best_template]})")
            return best_template
            
        except Exception as e:
            logger.error(f"Error detecting offer template: {e}")
            return OfferTemplate.GENERAL_SOFTWARE
    
    def generate_pricing_tiers(self, analysis: BusinessAnalysisResult, template: OfferTemplate) -> List[PricingTier]:
        """
        Generate pricing tiers based on project complexity and template
        
        Args:
            analysis: Business analysis result
            template: Offer template type
            
        Returns:
            List[PricingTier]: Generated pricing tiers
        """
        try:
            complexity = analysis.complexity_assessment.level
            budget_range = analysis.budget_range
            timeline = analysis.timeline_estimate
            
            # Parse budget range
            try:
                budget_parts = budget_range.replace("$", "").replace(",", "").split(" - ")
                min_budget = int(budget_parts[0])
                max_budget = int(budget_parts[1])
            except (ValueError, IndexError):
                min_budget, max_budget = 8000, 15000
            
            tiers = []
            
            if complexity == "simple":
                tiers = [
                    PricingTier(
                        name="Essential",
                        price_range=f"${min_budget:,} - ${int(min_budget * 1.3):,}",
                        description="Perfect for startups and small businesses",
                        features=self._get_tier_features(template, "essential"),
                        timeline=timeline,
                        support_level="Email support for 3 months",
                        recommended=True
                    ),
                    PricingTier(
                        name="Professional",
                        price_range=f"${int(min_budget * 1.3):,} - ${max_budget:,}",
                        description="Enhanced features for growing businesses",
                        features=self._get_tier_features(template, "professional"),
                        timeline=self._adjust_timeline(timeline, 1.2),
                        support_level="Priority email and phone support for 6 months"
                    )
                ]
            elif complexity == "moderate":
                tiers = [
                    PricingTier(
                        name="Standard",
                        price_range=f"${min_budget:,} - ${int((min_budget + max_budget) / 2):,}",
                        description="Comprehensive solution for established businesses",
                        features=self._get_tier_features(template, "standard"),
                        timeline=timeline,
                        support_level="Email and phone support for 3 months"
                    ),
                    PricingTier(
                        name="Professional",
                        price_range=f"${int((min_budget + max_budget) / 2):,} - ${max_budget:,}",
                        description="Advanced features with premium support",
                        features=self._get_tier_features(template, "professional"),
                        timeline=self._adjust_timeline(timeline, 1.1),
                        support_level="Priority support with dedicated account manager for 6 months",
                        recommended=True
                    ),
                    PricingTier(
                        name="Enterprise",
                        price_range=f"${max_budget:,} - ${int(max_budget * 1.4):,}",
                        description="Full-featured solution with enterprise support",
                        features=self._get_tier_features(template, "enterprise"),
                        timeline=self._adjust_timeline(timeline, 1.3),
                        support_level="24/7 support with SLA guarantee for 12 months"
                    )
                ]
            else:  # complex or enterprise
                tiers = [
                    PricingTier(
                        name="Professional",
                        price_range=f"${min_budget:,} - ${int((min_budget + max_budget) / 2):,}",
                        description="Scalable solution for growing enterprises",
                        features=self._get_tier_features(template, "professional"),
                        timeline=timeline,
                        support_level="Priority support for 6 months"
                    ),
                    PricingTier(
                        name="Enterprise",
                        price_range=f"${int((min_budget + max_budget) / 2):,} - ${max_budget:,}",
                        description="Complete enterprise solution with full customization",
                        features=self._get_tier_features(template, "enterprise"),
                        timeline=self._adjust_timeline(timeline, 1.2),
                        support_level="Dedicated support team with SLA for 12 months",
                        recommended=True
                    ),
                    PricingTier(
                        name="Enterprise Plus",
                        price_range=f"${max_budget:,} - ${int(max_budget * 1.5):,}",
                        description="Premium enterprise solution with white-glove service",
                        features=self._get_tier_features(template, "enterprise_plus"),
                        timeline=self._adjust_timeline(timeline, 1.4),
                        support_level="24/7 dedicated support with on-site assistance for 18 months"
                    )
                ]
            
            return tiers
            
        except Exception as e:
            logger.error(f"Error generating pricing tiers: {e}")
            return self._get_fallback_pricing_tiers()
    
    def _get_tier_features(self, template: OfferTemplate, tier_level: str) -> List[str]:
        """Get features for a specific tier and template"""
        base_features = {
            "essential": [
                "Core functionality implementation",
                "Basic user interface",
                "Standard database setup",
                "Basic testing and deployment",
                "3 months email support"
            ],
            "standard": [
                "Full functionality implementation",
                "Custom user interface design",
                "Optimized database architecture",
                "Comprehensive testing suite",
                "Standard deployment setup",
                "3 months support"
            ],
            "professional": [
                "Advanced functionality with customizations",
                "Premium UI/UX design",
                "Scalable database architecture",
                "Automated testing and CI/CD",
                "Production-ready deployment",
                "Performance optimization",
                "6 months priority support"
            ],
            "enterprise": [
                "Full feature set with enterprise integrations",
                "Custom branding and design system",
                "High-availability database setup",
                "Comprehensive test automation",
                "Multi-environment deployment",
                "Advanced security implementation",
                "Performance monitoring and analytics",
                "12 months dedicated support"
            ],
            "enterprise_plus": [
                "Complete custom solution development",
                "White-label branding and customization",
                "Enterprise-grade infrastructure",
                "Full test automation and quality assurance",
                "Multi-region deployment with failover",
                "Advanced security and compliance",
                "Real-time monitoring and alerting",
                "Custom integrations and APIs",
                "18 months premium support with SLA"
            ]
        }
        
        features = base_features.get(tier_level, base_features["standard"]).copy()
        
        # Add template-specific features
        if template == OfferTemplate.WEB_APPLICATION:
            if tier_level in ["professional", "enterprise", "enterprise_plus"]:
                features.append("SEO optimization and analytics")
                features.append("Progressive Web App (PWA) capabilities")
        elif template == OfferTemplate.MOBILE_APPLICATION:
            if tier_level in ["professional", "enterprise", "enterprise_plus"]:
                features.append("App store optimization and submission")
                features.append("Push notifications and offline support")
        elif template == OfferTemplate.ECOMMERCE_PLATFORM:
            if tier_level in ["standard", "professional", "enterprise", "enterprise_plus"]:
                features.append("Payment gateway integration")
                features.append("Inventory management system")
        elif template == OfferTemplate.AI_ML_SOLUTION:
            if tier_level in ["professional", "enterprise", "enterprise_plus"]:
                features.append("Custom AI model training")
                features.append("Advanced analytics and insights")
        
        return features
    
    def _adjust_timeline(self, timeline: str, multiplier: float) -> str:
        """Adjust timeline by multiplier"""
        try:
            # Simple timeline adjustment - in production, this would be more sophisticated
            if "week" in timeline:
                parts = timeline.split("-")
                if len(parts) == 2:
                    min_weeks = int(parts[0]) * multiplier
                    max_weeks = int(parts[1].split()[0]) * multiplier
                    return f"{int(min_weeks)}-{int(max_weeks)} weeks"
            elif "month" in timeline:
                parts = timeline.split("-")
                if len(parts) == 2:
                    min_months = int(parts[0]) * multiplier
                    max_months = int(parts[1].split()[0]) * multiplier
                    return f"{int(min_months)}-{int(max_months)} months"
            
            return timeline
        except:
            return timeline
    
    def generate_deliverables(self, analysis: BusinessAnalysisResult, template: OfferTemplate) -> List[DeliverableCategory]:
        """
        Generate comprehensive deliverables list based on analysis and template
        
        Args:
            analysis: Business analysis result
            template: Offer template type
            
        Returns:
            List[DeliverableCategory]: Organized deliverables
        """
        try:
            deliverables = []
            
            # Always include technical deliverables
            deliverables.append(DeliverableCategory(**self.base_deliverables["technical"]))
            
            # Add design deliverables for UI-focused projects
            if template in [OfferTemplate.WEB_APPLICATION, OfferTemplate.MOBILE_APPLICATION, 
                          OfferTemplate.ECOMMERCE_PLATFORM, OfferTemplate.ANALYTICS_DASHBOARD]:
                deliverables.append(DeliverableCategory(**self.base_deliverables["design"]))
            
            # Always include testing
            deliverables.append(DeliverableCategory(**self.base_deliverables["testing"]))
            
            # Add deployment for all projects
            deliverables.append(DeliverableCategory(**self.base_deliverables["deployment"]))
            
            # Add support
            deliverables.append(DeliverableCategory(**self.base_deliverables["support"]))
            
            # Add template-specific deliverables
            if template == OfferTemplate.AI_ML_SOLUTION:
                deliverables.append(DeliverableCategory(
                    category="AI/ML Specific",
                    description="Artificial intelligence and machine learning components",
                    items=[
                        "Trained machine learning models",
                        "Data preprocessing pipelines",
                        "Model evaluation and validation reports",
                        "AI model deployment and serving infrastructure",
                        "Data science notebooks and analysis"
                    ]
                ))
            elif template == OfferTemplate.ECOMMERCE_PLATFORM:
                deliverables.append(DeliverableCategory(
                    category="E-commerce Specific",
                    description="E-commerce platform specific components",
                    items=[
                        "Payment gateway integration and testing",
                        "Inventory management system",
                        "Order processing and fulfillment workflows",
                        "Customer account management",
                        "Analytics and reporting dashboard"
                    ]
                ))
            
            return deliverables
            
        except Exception as e:
            logger.error(f"Error generating deliverables: {e}")
            return [DeliverableCategory(**self.base_deliverables["technical"])]
    
    def generate_next_steps(self, analysis: BusinessAnalysisResult, template: OfferTemplate) -> List[CallToAction]:
        """
        Generate next steps and call-to-action items
        
        Args:
            analysis: Business analysis result
            template: Offer template type
            
        Returns:
            List[CallToAction]: Next steps with urgency and timeline
        """
        try:
            next_steps = []
            
            # Immediate next steps
            next_steps.append(CallToAction(
                action="Schedule Discovery Call",
                description="Book a 30-minute consultation to discuss your project requirements in detail",
                urgency="high",
                timeline="Within 24 hours"
            ))
            
            next_steps.append(CallToAction(
                action="Review Detailed Proposal",
                description="Receive and review a comprehensive project proposal with detailed specifications",
                urgency="high",
                timeline="2-3 business days"
            ))
            
            # Project-specific steps
            if analysis.complexity_assessment.level in ["complex", "enterprise"]:
                next_steps.append(CallToAction(
                    action="Technical Architecture Review",
                    description="Conduct detailed technical architecture planning session with our senior architects",
                    urgency="medium",
                    timeline="1 week"
                ))
            
            # Contract and planning steps
            next_steps.append(CallToAction(
                action="Finalize Project Scope",
                description="Confirm final project scope, timeline, and deliverables",
                urgency="medium",
                timeline="3-5 business days"
            ))
            
            next_steps.append(CallToAction(
                action="Sign Development Agreement",
                description="Execute project contract and establish project governance",
                urgency="medium",
                timeline="1 week"
            ))
            
            # Project initiation
            next_steps.append(CallToAction(
                action="Project Kickoff",
                description="Begin development with project kickoff meeting and team introductions",
                urgency="low",
                timeline="1-2 weeks"
            ))
            
            # Template-specific steps
            if template == OfferTemplate.AI_ML_SOLUTION:
                next_steps.insert(2, CallToAction(
                    action="Data Assessment",
                    description="Evaluate available data sources and quality for AI/ML implementation",
                    urgency="high",
                    timeline="1 week"
                ))
            elif template == OfferTemplate.ECOMMERCE_PLATFORM:
                next_steps.insert(2, CallToAction(
                    action="Payment Gateway Setup",
                    description="Configure and test payment processing systems",
                    urgency="medium",
                    timeline="1-2 weeks"
                ))
            
            return next_steps
            
        except Exception as e:
            logger.error(f"Error generating next steps: {e}")
            return self._get_fallback_next_steps()
    
    def format_comprehensive_offer(self, analysis: BusinessAnalysisResult, project_urls: Dict[str, str]) -> FormattedOffer:
        """
        Create a comprehensive formatted offer from business analysis
        
        Args:
            analysis: Business analysis result
            project_urls: Project URLs (plan_url, offer_url)
            
        Returns:
            FormattedOffer: Complete formatted offer
        """
        try:
            # Detect appropriate template
            template = self.detect_offer_template(analysis)
            
            # Generate offer components
            pricing_tiers = self.generate_pricing_tiers(analysis, template)
            deliverables = self.generate_deliverables(analysis, template)
            next_steps = self.generate_next_steps(analysis, template)
            
            # Create executive summary
            executive_summary = self._generate_executive_summary(analysis, template)
            
            # Generate title
            title = self._generate_offer_title(analysis, template)
            
            # Convert implementation phases
            implementation_phases = []
            for phase in analysis.implementation_strategy:
                implementation_phases.append({
                    "name": phase.name,
                    "duration": phase.duration,
                    "description": phase.description,
                    "deliverables": phase.deliverables,
                    "dependencies": phase.dependencies
                })
            
            offer = FormattedOffer(
                project_id=analysis.analysis_metadata.get("workflow_id", "unknown"),
                template_type=template,
                title=title,
                executive_summary=executive_summary,
                pricing_tiers=pricing_tiers,
                implementation_phases=implementation_phases,
                deliverables=deliverables,
                next_steps=next_steps,
                terms_and_conditions=self._generate_terms_and_conditions(),
                contact_information=self._get_contact_information(),
                validity_period="30 days",
                created_at=datetime.now().isoformat(),
                urls=project_urls
            )
            
            logger.info(f"Generated comprehensive offer for project {offer.project_id} using template {template.value}")
            return offer
            
        except Exception as e:
            logger.error(f"Error formatting comprehensive offer: {e}")
            return self._create_fallback_offer(analysis, project_urls)
    
    def _generate_executive_summary(self, analysis: BusinessAnalysisResult, template: OfferTemplate) -> str:
        """Generate executive summary for the offer"""
        summary_parts = []
        
        # Project overview
        summary_parts.append(f"This proposal outlines a comprehensive {template.value.replace('_', ' ')} solution based on your specific requirements.")
        
        # Complexity and scope
        complexity_desc = {
            "simple": "straightforward",
            "moderate": "moderately complex",
            "complex": "sophisticated",
            "enterprise": "enterprise-grade"
        }
        
        summary_parts.append(f"Our analysis indicates this is a {complexity_desc.get(analysis.complexity_assessment.level, 'custom')} project requiring {analysis.timeline_estimate} for completion.")
        
        # Key technologies
        if analysis.technology_stack:
            tech_list = [tech.technology for tech in analysis.technology_stack[:3]]
            summary_parts.append(f"The solution will leverage modern technologies including {', '.join(tech_list)} to ensure scalability and maintainability.")
        
        # Budget and value
        summary_parts.append(f"With an estimated investment range of {analysis.budget_range}, this solution will deliver significant value through improved efficiency and user experience.")
        
        return " ".join(summary_parts)
    
    def _generate_offer_title(self, analysis: BusinessAnalysisResult, template: OfferTemplate) -> str:
        """Generate offer title"""
        template_titles = {
            OfferTemplate.WEB_APPLICATION: "Custom Web Application Development",
            OfferTemplate.MOBILE_APPLICATION: "Mobile Application Development",
            OfferTemplate.API_SERVICE: "API Service Development",
            OfferTemplate.ECOMMERCE_PLATFORM: "E-commerce Platform Development",
            OfferTemplate.CRM_SYSTEM: "CRM System Development",
            OfferTemplate.ANALYTICS_DASHBOARD: "Analytics Dashboard Development",
            OfferTemplate.AI_ML_SOLUTION: "AI/ML Solution Development",
            OfferTemplate.ENTERPRISE_SYSTEM: "Enterprise System Development",
            OfferTemplate.GENERAL_SOFTWARE: "Custom Software Development"
        }
        
        base_title = template_titles.get(template, "Custom Software Development")
        complexity = analysis.complexity_assessment.level.title()
        
        return f"{base_title} - {complexity} Solution"
    
    def _generate_terms_and_conditions(self) -> List[str]:
        """Generate standard terms and conditions"""
        return [
            "Prices are valid for 30 days from the date of this proposal",
            "50% deposit required to begin development, remaining balance due upon completion",
            "All deliverables remain property of StateX until final payment is received",
            "Changes to project scope may result in additional costs and timeline adjustments",
            "Client is responsible for providing necessary access, content, and feedback in a timely manner",
            "Support period begins after successful project delivery and acceptance",
            "All work is covered by our standard warranty and service level agreements"
        ]
    
    def _get_contact_information(self) -> Dict[str, str]:
        """Get contact information"""
        return {
            "company": "StateX Development",
            "email": "projects@statex.cz",
            "phone": "+420 XXX XXX XXX",
            "website": "https://statex.cz",
            "address": "Prague, Czech Republic"
        }
    
    def _get_fallback_pricing_tiers(self) -> List[PricingTier]:
        """Fallback pricing tiers"""
        return [
            PricingTier(
                name="Standard",
                price_range="$8,000 - $12,000",
                description="Complete solution for your needs",
                features=["Full functionality", "Custom design", "Testing", "Deployment", "3 months support"],
                timeline="2-4 weeks",
                support_level="Email and phone support",
                recommended=True
            )
        ]
    
    def _get_fallback_next_steps(self) -> List[CallToAction]:
        """Fallback next steps"""
        return [
            CallToAction(
                action="Schedule Consultation",
                description="Book a call to discuss your project",
                urgency="high",
                timeline="24 hours"
            ),
            CallToAction(
                action="Review Proposal",
                description="Review detailed project proposal",
                urgency="medium",
                timeline="2-3 days"
            )
        ]
    
    def _create_fallback_offer(self, analysis: BusinessAnalysisResult, project_urls: Dict[str, str]) -> FormattedOffer:
        """Create fallback offer when main formatting fails"""
        return FormattedOffer(
            project_id=analysis.analysis_metadata.get("workflow_id", "unknown"),
            template_type=OfferTemplate.GENERAL_SOFTWARE,
            title="Custom Software Development Proposal",
            executive_summary="This proposal outlines a custom software solution based on your requirements.",
            pricing_tiers=self._get_fallback_pricing_tiers(),
            implementation_phases=[{
                "name": "Development",
                "duration": "2-4 weeks",
                "description": "Complete development and deployment",
                "deliverables": ["Working software", "Documentation"],
                "dependencies": []
            }],
            deliverables=[DeliverableCategory(**self.base_deliverables["technical"])],
            next_steps=self._get_fallback_next_steps(),
            terms_and_conditions=self._generate_terms_and_conditions(),
            contact_information=self._get_contact_information(),
            validity_period="30 days",
            created_at=datetime.now().isoformat(),
            urls=project_urls
        )

# Global offer formatter instance
offer_formatter = OfferFormatter()