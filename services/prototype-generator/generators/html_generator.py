"""
HTML structure generator using AI models.
"""

import httpx
import logging
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

class HTMLGenerator:
    """Generates HTML structure using AI models."""
    
    def __init__(self, free_ai_service_url: str = None):
        """Initialize HTML generator."""
        import os
        if free_ai_service_url is None:
            free_ai_port = os.getenv("FREE_AI_SERVICE_PORT", "3386")
            free_ai_service_url = f"http://localhost:{free_ai_port}"
        self.free_ai_service_url = free_ai_service_url
        
    async def generate_html(self, requirements: str, analysis: Dict[str, Any], prototype_type: str) -> str:
        """Generate HTML structure based on requirements."""
        try:
            # Create prompt for HTML generation
            prompt = self._create_html_prompt(requirements, analysis, prototype_type)
            
            logger.info(f"🚀 Calling Free AI Service for HTML generation: {prototype_type}")
            
            # Call Free AI Service using the correct /analyze endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.free_ai_service_url}/analyze",
                    json={
                        "text_content": prompt,
                        "analysis_type": "content_generation",
                        "provider": "mock"  # Use mock for now since Ollama/HuggingFace are unavailable
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("success", False):
                        # Extract HTML from the analysis response
                        analysis_data = result.get("analysis", {})
                        
                        # The Free AI Service mock provider returns business analysis, not code
                        # We need to use the insights to generate appropriate HTML
                        insights = analysis_data.get("key_insights", [])
                        recommendations = analysis_data.get("recommendations", [])
                        
                        if insights or recommendations:
                            # Generate HTML based on the AI insights
                            html_content = self._generate_html_from_insights(
                                insights, recommendations, prototype_type, requirements
                            )
                            logger.info(f"✅ Generated HTML from AI insights using {result.get('provider_used', 'unknown')}")
                        else:
                            logger.warning("⚠️ No insights from AI service, using fallback")
                            html_content = self._generate_fallback_html(prototype_type, requirements)
                        
                        # Clean and validate HTML
                        html_content = self._clean_html(html_content)
                        
                        logger.info(f"✅ Successfully generated HTML for {prototype_type} prototype using {result.get('provider_used', 'unknown')}")
                        return html_content
                    else:
                        logger.error(f"❌ AI service returned error: {result.get('error', 'Unknown error')}")
                        return self._generate_fallback_html(prototype_type, requirements)
                else:
                    logger.error(f"❌ AI service HTTP error: {response.status_code} - {response.text}")
                    return self._generate_fallback_html(prototype_type, requirements)
                    
        except httpx.TimeoutException:
            logger.error("❌ AI service timeout - using fallback HTML")
            return self._generate_fallback_html(prototype_type, requirements)
        except Exception as e:
            logger.error(f"❌ Error generating HTML: {e}", exc_info=True)
            return self._generate_fallback_html(prototype_type, requirements)
    
    def _create_html_prompt(self, requirements: str, analysis: Dict[str, Any], prototype_type: str) -> str:
        """Create prompt for HTML generation."""
        
        base_prompt = f"""
Generate a complete HTML5 structure for a {prototype_type} website based on these requirements:

REQUIREMENTS: {requirements}

PROTOTYPE TYPE: {prototype_type}

ANALYSIS: {json.dumps(analysis, indent=2)}

Please generate a complete HTML5 document with:
1. Proper DOCTYPE and meta tags
2. Semantic HTML structure
3. Responsive design considerations
4. SEO-optimized structure
5. Clean, modern layout
6. All necessary sections for a {prototype_type}

Return ONLY the HTML code, no explanations or markdown formatting.
"""
        
        if prototype_type == "ecommerce":
            base_prompt += """
Include these specific sections:
- Header with navigation and cart
- Hero section with featured products
- Product grid/catalog
- Shopping cart functionality
- Footer with links and info
"""
        elif prototype_type == "website":
            base_prompt += """
Include these specific sections:
- Header with navigation
- Hero section
- About/Services section
- Contact form
- Footer
"""
        elif prototype_type == "landing_page":
            base_prompt += """
Include these specific sections:
- Hero section with CTA
- Features/Benefits section
- Pricing section
- Contact form
- Footer
"""
        
        return base_prompt
    
    def _generate_html_from_insights(self, insights: List[str], recommendations: List[str], prototype_type: str, requirements: str) -> str:
        """Generate HTML based on AI insights and recommendations."""
        
        # Extract key themes from insights
        themes = []
        for insight in insights:
            if "digital transformation" in insight.lower():
                themes.append("modern_tech")
            elif "customer experience" in insight.lower():
                themes.append("user_focused")
            elif "multi-platform" in insight.lower():
                themes.append("responsive")
            elif "integration" in insight.lower():
                themes.append("connected")
        
        # Generate content based on themes and prototype type
        if prototype_type == "landing_page":
            return self._generate_landing_from_insights(themes, insights, recommendations, requirements)
        elif prototype_type == "ecommerce":
            return self._generate_ecommerce_from_insights(themes, insights, recommendations, requirements)
        else:  # website
            return self._generate_website_from_insights(themes, insights, recommendations, requirements)
    
    def _generate_landing_from_insights(self, themes: List[str], insights: List[str], recommendations: List[str], requirements: str) -> str:
        """Generate landing page HTML from AI insights."""
        
        # Create dynamic content based on insights
        hero_title = "Transform Your Business Today"
        hero_subtitle = "Discover the solution that will revolutionize your workflow"
        
        if "modern_tech" in themes:
            hero_title = "Next-Gen Technology Solutions"
            hero_subtitle = "Leverage cutting-edge technology to transform your business"
        elif "user_focused" in themes:
            hero_title = "Customer-Centric Solutions"
            hero_subtitle = "Put your customers first with our innovative approach"
        
        # Generate features based on recommendations
        features = []
        for rec in recommendations[:3]:  # Take first 3 recommendations
            if "MVP" in rec:
                features.append({"title": "Rapid MVP Development", "desc": "Get to market faster with our proven development process"})
            elif "feedback" in rec.lower():
                features.append({"title": "User Feedback Integration", "desc": "Continuous improvement through real user insights"})
            elif "scalability" in rec.lower():
                features.append({"title": "Scalable Architecture", "desc": "Built to grow with your business needs"})
            elif "security" in rec.lower():
                features.append({"title": "Enterprise Security", "desc": "Bank-level security for your peace of mind"})
        
        # Default features if none generated
        if not features:
            features = [
                {"title": "Modern Design", "desc": "Clean, professional interface that converts"},
                {"title": "Fast Performance", "desc": "Optimized for speed and user experience"},
                {"title": "Mobile Ready", "desc": "Perfect on all devices and screen sizes"}
            ]
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{hero_title}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">TechStartup Pro</div>
            <button class="cta-button">Get Started</button>
        </nav>
    </header>
    
    <main>
        <section class="hero">
            <h1>{hero_title}</h1>
            <p>{hero_subtitle}</p>
            <button class="cta-button primary">Start Free Trial</button>
            <button class="cta-button secondary">Learn More</button>
        </section>
        
        <section class="features">
            <h2>Why Choose Us</h2>
            <div class="feature-grid">
                {''.join([f'''
                <div class="feature-card">
                    <h3>{feature['title']}</h3>
                    <p>{feature['desc']}</p>
                </div>
                ''' for feature in features])}
            </div>
        </section>
        
        <section class="pricing">
            <h2>Simple Pricing</h2>
            <div class="pricing-card">
                <h3>Starter Plan</h3>
                <div class="price">$29/month</div>
                <ul>
                    <li>All core features</li>
                    <li>24/7 support</li>
                    <li>30-day free trial</li>
                </ul>
                <button class="cta-button">Choose Plan</button>
            </div>
        </section>
        
        <section class="contact">
            <h2>Ready to Get Started?</h2>
            <form class="contact-form">
                <input type="email" placeholder="Enter your email" required>
                <button type="submit">Get Started</button>
            </form>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 TechStartup Pro. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    def _generate_ecommerce_from_insights(self, themes: List[str], insights: List[str], recommendations: List[str], requirements: str) -> str:
        """Generate e-commerce HTML from AI insights."""
        
        # Create dynamic content based on insights
        store_name = "TechGadgets Store"
        hero_title = "Discover Amazing Products"
        hero_subtitle = "Shop the latest trends and find exactly what you're looking for"
        
        if "modern_tech" in themes:
            store_name = "AI Tech Store"
            hero_title = "Next-Gen Tech Gadgets"
            hero_subtitle = "Discover AI-powered devices and cutting-edge technology"
        elif "user_focused" in themes:
            store_name = "Customer-First Store"
            hero_title = "Products You'll Love"
            hero_subtitle = "Curated selection based on your preferences and needs"
        
        # Generate features based on recommendations
        features = []
        for rec in recommendations[:3]:  # Take first 3 recommendations
            if "MVP" in rec:
                features.append({"title": "Quick Delivery", "desc": "Get your products delivered fast with our express shipping"})
            elif "feedback" in rec.lower():
                features.append({"title": "Customer Reviews", "desc": "Real reviews from verified customers to help you decide"})
            elif "scalability" in rec.lower():
                features.append({"title": "Wide Selection", "desc": "Thousands of products to choose from"})
            elif "security" in rec.lower():
                features.append({"title": "Secure Shopping", "desc": "Bank-level security for safe transactions"})
        
        # Default features if none generated
        if not features:
            features = [
                {"title": "Fast Shipping", "desc": "Get your orders delivered quickly"},
                {"title": "Quality Guarantee", "desc": "We stand behind every product"},
                {"title": "24/7 Support", "desc": "Our team is here to help"}
            ]
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{store_name}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">{store_name}</div>
            <ul class="nav-links">
                <li><a href="#home">Home</a></li>
                <li><a href="#products">Products</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
            <div class="cart">Cart (0)</div>
        </nav>
    </header>
    
    <main>
        <section id="hero" class="hero">
            <h1>{hero_title}</h1>
            <p>{hero_subtitle}</p>
            <button class="cta-button">Shop Now</button>
        </section>
        
        <section id="products" class="products">
            <h2>Featured Products</h2>
            <div class="product-grid">
                <div class="product-card">
                    <img src="https://via.placeholder.com/300x200" alt="Product 1">
                    <h3>AI Smart Device</h3>
                    <p>$299.99</p>
                    <button class="add-to-cart">Add to Cart</button>
                </div>
                <div class="product-card">
                    <img src="https://via.placeholder.com/300x200" alt="Product 2">
                    <h3>Wireless Headphones</h3>
                    <p>$199.99</p>
                    <button class="add-to-cart">Add to Cart</button>
                </div>
                <div class="product-card">
                    <img src="https://via.placeholder.com/300x200" alt="Product 3">
                    <h3>Smart Watch</h3>
                    <p>$399.99</p>
                    <button class="add-to-cart">Add to Cart</button>
                </div>
            </div>
        </section>
        
        <section class="features">
            <h2>Why Shop With Us</h2>
            <div class="feature-grid">
                {''.join([f'''
                <div class="feature-card">
                    <h3>{feature['title']}</h3>
                    <p>{feature['desc']}</p>
                </div>
                ''' for feature in features])}
            </div>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 {store_name}. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    def _generate_website_from_insights(self, themes: List[str], insights: List[str], recommendations: List[str], requirements: str) -> str:
        """Generate business website HTML from AI insights."""
        
        # Create dynamic content based on insights
        company_name = "Professional Solutions"
        hero_title = "Professional Business Solutions"
        hero_subtitle = "We help businesses grow with innovative solutions and expert guidance"
        
        if "modern_tech" in themes:
            company_name = "Tech Solutions Inc"
            hero_title = "Next-Gen Business Solutions"
            hero_subtitle = "Leverage cutting-edge technology to transform your business operations"
        elif "user_focused" in themes:
            company_name = "Customer-Centric Solutions"
            hero_title = "Customer-First Business Solutions"
            hero_subtitle = "Put your customers at the center of everything we do"
        
        # Generate services based on recommendations
        services = []
        for rec in recommendations[:3]:  # Take first 3 recommendations
            if "MVP" in rec:
                services.append({"title": "Rapid Development", "desc": "Get your MVP to market faster with our proven process"})
            elif "feedback" in rec.lower():
                services.append({"title": "Customer Research", "desc": "Understand your customers with data-driven insights"})
            elif "scalability" in rec.lower():
                services.append({"title": "Growth Strategy", "desc": "Scale your business with our strategic planning"})
            elif "security" in rec.lower():
                services.append({"title": "Security Solutions", "desc": "Protect your business with enterprise-grade security"})
        
        # Default services if none generated
        if not services:
            services = [
                {"title": "Consulting", "desc": "Expert advice to help you make informed decisions"},
                {"title": "Strategy", "desc": "Comprehensive planning to drive your business forward"},
                {"title": "Implementation", "desc": "Turn your ideas into reality with our proven methodologies"}
            ]
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">{company_name}</div>
            <ul class="nav-links">
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#services">Services</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="hero" class="hero">
            <h1>{hero_title}</h1>
            <p>{hero_subtitle}</p>
            <button class="cta-button">Get Started</button>
        </section>
        
        <section id="about" class="about">
            <h2>About {company_name}</h2>
            <p>We are a dedicated team of professionals committed to delivering exceptional results. With years of experience and a passion for excellence, we help our clients achieve their goals.</p>
        </section>
        
        <section id="services" class="services">
            <h2>Our Services</h2>
            <div class="service-grid">
                {''.join([f'''
                <div class="service-card">
                    <h3>{service['title']}</h3>
                    <p>{service['desc']}</p>
                </div>
                ''' for service in services])}
            </div>
        </section>
        
        <section id="contact" class="contact">
            <h2>Let's Work Together</h2>
            <form class="contact-form">
                <input type="text" placeholder="Your Name" required>
                <input type="email" placeholder="Your Email" required>
                <textarea placeholder="Your Message" required></textarea>
                <button type="submit">Send Message</button>
            </form>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 {company_name}. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    def _clean_html(self, html_content: str) -> str:
        """Clean and validate generated HTML."""
        # Remove markdown code blocks if present
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.startswith("```"):
            html_content = html_content[3:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        
        # Ensure proper DOCTYPE
        if not html_content.strip().startswith("<!DOCTYPE"):
            html_content = "<!DOCTYPE html>\n" + html_content
        
        # Basic validation - ensure html, head, body tags
        if "<html" not in html_content:
            html_content = f"<html lang=\"en\">\n{html_content}\n</html>"
        
        if "<head" not in html_content:
            html_content = html_content.replace("<html", "<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<title>Generated Prototype</title>\n</head>\n<body>")
        
        if "</body>" not in html_content:
            html_content += "\n</body>"
        
        return html_content.strip()
    
    def _generate_fallback_html(self, prototype_type: str, requirements: str) -> str:
        """Generate fallback HTML when AI service fails."""
        
        if prototype_type == "ecommerce":
            return self._get_ecommerce_template(requirements)
        elif prototype_type == "website":
            return self._get_website_template(requirements)
        else:  # landing_page
            return self._get_landing_template(requirements)
    
    def _get_ecommerce_template(self, requirements: str) -> str:
        """E-commerce HTML template."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-commerce Store</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">Store</div>
            <ul class="nav-links">
                <li><a href="#home">Home</a></li>
                <li><a href="#products">Products</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
            <div class="cart">Cart (0)</div>
        </nav>
    </header>
    
    <main>
        <section id="hero" class="hero">
            <h1>Welcome to Our Store</h1>
            <p>Discover amazing products at great prices</p>
            <button class="cta-button">Shop Now</button>
        </section>
        
        <section id="products" class="products">
            <h2>Featured Products</h2>
            <div class="product-grid">
                <div class="product-card">
                    <img src="https://via.placeholder.com/300x200" alt="Product 1">
                    <h3>Product 1</h3>
                    <p>$29.99</p>
                    <button class="add-to-cart">Add to Cart</button>
                </div>
                <div class="product-card">
                    <img src="https://via.placeholder.com/300x200" alt="Product 2">
                    <h3>Product 2</h3>
                    <p>$39.99</p>
                    <button class="add-to-cart">Add to Cart</button>
                </div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 Store. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    def _get_website_template(self, requirements: str) -> str:
        """Business website HTML template."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business Website</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">Company</div>
            <ul class="nav-links">
                <li><a href="#home">Home</li>
                <li><a href="#about">About</li>
                <li><a href="#services">Services</li>
                <li><a href="#contact">Contact</li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="hero" class="hero">
            <h1>Welcome to Our Company</h1>
            <p>We provide excellent services to help your business grow</p>
            <button class="cta-button">Get Started</button>
        </section>
        
        <section id="about" class="about">
            <h2>About Us</h2>
            <p>We are a dedicated team focused on delivering quality solutions.</p>
        </section>
        
        <section id="services" class="services">
            <h2>Our Services</h2>
            <div class="service-grid">
                <div class="service-card">
                    <h3>Service 1</h3>
                    <p>Description of service 1</p>
                </div>
                <div class="service-card">
                    <h3>Service 2</h3>
                    <p>Description of service 2</p>
                </div>
            </div>
        </section>
        
        <section id="contact" class="contact">
            <h2>Contact Us</h2>
            <form class="contact-form">
                <input type="text" placeholder="Your Name" required>
                <input type="email" placeholder="Your Email" required>
                <textarea placeholder="Your Message" required></textarea>
                <button type="submit">Send Message</button>
            </form>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 Company. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    def _get_landing_template(self, requirements: str) -> str:
        """Landing page HTML template."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Landing Page</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">Brand</div>
            <button class="cta-button">Get Started</button>
        </nav>
    </header>
    
    <main>
        <section class="hero">
            <h1>Transform Your Business Today</h1>
            <p>Discover the solution that will revolutionize your workflow</p>
            <button class="cta-button primary">Start Free Trial</button>
            <button class="cta-button secondary">Learn More</button>
        </section>
        
        <section class="features">
            <h2>Why Choose Us</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>Feature 1</h3>
                    <p>Amazing feature description</p>
                </div>
                <div class="feature-card">
                    <h3>Feature 2</h3>
                    <p>Another great feature</p>
                </div>
                <div class="feature-card">
                    <h3>Feature 3</h3>
                    <p>Third awesome feature</p>
                </div>
            </div>
        </section>
        
        <section class="pricing">
            <h2>Simple Pricing</h2>
            <div class="pricing-card">
                <h3>Basic Plan</h3>
                <div class="price">$29/month</div>
                <ul>
                    <li>Feature 1</li>
                    <li>Feature 2</li>
                    <li>Feature 3</li>
                </ul>
                <button class="cta-button">Choose Plan</button>
            </div>
        </section>
        
        <section class="contact">
            <h2>Ready to Get Started?</h2>
            <form class="contact-form">
                <input type="email" placeholder="Enter your email" required>
                <button type="submit">Get Started</button>
            </form>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 Brand. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
