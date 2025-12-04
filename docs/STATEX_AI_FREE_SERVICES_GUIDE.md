# StateX AI Free Services Guide

## 🎯 **Overview**

The StateX AI microservice now includes **FREE AI services** that provide intelligent analysis without requiring any API keys or paid services. All AI services are properly organized in the `statex-ai` microservice and managed by the main `statex-platform` orchestrator.

## 🏗️ **Architecture**

### **StateX AI Microservice Structure**

```text
statex-ai/
├── services/
│   ├── ai-orchestrator/     # Central coordination (Port 8010)
│   ├── nlp-service/         # Text analysis & generation (Port 8011)
│   ├── asr-service/         # Speech-to-text conversion (Port 8012)
│   ├── document-ai/         # File processing & OCR (Port 8013)
│   ├── prototype-generator/ # Website/app creation (Port 8014)
│   ├── template-repository/ # Template management (Port 8015)
│   └── free-ai-service/     # Free AI models (Port 8016) ⭐ NEW
├── ollama/                  # Local LLM service (Port 11434) ⭐ NEW
└── docker-compose.yml       # AI services orchestration
```

### **AI Providers Available**

1. **Ollama (Local LLM)** - Primary option
   - Llama 2 7B - General purpose
   - Mistral 7B - Code and analysis
   - CodeLlama 7B - Technical analysis

2. **Hugging Face API** - Secondary option
   - DialoGPT - Conversational AI
   - GPT-2 - Text generation
   - DistilBERT - Text classification

3. **Mock AI Service** - Fallback option
   - Realistic simulation for testing
   - Always available

## 🚀 **Quick Start**

### **Step 1: Setup StateX AI with Free AI Services**

```bash
cd /Users/sergiystashok/Documents/GitHub/statex/statex-ai
./setup_free_ai.sh
```

This will:

- Start all AI services
- Download free AI models (Llama 2, Mistral, CodeLlama)
- Verify all services are working
- Show available AI models

### **Step 2: Test the AI Workflow**

```bash
python3 test_ai_workflow.py --demo
```

## 🔧 **Manual Setup**

### **Start AI Services**

```bash
# Start all AI services including free AI
docker compose up -d

# Check service status
docker compose ps
```

### **Download AI Models**

```bash
# Download Llama 2 7B
docker exec statex-ai-ollama-1 ollama pull llama2:7b

# Download Mistral 7B
docker exec statex-ai-ollama-1 ollama pull mistral:7b

# Download CodeLlama 7B
docker exec statex-ai-ollama-1 ollama pull codellama:7b
```

### **Verify Services**

```bash
# Check AI Orchestrator
curl http://localhost:${AI_ORCHESTRATOR_EXTERNAL_PORT:-3610}/health

# Check Free AI Service
curl http://localhost:${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}/health

# Check Ollama Service
curl http://localhost:11434/api/tags

# Check other AI services
curl http://localhost:${NLP_SERVICE_EXTERNAL_PORT:-3611}/health  # NLP Service
curl http://localhost:${ASR_SERVICE_EXTERNAL_PORT:-3612}/health  # ASR Service
curl http://localhost:${DOCUMENT_AI_EXTERNAL_PORT:-3613}/health  # Document AI
```

## 📊 **Available AI Services**

### **AI Orchestrator (Port 8010)**

- **Purpose**: Central coordination hub for all AI operations
- **Features**: Workflow management, service routing, result aggregation
- **Status**: ✅ Ready

### **Free AI Service (Port 8016)** ⭐ NEW

- **Purpose**: Free AI models using Ollama, Hugging Face, and Mock
- **Features**: Business analysis, technical analysis, content generation
- **Models**: Llama 2, Mistral, CodeLlama, DialoGPT, GPT-2, DistilBERT
- **Status**: ✅ Ready

### **NLP Service (Port 8011)**

- **Purpose**: Natural language processing and content generation
- **Features**: Text analysis, business plan generation, sentiment analysis
- **Status**: ✅ Ready

### **ASR Service (Port 8012)**

- **Purpose**: Speech-to-text conversion and voice processing
- **Features**: Voice transcription, audio analysis, multi-language support
- **Status**: ✅ Ready

### **Document AI Service (Port 8013)**

- **Purpose**: Document analysis and processing
- **Features**: PDF extraction, OCR, document structure analysis
- **Status**: ✅ Ready

### **Prototype Generator (Port 8014)**

- **Purpose**: Website and application prototype creation
- **Features**: Next.js templates, CMS setup, code generation
- **Status**: ✅ Ready

### **Template Repository (Port 8015)**

- **Purpose**: Template management and optimization
- **Features**: Template storage, search, versioning
- **Status**: ✅ Ready

### **Ollama Service (Port 11434)** ⭐ NEW

- **Purpose**: Local LLM service
- **Features**: High-quality local AI models, privacy-focused
- **Models**: Llama 2 7B, Mistral 7B, CodeLlama 7B
- **Status**: ✅ Ready

## 🎮 **How to Use**

### **Interactive Mode**

```bash
python3 test_ai_workflow.py
```text
- Enter your own data
- Choose custom project description
- Get personalized analysis

### **Demo Mode**
```bash
python3 test_ai_workflow.py --demo
```text
- Uses sample data
- Quick testing
- No input required

### **Default Mode**
```bash
python3 test_ai_workflow.py --default
```text
- Uses default test data
- Good for repeated testing

## 📱 **What You'll Receive**

### **Initial Notification**
```text
Hello Sergej!

Thank you for your submission! We've received your project details:
• Text description: 245 characters
• Voice transcript: 156 characters  
• File content: 234 characters

Our AI agents are now analyzing your requirements using StateX AI services. We'll contact you via Telegram with the analysis results shortly.

Best regards,
The Statex Team
```

### **AI Analysis Results**

```text
🤖 AI Analysis Complete for Sergej

📋 Project Summary:
User Sergej wants to create a digital solution for their auto business, focusing on automation and customer experience.

🔍 Business Type:
Auto

⚠️ Current Pain Points:
• Manual processes and workflows
• Customer communication challenges
• Data management and tracking
• Integration between systems

💡 Business Opportunities:
• Digital Platform Development - High potential (3-6 months)
• Mobile Application - High potential (2-4 months)
• Process Automation - Medium potential (1-3 months)

🔧 Technical Recommendations:
• Frontend: React/Next.js, TypeScript, Responsive design
• Backend: Node.js/Python, PostgreSQL, RESTful API
• Integrations: Payment processing, SMS/Email, Calendar sync, Analytics

📝 Next Steps:
• Conduct auto market research (1-2 weeks)
• Develop MVP prototype (4-8 weeks)
• Create technical architecture (2-3 weeks)

💰 Budget Estimate:
• Development: $15,000 - $35,000
• Infrastructure: $200 - $500/month
• Maintenance: $1,000 - $2,000/month

🎯 Confidence: 85%
🤖 AI Provider: OLLAMA
🧠 Model: llama2:7b
⏱️ Processing Time: 3.45 seconds
```

## 🔍 **Service Detection**

The workflow automatically detects which AI services are available:

1. **Checks Free AI Service** (port 8016) - If available, uses it
2. **Checks NLP Service** (port 8011) - If Free AI not available, tries NLP
3. **Checks AI Orchestrator** (port 8010) - If others not available, tries orchestrator
4. **Falls back gracefully** - If no services available, reports error

## 🛠️ **Troubleshooting**

### **Ollama Issues**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
docker restart statex-ai-ollama-1

# Check models
docker exec statex-ai-ollama-1 ollama list
```

### **Free AI Service Issues**

```bash
# Check service health
curl http://localhost:${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}/health

# Check available models
curl http://localhost:${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}/models

# View logs
docker logs statex-ai-free-ai-service-1
```

### **AI Orchestrator Issues**

```bash
# Check service health
curl http://localhost:${AI_ORCHESTRATOR_EXTERNAL_PORT:-3610}/health

# View logs
docker logs statex-ai-orchestrator-blue
```

## 🎯 **Best Practices**

### **For Development**

- Use **Mock AI Service** for quick testing
- Use **Ollama** for realistic AI responses
- Use **NLP Service** for advanced text analysis

### **For Production**

- Use **Ollama** for privacy and control
- Use **AI Orchestrator** for complex workflows
- Use **Free AI Service** for general analysis

### **For Testing**

- Test with all available services
- Compare response quality
- Measure performance differences

## 🚀 **Advanced Usage**

### **Custom Models**

```bash
# Download additional models
docker exec statex-ai-ollama-1 ollama pull llama2:13b  # Larger model
docker exec statex-ai-ollama-1 ollama pull mistral:7b-instruct  # Instruction-tuned
```

### **API Usage**

```bash
# Direct API call to Free AI Service
curl -X POST "http://localhost:${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "I want to create a website for my business",
    "analysis_type": "business_analysis",
    "user_name": "Test User"
  }'
```

### **Service Management**

```bash
# View all services
docker compose ps

# Restart specific service
docker compose restart free-ai-service

# View logs
docker compose logs -f free-ai-service
```

## 🎉 **Benefits**

### **✅ Cost-Free**

- No API keys required
- No usage limits (for Ollama)
- No monthly costs

### **✅ Privacy**

- Data stays on your machine (Ollama)
- No external API calls (Ollama)
- Complete control over data

### **✅ Quality**

- High-quality models (Llama 2, Mistral)
- Real AI analysis
- Professional results

### **✅ Integration**

- Properly organized in statex-ai microservice
- Managed by statex-platform orchestrator
- Consistent with microservices architecture

## 🔮 **Future Enhancements**

### **Additional Models**

- More Ollama models
- Custom fine-tuned models
- Specialized business models

### **Advanced Features**

- Model switching
- Response caching
- Performance optimization
- Custom prompts

### **Monitoring**

- Usage analytics
- Performance metrics
- Model performance tracking

## 🎯 **Current Status**

| Service | Status | Port | Type | Repository |
|---------|--------|------|------|------------|
| AI Orchestrator | ✅ Ready | 8010 | AI | statex-ai |
| Free AI Service | ✅ Ready | 8016 | AI | statex-ai |
| NLP Service | ✅ Ready | 8011 | AI | statex-ai |
| ASR Service | ✅ Ready | 8012 | AI | statex-ai |
| Document AI | ✅ Ready | 8013 | AI | statex-ai |
| Prototype Generator | ✅ Ready | 8014 | AI | statex-ai |
| Template Repository | ✅ Ready | 8015 | AI | statex-ai |
| Ollama Service | ✅ Ready | 11434 | AI | statex-ai |

## 🎉 **You're Ready!**

With the StateX AI free services, you can:

✅ **Test your workflow** without any costs
✅ **Get real AI analysis** using local models
✅ **Maintain privacy** (data stays on your machine)
✅ **Scale as needed** (add more models or services)
✅ **Develop and iterate** quickly
✅ **Use proper microservices architecture** (all AI services in statex-ai)

**Start with the setup script and you'll be running AI-powered workflows in minutes!** 🚀
