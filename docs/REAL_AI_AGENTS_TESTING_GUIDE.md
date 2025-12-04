# StateX Real AI Agents Performance Testing Guide

## 🎯 **Overview**

This guide explains how to test the complete workflow with **REAL AI agents** and measure their individual performance. The test simulates a real contact form submission and measures how each AI agent performs.

## 🚀 **What This Test Does**

### **1. Contact Form Simulation**

- Simulates user submitting text, voice, and file data
- Uses realistic auto repair business scenario
- Tests all data types (text, voice transcript, document)

### **2. Initial Notification**

- Sends confirmation message to Telegram
- Confirms data received and analysis started
- Sets expectations for follow-up

### **3. AI Agents Performance Testing**

- Tests **ALL available AI agents** in parallel
- Measures individual performance (speed, quality)
- Sends individual results to Telegram
- Compares agent performance

### **4. Performance Analysis**

- Measures processing time for each agent
- Tracks success/failure rates
- Identifies fastest and slowest agents
- Provides detailed performance metrics

## 🤖 **AI Agents Tested**

### **1. Free AI Agent (Port 8016)**

- **Purpose**: Business analysis using free models
- **Models**: Ollama (Llama 2, Mistral, CodeLlama), Hugging Face
- **Analysis**: Business insights, technical recommendations
- **Expected Time**: 3-10 seconds

### **2. NLP Agent (Port 8011)**

- **Purpose**: Natural language processing
- **Models**: OpenAI GPT-4, Anthropic Claude
- **Analysis**: Text analysis, content generation
- **Expected Time**: 2-8 seconds

### **3. ASR Agent (Port 8012)**

- **Purpose**: Speech-to-text analysis
- **Models**: OpenAI Whisper
- **Analysis**: Voice transcript processing
- **Expected Time**: 1-5 seconds

### **4. Document AI Agent (Port 8013)**

- **Purpose**: Document analysis and processing
- **Models**: Tesseract OCR, Unstructured
- **Analysis**: File content extraction and analysis
- **Expected Time**: 2-6 seconds

### **5. AI Orchestrator Agent (Port 8010)**

- **Purpose**: Multi-agent coordination
- **Models**: Coordinates all other agents
- **Analysis**: Comprehensive workflow management
- **Expected Time**: 5-15 seconds

## 📱 **What You'll Receive on Telegram**

### **1. Initial Confirmation**

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

### **2. Individual Agent Results**

```text
✅ Free AI Agent Results

⏱️ Processing Time: 4.23 seconds
🤖 Provider: Ollama
🧠 Model: llama2:7b
🎯 Confidence: 85%
📊 Response Length: 1,247 characters

📋 Analysis Summary:
User Sergej wants to create a digital solution for their auto business, focusing on automation and customer experience.

💡 Key Insights:
• High demand for digital transformation in auto repair
• Customer booking automation is critical
• Mobile app for technicians is essential

🔧 Recommendations:
• Implement online booking system
• Create customer management portal
• Develop mobile technician app
```

### **3. Performance Summary**

```text
📊 AI Agents Performance Summary

✅ Successful Agents: 4/5
❌ Failed Agents: 1/5

⏱️ Performance Metrics:
• Total Processing Time: 18.45 seconds
• Average Processing Time: 3.69 seconds
• Fastest Agent: ASR Agent (1.23s)
• Slowest Agent: AI Orchestrator (8.91s)

🤖 Agent Details:
• ✅ Free AI Agent: 4.23s (Ollama)
• ✅ NLP Agent: 3.45s (OpenAI GPT-4)
• ✅ ASR Agent: 1.23s (OpenAI Whisper)
• ✅ Document AI Agent: 2.67s (Tesseract OCR)
• ❌ AI Orchestrator: 8.91s (Failed)
```

## 🚀 **How to Run the Test**

### **Step 1: Start AI Services**

```bash
cd /Users/sergiystashok/Documents/GitHub/statex/statex-ai
./setup_free_ai.sh
```

### **Step 2: Run the Test**

```bash
# Interactive mode (enter your own data)
python3 test_real_ai_agents.py

# Demo mode (use sample data)
python3 test_real_ai_agents.py --demo

# Default mode (use default test data)
python3 test_real_ai_agents.py --default
```

### **Step 3: Monitor Results**

- Watch the console for real-time progress
- Check Telegram for individual agent results
- Review performance metrics

## 📊 **Performance Metrics Explained**

### **Processing Time**

- **Fastest**: ASR Agent (1-2 seconds)
- **Medium**: Document AI, NLP Agent (2-5 seconds)
- **Slowest**: Free AI Agent, AI Orchestrator (3-10 seconds)

### **Success Rate**

- **Expected**: 80-100% (depending on service availability)
- **Free AI Agent**: May fail if Ollama not running
- **NLP Agent**: May fail if OpenAI API not configured
- **ASR Agent**: May fail if OpenAI API not configured
- **Document AI**: Usually succeeds (local processing)
- **AI Orchestrator**: May fail if other services unavailable

### **Quality Indicators**

- **Response Length**: Longer responses usually indicate more detailed analysis
- **Confidence Score**: Higher confidence indicates more reliable results
- **Provider**: Different providers have different strengths

## 🔍 **Troubleshooting**

### **No AI Agents Available**

```bash
# Check if services are running
docker compose ps

# Restart services
docker compose restart

# Check logs
docker compose logs -f free-ai-service
```

### **Free AI Agent Fails**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama manually
docker exec statex-ai-ollama-1 ollama serve

# Download models
docker exec statex-ai-ollama-1 ollama pull llama2:7b
```

### **NLP/ASR Agents Fail**

```bash
# Check if OpenAI API key is set
echo $OPENAI_API_KEY

# Set API key in .env file
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### **Telegram Notifications Not Working**

```bash
# Check notification service

# Check Telegram bot
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe"
```

## 🎯 **Expected Results**

### **Best Case Scenario**

- All 5 agents succeed
- Total time: 15-25 seconds
- All agents send individual results
- Performance summary shows all green

### **Typical Scenario**

- 3-4 agents succeed
- Total time: 20-35 seconds
- Some agents may fail due to configuration
- Performance summary shows mixed results

### **Worst Case Scenario**

- 1-2 agents succeed
- Total time: 30-60 seconds
- Most agents fail due to service issues
- Performance summary shows mostly red

## 📈 **Performance Optimization**

### **For Faster Results**

- Use local models (Ollama) instead of cloud APIs
- Ensure all services are running before test
- Use smaller models for faster processing
- Run agents in parallel (already implemented)

### **For Better Quality**

- Use larger models (llama2:13b instead of 7b)
- Configure OpenAI API for NLP/ASR agents
- Ensure all services are properly configured
- Use specialized models for specific tasks

## 🎉 **Benefits of This Test**

### **✅ Real Performance Data**

- Actual processing times
- Real success/failure rates
- Genuine quality measurements

### **✅ Agent Comparison**

- Side-by-side performance comparison
- Identify best performing agents
- Understand agent strengths/weaknesses

### **✅ Workflow Validation**

- End-to-end workflow testing
- Real notification delivery
- Complete user experience simulation

### **✅ Performance Monitoring**

- Track improvements over time
- Identify bottlenecks
- Optimize agent configuration

## 🚀 **Next Steps**

After running the test:

1. **Analyze Results**: Review which agents performed best
2. **Optimize Configuration**: Fix failed agents, improve performance
3. **Scale Testing**: Run multiple tests with different data
4. **Monitor Trends**: Track performance over time
5. **Improve Quality**: Adjust models and parameters

## 🎯 **Success Criteria**

### **Minimum Success**

- At least 2 agents succeed
- Total time under 60 seconds
- At least 1 notification received

### **Good Success**

- At least 3 agents succeed
- Total time under 40 seconds
- All notifications received

### **Excellent Success**

- All 5 agents succeed
- Total time under 30 seconds
- All notifications received
- High quality analysis results

**This test gives you real performance data for all your AI agents and helps you optimize your StateX AI system!** 🚀
