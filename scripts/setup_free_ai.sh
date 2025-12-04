#!/bin/bash

# StateX AI Free AI Services Setup Script
# This script sets up free AI services in the statex-ai microservice

echo "🆓 Setting up StateX AI Free AI Services..."

# Check if we're in the statex-ai directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the statex-ai directory"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

echo "🐳 Starting AI services with free AI..."

# Start the AI services
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 20

# Check service health
echo "🔍 Checking service health..."

# AI Orchestrator
AI_ORCHESTRATOR_PORT=${AI_ORCHESTRATOR_EXTERNAL_PORT:-3610}
if curl -s http://localhost:$AI_ORCHESTRATOR_PORT/health > /dev/null; then
    echo "✅ AI Orchestrator ($AI_ORCHESTRATOR_PORT) - Ready"
else
    echo "❌ AI Orchestrator ($AI_ORCHESTRATOR_PORT) - Not ready"
fi

# Free AI Service
FREE_AI_PORT=${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}
if curl -s http://localhost:$FREE_AI_PORT/health > /dev/null; then
    echo "✅ Free AI Service ($FREE_AI_PORT) - Ready"
else
    echo "❌ Free AI Service ($FREE_AI_PORT) - Not ready"
fi

# Ollama Service (Port 11434) - DISABLED
# if curl -s http://localhost:11434/api/tags > /dev/null; then
#     echo "✅ Ollama Service (11434) - Ready"
#     
#     # Download free models
#     echo "📥 Downloading free AI models..."
#     echo "   Downloading Llama 2 7B..."
#     docker exec statex-ai-ollama-1 ollama pull llama2:7b
#     
#     echo "   Downloading Mistral 7B..."
#     docker exec statex-ai-ollama-1 ollama pull mistral:7b
#     
#     echo "   Downloading CodeLlama 7B..."
#     docker exec statex-ai-ollama-1 ollama pull codellama:7b
#     
#     echo "✅ AI models downloaded successfully"
# else
#     echo "❌ Ollama Service (11434) - Not ready"
# fi

# NLP Service
NLP_PORT=${NLP_SERVICE_EXTERNAL_PORT:-3611}
if curl -s http://localhost:$NLP_PORT/health > /dev/null; then
    echo "✅ NLP Service ($NLP_PORT) - Ready"
else
    echo "❌ NLP Service ($NLP_PORT) - Not ready"
fi

# ASR Service
ASR_PORT=${ASR_SERVICE_EXTERNAL_PORT:-3612}
if curl -s http://localhost:$ASR_PORT/health > /dev/null; then
    echo "✅ ASR Service ($ASR_PORT) - Ready"
else
    echo "❌ ASR Service ($ASR_PORT) - Not ready"
fi

# Document AI Service
DOC_AI_PORT=${DOCUMENT_AI_EXTERNAL_PORT:-3613}
if curl -s http://localhost:$DOC_AI_PORT/health > /dev/null; then
    echo "✅ Document AI Service ($DOC_AI_PORT) - Ready"
else
    echo "❌ Document AI Service ($DOC_AI_PORT) - Not ready"
fi

# Prototype Generator
PROTO_GEN_PORT=${PROTOTYPE_GENERATOR_PORT:-3614}
if curl -s http://localhost:$PROTO_GEN_PORT/health > /dev/null; then
    echo "✅ Prototype Generator ($PROTO_GEN_PORT) - Ready"
else
    echo "❌ Prototype Generator ($PROTO_GEN_PORT) - Not ready"
fi

# Template Repository
TEMPLATE_PORT=${TEMPLATE_REPOSITORY_PORT:-3615}
if curl -s http://localhost:$TEMPLATE_PORT/health > /dev/null; then
    echo "✅ Template Repository ($TEMPLATE_PORT) - Ready"
else
    echo "❌ Template Repository ($TEMPLATE_PORT) - Not ready"
fi

# Check available AI models
echo ""
echo "🤖 Available AI Models:"
curl -s http://localhost:$FREE_AI_PORT/models | jq '.models' 2>/dev/null || echo "   Could not fetch model list"

echo ""
echo "🎉 StateX AI Free AI Services Setup Complete!"
echo ""
echo "📋 Available AI Services:"
echo "  • AI Orchestrator: http://localhost:$AI_ORCHESTRATOR_PORT"
echo "  • Free AI Service: http://localhost:$FREE_AI_PORT"
echo "  • NLP Service: http://localhost:$NLP_PORT"
echo "  • ASR Service: http://localhost:$ASR_PORT"
echo "  • Document AI: http://localhost:$DOC_AI_PORT"
echo "  • Prototype Generator: http://localhost:$PROTO_GEN_PORT"
echo "  • Template Repository: http://localhost:$TEMPLATE_PORT"
OLLAMA_PORT=${OLLAMA_PORT:-11434}
echo "  • Ollama Service: http://localhost:$OLLAMA_PORT"
echo ""
echo "🔧 To test the AI services:"
echo "  curl http://localhost:$FREE_AI_PORT/health"
echo "  curl http://localhost:$FREE_AI_PORT/models"
echo ""
echo "🧪 To test real AI agents performance:"
echo "  python3 test_real_ai_agents.py --demo"
echo ""
echo "📊 To view service status:"
echo "  docker compose ps"
echo ""
echo "🛑 To stop services:"
echo "  docker compose down"
