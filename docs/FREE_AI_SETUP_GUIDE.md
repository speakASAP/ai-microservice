# StateX Free AI Services Setup Guide

This guide explains how to set up and use the free AI services for development in the StateX platform.

## Overview

The StateX platform now supports multiple free AI services for development:

1. **Ollama** - Local AI models (Llama 2, Mistral, CodeLlama)
2. **HuggingFace** - Free API access to various models
3. **Local Whisper** - Free speech-to-text processing
4. **Mock AI** - Testing and development fallback

## Quick Start

### 1. Start Infrastructure Services

```bash
# Start infrastructure with Ollama
cd statex-infrastructure
docker compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
docker compose -f docker-compose.dev.yml logs -f ollama
```

### 2. Set up Ollama Models

```bash
# Run the Ollama setup script
./scripts/setup-ollama.sh

# Or manually pull models
docker exec statex_ollama_dev ollama pull llama2:7b
docker exec statex_ollama_dev ollama pull mistral:7b
docker exec statex_ollama_dev ollama pull codellama:7b
```

### 3. Configure Environment

```bash
# Copy environment template
cd statex-ai
cp .env.example .env

# Edit configuration
nano .env
```

Set these variables for free AI services:

```env
# Use free AI services only
AI_MODE=free
ASR_MODE=free

# Optional: HuggingFace API key for better rate limits
HUGGINGFACE_API_KEY=your_free_key_here

# Keep these empty for development
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### 4. Start AI Services

```bash
# Start all AI services
cd statex-ai
docker compose up -d

# Check service health
curl http://localhost:${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}/health  # Free AI Service
curl http://localhost:${ASR_SERVICE_EXTERNAL_PORT:-3612}/health  # ASR Service
```

## Service Details

### Free AI Service (Port 8016)

**Features:**

- Ollama integration with local models
- HuggingFace API fallback
- Automatic model selection based on task type
- Comprehensive error handling and fallbacks

**Available Models:**

- `llama2:7b` - General purpose analysis
- `mistral:7b` - Efficient analysis and coding
- `codellama:7b` - Technical and code analysis

**API Endpoints:**

- `GET /health` - Service health and provider status
- `GET /models` - Available models and providers
- `POST /analyze` - AI analysis with automatic fallback

### ASR Service (Port 8012)

**Features:**

- Local Whisper model for free transcription
- OpenAI Whisper API fallback (if configured)
- Multiple audio format support
- Voice quality analysis

**Supported Formats:**

- WAV, MP3, M4A, FLAC
- Max file size: 25MB
- Max duration: 25 minutes

**API Endpoints:**

- `GET /health` - Service health and provider status
- `GET /api/providers` - Available ASR providers
- `POST /api/transcribe` - Audio transcription
- `POST /api/upload-audio` - Audio file upload

### Ollama Service (Port 11434)

**Features:**

- Local AI model hosting
- No API keys required
- Privacy-focused (all processing local)
- Multiple model support

**Models Included:**

- Llama 2 7B (1.4GB download)
- Mistral 7B (4.1GB download)
- CodeLlama 7B (3.8GB download)

## Testing

### Test All Services

```bash
# Test Ollama integration
cd statex-ai
python test_ollama_integration.py

# Test HuggingFace integration
python test_huggingface_integration.py

# Test ASR integration
python test_asr_integration.py
```

### Test Individual Components

```bash
# Test Ollama directly
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b",
    "prompt": "Analyze this business request: I want to create a restaurant app",
    "stream": false
  }'

# Test Free AI Service
curl -X POST http://localhost:${FREE_AI_SERVICE_EXTERNAL_PORT:-3616}/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "I want to create a restaurant management system",
    "analysis_type": "business_analysis",
    "user_name": "Test User"
  }'
```

## Configuration Options

### AI_MODE Settings

- `free` - Use only free services (Ollama, local Whisper, HuggingFace free tier)
- `paid` - Use paid services (OpenAI, Anthropic) with free fallbacks
- `hybrid` - Prefer free services, fallback to paid if needed

### ASR_MODE Settings

- `free` - Use local Whisper model only
- `paid` - Use OpenAI Whisper API with local fallback
- `hybrid` - Prefer local Whisper, fallback to OpenAI API

## Troubleshooting

### Ollama Issues

```bash
# Check Ollama status
docker logs statex_ollama_dev

# Restart Ollama
docker restart statex_ollama_dev

# Check available models
curl http://localhost:11434/api/tags
```

### Model Download Issues

```bash
# Check disk space (models are large)
df -h

# Manually pull models
docker exec -it statex_ollama_dev ollama pull llama2:7b

# Check model status
docker exec -it statex_ollama_dev ollama list
```

### Memory Issues

```bash
# Check memory usage
docker stats

# Reduce concurrent models (edit docker-compose.yml)
# Add memory limits to Ollama service:
deploy:
  resources:
    limits:
      memory: 4G
```

### HuggingFace Rate Limits

```bash
# Get free API key from https://huggingface.co/settings/tokens
# Add to .env file:
HUGGINGFACE_API_KEY=hf_your_key_here

# Check rate limit status
curl -H "Authorization: Bearer $HUGGINGFACE_API_KEY" \
  https://api-inference.huggingface.co/models/gpt2
```

## Performance Tips

### Optimize Ollama

1. **Use appropriate model sizes:**
   - Development: 7B models (faster, less memory)
   - Production: 13B+ models (better quality, more memory)

2. **Adjust model parameters:**

   ```json
   {
     "temperature": 0.7,
     "top_p": 0.9,
     "num_predict": 500
   }
   ```

3. **Enable GPU acceleration (if available):**

   ```bash
   # Install NVIDIA Docker support
   # Add to docker-compose.yml:
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

### Optimize ASR

1. **Use appropriate audio formats:**
   - Best: WAV (uncompressed)
   - Good: FLAC (lossless)
   - Acceptable: MP3 (compressed)

2. **Optimize audio quality:**
   - Sample rate: 16kHz or 44.1kHz
   - Bit depth: 16-bit minimum
   - Mono audio for speech

## Cost Comparison

| Service | Free Tier | Paid Alternative | Cost Savings |
|---------|-----------|------------------|--------------|
| Ollama | Unlimited local | OpenAI GPT-4 | $0.03/1K tokens |
| Local Whisper | Unlimited | OpenAI Whisper | $0.006/minute |
| HuggingFace | 1000 requests/month | Various APIs | Varies |

## Next Steps

1. **Production Setup:** Configure paid services for production use
3. **Scaling:** Implement load balancing for multiple Ollama instances
4. **Fine-tuning:** Train custom models for specific business domains

## Support

- **Documentation:** Check service logs for detailed error messages
- **Community:** Ollama GitHub repository and Discord
- **Issues:** Report bugs in the StateX repository
