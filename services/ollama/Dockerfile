# Ollama server for ai-microservice (built by compose; deploy.sh → prepare-green-smart.sh builds all services sequentially).
# Keep base image bumps intentional; pin digest in production if required by policy.
FROM ollama/ollama:latest

# Bind API on all interfaces so LiteLLM and other containers on nginx-network can reach :11434.
ENV OLLAMA_HOST=0.0.0.0:11434

EXPOSE 11434
