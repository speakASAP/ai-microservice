# Ollama server for ai-microservice (built by compose; deploy.sh → prepare-green-smart.sh builds all services sequentially).
# Keep base image bumps intentional; pin digest in production if required by policy.
FROM ollama/ollama:latest

# Bind API on all interfaces so LiteLLM and other containers on nginx-network can reach :11435.
# Port 11434 is reserved for the host systemd ollama.service.
ENV OLLAMA_HOST=0.0.0.0:11435

EXPOSE 11435
