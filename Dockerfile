# AI Microservice Root Dockerfile
# Kubernetes-optimized multi-service builder
# Builds ai-orchestrator as the primary K8s-deployed service
# Includes shared modules at /app/shared for dynamic imports

FROM python:3.11-slim

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from ai-orchestrator and install Python dependencies
COPY services/ai-orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules to /app/shared (exact path needed for importlib.util)
# This allows dynamic imports via sys.path manipulation in ai-orchestrator/app/main.py
COPY shared/ ./shared/

# Copy ai-orchestrator application code
COPY services/ai-orchestrator/app/ ./app/

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Set ownership of all application code to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for ai-orchestrator service
EXPOSE 3380

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3380/health || exit 1

# Run the application
# Service listens on port 3380 (AI_ORCHESTRATOR_PORT from .env)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3380"]
