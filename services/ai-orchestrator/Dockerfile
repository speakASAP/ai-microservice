# AI Orchestrator Dockerfile
# OPTIMIZED FOR HYBRID DEVELOPMENT APPROACH
# This Dockerfile is used with volume mounts for instant startup and hot reload
# For production, use the same Dockerfile but with different volume configuration
# Recommended to run with --cpus='1.0' and --memory='512m' to prevent resource overuse

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

# Copy requirements and install Python dependencies
COPY --chown=appuser:appuser statex-ai/services/ai-orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared resources (needed for http_clients, etc.)
# Note: Build context is now root (.), so paths are relative to root
COPY --chown=appuser:appuser statex-ai/shared /app/shared
COPY --chown=appuser:appuser statex-platform/shared /app/statex-platform/shared
COPY --chown=appuser:appuser utils /app/utils

# Copy application code with correct ownership
COPY --chown=appuser:appuser statex-ai/services/ai-orchestrator /app

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Expose port (internal port)
EXPOSE 8010

# Health check (uses internal port that service listens on)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Run the application
# NOTE: This container is designed to work with volume mounts
# The actual source code will be mounted from the host for instant hot reload
# Service listens on AI_ORCHESTRATOR_INTERNAL_PORT (8010)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
