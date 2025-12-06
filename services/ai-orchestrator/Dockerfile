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
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with correct ownership
# Note: Shared resources are mounted as volumes at runtime (./shared:/app/shared:ro)
COPY --chown=appuser:appuser . .

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Expose port (internal port)
EXPOSE 3380

# Health check (uses port that service listens on)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3380/health || exit 1

# Run the application
# NOTE: This container is designed to work with volume mounts
# The actual source code will be mounted from the host for instant hot reload
# Service listens on AI_ORCHESTRATOR_PORT (3380, configured in ai-microservice/.env)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3380"]
