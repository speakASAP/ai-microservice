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
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check (service listens on AI_WORKERS_PORT - 3387, configured in ai-microservice/.env)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3387/health || exit 1

# Expose port (service listens on AI_WORKERS_PORT - 3387)
EXPOSE 3387

# Start application (service listens on AI_WORKERS_PORT - 3387)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3387"]
