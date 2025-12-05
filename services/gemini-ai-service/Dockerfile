FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create logs directory
RUN mkdir -p /app/logs

# Expose port (service listens on GEMINI_AI_SERVICE_PORT - 3388, configured in ai-microservice/.env)
EXPOSE 3388

# Health check (uses port that service listens on)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3388/health || exit 1

# Run the application
CMD ["python", "app/main.py"]
