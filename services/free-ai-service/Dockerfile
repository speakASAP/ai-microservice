FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port (service listens on FREE_AI_SERVICE_EXTERNAL_PORT - 8016)
EXPOSE 8016

# Health check (service listens on FREE_AI_SERVICE_EXTERNAL_PORT - 8016)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8016/health || exit 1

# Run the application (service listens on FREE_AI_SERVICE_EXTERNAL_PORT - 8016)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8016"]
