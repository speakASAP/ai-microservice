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

# Expose port (service listens on FREE_AI_SERVICE_PORT - 3386, configured in ai-microservice/.env)
EXPOSE 3386

# Health check (service listens on FREE_AI_SERVICE_PORT - 3386)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3386/health || exit 1

# Run the application (service listens on FREE_AI_SERVICE_PORT - 3386)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3386"]
