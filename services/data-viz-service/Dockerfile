FROM python:3.11-slim

WORKDIR /app

# Install system deps + build deps for numpy (for CPU compatibility on older hosts)
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Rebuild numpy from source so it matches the build host CPU (fixes X86_V2 on older prod servers)
RUN pip install --no-cache-dir --no-binary numpy numpy

# Copy application code
COPY app/ ./app/

# Create logs directory
RUN mkdir -p /app/logs

# Expose port (service listens on DATA_VIZ_SERVICE_PORT - 3389, configured in ai-microservice/.env)
EXPOSE 3389

# Health check (uses port that service listens on)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3389/health || exit 1

# Run the application
CMD ["python", "app/main.py"]
