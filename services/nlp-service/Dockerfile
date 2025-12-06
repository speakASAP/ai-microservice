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

# Download NLTK data
# Download wordnet manually first (before importing nltk) to avoid import-triggered loading issues
RUN python -c "import urllib.request, zipfile, os; os.makedirs('/root/nltk_data/corpora', exist_ok=True); urllib.request.urlretrieve('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip', '/tmp/wordnet.zip'); zipfile.ZipFile('/tmp/wordnet.zip').extractall('/root/nltk_data/corpora/'); os.remove('/tmp/wordnet.zip')"
# Then download other NLTK data packages
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True)"

# Copy application code with correct ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port (internal port)
EXPOSE 3381

# Health check (uses internal port that service listens on)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3381/health || exit 1

# Run the application (service listens on NLP_SERVICE_PORT - 3381, configured in ai-microservice/.env)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3381"]
