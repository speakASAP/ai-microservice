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
# Install NumPy from source so it matches build host CPU (avoids RuntimeError: NumPy was built with
# baseline optimizations X86_V2 but your machine doesn't support X86_V2 when building on older VMs/emulators)
RUN pip install --no-cache-dir "numpy>=1.26.0" --no-binary numpy
# Keep --no-binary numpy so pip does not overwrite with a wheel when installing nltk/spacy deps
RUN pip install --no-cache-dir --no-binary numpy -r requirements.txt

# Download NLTK data manually (no "import nltk" at build time to avoid NumPy X86_V2 wheel load)
ENV NLTK_DATA=/app/nltk_data
RUN python -c "\
import urllib.request, zipfile, os; \
base='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages'; \
dest='/app/nltk_data'; \
pkgs=[('corpora/wordnet', 'wordnet'), ('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords'), ('sentiment/vader_lexicon', 'vader_lexicon')]; \
[(os.makedirs(os.path.join(dest, os.path.dirname(p)), exist_ok=True), urllib.request.urlretrieve(f'{base}/{p}.zip', f'/tmp/{n}.zip'), zipfile.ZipFile(f'/tmp/{n}.zip').extractall(os.path.join(dest, os.path.dirname(p))), os.remove(f'/tmp/{n}.zip')) for p, n in pkgs]\
" && chown -R appuser:appuser /app/nltk_data

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
