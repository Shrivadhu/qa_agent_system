FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# Pre-download models for faster startup if network is available
RUN python -c "try:\n    from sentence_transformers import SentenceTransformer\n    SentenceTransformer('all-MiniLM-L6-v2')\n    from transformers import pipeline\n    pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')\nexcept Exception as e:\n    print(f'Model pre-download notice (fallback mode ready): {e}')"

# Copy application code
COPY . .

# Create non-root user and setup directories with proper permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/.cache/huggingface && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# Run the application
CMD ["python", "main.py", "server"]
