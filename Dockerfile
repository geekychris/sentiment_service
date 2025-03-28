# Multi-stage build Dockerfile for Entity Sentiment Analysis
# Stage 1: Base image with Python
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    NLTK_DATA=/usr/share/nltk_data

# Stage 2: Build dependencies
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 3: Final image
FROM base

# Copy built wheels from builder stage
COPY --from=builder /app/wheels /wheels

# Install dependencies
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Create NLTK data directory with proper permissions
RUN mkdir -p /usr/share/nltk_data && \
    chmod -R 777 /usr/share/nltk_data

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')"

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0"]

# Default command (can be overridden)
CMD ["--workers", "1"]

