FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5000

# Create user and set working directory
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only first (for better Docker layer caching)
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements-docker.txt requirements.txt

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --timeout=1000 \
    || (echo "pip install failed" && exit 1)

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/cache /app/uploads \
    && chown -R appuser:appuser /app \
    && chmod 755 /app/logs /app/data /app/cache /app/uploads

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 5000

# Health check (updated for new port)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]
