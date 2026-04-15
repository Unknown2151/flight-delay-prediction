# Multi-stage build for production-grade Flight Delay Predictor API
# Stage 1: Build stage for dependency installation
FROM python:3.11-slim as builder

LABEL maintainer="Flight Delay Prediction Team"
LABEL description="Flight Delay Predictor API - Production Docker Image"

# Install system dependencies with minimal bloat
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /tmp

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Verify critical packages are installed
RUN python -c "import redis, fastapi, gunicorn, pandas, lightgbm; print('✅ Critical dependencies verified')"


# Stage 2: Runtime stage - minimal image
FROM python:3.11-slim

LABEL maintainer="Flight Delay Prediction Team"
LABEL version="2.0.0"
LABEL description="Flight Delay Predictor API with Redis caching and circuit breaker patterns"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security (principle of least privilege)
RUN useradd -m -u 1000 -s /sbin/nologin appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application files
COPY main.py /app/
COPY config.py /app/
COPY gunicorn_conf.py /app/
COPY artifacts /app/artifacts/

# Set proper file permissions
RUN chown -R appuser:appuser /app
RUN chmod -R 755 /app

# Switch to non-root user
USER appuser

# Add Python local packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

EXPOSE 8000

# Health check - ensures container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production-grade command using gunicorn with Uvicorn workers
CMD ["gunicorn", \
     "-c", "gunicorn_conf.py", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "main:app"]

