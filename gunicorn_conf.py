"""
Gunicorn Configuration for Flight Delay Predictor API

This configuration file optimizes Gunicorn for production deployment with:
- Dynamic worker scaling based on CPU cores
- Uvicorn workers for async/await support
- Request timeout handling
- Graceful shutdown
- Comprehensive logging

Worker sizing:
  For optimal performance with async ML predictions:
  workers = (2 × CPU cores) + 1
  This balances concurrency and resource usage.

Timeout considerations:
  ML model inference + external API calls can take 10-15 seconds
  Set timeout to 60s to allow sufficient time for predictions

Reference: https://docs.gunicorn.org/en/stable/settings.html
"""

import multiprocessing
import os
import logging

# ============================================================================
# Basic Server Configuration
# ============================================================================

# Port binding - defaults to 8000, can be overridden via PORT env var
port = int(os.getenv("PORT", 8000))
bind = f"0.0.0.0:{port}"

# Server socket backlog - queue for pending connections
backlog = 2048

# ============================================================================
# Worker Configuration
# ============================================================================

# Worker class: UvicornWorker for async FastAPI support
worker_class = "uvicorn.workers.UvicornWorker"

# Worker processes - formula: (2 × CPU cores) + 1
# For production servers with 4+ cores: typically 9-17 workers
# Adjust down for memory-constrained environments
workers = multiprocessing.cpu_count() * 2 + 1

# Worker connections - max simultaneous connections per worker
worker_connections = 1000

# Max requests per worker before restart (memory leak prevention)
max_requests = 1000

# Jitter to prevent thundering herd on max_requests
max_requests_jitter = 50

# ============================================================================
# Timeout Configuration
# ============================================================================

# Request timeout in seconds
# Critical: Must accommodate:
# - ML model inference (~5-10s)
# - External API calls (5-8s each)
# - Total: 60s provides safe margin
timeout = 60

# Graceful shutdown timeout
graceful_timeout = 30

# Keep-alive timeout for persistent connections
keepalive = 2

# ============================================================================
# Logging Configuration
# ============================================================================

# Log level
loglevel = os.getenv("LOG_LEVEL", "info")

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Log to stdout for container/cloud deployment
accesslog = "-"
errorlog = "-"

# Capture stdout in logs
capture_output = True

# ============================================================================
# Performance Tuning
# ============================================================================

# Pre-fork vs async tradeoffs
# Using Uvicorn workers for async/await support in FastAPI
# Pre-fork mode: max throughput but higher memory
# Async mode: lower memory, good for many concurrent connections

# Daemon mode (off in containers)
daemon = False

# Process naming for monitoring
proc_name = "flight-delay-predictor"

# ============================================================================
# Application Configuration
# ============================================================================

# Server hooks for monitoring/initialization
def on_starting(server):
    """Called when Gunicorn server starts."""
    logging.info(f"🚀 Gunicorn starting with {workers} workers")
    logging.info(f"🔗 Listening on {bind}")


def when_ready(server):
    """Called when Gunicorn is ready to accept requests."""
    logging.info("✅ Gunicorn ready, accepting requests")


def on_exit(server):
    """Called when Gunicorn exits gracefully."""
    logging.info("👋 Gunicorn shutting down gracefully")


# ============================================================================
# Security Configuration
# ============================================================================

# Limit request header size to prevent attacks
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# ============================================================================
# Notes for Production Deployment
# ============================================================================
# 
# Environment Variables:
#   PORT: Port to bind (default: 8000)
#   LOG_LEVEL: Logging level (debug, info, warning, error, critical)
#
# Docker Deployment:
#   CMD ["gunicorn", "-c", "gunicorn_conf.py", "main:app"]
#
# Cloud Platforms (Render, Heroku, AWS):
#   - Workers auto-scale based on available CPU
#   - Set PORT env var for platform-specific port
#   - Use managed Redis for caching layer
#
# Monitoring:
#   Track metrics: requests/sec, avg response time, worker utilization
#   Use tools: DataDog, New Relic, CloudWatch, etc.
#
# Troubleshooting:
#   - High 502/503 errors → Increase workers or timeout
#   - Memory growth → Enable max_requests restart
#   - Slow requests → Check external API performance
#
