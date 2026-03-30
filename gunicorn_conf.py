import os
import logging

# ============================================================================      
# Basic Server Configuration
# ============================================================================      

port = int(os.getenv("PORT", 10000))
bind = f"0.0.0.0:{port}"
worker_class = "uvicorn.workers.UvicornWorker"

# ============================================================================      
# Worker Configuration (Optimized for 512MB RAM)
# ============================================================================      

# Use Render's env var, but force a hard cap of 1 for memory safety
workers = int(os.getenv("WEB_CONCURRENCY", 1))
if workers > 1:
    workers = 1

# Use threads for light concurrency without loading the ML model multiple times
threads = 2 
timeout = 120
keepalive = 5

# ============================================================================      
# Logging & Hooks
# ============================================================================      

accesslog = "-"
errorlog = "-"
loglevel = "info"

def on_starting(server):
    logging.info(f"🚀 Gunicorn starting | Workers: {workers} | Port: {port}")

def when_ready(server):
    logging.info("✅ Flight Delay API is ready to accept requests")