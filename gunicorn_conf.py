import multiprocessing
import os

# Port configuration
port = os.getenv("PORT", "8000")
bind = f"0.0.0.0:{port}"

# Worker configuration
# Rule of thumb: (2 x number of cores) + 1
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class: Use the high-performance Uvicorn worker
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout: Set to 60s because ML models and external APIs can be slow
timeout = 60

# Logging
accesslog = "-" # Log to stdout
errorlog = "-"  # Log to stderr