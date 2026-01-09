# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
#
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./main.py /app/
COPY ./artifacts /app/artifacts/

# Optimized command for Render/Railway using Gunicorn
# -w 4: Spawns 4 worker processes
# -k uvicorn.workers.UvicornWorker: High-performance async workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]