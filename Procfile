# Flight Delay Predictor API - Procfile for Render/Heroku Deployment
# Format: process_type: command
# 
# Render uses 'web' process type for HTTP services
# This command will be executed in the deployed container

# Primary web service
web: gunicorn -c gunicorn_conf.py main:app
