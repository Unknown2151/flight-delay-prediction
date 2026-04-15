"""
Smoke tests for the Flight Delay Predictor API.

Validates that monitoring endpoints (/health, /info) are operational.

Configuration:
    API_BASE_URL: Base URL of the API (default: http://127.0.0.1:8000)
"""

import os
import requests
import time


# Configurable base URL — no hardcoded localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def test_endpoints():
    """Run smoke tests against monitoring endpoints."""
    # Wait a few seconds for the server to start up in the background
    time.sleep(5)

    # Test 1: Health Check
    print(f"Testing {API_BASE_URL}/health...")
    r = requests.get(f"{API_BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # Test 2: Info Endpoint
    print(f"Testing {API_BASE_URL}/info...")
    r = requests.get(f"{API_BASE_URL}/info")
    assert r.status_code == 200
    assert "version" in r.json()

    print("All monitoring endpoints passed!")


if __name__ == "__main__":
    test_endpoints()