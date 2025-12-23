import requests
import time


def test_endpoints():
    # Wait a few seconds for the server to start up in the background
    time.sleep(5)

    base_url = "http://127.0.0.1:8000"

    # Test 1: Health Check
    print("Testing /health...")
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # Test 2: Info Endpoint
    print("Testing /info...")
    r = requests.get(f"{base_url}/info")
    assert r.status_code == 200
    assert "version" in r.json()

    print("All monitoring endpoints passed!")


if __name__ == "__main__":
    test_endpoints()