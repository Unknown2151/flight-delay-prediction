import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming your file is named main.py

client = TestClient(app)


# 1. Test Input Validation (Should Fail)
def test_invalid_carrier_code():
    response = client.post("/predict", json={
        "carrierCode": "123",  # Should be uppercase letters
        "flightNumber": "400",
        "scheduledDepartureDate": "2026-01-10"
    })
    assert response.status_code == 422
    assert "carrierCode" in response.json()["detail"][0]["loc"]


def test_past_date_validation():
    response = client.post("/predict", json={
        "carrierCode": "LH",
        "flightNumber": "400",
        "scheduledDepartureDate": "2020-01-10"  # Date in the past
    })
    assert response.status_code == 422
    assert "Departure date cannot be in the past" in response.json()["detail"][0]["msg"]


# 2. Test Monitoring Endpoints (Should Pass)
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# 3. Test Caching Flow (Integration)
def test_prediction_caching():
    payload = {
        "carrierCode": "IB",
        "flightNumber": "532",
        "scheduledDepartureDate": "2026-01-11"
    }

    # First call - Cold Start
    res1 = client.post("/predict", json=payload)

    # If Amadeus sandbox is working, check cache
    if res1.status_code == 200:
        # Second call - Should hit Redis
        res2 = client.post("/predict", json=payload)
        assert res2.status_code == 200
        assert res2.json()["is_cached"] is True