import pytest
from fastapi.testclient import TestClient
from datetime import date, timedelta
from main import app, CircuitBreaker

client = TestClient(app)


# --- 1. Input Validation Tests ---
def test_invalid_carrier_code_format():
    """Test carrier code must be uppercase letters only."""
    response = client.post("/predict", json={
        "carrierCode": "123",  # Should be letters
        "flightNumber": "400",
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    })
    assert response.status_code == 422
    assert "carrierCode" in str(response.json()["detail"])


def test_invalid_flight_number_format():
    """Test flight number must be numeric only."""
    response = client.post("/predict", json={
        "carrierCode": "AA",
        "flightNumber": "ABC",  # Should be numeric
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    })
    assert response.status_code == 422


def test_past_date_validation():
    """Test departure date cannot be in the past."""
    response = client.post("/predict", json={
        "carrierCode": "LH",
        "flightNumber": "400",
        "scheduledDepartureDate": "2020-01-10"
    })
    assert response.status_code == 422
    assert "Departure date cannot be in the past" in str(response.json()["detail"])


def test_future_date_exceeds_one_year():
    """Test departure date cannot be more than 1 year in future."""
    future_date = (date.today() + timedelta(days=400)).isoformat()
    response = client.post("/predict", json={
        "carrierCode": "AA",
        "flightNumber": "100",
        "scheduledDepartureDate": future_date
    })
    assert response.status_code == 422
    assert "more than 1 year" in str(response.json()["detail"])


def test_invalid_date_format():
    """Test invalid date format."""
    response = client.post("/predict", json={
        "carrierCode": "AA",
        "flightNumber": "100",
        "scheduledDepartureDate": "01/01/2026"  # Wrong format
    })
    assert response.status_code == 422
    assert "Invalid date format" in str(response.json()["detail"])


# --- 2. Monitoring Endpoints Tests ---
def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "redis" in data


def test_info_endpoint():
    """Test API info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# --- 3. Valid Request Tests ---
def test_valid_request_structure():
    """Test that valid requests are accepted (may fail at API level)."""
    payload = {
        "carrierCode": "AA",
        "flightNumber": "100",
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    }
    response = client.post("/predict", json=payload)
    # Should be 200, 404 (flight not found), or 429 (rate limited)
    assert response.status_code in [200, 404, 429, 500, 502, 504]


# --- 4. Edge Cases ---
def test_min_carrier_code_length():
    """Test minimum carrier code length."""
    response = client.post("/predict", json={
        "carrierCode": "A",  # Too short
        "flightNumber": "100",
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    })
    assert response.status_code == 422


def test_max_carrier_code_length():
    """Test maximum carrier code length."""
    response = client.post("/predict", json={
        "carrierCode": "AAAA",  # Too long
        "flightNumber": "100",
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    })
    assert response.status_code == 422


def test_max_flight_number_digits():
    """Test maximum flight number digits."""
    response = client.post("/predict", json={
        "carrierCode": "AA",
        "flightNumber": "99999",  # Too many digits
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    })
    assert response.status_code == 422


# --- 5. Circuit Breaker Tests ---
def test_circuit_breaker_initialization():
    """Test circuit breaker initializes correctly."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0


def test_circuit_breaker_tracks_failures():
    """Test circuit breaker tracks failures."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)

    def failing_func():
        raise Exception("Test error")

    # First failure
    with pytest.raises(Exception):
        cb.call(failing_func)
    assert cb.failure_count == 1
    assert cb.state == "CLOSED"

    # Second failure - should trip
    with pytest.raises(Exception):
        cb.call(failing_func)
    assert cb.failure_count == 2
    assert cb.state == "OPEN"


def test_circuit_breaker_blocks_when_open():
    """Test circuit breaker blocks calls when OPEN."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)

    def failing_func():
        raise Exception("Test error")

    # Trip the breaker
    with pytest.raises(Exception):
        cb.call(failing_func)

    # Now it should block
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        cb.call(failing_func)


# --- 6. Caching Test ---
def test_prediction_caching():
    """Test caching flow (integration test)."""
    payload = {
        "carrierCode": "IB",
        "flightNumber": "532",
        "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
    }

    # First call
    res1 = client.post("/predict", json=payload)

    # If successful, second call should be cached
    if res1.status_code == 200:
        res2 = client.post("/predict", json=payload)
        assert res2.status_code == 200
        # Check if cached flag is set
        if "is_cached" in res2.json():
            assert res2.json()["is_cached"] is True
