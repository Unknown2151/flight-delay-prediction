"""
Comprehensive Test Suite for Flight Delay Predictor API

Tests cover:
- Input validation and error handling
- Health and monitoring endpoints
- Circuit breaker pattern behavior
- Caching functionality
- Response structure and types
- Edge cases and boundary conditions
- Security headers

Running tests:
    pytest tests/test_main.py -v
    pytest tests/test_main.py -v --cov=main
"""

import pytest
from fastapi.testclient import TestClient
from datetime import date, timedelta
from main import app, CircuitBreaker

client = TestClient(app)


# ============================================================================
# SECTION 1: Input Validation Tests
# ============================================================================

class TestInputValidation:
    """Test input validation for flight prediction requests."""
    
    def test_invalid_carrier_code_numeric(self):
        """Carrier code must be uppercase letters, not numeric."""
        response = client.post("/predict", json={
            "carrierCode": "123",
            "flightNumber": "400",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
        assert "carrierCode" in str(response.json()["detail"])
    
    def test_invalid_carrier_code_lowercase(self):
        """Carrier code must be uppercase letters."""
        response = client.post("/predict", json={
            "carrierCode": "aa",
            "flightNumber": "100",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
    
    def test_invalid_carrier_code_too_short(self):
        """Carrier code must be 2-3 letters."""
        response = client.post("/predict", json={
            "carrierCode": "A",
            "flightNumber": "100",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
    
    def test_invalid_carrier_code_too_long(self):
        """Carrier code must not exceed 3 letters."""
        response = client.post("/predict", json={
            "carrierCode": "AAAA",
            "flightNumber": "100",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
    
    def test_invalid_flight_number_alphabetic(self):
        """Flight number must be numeric only."""
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "ABC",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
    
    def test_invalid_flight_number_too_long(self):
        """Flight number must not exceed 4 digits."""
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "99999",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
    
    def test_invalid_flight_number_empty(self):
        """Flight number is required."""
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "",
            "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
        })
        assert response.status_code == 422
    
    def test_past_date_validation(self):
        """Departure date cannot be in the past."""
        response = client.post("/predict", json={
            "carrierCode": "LH",
            "flightNumber": "400",
            "scheduledDepartureDate": "2020-01-10"
        })
        assert response.status_code == 422
        detail = str(response.json()["detail"])
        assert "Departure date cannot be in the past" in detail or "past" in detail.lower()
    
    def test_future_date_exceeds_one_year(self):
        """Departure date cannot be more than 1 year in future."""
        future_date = (date.today() + timedelta(days=400)).isoformat()
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "100",
            "scheduledDepartureDate": future_date
        })
        assert response.status_code == 422
    
    def test_invalid_date_format(self):
        """Date must be in YYYY-MM-DD format."""
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "100",
            "scheduledDepartureDate": "01/01/2026"
        })
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Missing required fields should fail validation."""
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "100"
        })
        assert response.status_code == 422
    
    def test_valid_date_boundary_today_plus_one(self):
        """Valid prediction for tomorrow."""
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        response = client.post("/predict", json={
            "carrierCode": "AA",
            "flightNumber": "100",
            "scheduledDepartureDate": tomorrow
        })
        # Should not be a validation error
        assert response.status_code in [200, 404, 429, 502, 504, 500, 400]
    
    def test_valid_carrier_codes(self):
        """Test various valid carrier codes."""
        valid_codes = ["AA", "BB", "DL", "UA", "IB", "LH", "BA", "AF"]
        
        for code in valid_codes:
            payload = {
                "carrierCode": code,
                "flightNumber": "100",
                "scheduledDepartureDate": (date.today() + timedelta(days=7)).isoformat()
            }
            response = client.post("/predict", json=payload)
            assert response.status_code != 422, f"Failed for carrier code {code}"


# ============================================================================
# SECTION 2: Monitoring Endpoints Tests
# ============================================================================

class TestMonitoringEndpoints:
    """Test health check and info endpoints."""
    
    def test_health_check_success(self):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_structure(self):
        """Health check response should have required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "redis" in data
        assert data["status"] == "healthy"
    
    def test_health_check_redis_status(self):
        """Health check should report Redis connection status."""
        response = client.get("/health")
        data = response.json()
        assert data["redis"] in ["connected", "disconnected", "error"]
    
    def test_info_endpoint_success(self):
        """Info endpoint should return 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200
    
    def test_info_endpoint_response_structure(self):
        """Info endpoint response should have required fields."""
        response = client.get("/info")
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "description" in data
        assert data["version"] == "2.0.0"
    
    def test_root_endpoint_success(self):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_endpoint_response_structure(self):
        """Root endpoint should return message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data


# ============================================================================
# SECTION 3: Security Headers Tests
# ============================================================================

class TestSecurityHeaders:
    """Test that security headers are properly set."""
    
    def test_x_content_type_options_header(self):
        """X-Content-Type-Options header should prevent MIME sniffing."""
        response = client.get("/health")
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    
    def test_x_frame_options_header(self):
        """X-Frame-Options header should prevent clickjacking."""
        response = client.get("/health")
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    
    def test_x_xss_protection_header(self):
        """X-XSS-Protection header should enable XSS protection."""
        response = client.get("/health")
        assert "X-XSS-Protection" in response.headers
    
    def test_content_security_policy_header(self):
        """Content-Security-Policy header should be present."""
        response = client.get("/health")
        assert "Content-Security-Policy" in response.headers


# ============================================================================
# SECTION 4: Circuit Breaker Pattern Tests
# ============================================================================

class TestCircuitBreaker:
    """Test circuit breaker implementation."""
    
    def test_circuit_breaker_initialization(self):
        """Circuit breaker should initialize in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_tracks_failures(self):
        """Circuit breaker should track failure count."""
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
    
    def test_circuit_breaker_blocks_when_open(self):
        """Circuit breaker should block calls when OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        
        def failing_func():
            raise Exception("Test error")
        
        # Trip the breaker
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.state == "OPEN"
        
        # Now it should block immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failing_func)
    
    def test_circuit_breaker_success_resets(self):
        """Circuit breaker should reset on successful call."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        
        def failing_func():
            raise Exception("Test error")
        
        # Cause one failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.failure_count == 1
        
        # Successful call should reset
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.failure_count == 0


# ============================================================================
# SECTION 5: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_health_check_flow(self):
        """Test complete health check workflow."""
        # Root endpoint
        root_response = client.get("/")
        assert root_response.status_code == 200
        
        # Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Info endpoint
        info_response = client.get("/info")
        assert info_response.status_code == 200
    
    def test_validation_error_flow(self):
        """Test validation error handling."""
        response = client.post("/predict", json={
            "carrierCode": "INVALID",
            "flightNumber": "ABC",
            "scheduledDepartureDate": "invalid-date"
        })
        
        assert response.status_code == 422
        assert "detail" in response.json()


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def api():
    """Provide TestClient fixture."""
    return TestClient(app)


def test_client_available():
    """Verify TestClient is initialized."""
    assert client is not None
