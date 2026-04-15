"""
Locust Load Testing Suite for Flight Delay Predictor API

This module provides comprehensive load testing for the Flight Delay Prediction API,
testing both happy paths and error scenarios to ensure >80% success rate under load.

Features:
    - Multiple user types simulating different traffic patterns
    - Externalized flight test data (test_flights.json)
    - Circuit breaker testing
    - Cache effectiveness monitoring
    - Performance metrics collection
    - Success rate tracking

Configuration (via environment variables):
    LOCUST_HOST: Target API URL (default: http://localhost:8000)
    TARGET_SUCCESS_RATE: Success rate threshold (default: 80%)
    LOCUST_TEST_FLIGHTS_PATH: Path to flight test data JSON (default: test_flights.json)
    SLOW_REQUEST_THRESHOLD_MS: Threshold for logging slow requests (default: 10000ms)

Usage:
    # Headless mode with specified users and duration
    locust -f locustfile.py -u 100 -r 10 --run-time 5m --headless

    # Web UI mode (default)
    locust -f locustfile.py

    # With custom host
    LOCUST_HOST=https://api.example.com locust -f locustfile.py

Author: Flight Delay Prediction Team
Version: 2.0.0
"""

import random
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import date, timedelta
from locust import HttpUser, task, between, events, TaskSet

# ============================================================================
# Configuration — All configurable via environment variables
# ============================================================================

DEFAULT_HOST = os.getenv("LOCUST_HOST", "http://localhost:8000")
TARGET_SUCCESS_RATE = float(os.getenv("TARGET_SUCCESS_RATE", 80))
TEST_FLIGHTS_PATH = os.getenv("LOCUST_TEST_FLIGHTS_PATH", "test_flights.json")
SLOW_REQUEST_THRESHOLD_MS = int(os.getenv("SLOW_REQUEST_THRESHOLD_MS", 10000))
PREDICT_TIMEOUT = int(os.getenv("LOCUST_PREDICT_TIMEOUT", 30))
INVALID_TIMEOUT = int(os.getenv("LOCUST_INVALID_TIMEOUT", 10))
RESILIENCE_TIMEOUT = int(os.getenv("LOCUST_RESILIENCE_TIMEOUT", 60))

# ============================================================================
# Test Flight Data — Loaded from external JSON file
# ============================================================================

# Default fallback flights if the JSON file is missing
_DEFAULT_FLIGHTS: List[Dict[str, str]] = [
    {"carrierCode": "IB", "flightNumber": "532"},
    {"carrierCode": "BA", "flightNumber": "490"},
    {"carrierCode": "LH", "flightNumber": "400"},
    {"carrierCode": "AA", "flightNumber": "100"},
    {"carrierCode": "DL", "flightNumber": "1"},
]


def _load_test_flights(file_path: str) -> List[Dict[str, str]]:
    """
    Load flight test data from an external JSON file.

    The JSON file should have the structure:
    {
        "flights": [
            {"carrierCode": "AA", "flightNumber": "100"},
            ...
        ]
    }

    Args:
        file_path: Path to the JSON file containing flight data.

    Returns:
        List of flight dictionaries with carrierCode and flightNumber.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_absolute():
        # Resolve relative to this file's directory
        resolved_path = Path(__file__).parent / file_path

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            flights = data.get("flights", [])
            if not flights:
                raise ValueError("'flights' array is empty")
            logging.info(f"✅ Loaded {len(flights)} test flights from {resolved_path}")
            return flights
    except FileNotFoundError:
        logging.warning(
            f"⚠️  Flight test data file not found: {resolved_path}. "
            f"Using {len(_DEFAULT_FLIGHTS)} built-in defaults. "
            f"Create '{file_path}' to customize test flights."
        )
        return _DEFAULT_FLIGHTS
    except (json.JSONDecodeError, ValueError) as e:
        logging.warning(f"⚠️  Failed to parse {resolved_path}: {e}. Using defaults.")
        return _DEFAULT_FLIGHTS


KNOWN_FLIGHTS: List[Dict[str, str]] = _load_test_flights(TEST_FLIGHTS_PATH)

# Logging
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Collection & Tracking
# ============================================================================

class SuccessRateTracker:
    """
    Tracks API success rate and reports progress.

    Attributes:
        total_requests: Total requests sent
        successful_requests: Requests that completed successfully
        failed_requests: Requests that failed
    """

    def __init__(self):
        """Initialize the tracker."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    def record_request(self, success: bool) -> None:
        """
        Record a request outcome.

        Args:
            success: Whether the request was successful
        """
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    @property
    def success_rate(self) -> float:
        """
        Calculate current success rate as percentage.

        Returns:
            float: Success rate (0-100%)
        """
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def __str__(self) -> str:
        """String representation of success rate."""
        return (f"Success Rate: {self.success_rate:.2f}% "
                f"({self.successful_requests}/{self.total_requests})")


# Global tracker instance
success_tracker = SuccessRateTracker()


# ============================================================================
# Base User Classes
# ============================================================================

class FlightApiBaseSimulation(TaskSet):
    """
    Base task set containing shared behaviors for all user types.
    """

    def _generate_future_date(self, min_days: int = 1, max_days: int = 14) -> str:
        """
        Generate a random future date within acceptable range.

        Args:
            min_days: Minimum days in future
            max_days: Maximum days in future

        Returns:
            str: ISO format date string
        """
        future_date = date.today() + timedelta(days=random.randint(min_days, max_days))
        return future_date.isoformat()

    def _create_valid_prediction_payload(self) -> Dict[str, str]:
        """
        Create a valid flight prediction request.

        Returns:
            Dict: Valid prediction payload
        """
        flight = random.choice(KNOWN_FLIGHTS)
        return {
            "carrierCode": flight["carrierCode"],
            "flightNumber": flight["flightNumber"],
            "scheduledDepartureDate": self._generate_future_date(),
        }


# ============================================================================
# Nominal User Simulations
# ============================================================================

class NormalFlightApiUser(FlightApiBaseSimulation):
    """
    Simulates normal user behavior - mostly valid predictions with some monitoring.
    This is the primary load test user type.

    Task distribution:
    - 70%: Valid predictions
    - 15%: Health checks
    - 10%: Info endpoint
    - 5%: Invalid requests
    """

    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Called when user starts."""
        logger.info(f"User {self.user.client_id} starting")

    @task(70)
    def predict_valid_flight(self) -> None:
        """
        Predict delay for a valid flight (primary operation).

        This task performs successful API calls that should result in
        200 responses with valid predictions.
        """
        payload = self._create_valid_prediction_payload()

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [valid]",
            catch_response=True,
            timeout=PREDICT_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                success_tracker.record_request(True)
                response.success()
            elif response.status_code >= 500:
                success_tracker.record_request(False)
                response.failure(f"Server error: {response.status_code}")
            else:
                success_tracker.record_request(False)
                response.failure(f"Unexpected status: {response.status_code}")

    @task(15)
    def health_check(self) -> None:
        """
        Check API health (monitoring endpoint).

        Should consistently return 200 with healthy status.
        """
        self.client.get("/health", name="/health")

    @task(10)
    def get_api_info(self) -> None:
        """
        Retrieve API information and version.

        Should return 200 with API metadata.
        """
        self.client.get("/info", name="/info")

    @task(5)
    def test_invalid_request(self) -> None:
        """
        Test error handling with intentionally invalid requests.

        These should return 422 (validation error) which is expected
        and counts as successful error handling.
        """
        choice = random.random()

        if choice < 0.33:
            # Invalid carrier code
            payload = {
                "carrierCode": "123",
                "flightNumber": "100",
                "scheduledDepartureDate": self._generate_future_date(),
            }
        elif choice < 0.66:
            # Invalid flight number
            payload = {
                "carrierCode": "AA",
                "flightNumber": "ABCD",
                "scheduledDepartureDate": self._generate_future_date(),
            }
        else:
            # Past date
            payload = {
                "carrierCode": "AA",
                "flightNumber": "100",
                "scheduledDepartureDate": (date.today() - timedelta(days=1)).isoformat(),
            }

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [invalid request]",
            catch_response=True,
            timeout=INVALID_TIMEOUT,
        ) as response:
            if response.status_code == 422:
                # 422 is expected for invalid requests - counts as success
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class CacheOptimizationUser(FlightApiBaseSimulation):
    """
    Simulates user behavior that optimizes cache hit rates.

    Strategy: Repeatedly query the same flights to maximize cache hits
    and tests caching layer effectiveness.

    Task distribution:
    - 80%: Repeated predictions (cache hits)
    - 20%: Health/info checks
    """

    wait_time = between(0.5, 2)

    def __init__(self, environment):
        """Initialize with a set of "favorite" flights."""
        super().__init__(environment)
        # Select up to 5 random flights to repeatedly query
        sample_size = min(5, len(KNOWN_FLIGHTS))
        self.favorite_flights = random.sample(KNOWN_FLIGHTS, sample_size)
        self.flight_index = 0

    @task(80)
    def predict_cached_flight(self) -> None:
        """
        Predict delay for frequently-queried flights (cache test).

        By querying the same flights repeatedly, we test:
        - Cache hit efficiency
        - Redis connection stability
        - Cache expiration handling
        """
        # Cycle through favorite flights
        flight = self.favorite_flights[self.flight_index % len(self.favorite_flights)]
        self.flight_index += 1

        # Use same date to maximize cache hits
        payload = {
            "carrierCode": flight["carrierCode"],
            "flightNumber": flight["flightNumber"],
            "scheduledDepartureDate": self._generate_future_date(min_days=5, max_days=7),
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [cached]",
            catch_response=True,
            timeout=PREDICT_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                # Check if response was cached
                resp_data = response.json()
                is_cached = resp_data.get("is_cached", False)
                logger.debug(f"Cache hit: {is_cached}")
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(20)
    def monitoring_endpoints(self) -> None:
        """Monitor API health while stress-testing cache."""
        endpoints = ["/health", "/info"]
        self.client.get(random.choice(endpoints))


class CircuitBreakerTestUser(FlightApiBaseSimulation):
    """
    Simulates user behavior during API degradation to test circuit breaker.

    Tests that the API gracefully degrades when external services fail,
    returning predictions with fallback data instead of 500 errors.
    """

    wait_time = between(2, 5)

    @task(100)
    def predict_with_potential_fallback(self) -> None:
        """
        Predict flight delays (tests graceful degradation).

        If external APIs (Amadeus, Tomorrow.io) fail, the API should:
        - Return 200 (not 500)
        - Provide prediction with default data
        - Indicate fallback in response note
        - Maintain >80% success rate
        """
        payload = self._create_valid_prediction_payload()

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [resilience]",
            catch_response=True,
            timeout=RESILIENCE_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                success_tracker.record_request(True)
                resp_data = response.json()
                # Check if prediction included fallback note
                note = resp_data.get("note", "")
                if note:
                    logger.debug(f"Fallback used: {note}")
                response.success()
            elif response.status_code >= 500:
                success_tracker.record_request(False)
                response.failure(f"Server error: {response.status_code}")
            else:
                success_tracker.record_request(False)
                response.failure(f"Status: {response.status_code}")


# ============================================================================
# Primary HTTP User Configuration
# ============================================================================

class FlightApiUser(HttpUser):
    """
    Main HTTP User for load testing the Flight Delay Predictor API.

    This user profile represents a typical application instance making
    predictions and monitoring the API health.
    """

    # Task set - contains all test behaviors
    tasks = [NormalFlightApiUser]

    # Wait time between tasks (1-5 seconds)
    wait_time = between(1, 5)

    def on_start(self) -> None:
        """Called when user starts - set base URL."""
        if not self.client.base_url or self.client.base_url == "http://localhost":
            self.client.base_url = DEFAULT_HOST
            logger.info(f"Set base URL: {self.client.base_url}")


# Optional: Additional user types can be defined here
# class CacheOptimizationTestUser(HttpUser):
#     tasks = [CacheOptimizationUser]
#     wait_time = between(0.5, 2)
#
# class CircuitBreakerTestHttpUser(HttpUser):
#     tasks = [CircuitBreakerTestUser]
#     wait_time = between(2, 5)


# ============================================================================
# Event Handlers & Monitoring
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """
    Called when test starts - initialize monitoring.

    Args:
        environment: Locust environment
    """
    print("\n" + "=" * 70)
    print("🚀 Flight Delay Predictor API - Load Test Started")
    print("=" * 70)
    print(f"Target API: {DEFAULT_HOST}")
    print(f"Target Success Rate: {TARGET_SUCCESS_RATE}%")
    print(f"Test Flights Loaded: {len(KNOWN_FLIGHTS)}")
    print(f"Start Time: {date.today().isoformat()}")
    print("=" * 70 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """
    Called when test stops - generate final report.

    Args:
        environment: Locust environment
    """
    print("\n" + "=" * 70)
    print("✅ Load Test Completed")
    print("=" * 70)

    stats = environment.stats

    # Summary statistics
    print(f"\nTotal Requests: {stats.total.num_requests:,}")
    print(f"Failed Requests: {stats.total.num_failures:,}")
    print(f"Success Rate: {success_tracker.success_rate:.2f}%")

    # Response time metrics
    print(f"\nResponse Times:")
    print(f"  Average: {stats.total.avg_response_time:.0f}ms")
    print(f"  50th percentile: {stats.total.get_response_time_percentile(0.50):.0f}ms")
    print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"  Max: {stats.total.max_response_time:.0f}ms")

    # Throughput metrics
    rps = stats.total.total_rps
    print(f"\nThroughput: {rps:.2f} req/sec")

    # Success rate evaluation
    print("\n" + "-" * 70)
    if success_tracker.success_rate >= TARGET_SUCCESS_RATE:
        print(f"✅ TARGET MET: Success rate {success_tracker.success_rate:.2f}% >= {TARGET_SUCCESS_RATE}%")
    else:
        print(f"❌ TARGET MISSED: Success rate {success_tracker.success_rate:.2f}% < {TARGET_SUCCESS_RATE}%")
    print("-" * 70 + "\n")

    # Per-endpoint breakdown
    print("Per-Endpoint Breakdown:")
    for name, stats_item in environment.stats.entries.items():
        if stats_item.num_requests > 0:
            failure_rate = (stats_item.num_failures / stats_item.num_requests) * 100
            print(f"  {name}: {stats_item.num_requests} requests, "
                  f"{failure_rate:.1f}% failures, "
                  f"{stats_item.avg_response_time:.0f}ms avg response")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, success, **kwargs) -> None:
    """
    Called after each request - logs slow requests.

    Args:
        request_type: HTTP method
        name: Request name/endpoint
        response_time: Response time in ms
        response_length: Response body length
        success: Whether request succeeded
    """
    if response_time > SLOW_REQUEST_THRESHOLD_MS:
        logger.warning(f"Slow request: {name} took {response_time:.0f}ms")
