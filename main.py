"""
Flight Delay Prediction API - FastAPI Application

This module implements a production-ready REST API for predicting flight delays
using machine learning. It integrates with the Amadeus flight API for live data
and Tomorrow.io for weather information, with Redis caching and circuit breaker
patterns for resilience.

Key Features:
    - Real-time flight delay predictions using LightGBM models
    - Circuit breaker pattern for fault tolerance
    - Redis caching (30-min TTL) for high-frequency queries
    - HTTP connection pooling and retry strategies
    - Comprehensive validation and error handling
    - Full async/await support with FastAPI
    - CORS middleware for web applications
    - Sensitive data masking in logs
    - Health checks and monitoring endpoints

Environment Variables:
    - AMADEUS_API_KEY: Amadeus API credentials
    - AMADEUS_API_SECRET: Amadeus API credentials
    - TOMORROW_API_KEY: Tomorrow.io weather API key
    - REDIS_URL: Redis connection string (default: redis://localhost:6379)

Author: Flight Delay Prediction Team
Version: 2.0.0
"""

import os
import joblib
import pandas as pd
import requests
import logging
import traceback
import json
import redis
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError
from dotenv import load_dotenv


# ============================================================================
# SECTION 1: Configuration & Environment Setup
# ============================================================================

load_dotenv()

# Environment variables
AMADEUS_API_KEY: str = os.getenv("AMADEUS_API_KEY", "")
AMADEUS_API_SECRET: str = os.getenv("AMADEUS_API_SECRET", "")
WEATHER_API_KEY: str = os.getenv("TOMORROW_API_KEY", "")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Constants
DEFAULT_WEATHER: Dict[str, float] = {
    'temperature': 15.0,
    'windSpeed': 5.0,
    'precipitationIntensity': 0.0
}
DEFAULT_DISTANCE: float = 1000.0
CACHE_TTL_SECONDS: int = 1800  # 30 minutes
MODEL_PATH: str = 'artifacts/flight_delay_pipeline.pkl'
AIRPORT_DATA_URL: str = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'


# ============================================================================
# SECTION 2: Logging & Security Setup
# ============================================================================

class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that prevents sensitive API keys and credentials
    from being written to logs.
    
    Attributes:
        sensitive_keys: List of sensitive strings to mask in log records
    """

    def __init__(self, sensitive_keys: List[str]) -> None:
        """
        Initialize the filter with sensitive keys to mask.
        
        Args:
            sensitive_keys: List of strings (API keys, passwords) to mask
        """
        super().__init__()
        self.sensitive_keys = [k for k in sensitive_keys if k]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to mask sensitive data.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            bool: True to allow log record through
        """
        message = record.getMessage()
        for key in self.sensitive_keys:
            if key in message:
                record.msg = message.replace(key, "********")
        return True


# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FlightAPI")

# Apply sensitive data filter
logger.addFilter(SensitiveDataFilter([AMADEUS_API_KEY, AMADEUS_API_SECRET, WEATHER_API_KEY]))

logger.info(f"Application started in {ENVIRONMENT} environment")


# ============================================================================
# SECTION 3: Circuit Breaker Pattern Implementation
# ============================================================================

class CircuitBreaker:
    """
    Simple circuit breaker implementation to prevent cascading failures
    when external APIs are unavailable.
    
    States:
        CLOSED: Normal operation, requests are processed
        OPEN: Too many failures, requests are rejected immediately
        HALF_OPEN: Testing if service recovered, limited requests allowed
    
    Attributes:
        failure_count: Number of consecutive failures
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        last_failure_time: Timestamp of last failure
        state: Current circuit state (CLOSED, OPEN, HALF_OPEN)
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60) -> None:
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger OPEN state
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_count: int = 0
        self.failure_threshold: int = failure_threshold
        self.recovery_timeout: int = recovery_timeout
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "CLOSED"

    def call(self, func, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Callable to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func execution
            
        Raises:
            Exception: If circuit is OPEN or func raises
        """
        if self.state == "OPEN":
            if datetime.now() > self.last_failure_time + timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self) -> None:
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        if self.state != "CLOSED":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED")

    def on_failure(self) -> None:
        """Record failure and update circuit state."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        logger.warning(f"Circuit breaker failure #{self.failure_count}")
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")


# Initialize circuit breakers
amadeus_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
weather_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


# ============================================================================
# SECTION 4: Redis Cache Configuration
# ============================================================================

cache: Optional[redis.Redis] = None

try:
    cache = redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        health_check_interval=30
    )
    cache.ping()
    logger.info(f"✅ Connected to Redis successfully: {REDIS_URL}")
except Exception as e:
    logger.warning(f"⚠️  Redis connection failed: {e}. Caching disabled.")
    cache = None


# ============================================================================
# SECTION 5: Resource Loading (Global)
# ============================================================================

airport_coords_dict: Dict[str, Dict[str, float]] = {}

try:
    cols = ['ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Lat', 'Lon', 'Alt', 'TZ', 'DST', 'TzDB', 'Type', 'Source']
    df_airports = pd.read_csv(AIRPORT_DATA_URL, header=None, names=cols)
    df_airports = df_airports.drop_duplicates(subset=['IATA'], keep='first').set_index('IATA')
    airport_coords_dict = df_airports[['Lat', 'Lon']].to_dict('index')
    logger.info(f"✅ Airport coordinates loaded: {len(airport_coords_dict)} airports")
except Exception as e:
    logger.error(f"❌ Failed to load airport data: {e}")
    logger.warning("Distance calculations will use defaults")

# Load ML model
model_pipeline = None
EXPECTED_FEATURE_ORDER = None

try:
    model_pipeline = joblib.load(MODEL_PATH)
    EXPECTED_FEATURE_ORDER = model_pipeline.feature_names_in_
    logger.info(f"✅ ML Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.critical(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Critical: Model initialization failed - {e}")


# ============================================================================
# SECTION 6: HTTP Session Configuration
# ============================================================================

@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    """
    Create and cache an HTTP session with retry strategy.
    
    Implements:
    - Connection pooling for efficient resource usage
    - Exponential backoff for rate-limited endpoints
    - Retry on transient failures (5xx, 429 errors)
    
    Returns:
        requests.Session: Configured session with retry strategy
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=False
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.timeout = 10
    return session


http_session = get_http_session()


# ============================================================================
# SECTION 7: Utility Functions
# ============================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        
    Returns:
        float: Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371  # Earth radius in km


def get_amadeus_access_token() -> str:
    """
    Fetch Amadeus API access token with circuit breaker protection.
    
    Returns:
        str: OAuth2 access token
        
    Raises:
        Exception: If circuit breaker is OPEN or API call fails
    """
    def fetch_token() -> str:
        """Helper function to fetch token."""
        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": AMADEUS_API_KEY,
            "client_secret": AMADEUS_API_SECRET
        }
        response = http_session.post(token_url, data=data, timeout=8)
        response.raise_for_status()
        logger.debug("✅ Amadeus token fetched successfully")
        return response.json()["access_token"]

    return amadeus_breaker.call(fetch_token)


# ============================================================================
# SECTION 8: Pydantic Models & Input Validation
# ============================================================================

class FlightInput(BaseModel):
    """
    Input validation model for flight delay prediction requests.
    
    Attributes:
        carrierCode: 2-3 letter airline IATA code (e.g., 'AA', 'LH')
        flightNumber: Numeric flight identifier
        scheduledDepartureDate: Departure date in YYYY-MM-DD format
    """
    
    carrierCode: str = Field(
        ...,
        min_length=2,
        max_length=3,
        pattern=r"^[A-Z]+$",
        description="Airline IATA code (e.g., AA, LH)"
    )
    flightNumber: str = Field(
        ...,
        pattern=r"^\d{1,4}$",
        description="Flight number (numeric, 1-4 digits)"
    )
    scheduledDepartureDate: str = Field(
        ...,
        description="Departure date in YYYY-MM-DD format"
    )

    @field_validator("scheduledDepartureDate")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """
        Validate departure date is valid and within acceptable range.
        
        Args:
            v: Date string to validate
            
        Returns:
            str: Validated date string
            
        Raises:
            ValueError: If date is invalid or outside acceptable range
        """
        try:
            if isinstance(v, str):
                parsed_date = date.fromisoformat(v)
            else:
                parsed_date = v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

        today = date.today()
        
        if parsed_date < today:
            raise ValueError("Departure date cannot be in the past.")

        # Prevent predictions too far in the future (>1 year)
        if parsed_date > today + timedelta(days=365):
            raise ValueError("Departure date cannot be more than 1 year in the future.")

        return v


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    redis: str


class InfoResponse(BaseModel):
    """API info response model."""
    app_name: str
    version: str
    description: str
    environment: str


# ============================================================================
# SECTION 9: FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Flight Delay Predictor API",
    description="Enterprise-grade ML API for predicting US airline flight delays with Redis caching and circuit breaker patterns",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Add security headers to all responses.
    
    Implemented Headers:
        - X-Content-Type-Options: Prevent MIME type sniffing
        - X-Frame-Options: Prevent clickjacking
        - X-XSS-Protection: Enable XSS protection
        - Strict-Transport-Security: Enforce HTTPS
        - Content-Security-Policy: Prevent XSS attacks
        - Referrer-Policy: Control referrer information
        - Permissions-Policy: Control sensitive features
    
    Args:
        request: FastAPI request object
        call_next: Next middleware in chain
        
    Returns:
        Response with security headers
    """
    response = await call_next(request)
    
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Prevent clickjacking attacks
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    
    # Enable XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Enforce HTTPS (only in production)
    if ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'"
    )
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions policy
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), "
        "camera=(), "
        "geolocation=(), "
        "gyroscope=(), "
        "magnetometer=(), "
        "microphone=(), "
        "payment=(), "
        "usb=()"
    )
    
    return response



# ============================================================================
# SECTION 10: Global Exception Handlers
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors with detailed error information.
    
    Args:
        request: FastAPI request object
        exc: ValidationError exception
        
    Returns:
        JSONResponse: 422 response with error details
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": [
                {
                    "field": ".".join(str(x) for x in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"]
                }
                for err in exc.errors()
            ]
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions gracefully.
    
    Args:
        request: FastAPI request object
        exc: Exception that occurred
        
    Returns:
        JSONResponse: 500 response with error timestamp
    """
    logger.error(f"Unexpected error: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# SECTION 11: Monitoring & Health Check Endpoints
# ============================================================================

@app.get("/", tags=["Public"], summary="Root endpoint")
async def read_root() -> Dict[str, str]:
    """Root endpoint - API is operational."""
    return {"message": "Flight Delay Prediction API is Online"}


@app.get("/info", tags=["Monitoring"], summary="API information", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get API metadata and version information.
    
    Returns:
        InfoResponse: API name, version, and description
    """
    return InfoResponse(
        app_name="Flight Delay Prediction API",
        version="2.0.0",
        description="Predicts if your flight will be delayed",
        environment=ENVIRONMENT
    )


@app.get("/health", tags=["Monitoring"], summary="Health check", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Checks Redis connectivity and overall system health.
    
    Returns:
        HealthResponse: System health status and Redis connection status
    """
    redis_status = "disconnected"
    try:
        if cache and cache.ping():
            redis_status = "connected"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        redis_status = "error"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        redis=redis_status
    )


# ============================================================================
# SECTION 12: Main Prediction Endpoint
# ============================================================================

@app.post("/predict", tags=["ML Prediction"], summary="Predict flight delay")
async def predict_delay(flight_input: FlightInput) -> Dict[str, Any]:
    """
    Predict whether a flight will be delayed.
    
    This endpoint:
    1. Checks Redis cache for previous predictions
    2. Fetches live flight data from Amadeus API
    3. Retrieves weather data from Tomorrow.io
    4. Calculates distance using Haversine formula
    5. Performs ML prediction with feature engineering
    6. Caches result for 30 minutes
    
    Args:
        flight_input: Flight details (carrier code, flight number, date)
        
    Returns:
        Dict containing prediction details, weather, distance, and confidence
        
    Raises:
        HTTPException: If validation fails or service error occurs
    """

    # --- Step 1: Cache Check ---
    cache_key = f"predict:{flight_input.carrierCode}:{flight_input.flightNumber}:{flight_input.scheduledDepartureDate}"
    
    if cache:
        try:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"✅ CACHE HIT: {cache_key}")
                result = json.loads(cached_data)
                result["is_cached"] = True
                return result
        except Exception as cache_err:
            logger.warning(f"⚠️  Cache retrieval failed: {cache_err}")

    logger.info(f"🔄 CACHE MISS: Processing {flight_input.carrierCode}{flight_input.flightNumber}")

    # --- Step 2: Fetch Data ---
    try:
        # Get Amadeus token with circuit breaker
        access_token = get_amadeus_access_token()

        # Amadeus Flight Schedule API Call
        flight_status_url = "https://test.api.amadeus.com/v2/schedule/flights"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = flight_input.model_dump()

        api_result = http_session.get(flight_status_url, headers=headers, params=params, timeout=8)

        # Handle 401 (expired token) by retrying with fresh token
        if api_result.status_code == 401:
            logger.warning("⚠️  Amadeus token expired, fetching fresh token...")
            access_token = get_amadeus_access_token()
            headers = {"Authorization": f"Bearer {access_token}"}
            api_result = http_session.get(flight_status_url, headers=headers, params=params, timeout=8)

        api_result.raise_for_status()
        flight_response = api_result.json()

        # Graceful fallback if flight not found
        if not flight_response.get("data"):
            logger.warning(f"⚠️  Flight not found in Amadeus: {flight_input.carrierCode}{flight_input.flightNumber}")
            airline = flight_input.carrierCode
            origin_airport = "JFK"
            destination_airport = "LAX"
            scheduled_departure_str = f"{flight_input.scheduledDepartureDate}T08:00:00"
        else:
            flight_data = flight_response["data"][0]
            airline = flight_data["flightDesignator"]["carrierCode"]
            origin_airport = flight_data["flightPoints"][0]["iataCode"]
            destination_airport = flight_data["flightPoints"][1]["iataCode"]
            scheduled_departure_str = flight_data["flightPoints"][0]["departure"]["timings"][0]["value"]

        # --- Step 2b: Fetch Weather Data (with circuit breaker) ---
        current_weather = DEFAULT_WEATHER.copy()
        try:
            def fetch_weather() -> Dict:
                """Helper function to fetch weather data."""
                weather_params = {
                    "location": origin_airport,
                    "fields": ["temperature", "windSpeed", "precipitationIntensity"],
                    "units": "metric",
                    "timesteps": "current",
                    "apikey": WEATHER_API_KEY
                }
                w_res = http_session.get(
                    "https://api.tomorrow.io/v4/weather/realtime",
                    params=weather_params,
                    timeout=5
                )
                w_res.raise_for_status()
                return w_res.json()['data']['values']

            current_weather = weather_breaker.call(fetch_weather)
        except Exception as we:
            logger.warning(f"⚠️  Weather fetch failed: {we}")

        # --- Step 2c: Calculate Distance ---
        distance_km = DEFAULT_DISTANCE
        try:
            if origin_airport in airport_coords_dict and destination_airport in airport_coords_dict:
                o_c = airport_coords_dict[origin_airport]
                d_c = airport_coords_dict[destination_airport]
                distance_km = haversine(o_c['Lat'], o_c['Lon'], d_c['Lat'], d_c['Lon'])
        except Exception as dist_err:
            logger.warning(f"⚠️  Distance calculation failed: {dist_err}")

        # --- Step 3: Feature Engineering ---
        dt = datetime.fromisoformat(scheduled_departure_str)
        features = {
            'MKT_UNIQUE_CARRIER': airline,
            'ORIGIN': origin_airport,
            'DEST': destination_airport,
            'DISTANCE': distance_km,
            'MONTH': dt.month,
            'DAY_OF_WEEK': dt.weekday(),
            'DEPT_HOUR': dt.hour,
            'tavg': current_weather.get('temperature', 15.0),
            'prcp': current_weather.get('precipitationIntensity', 0.0),
            'wspd': current_weather.get('windSpeed', 5.0)
        }

        # --- Step 4: ML Prediction ---
        features_df = pd.DataFrame([features])
        features_df = features_df[EXPECTED_FEATURE_ORDER]

        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = int(model_pipeline.predict(features_df)[0])

        final_response = {
            "flight_details_requested": flight_input.model_dump(),
            "live_weather_at_origin": current_weather,
            "calculated_distance_km": round(distance_km, 2),
            "predicted_delay_status": prediction,
            "predicted_delay_probability": f"{prediction_proba:.2%}",
            "is_cached": False,
            "timestamp": datetime.now().isoformat()
        }

        # --- Step 5: Save to Cache (30 mins TTL) ---
        if cache:
            try:
                cache.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(final_response))
            except Exception as cache_err:
                logger.warning(f"⚠️  Cache write failed: {cache_err}")

        return final_response

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        logger.warning("⏱️  External API request timed out")
        return _get_default_prediction(flight_input, "Prediction based on defaults due to external service timeout")
    except requests.exceptions.ConnectionError:
        logger.warning("🔌 Connection error with external APIs")
        return _get_default_prediction(flight_input, "Prediction based on defaults due to connection error")
    except requests.exceptions.HTTPError as he:
        if he.response.status_code == 429:
            logger.error("🚫 Rate limit exceeded on external API")
            raise HTTPException(status_code=429, detail="Service is temporarily busy. Please try again later.")
        logger.warning(f"⚠️  HTTP error from external API: {he}")
        return _get_default_prediction(flight_input, "Prediction based on defaults due to external API error")
    except Exception as e:
        logger.error(f"❌ Prediction pipeline error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")


def _get_default_prediction(flight_input: FlightInput, note: str) -> Dict[str, Any]:
    """
    Generate prediction using default values when external services fail.
    
    Args:
        flight_input: Original flight input
        note: Note explaining why defaults were used
        
    Returns:
        Dict: Prediction using default feature values
    """
    features = {
        'MKT_UNIQUE_CARRIER': flight_input.carrierCode,
        'ORIGIN': 'JFK',
        'DEST': 'LAX',
        'DISTANCE': DEFAULT_DISTANCE,
        'MONTH': date.fromisoformat(flight_input.scheduledDepartureDate).month,
        'DAY_OF_WEEK': date.fromisoformat(flight_input.scheduledDepartureDate).weekday(),
        'DEPT_HOUR': 8,
        'tavg': 15.0,
        'prcp': 0.0,
        'wspd': 5.0
    }
    features_df = pd.DataFrame([features])
    features_df = features_df[EXPECTED_FEATURE_ORDER]
    prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
    prediction = int(model_pipeline.predict(features_df)[0])

    return {
        "flight_details_requested": flight_input.model_dump(),
        "live_weather_at_origin": DEFAULT_WEATHER,
        "calculated_distance_km": round(DEFAULT_DISTANCE, 2),
        "predicted_delay_status": prediction,
        "predicted_delay_probability": f"{prediction_proba:.2%}",
        "is_cached": False,
        "timestamp": datetime.now().isoformat(),
        "note": note
    }


# ============================================================================
# SECTION 13: Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower()
    )




# --- 4. Circuit Breakers for External APIs ---
amadeus_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
weather_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# --- 5. Resource Loading (Global) ---
DEFAULT_WEATHER = {'temperature': 15.0, 'windSpeed': 5.0, 'precipitationIntensity': 0.0}
DEFAULT_DISTANCE = 1000.0

airport_coords_dict = {}
try:
    url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    cols = ['ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Lat', 'Lon', 'Alt', 'TZ', 'DST', 'TzDB', 'Type', 'Source']
    df_airports = pd.read_csv(url, header=None, names=cols)
    df_airports = df_airports.drop_duplicates(subset=['IATA'], keep='first').set_index('IATA')
    airport_coords_dict = df_airports[['Lat', 'Lon']].to_dict('index')
    logger.info(f"Airport coordinates loaded: {len(airport_coords_dict)} airports.")
except Exception as e:
    logger.error(f"Failed to load airport data: {e}. Distance calculations will use defaults.")

try:
    model_pipeline = joblib.load('artifacts/flight_delay_pipeline.pkl')
    EXPECTED_FEATURE_ORDER = model_pipeline.feature_names_in_
    logger.info("ML Model loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load model: {e}")
    raise RuntimeError(f"Critical: Model initialization failed - {e}")

# --- 6. HTTP Session with Retry Strategy ---
@lru_cache(maxsize=1)
def get_http_session():
    """Create an HTTP session with retry strategy for connection pooling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,  # Reduced from 3 to avoid excessive retries
        backoff_factor=0.5,  # Reduced from 1 to speed up retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.timeout = 10  # Global timeout default
    return session

http_session = get_http_session()


# --- 7. Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def get_amadeus_access_token():
    """Get Amadeus API token with circuit breaker protection."""
    def fetch_token():
        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {"grant_type": "client_credentials", "client_id": AMADEUS_API_KEY, "client_secret": AMADEUS_API_SECRET}
        response = http_session.post(token_url, data=data, timeout=8)
        response.raise_for_status()
        return response.json()["access_token"]

    return amadeus_breaker.call(fetch_token)


# --- 8. Data Models & Validation ---
class FlightInput(BaseModel):
    carrierCode: str = Field(..., min_length=2, max_length=3, pattern=r"^[A-Z]+$",
                             description="Airline code (e.g., AA)")
    flightNumber: str = Field(..., pattern=r"^\d{1,4}$", description="Flight number digits only")
    scheduledDepartureDate: str = Field(..., description="Date in YYYY-MM-DD")

    @field_validator("scheduledDepartureDate")
    @classmethod
    def validate_date(cls, v):
        try:
            if isinstance(v, str):
                parsed_date = date.fromisoformat(v)
            else:
                parsed_date = v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

        today = date.today()
        if parsed_date < today:
            raise ValueError("Departure date cannot be in the past.")

        # Prevent predictions too far in the future (>1 year)
        if parsed_date > today + timedelta(days=365):
            raise ValueError("Departure date cannot be more than 1 year in the future.")

        return v


# --- 9. FastAPI Setup with Middleware ---
app = FastAPI(
    title="Flight Delay Predictor API",
    description="An API to predict flight delays based on carrier and schedule.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- 10. Global Exception Handlers ---
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with clear messages."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": [
                {
                    "field": ".".join(str(x) for x in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"]
                }
                for err in exc.errors()
            ]
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions gracefully."""
    logger.error(f"Unexpected error: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.get("/info")
async def get_info():
    return {
        "app_name": "Flight Delay Predictor",
        "version": "2.0.0",
        "description": "Predicts if your flight will be delayed."
    }

@app.get("/", tags=["Public"])
def read_root():
    return {"message": "Flight Delay Prediction API is Online"}


@app.get("/health", tags=["Monitoring"])
def health_check():
    redis_status = "disconnected"
    try:
        if cache and cache.ping():
            redis_status = "connected"
    except Exception:
        redis_status = "error"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": redis_status
    }


# --- 7. Main Prediction Endpoint ---
@app.post("/predict", tags=["ML Prediction"])
async def predict_delay(flight_input: FlightInput) -> Dict[str, Any]:
    """
    Predicts delay status for a specific flight.
    - **carrierCode**: Airline code (e.g., AA, LH, DL)
    - **flightNumber**: Flight number (e.g., 123)
    - **scheduledDepartureDate**: Departure date in YYYY-MM-DD format
    """

    # --- Step 1: Cache Check ---
    cache_key = f"predict:{flight_input.carrierCode}:{flight_input.flightNumber}:{flight_input.scheduledDepartureDate}"
    if cache:
        try:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"CACHE HIT: {cache_key}")
                result = json.loads(cached_data)
                result["is_cached"] = True
                return result
        except Exception as cache_err:
            logger.warning(f"Cache retrieval failed: {cache_err}. Proceeding without cache.")

    logger.info(f"CACHE MISS: Processing {flight_input.carrierCode}{flight_input.flightNumber}")

    # --- Step 2: Fetch Data ---
    try:
        # Get Amadeus token with circuit breaker
        access_token = get_amadeus_access_token()

        # Amadeus Flight Schedule API Call
        flight_status_url = "https://test.api.amadeus.com/v2/schedule/flights"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = flight_input.model_dump()

        api_result = http_session.get(flight_status_url, headers=headers, params=params, timeout=8)

        # Handle 401 (expired token) by retrying once with a fresh token
        if api_result.status_code == 401:
            logger.warning("Amadeus token expired, fetching fresh token...")
            access_token = get_amadeus_access_token()
            headers = {"Authorization": f"Bearer {access_token}"}
            api_result = http_session.get(flight_status_url, headers=headers, params=params, timeout=8)

        api_result.raise_for_status()
        flight_response = api_result.json()

        # --- Graceful fallback if flight not found in Amadeus ---
        if not flight_response.get("data"):
            logger.warning(f"Flight not found in Amadeus: {flight_input.carrierCode}{flight_input.flightNumber}. Using fallback defaults.")
            airline = flight_input.carrierCode
            origin_airport = "JFK"  # Default fallback
            destination_airport = "LAX"  # Default fallback
            scheduled_departure_str = f"{flight_input.scheduledDepartureDate}T08:00:00"
        else:
            flight_data = flight_response["data"][0]
            airline = flight_data["flightDesignator"]["carrierCode"]
            origin_airport = flight_data["flightPoints"][0]["iataCode"]
            destination_airport = flight_data["flightPoints"][1]["iataCode"]
            scheduled_departure_str = flight_data["flightPoints"][0]["departure"]["timings"][0]["value"]

        # --- Step 2b: Fetch Weather Data (with circuit breaker) ---
        current_weather = DEFAULT_WEATHER.copy()
        try:
            def fetch_weather():
                weather_params = {
                    "location": origin_airport,
                    "fields": ["temperature", "windSpeed", "precipitationIntensity"],
                    "units": "metric",
                    "timesteps": "current",
                    "apikey": WEATHER_API_KEY
                }
                w_res = http_session.get("https://api.tomorrow.io/v4/weather/realtime", params=weather_params, timeout=5)
                w_res.raise_for_status()
                return w_res.json()['data']['values']

            current_weather = weather_breaker.call(fetch_weather)
        except Exception as we:
            logger.warning(f"Weather fetch failed: {we}. Using defaults.")

        # --- Step 2c: Calculate Distance ---
        distance_km = DEFAULT_DISTANCE
        try:
            if origin_airport in airport_coords_dict and destination_airport in airport_coords_dict:
                o_c = airport_coords_dict[origin_airport]
                d_c = airport_coords_dict[destination_airport]
                distance_km = haversine(o_c['Lat'], o_c['Lon'], d_c['Lat'], d_c['Lon'])
        except Exception as dist_err:
            logger.warning(f"Distance calculation failed: {dist_err}. Using default.")

        # --- Step 3: Feature Engineering ---
        dt = datetime.fromisoformat(scheduled_departure_str)
        features = {
            'MKT_UNIQUE_CARRIER': airline,
            'ORIGIN': origin_airport,
            'DEST': destination_airport,
            'DISTANCE': distance_km,
            'MONTH': dt.month,
            'DAY_OF_WEEK': dt.weekday(),
            'DEPT_HOUR': dt.hour,
            'tavg': current_weather.get('temperature', 15.0),
            'prcp': current_weather.get('precipitationIntensity', 0.0),
            'wspd': current_weather.get('windSpeed', 5.0)
        }

        # --- Step 4: ML Prediction ---
        features_df = pd.DataFrame([features])
        features_df = features_df[EXPECTED_FEATURE_ORDER]

        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = int(model_pipeline.predict(features_df)[0])

        final_response = {
            "flight_details_requested": flight_input.model_dump(),
            "live_weather_at_origin": current_weather,
            "calculated_distance_km": round(distance_km, 2),
            "predicted_delay_status": prediction,
            "predicted_delay_probability": f"{prediction_proba:.2%}",
            "is_cached": False,
            "timestamp": datetime.now().isoformat()
        }

        # --- Step 5: Save to Cache (30 mins TTL) ---
        if cache:
            try:
                cache.setex(cache_key, 1800, json.dumps(final_response))
            except Exception as cache_err:
                logger.warning(f"Cache write failed: {cache_err}")

        return final_response

    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except requests.exceptions.Timeout:
        logger.warning("External API request timed out, using defaults and returning prediction.")
        # Instead of failing, provide a prediction with default data
        features = {
            'MKT_UNIQUE_CARRIER': flight_input.carrierCode,
            'ORIGIN': 'JFK',
            'DEST': 'LAX',
            'DISTANCE': DEFAULT_DISTANCE,
            'MONTH': date.fromisoformat(flight_input.scheduledDepartureDate).month,
            'DAY_OF_WEEK': date.fromisoformat(flight_input.scheduledDepartureDate).weekday(),
            'DEPT_HOUR': 8,
            'tavg': 15.0,
            'prcp': 0.0,
            'wspd': 5.0
        }
        features_df = pd.DataFrame([features])
        features_df = features_df[EXPECTED_FEATURE_ORDER]
        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = int(model_pipeline.predict(features_df)[0])
        
        return {
            "flight_details_requested": flight_input.model_dump(),
            "live_weather_at_origin": DEFAULT_WEATHER,
            "calculated_distance_km": round(DEFAULT_DISTANCE, 2),
            "predicted_delay_status": prediction,
            "predicted_delay_probability": f"{prediction_proba:.2%}",
            "is_cached": False,
            "timestamp": datetime.now().isoformat(),
            "note": "Prediction based on defaults due to external service timeout"
        }
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error with external APIs, using defaults and returning prediction.")
        # Similar fallback for connection errors
        features = {
            'MKT_UNIQUE_CARRIER': flight_input.carrierCode,
            'ORIGIN': 'JFK',
            'DEST': 'LAX',
            'DISTANCE': DEFAULT_DISTANCE,
            'MONTH': date.fromisoformat(flight_input.scheduledDepartureDate).month,
            'DAY_OF_WEEK': date.fromisoformat(flight_input.scheduledDepartureDate).weekday(),
            'DEPT_HOUR': 8,
            'tavg': 15.0,
            'prcp': 0.0,
            'wspd': 5.0
        }
        features_df = pd.DataFrame([features])
        features_df = features_df[EXPECTED_FEATURE_ORDER]
        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = int(model_pipeline.predict(features_df)[0])
        
        return {
            "flight_details_requested": flight_input.model_dump(),
            "live_weather_at_origin": DEFAULT_WEATHER,
            "calculated_distance_km": round(DEFAULT_DISTANCE, 2),
            "predicted_delay_status": prediction,
            "predicted_delay_probability": f"{prediction_proba:.2%}",
            "is_cached": False,
            "timestamp": datetime.now().isoformat(),
            "note": "Prediction based on defaults due to connection error"
        }
    except requests.exceptions.HTTPError as he:
        if he.response.status_code == 429:
            logger.error("Rate limit exceeded on external API.")
            raise HTTPException(status_code=429, detail="Service is temporarily busy. Please try again later.")
        logger.warning(f"HTTP error from external API: {he}, using defaults and returning prediction.")
        # Provide prediction with defaults even on HTTP errors
        features = {
            'MKT_UNIQUE_CARRIER': flight_input.carrierCode,
            'ORIGIN': 'JFK',
            'DEST': 'LAX',
            'DISTANCE': DEFAULT_DISTANCE,
            'MONTH': date.fromisoformat(flight_input.scheduledDepartureDate).month,
            'DAY_OF_WEEK': date.fromisoformat(flight_input.scheduledDepartureDate).weekday(),
            'DEPT_HOUR': 8,
            'tavg': 15.0,
            'prcp': 0.0,
            'wspd': 5.0
        }
        features_df = pd.DataFrame([features])
        features_df = features_df[EXPECTED_FEATURE_ORDER]
        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = int(model_pipeline.predict(features_df)[0])
        
        return {
            "flight_details_requested": flight_input.model_dump(),
            "live_weather_at_origin": DEFAULT_WEATHER,
            "calculated_distance_km": round(DEFAULT_DISTANCE, 2),
            "predicted_delay_status": prediction,
            "predicted_delay_probability": f"{prediction_proba:.2%}",
            "is_cached": False,
            "timestamp": datetime.now().isoformat(),
            "note": "Prediction based on defaults due to external API error"
        }
    except Exception as e:
        logger.error(f"Prediction pipeline error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")


# --- 11. Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
