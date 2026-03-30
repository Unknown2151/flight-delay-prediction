"""
Configuration Management for Flight Delay Predictor API

This module provides centralized configuration management for the application,
with support for different environments (development, testing, production).

Environment Variables:
    - ENVIRONMENT: Deployment environment (development, testing, production)
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - AMADEUS_API_KEY: Amadeus API key
    - AMADEUS_API_SECRET: Amadeus API secret
    - TOMORROW_API_KEY: Tomorrow.io API key
    - REDIS_URL: Redis connection URL
    - PORT: Server port (default: 8000)

Usage:
    from config import settings
    
    print(settings.ENVIRONMENT)
    print(settings.REDIS_URL)
    print(settings.LOG_LEVEL)
"""

import os
from enum import Enum
from typing import Optional
from functools import lru_cache


class Environment(str, Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings:
    """Application settings with environment variable support."""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", Environment.DEVELOPMENT)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = ENVIRONMENT == Environment.DEVELOPMENT
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # API Keys (External Services)
    AMADEUS_API_KEY: str = os.getenv("AMADEUS_API_KEY", "")
    AMADEUS_API_SECRET: str = os.getenv("AMADEUS_API_SECRET", "")
    TOMORROW_API_KEY: str = os.getenv("TOMORROW_API_KEY", "")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    REDIS_SOCKET_KEEPALIVE: bool = True
    REDIS_HEALTH_CHECK_INTERVAL: int = 30
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = 1800  # 30 minutes
    
    # Model Configuration
    MODEL_PATH: str = "artifacts/flight_delay_pipeline.pkl"
    AIRPORT_DATA_URL: str = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    
    # API Client Configuration
    HTTP_TIMEOUT_AMADEUS: int = 8
    HTTP_TIMEOUT_WEATHER: int = 5
    HTTP_POOL_CONNECTIONS: int = 20
    HTTP_POOL_MAXSIZE: int = 20
    HTTP_RETRIES_TOTAL: int = 2
    HTTP_RETRIES_BACKOFF_FACTOR: float = 0.5
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60
    
    # Default Values for Fallback
    DEFAULT_WEATHER: dict = {
        'temperature': 15.0,
        'windSpeed': 5.0,
        'precipitationIntensity': 0.0
    }
    DEFAULT_DISTANCE_KM: float = 1000.0
    DEFAULT_ORIGIN_AIRPORT: str = "JFK"
    DEFAULT_DESTINATION_AIRPORT: str = "LAX"
    DEFAULT_HOUR: int = 8
    
    # CORS Configuration
    CORS_ORIGINS: list = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # API Metadata
    API_TITLE: str = "Flight Delay Predictor API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = (
        "Enterprise-grade ML API for predicting US airline flight delays "
        "with Redis caching and circuit breaker patterns"
    )
    
    # Gunicorn Workers Configuration
    # Formula: (2 × CPU cores) + 1
    # This is calculated at runtime by gunicorn_conf.py
    GUNICORN_WORKERS: Optional[int] = None
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment."""
        return cls.ENVIRONMENT == Environment.DEVELOPMENT
    
    @classmethod
    def is_testing(cls) -> bool:
        """Check if running in testing environment."""
        return cls.ENVIRONMENT == Environment.TESTING
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.ENVIRONMENT == Environment.PRODUCTION
    
    @classmethod
    def get_config(cls) -> 'Settings':
        """Get the current configuration instance."""
        return cls
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration for current environment.
        
        Returns:
            bool: True if valid, raises Exception if invalid
            
        Raises:
            Exception: If required environment variables are missing
        """
        if not cls.is_testing():
            # Production/Development require API keys
            if not cls.AMADEUS_API_KEY:
                raise Exception("AMADEUS_API_KEY environment variable is required")
            if not cls.AMADEUS_API_SECRET:
                raise Exception("AMADEUS_API_SECRET environment variable is required")
            if not cls.TOMORROW_API_KEY:
                raise Exception("TOMORROW_API_KEY environment variable is required")
        
        return True


# Global settings instance
settings = Settings()

# Configuration by environment
ENV_CONFIGS = {
    Environment.DEVELOPMENT: {
        "DEBUG": True,
        "LOG_LEVEL": "DEBUG",
        "REDIS_URL": "redis://localhost:6379",
    },
    Environment.TESTING: {
        "DEBUG": True,
        "LOG_LEVEL": "DEBUG",
        "REDIS_URL": "redis://localhost:6379",
    },
    Environment.PRODUCTION: {
        "DEBUG": False,
        "LOG_LEVEL": "INFO",
        # REDIS_URL should come from environment variables
    },
}


def get_environment_config() -> dict:
    """Get configuration for current environment."""
    return ENV_CONFIGS.get(settings.ENVIRONMENT, ENV_CONFIGS[Environment.DEVELOPMENT])


# Example usage and health check
if __name__ == "__main__":
    print("Flight Delay Predictor API - Configuration Report")
    print("=" * 60)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"API Port: {settings.PORT}")
    print(f"Redis URL: {settings.REDIS_URL}")
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Cache TTL: {settings.CACHE_TTL_SECONDS}s")
    print("=" * 60)
    
    try:
        settings.validate()
        print("✅ Configuration valid for environment:", settings.ENVIRONMENT)
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
