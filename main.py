import os
import joblib
import pandas as pd
import requests
import logging
import traceback
import json
import redis
import numpy as np
from datetime import datetime, date
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# --- 1. Setup Logging & Secret Masking ---
load_dotenv()


class SensitiveDataFilter(logging.Filter):
    """Prevents API keys from leaking into logs."""

    def __init__(self, sensitive_keys: list):
        super().__init__()
        self.sensitive_keys = [k for k in sensitive_keys if k]

    def filter(self, record):
        message = record.getMessage()
        for key in self.sensitive_keys:
            if key in message:
                record.msg = message.replace(key, "********")
        return True


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FlightAPI")

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
WEATHER_API_KEY = os.getenv("TOMORROW_API_KEY")

logger.addFilter(SensitiveDataFilter([AMADEUS_API_KEY, AMADEUS_API_SECRET, WEATHER_API_KEY]))

# --- 2. Redis Connection ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    cache = redis.from_url(REDIS_URL, decode_responses=True)
    cache.ping()
    logger.info("Connected to Redis successfully.")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
    cache = None

# --- 3. Resource Loading (Global) ---
DEFAULT_WEATHER = {'temperature': 15.0, 'windSpeed': 5.0, 'precipitationIntensity': 0.0}
DEFAULT_DISTANCE = 1000.0

airport_coords_dict = {}
try:
    url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    cols = ['ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Lat', 'Lon', 'Alt', 'TZ', 'DST', 'TzDB', 'Type', 'Source']
    df_airports = pd.read_csv(url, header=None, names=cols)
    df_airports = df_airports.drop_duplicates(subset=['IATA'], keep='first').set_index('IATA')
    airport_coords_dict = df_airports[['Lat', 'Lon']].to_dict('index')
    logger.info("Airport coordinates loaded.")
except Exception as e:
    logger.error(f"Failed to load airport data: {e}")

try:
    model_pipeline = joblib.load('artifacts/flight_delay_pipeline.pkl')
    EXPECTED_FEATURE_ORDER = model_pipeline.feature_names_in_
    logger.info("ML Model loaded.")
except Exception as e:
    logger.critical(f"Failed to load model: {e}")
    raise RuntimeError(e)


# --- 4. Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def get_amadeus_access_token():
    token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {"grant_type": "client_credentials", "client_id": AMADEUS_API_KEY, "client_secret": AMADEUS_API_SECRET}
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


# --- 5. Data Models & Validation ---
class FlightInput(BaseModel):
    carrierCode: str = Field(..., min_length=2, max_length=3, pattern=r"^[A-Z]+$",
                             description="Airline code (e.g., AA)")
    flightNumber: str = Field(..., pattern=r"^\d{1,4}$", description="Flight number digits only")
    scheduledDepartureDate: str = Field(..., description="Date in YYYY-MM-DD")

    @field_validator("scheduledDepartureDate")
    @classmethod
    def validate_date(cls, v):
        try:
            # 1. First, try to parse the date
            if isinstance(v, str):
                parsed_date = date.fromisoformat(v)
            else:
                parsed_date = v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

            # 2. Then, check if it's in the past
        if parsed_date < date.today():
            raise ValueError("Departure date cannot be in the past")
        return v


# --- 6. FastAPI Setup ---
app = FastAPI(
    title="Flight Delay Predictor API",
    description="An API to predict flight delays based on carrier and schedule.",
    version="1.0.0"
)

@app.get("/info")
async def get_info():
    return {
        "app_name": "Flight Delay Predictor",
        "version": "1.0.0",
        "description": "Predicts if your flight will be delayed."
    }

@app.get("/", tags=["Public"])
def read_root():
    return {"message": "Flight Delay Prediction API is Online"}


@app.get("/health", tags=["Monitoring"])
def health_check():
    redis_status = "connected" if cache and cache.ping() else "disconnected"
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
        - **carrierCode**: e.g., AA, LH, DL
        - **flightNumber**: e.g., 123
    """

    # --- Step 1: Cache Check ---
    cache_key = f"predict:{flight_input.carrierCode}:{flight_input.flightNumber}:{flight_input.scheduledDepartureDate}"
    if cache:
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.info(f"CACHE HIT: Returning cached result for {cache_key}")
            result = json.loads(cached_data)
            result["is_cached"] = True
            return result

    logger.info(f"CACHE MISS: Processing request for {flight_input.carrierCode}{flight_input.flightNumber}")

    # --- Step 2: Fetch Data ---
    try:
        access_token = get_amadeus_access_token()

        # Amadeus Call
        flight_status_url = "https://test.api.amadeus.com/v2/schedule/flights"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = flight_input.model_dump()

        api_result = requests.get(flight_status_url, headers=headers, params=params, timeout=15)
        api_result.raise_for_status()
        flight_response = api_result.json()

        if not flight_response.get("data"):
            raise HTTPException(status_code=404, detail="Flight schedule not found in Amadeus.")

        flight_data = flight_response["data"][0]
        airline = flight_data["flightDesignator"]["carrierCode"]
        origin_airport = flight_data["flightPoints"][0]["iataCode"]
        destination_airport = flight_data["flightPoints"][1]["iataCode"]
        scheduled_departure_str = flight_data["flightPoints"][0]["departure"]["timings"][0]["value"]

        # Weather Call (Tomorrow.io)
        weather_params = {
            "location": origin_airport,
            "fields": ["temperature", "windSpeed", "precipitationIntensity"],
            "units": "metric",
            "timesteps": "current",
            "apikey": WEATHER_API_KEY
        }
        current_weather = DEFAULT_WEATHER.copy()
        try:
            w_res = requests.get("https://api.tomorrow.io/v4/weather/realtime", params=weather_params, timeout=10)
            w_res.raise_for_status()
            current_weather = w_res.json()['data']['values']
        except Exception as we:
            logger.warning(f"Weather fetch failed: {we}. Using defaults.")

        # Distance Calculation
        distance_km = DEFAULT_DISTANCE
        try:
            o_c = airport_coords_dict[origin_airport]
            d_c = airport_coords_dict[destination_airport]
            distance_km = haversine(o_c['Lat'], o_c['Lon'], d_c['Lat'], d_c['Lon'])
        except Exception:
            logger.warning("Distance calculation failed. Using default.")

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
        prediction = model_pipeline.predict(features_df)[0]

        final_response = {
            "flight_details_requested": flight_input.model_dump(),
            "live_weather_at_origin": current_weather,
            "calculated_distance_km": round(distance_km, 2),
            "predicted_delay_status": int(prediction),
            "predicted_delay_probability": f"{prediction_proba:.2%}",
            "is_cached": False,
            "timestamp": datetime.now().isoformat()
        }

        # --- Step 5: Save to Cache (30 mins) ---
        if cache:
            cache.setex(cache_key, 1800, json.dumps(final_response))

        return final_response

    except HTTPException as he:
        # Let FastAPI handle our custom 404 or 429 errors directly
        raise he
    except Exception as e:
        # Only catch unexpected system/logic crashes here
        logger.error(f"Prediction Failure: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Prediction Pipeline Error")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error("Rate limit hit on external API.")
            raise HTTPException(status_code=429, detail="Service busy. Please try again later.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction Failure: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Prediction Pipeline Error")