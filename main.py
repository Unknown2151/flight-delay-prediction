import os
import joblib
import pandas as pd
import requests
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from typing import Dict, Any

# --- 1. Setup Logging & Environment ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FlightAPI")

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
WEATHER_API_KEY = os.getenv("TOMORROW_API_KEY")

# Constants
DEFAULT_WEATHER = {'temperature': 15.0, 'windSpeed': 5.0, 'precipitationIntensity': 0.0}
DEFAULT_DISTANCE = 1000.0

# --- 2. Load Resources (Global) ---
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

# --- 3. Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371

def get_amadeus_access_token():
    token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {"grant_type": "client_credentials", "client_id": AMADEUS_API_KEY, "client_secret": AMADEUS_API_SECRET}
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

# --- 4. FastAPI Setup ---
app = FastAPI(title="Flight Delay Prediction API v3.0")

class FlightInput(BaseModel):
    carrierCode: str
    flightNumber: str
    scheduledDepartureDate: str

# --- 5. Monitoring Endpoints ---

@app.get("/health")
def health_check():
    """Returns the health status of the API."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/info")
def get_info():
    """Returns metadata about the deployed model."""
    return {
        "version": "3.0",
        "model_type": "LightGBM",
        "features": list(EXPECTED_FEATURE_ORDER),
        "python_version": "3.13" # Or your current version
    }

# --- 6. Prediction Endpoint ---

@app.post("/predict")
def predict_delay(flight_input: FlightInput) -> Dict[str, Any]:
    # Log the incoming request details
    logger.info(f"New prediction request: {flight_input.carrierCode}{flight_input.flightNumber} on {flight_input.scheduledDepartureDate}")
    
    access_token = get_amadeus_access_token()

    # --- 1. Fetch Flight Data (Amadeus) ---
    flight_status_url = "https://test.api.amadeus.com/v2/schedule/flights"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "carrierCode": flight_input.carrierCode,
        "flightNumber": flight_input.flightNumber,
        "scheduledDepartureDate": flight_input.scheduledDepartureDate,
    }
    try:
        api_result = requests.get(flight_status_url, headers=headers, params=params, timeout=15)
        api_result.raise_for_status()
        flight_response = api_result.json()

        if not flight_response.get("data"):
            logger.warning(f"Flight {flight_input.flightNumber} not found.")
            raise HTTPException(status_code=404, detail="Flight schedule not found.")

        flight_data = flight_response["data"][0]
        airline = flight_data["flightDesignator"]["carrierCode"]
        origin_airport = flight_data["flightPoints"][0]["iataCode"]
        destination_airport = flight_data["flightPoints"][1]["iataCode"]
        scheduled_departure_str = flight_data["flightPoints"][0]["departure"]["timings"][0]["value"]

    except Exception as e:
        logger.error(f"Amadeus Error: {e}")
        raise HTTPException(status_code=500, detail="External Flight API Error")

    # --- 2. Fetch Weather Data (Tomorrow.io) ---
    weather_params = {
        "location": origin_airport,
        "fields": ["temperature", "windSpeed", "precipitationIntensity"],
        "units": "metric",
        "timesteps": "current",
        "apikey": WEATHER_API_KEY
    }
    current_weather = DEFAULT_WEATHER.copy()
    try:
        weather_result = requests.get("https://api.tomorrow.io/v4/weather/realtime", params=weather_params, timeout=10)
        weather_result.raise_for_status()
        current_weather = weather_result.json()['data']['values']
    except Exception as e:
        logger.warning(f"Weather Fetch failed: {e}. Using defaults.")

    # --- 3. Calculate Distance ---
    distance_km = DEFAULT_DISTANCE
    try:
        origin_coords = airport_coords_dict[origin_airport]
        dest_coords = airport_coords_dict[destination_airport]
        distance_km = haversine(origin_coords['Lat'], origin_coords['Lon'],
                                dest_coords['Lat'], dest_coords['Lon'])
    except Exception as e:
        logger.warning(f"Distance calculation failed: {e}")

    # --- 4. Feature Engineering & Prediction ---
    try:
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

        features_df = pd.DataFrame([features])
        features_df = features_df[EXPECTED_FEATURE_ORDER]

        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = model_pipeline.predict(features_df)[0]
        
        # Log successful prediction
        logger.info(f"Result for {airline}{flight_input.flightNumber}: Delay Prob {prediction_proba:.2%}")

    except Exception as e:
        logger.error(f"Pipeline Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Prediction Pipeline Failure")

    return {
        "flight_details_requested": flight_input.dict(),
        "live_weather_at_origin": current_weather,
        "calculated_distance_km": round(distance_km, 2),
        "model_input_features": features,
        "predicted_delay_status": int(prediction),
        "predicted_delay_probability": f"{prediction_proba:.2%}"
    }

@app.get("/")
def read_root():
    return {"message": "Flight Delay Prediction API v3.0 is Online"}