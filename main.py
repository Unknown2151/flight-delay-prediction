"""
FastAPI application for predicting US airline flight delays.

This API takes flight details, fetches relevant live data (flight schedule, weather),
calculates features, and uses a pre-trained LightGBM model pipeline to predict
the probability of the flight being delayed by more than 15 minutes.
"""

import os
import joblib
import pandas as pd
import requests
import numpy as np
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any

# --- Configuration & Initialization ---

# Load environment variables (API keys) from a .env file
load_dotenv()
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
WEATHER_API_KEY = os.getenv("TOMORROW_API_KEY")

# Default placeholder weather values if the live API fails
DEFAULT_WEATHER = {'temperature': 15.0, 'precipitationIntensity': 0.0, 'windSpeed': 5.0}
DEFAULT_DISTANCE = 1000.0  # Default distance in km

# --- Load Model and Airport Data ---

try:
    # Load the pre-trained model pipeline (preprocessing + classifier)
    model_pipeline = joblib.load('artifacts/flight_delay_pipeline.pkl')
    # Get the expected feature order from the loaded pipeline
    EXPECTED_FEATURE_ORDER = model_pipeline.feature_names_in_
except FileNotFoundError:
    raise RuntimeError("Model file 'artifacts/flight_delay_pipeline.pkl' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading model pipeline: {e}")

airport_coords_dict: Dict[str, Dict[str, float]] = {}
try:
    # Load airport coordinate data for distance calculation
    url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    airport_cols = ['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude',
                    'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source']
    df_airports_raw = pd.read_csv(url, header=None, names=airport_cols)
    # Handle potential duplicate IATA codes, keeping the first entry
    df_airports = df_airports_raw.drop_duplicates(subset=['IATA'], keep='first').set_index('IATA')
    airport_coords_dict = df_airports[['Latitude', 'Longitude']].to_dict('index')
    print("Airport coordinates loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load airport coordinates. Distance calculation will use placeholder. Error: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(title="Flight Delay Prediction API v3.0")


# --- Helper Functions ---

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def get_amadeus_access_token() -> str:
    """Authenticates with Amadeus and returns an access token."""
    token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET,
    }
    try:
        response = requests.post(token_url, data=token_data, timeout=10)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Amadeus Auth Error: {e}")
        raise HTTPException(status_code=503, detail=f"Amadeus authentication error: {e}")


# --- API Input Model ---

class FlightInput(BaseModel):
    carrierCode: str
    flightNumber: str
    scheduledDepartureDate: str  # Format: YYYY-MM-DD


# --- API Endpoints ---

@app.get("/info")
def get_model_info():
    """Returns information about the deployed model."""
    # In a real MLOps system, this info might be loaded from a config file
    # or stored alongside the model artifact.
    return {
        "model_description": "LightGBM Classifier for Flight Delay Prediction",
        "model_version": "2.0 (trained with weather data)", # Update this if you retrain
        "training_date": "2025-10-23", # Replace with the actual date you trained v2
        "sklearn_version_trained": "1.6.1", # The version used for training
        "input_features_expected": list(EXPECTED_FEATURE_ORDER)
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Flight Delay Prediction API v3.0!"}

@app.post("/predict")
def predict_delay(flight_input: FlightInput) -> Dict[str, Any]:
    """
    Predicts flight delay probability based on flight details.
    Fetches flight schedule and current weather, calculates features,
    and returns the prediction from the loaded model.
    """
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

        if not flight_response.get("data") or len(flight_response.get("data")) == 0:
            raise HTTPException(status_code=404,
                                detail="Flight schedule not found with Amadeus API for the given details.")

        flight_data = flight_response["data"][0]

        airline = flight_data["flightDesignator"]["carrierCode"]
        origin_airport = flight_data["flightPoints"][0]["iataCode"]
        destination_airport = flight_data["flightPoints"][1]["iataCode"]
        scheduled_departure_str = flight_data["flightPoints"][0]["departure"]["timings"][0]["value"]

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Amadeus API communication error: {e}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing Amadeus response: {e}\nResponse: {flight_response}")
        raise HTTPException(status_code=500, detail=f"Error parsing Amadeus response structure: {e}")

    # --- 2. Fetch Weather Data (Tomorrow.io) ---
    weather_params = {
        "location": origin_airport,
        "fields": ["temperature", "windSpeed", "precipitationIntensity"],
        "units": "metric",
        "timesteps": "current",
        "apikey": WEATHER_API_KEY
    }
    current_weather = DEFAULT_WEATHER.copy()  # Start with defaults
    try:
        weather_result = requests.get("https://api.tomorrow.io/v4/weather/realtime", params=weather_params, timeout=10)
        weather_result.raise_for_status()
        weather_response = weather_result.json()
        current_weather = weather_response['data']['values']
        print(f"Fetched weather for {origin_airport}: {current_weather}")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Weather API error: {e}. Using placeholder weather.")
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Error parsing weather response: {e}. Using placeholder weather.")

    # --- 3. Calculate Distance ---
    distance_km = DEFAULT_DISTANCE
    try:
        origin_coords = airport_coords_dict[origin_airport]
        dest_coords = airport_coords_dict[destination_airport]
        distance_km = haversine(origin_coords['Latitude'], origin_coords['Longitude'],
                                dest_coords['Latitude'], dest_coords['Longitude'])
        print(f"Calculated distance {origin_airport}-{destination_airport}: {distance_km:.2f} km")
    except KeyError:
        print(
            f"Warning: Could not find coordinates for {origin_airport} or {destination_airport}. Using placeholder distance.")
    except Exception as e:
        print(f"Warning: Error calculating distance: {e}. Using placeholder distance.")

    # --- 4. Feature Engineering & Prediction ---
    try:
        scheduled_departure_dt = datetime.fromisoformat(scheduled_departure_str)

        features = {
            'MKT_UNIQUE_CARRIER': airline,
            'ORIGIN': origin_airport,
            'DEST': destination_airport,
            'DISTANCE': distance_km,
            'MONTH': scheduled_departure_dt.month,
            'DAY_OF_WEEK': scheduled_departure_dt.weekday(),
            'DEPT_HOUR': scheduled_departure_dt.hour,
            'tavg': current_weather.get('temperature', DEFAULT_WEATHER['temperature']),
            'prcp': current_weather.get('precipitationIntensity', DEFAULT_WEATHER['precipitationIntensity']),
            'wspd': current_weather.get('windSpeed', DEFAULT_WEATHER['windSpeed'])
        }

        features_df = pd.DataFrame([features])

        # Ensure columns are in the exact order the model expects
        features_df = features_df[EXPECTED_FEATURE_ORDER]

        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]  # Probability of class 1 (delay)
        prediction = model_pipeline.predict(features_df)[0]  # Prediction (0 or 1)

    except Exception as e:
        print(f"Error during prediction pipeline: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during prediction pipeline: {e}")

    # --- 5. Format and Return Response ---
    return {
        "flight_details_requested": flight_input.dict(),
        "live_weather_at_origin": current_weather,
        "calculated_distance_km": round(distance_km, 2),
        "model_input_features": features,
        "predicted_delay_status": int(prediction),  # 0 = On Time, 1 = Delayed
        "predicted_delay_probability": f"{prediction_proba:.2%}"
    }
